"""Approach 1: one hierarchical DG space of degree pmax, restricted algebraically.

The orthonormal (Legendre) DG basis of Basix is hierarchical on simplices: the first
dim(P_k) basis functions of the degree k+1 element span P_k. A space with cell-wise degree p_K
is therefore the subset of dofs ``dofmap[K, :dim(P_{p_K})]`` of the degree pmax space, and the
matrices are sub-matrices of the matrices assembled for degree pmax.
"""

from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
from dolfinx.fem import create_form

from ..degrees import DegreeField
from ..forms import Parameters, Penalty, compile_single_forms
from ..problems import Problem
from .base import Approach


class HierarchicalApproach(Approach):
    name = "hierarchical"

    def __init__(self, degrees: DegreeField, params: Parameters, problem: Problem | None = None):
        if params.continuous:
            raise NotImplementedError(
                "The hierarchical approach is restricted to DG: Basix has no hierarchical "
                "continuous Lagrange basis, and a continuous fine space cannot be broken at the "
                "(moving) interfaces between degrees by selecting rows and columns."
            )
        super().__init__(degrees, params, problem)
        fields = self.problem.fields
        if len(fields) != 1 or fields[0].shape != ():
            raise NotImplementedError(
                "The hierarchical approach is only implemented for scalar problems, use the "
                "submesh approach"
            )
        cell = self.mesh.basix_cell()
        if cell not in (
            basix.CellType.interval,
            basix.CellType.triangle,
            basix.CellType.tetrahedron,
        ):
            raise NotImplementedError(
                "Legendre DG is only hierarchical in this ordering on simplices"
            )
        element = basix.ufl.element(
            "DG", cell, params.pmax, basix.LagrangeVariant.legendre, discontinuous=True
        )
        self.V = dolfinx.fem.functionspace(self.mesh, element)
        self.uh = dolfinx.fem.Function(self.V, name="u")
        # Number of basis functions for each degree
        self.dims = np.array(
            [
                basix.create_element(
                    basix.ElementFamily.P,
                    cell,
                    k,
                    basix.LagrangeVariant.legendre,
                    discontinuous=True,
                ).dim
                if k >= params.pmin
                else 0
                for k in range(params.pmax + 1)
            ],
            dtype=np.int32,
        )

        compiled = compile_single_forms(self.V, self.h, degrees.p, params, self.problem)
        coeffs = {self.h: self.h, degrees.p: degrees.p, compiled.u: self.uh}
        consts = self._constants
        rhs_spaces = [self.V, self.V] if self.problem.is_eigenvalue_problem else [self.V]
        self.a = create_form(compiled.a, [self.V, self.V], self.mesh, {}, coeffs, consts)
        self.rhs = create_form(compiled.rhs, rhs_spaces, self.mesh, {}, coeffs, consts)
        self.eta = create_form(compiled.eta, [self.h.function_space], self.mesh, {}, coeffs, consts)
        self.error_forms = {
            name: [create_form(form, [], self.mesh, {}, coeffs, consts)]
            for name, form in compiled.errors.items()
        }
        self.A_full = None
        self.B_full = None
        self._is = None

    def update(self):
        num_cells = len(self.degrees.cell_degrees)
        dofmap = self.V.dofmap.list[:num_cells]
        keep = np.arange(dofmap.shape[1])[None, :] < self.dims[self.degrees.cell_degrees][:, None]
        dofs = np.sort(dofmap[keep]).astype(np.int32)
        imap = self.V.dofmap.index_map
        self.sub_map, self.sub_to_parent, owners_changed = dolfinx.common.create_sub_index_map(
            imap, dofs
        )
        # DG dofs are owned by the owner of the cell, and the degrees are consistent on ghosts
        assert not self.mesh.comm.allreduce(owners_changed, op=MPI.LOR)
        owned = self.sub_to_parent[: self.sub_map.size_local]
        if self._is is not None:
            self._is.destroy()
        self._is = PETSc.IS().createGeneral(
            imap.local_to_global(owned).astype(PETSc.IntType), comm=self.mesh.comm
        )
        if self.params.penalty == Penalty.local and self.A_full is not None:
            # The penalty depends on the degrees, so the fine matrix (and the Nitsche terms of
            # the load vector) must be reassembled
            self.A_full.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self.A_full, self.a)
            self.A_full.assemble()
            if not self.problem.is_eigenvalue_problem:
                self._assemble_load_vector()

    def _assemble_load_vector(self):
        with self.B_full.localForm() as b:
            b.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self.B_full, self.rhs)
        self.B_full.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def assemble(self):
        if self.A_full is None:
            self.A_full = dolfinx.fem.petsc.assemble_matrix(self.a)
            self.A_full.assemble()
            if self.problem.is_eigenvalue_problem:
                self.B_full = dolfinx.fem.petsc.assemble_matrix(self.rhs)
                self.B_full.assemble()
            else:
                self.B_full = dolfinx.fem.petsc.create_vector(self.V)
                self._assemble_load_vector()
        A = self.A_full.createSubMatrix(self._is, self._is)
        if self.problem.is_eigenvalue_problem:
            B = self.B_full.createSubMatrix(self._is, self._is)
        else:
            sub = self.B_full.getSubVector(self._is)
            B = sub.copy()
            self.B_full.restoreSubVector(self._is, sub)
        return self._replace_system(A, B)

    def set_solution(self, x):
        self.uh.x.array[:] = 0
        self.uh.x.array[self.sub_to_parent[: self.sub_map.size_local]] = x.array_r
        self.uh.x.scatter_forward()

    def _estimator_forms(self):
        return [self.eta]

    def _error_forms(self):
        return self.error_forms

    def interpolate_solution(self, u, field=None):
        u.interpolate(self.uh)
        u.x.scatter_forward()

    @property
    def num_dofs(self):
        return self.sub_map.size_global
