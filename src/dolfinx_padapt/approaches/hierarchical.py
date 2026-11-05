"""Approach 1: one hierarchical DG space of degree pmax, restricted algebraically.

The space uses an orthonormal DG element whose basis functions are ordered by degree
(:func:`hierarchical_element`), such that the first ``dim(P_k)`` (on quadrilaterals and hexahedra
``dim(Q_k)``) basis functions span the space of degree k. A space with cell-wise degree p_K is
therefore the subset of dofs ``dofmap[K, :dim(P_{p_K})]`` of the degree pmax space, and the
matrices are sub-matrices of the matrices assembled for degree pmax.
"""

import itertools

from mpi4py import MPI
from petsc4py import PETSc

import basix
import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
from dolfinx.fem import create_form

from ..forms import Parameters, Penalty, compile_single_forms
from ..problems import Problem
from .base import Approach


def _polynomial_degrees(cell: basix.CellType, degree: int) -> npt.NDArray[np.int32]:
    """The degree of each function of the orthonormal polynomial set of Basix: the total degree
    on simplices, and the largest degree in each direction on quadrilaterals and hexahedra."""
    tdim = len(basix.cell.topology(cell)) - 1
    if cell in (basix.CellType.quadrilateral, basix.CellType.hexahedron):
        return np.array(
            [max(index) for index in itertools.product(range(degree + 1), repeat=tdim)],
            dtype=np.int32,
        )
    # On simplices, the set is ordered by degree
    dims = [
        basix.polynomials.dim(basix.PolynomialType.legendre, cell, k) for k in range(degree + 1)
    ]
    return np.searchsorted(dims, np.arange(dims[-1]), side="right").astype(np.int32)


def hierarchical_element(
    cell: basix.CellType, degree: int
) -> tuple[basix.ufl._ElementBase, npt.NDArray[np.int32]]:
    """An orthonormal discontinuous element whose basis functions are ordered by degree.

    The basis functions are the orthonormal polynomials of Basix (Dubiner polynomials on
    simplices and tensor products of Legendre polynomials on quadrilaterals and hexahedra), sorted
    by degree, such that the first ``dims[k]`` basis functions span the space of degree ``k``
    (``P_k`` on simplices, ``Q_k`` on quadrilaterals and hexahedra). The degrees of freedom are
    the moments against the basis functions.

    Args:
        cell: The cell type.
        degree: The degree of the element.

    Returns:
        The element, and the number of basis functions of degree at most ``k`` for each ``k``.
    """
    topology = basix.cell.topology(cell)
    tdim = len(topology) - 1
    points, weights = basix.make_quadrature(cell, 2 * degree)
    polynomials = basix.polynomials.tabulate_polynomial_set(
        cell, basix.PolysetType.standard, degree, 0, points
    )[0]
    degrees = _polynomial_degrees(cell, degree)
    order = np.argsort(degrees, kind="stable")
    num_dofs = len(order)
    wcoeffs = np.eye(num_dofs)[order]
    # All degrees of freedom are moments on the interior of the cell
    x = [[np.zeros((0, tdim)) for _ in topology[d]] for d in range(tdim)] + [[points]]
    M = [[np.zeros((0, 1, 0, 1)) for _ in topology[d]] for d in range(tdim)]
    M.append([(weights[None, :] * polynomials[order]).reshape(num_dofs, 1, len(weights), 1)])
    element = basix.ufl.custom_element(
        cell,
        [],
        wcoeffs,
        x,
        M,
        0,
        basix.MapType.identity,
        basix.SobolevSpace.L2,
        discontinuous=True,
        embedded_subdegree=degree,
        embedded_superdegree=degree,
    )
    dims = np.array([np.sum(degrees <= k) for k in range(degree + 1)], dtype=np.int32)
    return element, dims


class HierarchicalApproach(Approach):
    name = "hierarchical"

    def __init__(self, problem: Problem, params: Parameters):
        if params.continuous:
            raise NotImplementedError(
                "The hierarchical approach is restricted to DG: Basix has no hierarchical "
                "continuous Lagrange basis, and a continuous fine space cannot be broken at the "
                "(moving) interfaces between degrees by selecting rows and columns."
            )
        super().__init__(problem, params)
        fields = self._problem.fields
        if len(fields) != 1 or fields[0].shape != ():
            raise NotImplementedError(
                "The hierarchical approach is only implemented for scalar problems, use the "
                "submesh approach"
            )
        # The first dims[k] local dofs of a cell span the space of degree k
        self._element, self._dims = hierarchical_element(self._mesh.basix_cell(), params.pmax)
        self._V = dolfinx.fem.functionspace(self._mesh, self._element)
        # The cell degrees in the compiled forms (replaced by those of the current mesh)
        self._p_template = self._degrees.p
        self._compiled = compile_single_forms(
            self._V, self._h, self._p_template, params, self._problem
        )
        self._create_forms()

    def _mesh_changed(self):
        self._V = dolfinx.fem.functionspace(self._mesh, self._element)
        self._create_forms()

    def _create_forms(self):
        """Create the forms on the current mesh from the compiled forms."""
        compiled = self._compiled
        self._uh = dolfinx.fem.Function(self._V, name="u")
        coeffs = {
            self._h_template: self._h,
            self._p_template: self._degrees.p,
            compiled.u: self._uh,
        }
        consts = self._constants
        rhs_spaces = [self._V, self._V] if self._problem.is_eigenvalue_problem else [self._V]
        self._a = create_form(compiled.a, [self._V, self._V], self._mesh, {}, coeffs, consts)
        self._rhs = create_form(compiled.rhs, rhs_spaces, self._mesh, {}, coeffs, consts)
        self._eta = create_form(
            compiled.eta, [self._h.function_space], self._mesh, {}, coeffs, consts
        )
        self._error_functionals = {
            name: [create_form(form, [], self._mesh, {}, coeffs, consts)]
            for name, form in compiled.errors.items()
        }
        for petsc_object in ("_A_full", "_B_full", "_is"):
            if getattr(self, petsc_object, None) is not None:
                getattr(self, petsc_object).destroy()
        self._A_full = None
        self._B_full = None
        self._is = None

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        """The (hierarchical) space of the maximal degree."""
        return self._V

    @property
    def dims(self) -> npt.NDArray[np.int32]:
        """The number of local degrees of freedom of a cell of each degree."""
        return self._dims

    def update(self):
        num_cells = len(self._degrees.cell_degrees)
        dofmap = self._V.dofmap.list[:num_cells]
        keep = np.arange(dofmap.shape[1])[None, :] < self._dims[self._degrees.cell_degrees][:, None]
        dofs = np.sort(dofmap[keep]).astype(np.int32)
        imap = self._V.dofmap.index_map
        self._sub_map, self._sub_to_parent, owners_changed = dolfinx.common.create_sub_index_map(
            imap, dofs
        )
        # DG dofs are owned by the owner of the cell, and the degrees are consistent on ghosts
        assert not self._mesh.comm.allreduce(owners_changed, op=MPI.LOR)
        owned = self._sub_to_parent[: self._sub_map.size_local]
        if self._is is not None:
            self._is.destroy()
        self._is = PETSc.IS().createGeneral(
            imap.local_to_global(owned).astype(PETSc.IntType), comm=self._mesh.comm
        )
        if self._params.penalty == Penalty.local and self._A_full is not None:
            # The penalty depends on the degrees, so the fine matrix (and the Nitsche terms of
            # the load vector) must be reassembled
            self._A_full.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self._A_full, self._a)
            self._A_full.assemble()
            if not self._problem.is_eigenvalue_problem:
                self._assemble_load_vector()

    def _assemble_load_vector(self):
        with self._B_full.localForm() as b:
            b.set(0.0)
        dolfinx.fem.petsc.assemble_vector(self._B_full, self._rhs)
        self._B_full.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def assemble(self):
        if self._A_full is None:
            self._A_full = dolfinx.fem.petsc.assemble_matrix(self._a)
            self._A_full.assemble()
            if self._problem.is_eigenvalue_problem:
                self._B_full = dolfinx.fem.petsc.assemble_matrix(self._rhs)
                self._B_full.assemble()
            else:
                self._B_full = dolfinx.fem.petsc.create_vector(self._V)
                self._assemble_load_vector()
        A = self._A_full.createSubMatrix(self._is, self._is)
        if self._problem.is_eigenvalue_problem:
            B = self._B_full.createSubMatrix(self._is, self._is)
        else:
            sub = self._B_full.getSubVector(self._is)
            B = sub.copy()
            self._B_full.restoreSubVector(self._is, sub)
        return self._replace_system(A, B)

    def set_solution(self, x):
        self._uh.x.array[:] = 0
        self._uh.x.array[self._sub_to_parent[: self._sub_map.size_local]] = x.array_r
        self._uh.x.scatter_forward()

    def _estimator_forms(self):
        return [self._eta]

    def _error_forms(self):
        return self._error_functionals

    def interpolate_solution(self, u, field=None):
        u.interpolate(self._uh)
        u.x.scatter_forward()

    @property
    def num_dofs(self):
        return self._sub_map.size_global
