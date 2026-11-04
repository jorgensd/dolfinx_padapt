"""Common interface for the P-adaptive approaches, and the shared block assembly of approaches
2 (restricted dofmaps) and 3 (submeshes)."""

import abc

from mpi4py import MPI
from petsc4py import PETSc

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.fem import create_form

from ..degrees import DegreeField
from ..forms import BlockForms, Parameters, cell_size, compile_block_forms
from ..problems import EigenvalueProblem, Problem


class Approach(abc.ABC):
    """A representation of a space with a cell-wise varying polynomial degree.

    Args:
        degrees: The degree of each cell.
        params: Discretisation parameters.
        problem: The problem to solve, by default the Laplace eigenvalue problem.
    """

    name: str

    def __init__(self, degrees: DegreeField, params: Parameters, problem: Problem | None = None):
        self.degrees = degrees
        self.mesh = degrees.mesh
        self.params = params
        self.problem = problem if problem is not None else EigenvalueProblem(self.mesh)
        self.h = cell_size(self.mesh)
        self.A: PETSc.Mat | None = None
        self.B: PETSc.Mat | PETSc.Vec | None = None

    @property
    def _constants(self) -> dict:
        return {c: c for c in self.problem.constants}

    @abc.abstractmethod
    def update(self):
        """Rebuild the discrete space for the current degree distribution."""

    @abc.abstractmethod
    def assemble(self) -> tuple[PETSc.Mat, PETSc.Mat | PETSc.Vec]:
        """Stiffness matrix and right hand side (mass matrix of an eigenvalue problem, load
        vector otherwise) for the current degree distribution."""

    @abc.abstractmethod
    def set_solution(self, x: PETSc.Vec):
        """Store the solution vector (in the layout of :meth:`assemble`)."""

    @abc.abstractmethod
    def _estimator_forms(self) -> list[dolfinx.fem.Form]:
        """Linear forms (tested with DG-0) whose sum is the squared error indicator."""

    @abc.abstractmethod
    def _error_forms(self) -> dict[str, list[dolfinx.fem.Form]]:
        """Functionals whose sum is each squared error measure of the problem."""

    @abc.abstractmethod
    def interpolate_solution(self, u: dolfinx.fem.Function, field: str | None = None):
        """Interpolate a field of the solution (by default the first) into ``u`` (a broken space
        on the parent mesh)."""

    def nullspace(self) -> PETSc.NullSpace | None:
        """The nullspace of the stiffness matrix (constants of the
        :attr:`~dolfinx_padapt.problems.Problem.nullspace_fields`), if any."""
        return None

    @property
    @abc.abstractmethod
    def num_dofs(self) -> int:
        """Global number of degrees of freedom."""

    def estimate(self) -> npt.NDArray[np.float64]:
        """Error indicator η_K for every owned cell (indexed by local cell index)."""
        Q = self.h.function_space
        eta2 = dolfinx.la.vector(Q.dofmap.index_map, dtype=dolfinx.default_scalar_type)
        for form in self._estimator_forms():
            eta2.array[:] += dolfinx.fem.assemble_vector(form).array
        eta2.scatter_reverse(dolfinx.la.InsertMode.add)
        num_owned = self.degrees.num_owned_cells
        return np.sqrt(np.abs(eta2.array[Q.dofmap.list[:num_owned, 0]]))

    def errors(self) -> dict[str, float]:
        """The global value of each squared error measure of the problem."""
        return {
            name: self.mesh.comm.allreduce(
                sum(dolfinx.fem.assemble_scalar(form) for form in forms), op=MPI.SUM
            )
            for name, forms in self._error_forms().items()
        }

    def _replace_system(self, A: PETSc.Mat, B: PETSc.Mat | PETSc.Vec):
        for old in (self.A, self.B):
            if old is not None:
                old.destroy()
        self.A, self.B = A, B
        return A, B


class BlockApproach(Approach):
    """One space per polynomial degree, coupled through interface integrals.

    All forms are compiled once (:func:`compile_block_forms`). Every update only creates
    spaces for the active degrees and calls ``create_form`` with the new spaces and integration
    entities.
    """

    def __init__(self, degrees: DegreeField, params: Parameters, problem: Problem | None = None):
        super().__init__(degrees, params, problem)
        cell = self.mesh.basix_cell()
        family = "Lagrange" if params.continuous else "DG"
        self.elements = {}
        for i in params.degrees:
            for field in self.problem.fields:
                degree = field.degree(i)
                if degree < 0 or (params.continuous and degree < 1):
                    raise ValueError(f"Degree {degree} of field {field.name} is not supported")
                self.elements[(i, field.name)] = basix.ufl.element(
                    family, cell, degree, shape=field.shape, discontinuous=not params.continuous
                )
        self.forms: BlockForms = compile_block_forms(
            self.mesh, self._ufl_spaces(), self.h, params, self.problem
        )
        self.active: list[int] = []

    @property
    def _field_names(self) -> list[str]:
        return [field.name for field in self.problem.fields]

    @property
    def _constants(self) -> dict:
        return {c: c for c in self.problem.constants + self.forms.constants}

    @abc.abstractmethod
    def _ufl_spaces(self) -> dict[tuple[int, str], ufl.FunctionSpace]:
        """UFL spaces used to compile the forms, for each ``(degree, field)``."""

    @abc.abstractmethod
    def _create_spaces(
        self, active: list[int]
    ) -> tuple[dict[tuple[int, str], dolfinx.fem.FunctionSpace], list[dolfinx.mesh.EntityMap]]:
        """Function spaces of each field for the active degrees, and the entity maps needed to
        assemble them over the parent mesh."""

    def update(self):
        self.active = self.degrees.active_degrees()
        entities = self.degrees.integration_entities(self.params.continuous)
        self.spaces, self.entity_maps = self._create_spaces(self.active)
        #: The blocks of the system: the fields of each active degree
        self.blocks = [(i, name) for i in self.active for name in self._field_names]
        self.uh = {
            key: dolfinx.fem.Function(self.spaces[key], name=f"{key[1]}_{key[0]}")
            for key in self.blocks
        }
        coeffs = {self.h: self.h} | {self.forms.u[key]: self.uh[key] for key in self.blocks}
        consts = self._constants

        def make(compiled, spaces):
            if compiled is None:
                return None
            return create_form(
                compiled, spaces, self.mesh, entities, coeffs, consts, self.entity_maps
            )

        index = self.forms.index
        self.a = [
            [
                make(self.forms.a[index(r)][index(c)], [self.spaces[r], self.spaces[c]])
                for c in self.blocks
            ]
            for r in self.blocks
        ]
        if self.problem.is_eigenvalue_problem:
            self.rhs = [
                [
                    make(self.forms.rhs[index(r)], [self.spaces[r], self.spaces[r]])
                    if r == c
                    else None
                    for c in self.blocks
                ]
                for r in self.blocks
            ]
        else:
            self.rhs = [make(self.forms.rhs[index(r)], [self.spaces[r]]) for r in self.blocks]
        Q = self.h.function_space
        self.eta = [
            make(compiled, [Q])
            for key, compiled in self.forms.eta.items()
            if all(d in self.active for d in key)
        ]
        self.error_forms: dict[str, list[dolfinx.fem.Form]] = {}
        for i in self.active:
            for name, compiled in self.forms.errors[i].items():
                self.error_forms.setdefault(name, []).append(make(compiled, []))

    def assemble(self):
        A = dolfinx.fem.petsc.assemble_matrix(self.a)
        A.assemble()
        if self.problem.is_eigenvalue_problem:
            B = dolfinx.fem.petsc.assemble_matrix(self.rhs)
            B.assemble()
        else:
            # Owned entries of each block stacked (as the matrix), then the ghosts.
            # A single form is not assembled as a blocked vector.
            rhs = self.rhs[0] if len(self.rhs) == 1 else self.rhs
            B = dolfinx.fem.petsc.assemble_vector(rhs)
            B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return self._replace_system(A, B)

    def _block_sizes(self) -> list[int]:
        """Number of owned entries of each block in the monolithic layout."""
        return [
            self.spaces[key].dofmap.index_map.size_local * self.spaces[key].dofmap.index_map_bs
            for key in self.blocks
        ]

    def set_solution(self, x):
        # Monolithic block layout: the owned dofs of each block, stacked per process
        values = x.array_r
        start = 0
        for key, n in zip(self.blocks, self._block_sizes()):
            uh = self.uh[key]
            uh.x.array[:n] = values[start : start + n]
            uh.x.scatter_forward()
            start += n
        assert start == len(values)

    def nullspace(self):
        if not self.problem.nullspace_fields:
            return None
        assert self.A is not None
        vector = self.A.createVecRight()
        values = np.zeros(vector.getLocalSize(), dtype=dolfinx.default_scalar_type)
        start = 0
        for (_, name), n in zip(self.blocks, self._block_sizes()):
            values[start : start + n] = 1.0 if name in self.problem.nullspace_fields else 0.0
            start += n
        vector.setArray(values)
        vector.normalize()
        return PETSc.NullSpace().create(vectors=[vector], comm=self.mesh.comm)

    def _estimator_forms(self):
        return self.eta

    def _error_forms(self):
        return self.error_forms

    @property
    def num_dofs(self):
        return sum(
            V.dofmap.index_map.size_global * V.dofmap.index_map_bs for V in self.spaces.values()
        )

    def interpolate_solution(self, u, field=None):
        field = field or self._field_names[0]
        for i in self.active:
            self._interpolate(u, self.uh[(i, field)], i)
        u.x.scatter_forward()

    @abc.abstractmethod
    def _interpolate(self, u: dolfinx.fem.Function, u_i: dolfinx.fem.Function, degree: int):
        """Interpolate ``u_i`` (on the cells of the given degree) into ``u``."""
