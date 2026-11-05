"""The interface of a problem, discretised with the symmetric interior penalty (SIPG) method and
Nitsche's method for the Dirichlet conditions (see :mod:`dolfinx_padapt.dg`).

A problem has one or more fields (e.g. a velocity and a pressure), each with a value shape and a
degree relative to the degree of the cell. It provides the residual :math:`F(u; v)` on the cells
and the boundary of one degree region, the coupling terms on the interfaces between two regions,
the error indicator and the error measures. The model problems are in
:mod:`dolfinx_padapt.models`.
"""

import abc
import typing
from dataclasses import dataclass

from petsc4py import PETSc

import dolfinx
import ufl

from .solvers import solve_linear

Expr = ufl.core.expr.Expr
#: The fields of a problem by name, e.g. ``{"u": u, "p": p}``
Fields = dict[str, Expr]
ExactSolution = typing.Callable[[ufl.SpatialCoordinate], Expr]


@dataclass(frozen=True)
class Field:
    """A field of a problem.

    Args:
        name: Name of the field.
        shape: Value shape, e.g. ``(gdim,)`` for a vector.
        degree_offset: The degree of the field is the degree of the cell plus this offset, e.g.
            ``-1`` for the pressure of the Stokes equations.
        boundary: If given, the field lives on the exterior facets with this marker (see
            :attr:`Problem.facet_tags`), with the degree of the cell of each facet, and the
            continuity of the fields on the cells (such that it matches their trace). Fields of
            different degrees are not coupled, so the forms cannot have derivatives along the
            boundary. Only supported by the submesh approach.
    """

    name: str
    shape: tuple[int, ...] = ()
    degree_offset: int = 0
    boundary: int | None = None

    def degree(self, cell_degree: int) -> int:
        return cell_degree + self.degree_offset


class Problem(abc.ABC):
    """A problem discretised with SIPG and Nitsche's method.

    In the methods below, ``u`` and ``v`` are the trial (or solution) and test functions of each
    field, ``n`` the facet normal, ``h`` the cell size, ``sigma`` the penalty parameter (scaled
    with the degree), ``gamma`` the penalty parameter (not scaled) and ``p`` the degree of the
    cells.

    A problem with :attr:`boundary_markers` also gets the keyword argument ``ds_marker`` in
    :meth:`residual` and :meth:`estimator`, a function returning the measure of the facets with
    a given marker (of the cells of the current degree). A nonlinear problem gets the keyword
    argument ``u_prev`` with the fields of a previous iterate, see :meth:`solve_nonlinear`.
    """

    #: Whether the right hand side is a (mass) matrix rather than a vector
    is_eigenvalue_problem: bool = False
    #: Fields that are only determined up to a constant, e.g. the pressure of the Stokes
    #: equations with Dirichlet conditions on the whole boundary
    nullspace_fields: tuple[str, ...] = ()
    #: Whether the residual is nonlinear in the fields. Then ``u`` in :meth:`residual` is the
    #: current iterate, the Jacobian is computed with :func:`ufl.derivative`, and the problem is
    #: solved with :meth:`solve_nonlinear`
    is_nonlinear: bool = False

    def __init__(self, mesh: dolfinx.mesh.Mesh):
        self._mesh = mesh
        self._x = ufl.SpatialCoordinate(mesh)
        self._fields: tuple[Field, ...] = (Field("u"),)

    @property
    def mesh(self) -> dolfinx.mesh.Mesh:
        """The mesh."""
        return self._mesh

    @property
    def fields(self) -> tuple[Field, ...]:
        """The fields of the problem."""
        return self._fields

    @property
    def constants(self) -> list[dolfinx.fem.Constant]:
        """Constants in the forms of the problem."""
        return []

    @property
    def petsc_options(self) -> dict:
        """PETSc options of the solver: a direct solver (MUMPS), and for a nonlinear problem
        Newton's method with a backtracking line search (PETSc SNES)."""
        options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        }
        if self.nullspace_fields:
            # Let MUMPS detect the null pivot of the constant nullspace, and return a solution
            options |= {"mat_mumps_icntl_24": 1, "mat_mumps_icntl_25": 0}
        if self.is_nonlinear:
            options |= {
                "snes_type": "newtonls",
                "snes_linesearch_type": "bt",
                "snes_rtol": 1e-10,
                "snes_atol": 1e-10,
                "snes_max_it": 50,
                "snes_error_if_not_converged": True,
            }
        return options

    @property
    def facet_tags(self) -> dolfinx.mesh.MeshTags | None:
        """Markers of the exterior facets, for :attr:`boundary_markers` and boundary fields."""
        return None

    @property
    def boundary_markers(self) -> tuple[int, ...]:
        """Markers of the exterior facets with their own integrals (``ds_marker``)."""
        return ()

    @abc.abstractmethod
    def residual(self, u: Fields, v: Fields, n, h, sigma, dx, ds) -> ufl.Form:
        """The residual on the cells (``dx``) and the boundary facets (``ds``) of one degree."""

    @abc.abstractmethod
    def interface(self, u_p: Fields, u_m: Fields, v_p: Fields, v_m: Fields, n, h, sigma, dS):
        """The coupling terms on interior facets, between the fields ``u_p``, ``v_p`` on the "+"
        side and ``u_m``, ``v_m`` on the "-" side.

        Called for the facets between cells of every pair of degrees ``i <= j`` (the lower degree
        on the "+" side), see :func:`dolfinx_padapt.degrees.degree_pairs`. This includes the
        facets inside a region of degree ``i`` (``i == j``, where both sides are the fields of
        the same space) for discontinuous spaces, but not for continuous spaces, where the jumps
        vanish inside a region."""

    def mass(self, u: Fields, v: Fields, dx) -> ufl.Form:
        """The mass matrix of an eigenvalue problem."""
        raise NotImplementedError

    @abc.abstractmethod
    def estimator(self, u: Fields, h, p, gamma, w, dx, ds) -> ufl.Form:
        """The cell and boundary terms of the squared error indicator, tested with the DG-0
        function ``w``."""

    @abc.abstractmethod
    def estimator_interface(self, u_p: Fields, u_m: Fields, n, h, p_p, p_m, gamma, w, dS):
        """The interior facet terms of the squared error indicator, on the facets between cells
        of every pair of degrees ``i <= j``, also inside a region (``i == j``) for continuous
        spaces, where the flux jumps do not vanish."""

    def error_integrands(self, u: Fields) -> dict[str, Expr]:
        """Integrands of the error measures of a solution ``u``."""
        return {}

    def solve(
        self, A: PETSc.Mat, B: PETSc.Mat | PETSc.Vec, nullspace: PETSc.NullSpace | None = None
    ) -> PETSc.Vec:
        """Solve the discrete problem with the stiffness matrix ``A`` and the right hand side
        ``B`` (mass matrix or load vector), for approaches that assemble the system themselves.
        By default, a linear system is solved with :attr:`petsc_options`."""
        return solve_linear(A, B, nullspace, self.petsc_options)

    def solve_nonlinear(self, approach) -> None:
        """Solve a nonlinear problem, leaving the solution in the approach. Called by
        :meth:`dolfinx_padapt.approaches.BlockApproach.solve`. By default, a single solve with
        Newton's method (:meth:`dolfinx_padapt.approaches.BlockApproach.newton`, with
        :class:`dolfinx.fem.petsc.NonlinearProblem`). Override for an outer iteration."""
        approach.newton()

    @abc.abstractmethod
    def errors(self, integrals: dict[str, float]) -> tuple[float | None, dict[str, float]]:
        """The error used to stop the adaptive loop, and the values to report. Without an error
        measure (e.g. without an exact solution), the error is ``None``, and the adaptive loop
        uses the error indicator.

        Args:
            integrals: The global integral of each of :meth:`error_integrands`.
        """
