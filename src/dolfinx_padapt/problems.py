"""Model problems, discretised with the symmetric interior penalty (SIPG) method and Nitsche's
method for the Dirichlet conditions (see :mod:`dolfinx_padapt.dg`).

A problem has one or more fields (e.g. a velocity and a pressure), each with a value shape and a
degree relative to the degree of the cell. It provides the residual F(u; v) on the cells and the
boundary of one degree region, the coupling terms on the interfaces between two regions, the
error indicator and the error measures:

- :class:`EigenvalueProblem`: find (u, λ) with -Δu = λu in Ω and u = 0 on ∂Ω.
- :class:`PoissonProblem`: -Δu = f in Ω, u = g on ∂Ω.
- :class:`ElasticityProblem`: -∇·σ(u) = f in Ω, u = g on ∂Ω.
- :class:`StokesProblem`: -νΔu + ∇p = f, ∇·u = 0 in Ω, u = g on ∂Ω.

The source problems use a manufactured solution, such that the error can be measured.
"""

import abc
import typing
from dataclasses import dataclass

from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl

from .dg import average, estimator_jumps, nitsche_boundary, normal_jump, sipg_interior
from .eigen import solve_eigenproblem
from .mesh import EXACT_EIGENVALUE

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
    """

    name: str
    shape: tuple[int, ...] = ()
    degree_offset: int = 0

    def degree(self, cell_degree: int) -> int:
        return cell_degree + self.degree_offset


class Problem(abc.ABC):
    """A problem discretised with SIPG and Nitsche's method.

    In the methods below, ``u`` and ``v`` are the trial (or solution) and test functions of each
    field, ``n`` the facet normal, ``h`` the cell size, ``sigma`` the penalty parameter (scaled
    with the degree), ``gamma`` the penalty parameter (not scaled) and ``p`` the degree of the
    cells.
    """

    #: Whether the right hand side is a (mass) matrix rather than a vector
    is_eigenvalue_problem: bool = False
    #: Fields that are only determined up to a constant, e.g. the pressure of the Stokes
    #: equations with Dirichlet conditions on the whole boundary
    nullspace_fields: tuple[str, ...] = ()

    def __init__(self, mesh: dolfinx.mesh.Mesh):
        self.mesh = mesh
        self.x = ufl.SpatialCoordinate(mesh)
        self.fields: tuple[Field, ...] = (Field("u"),)

    @property
    def constants(self) -> list[dolfinx.fem.Constant]:
        """Constants in the forms of the problem."""
        return []

    @abc.abstractmethod
    def residual(self, u: Fields, v: Fields, n, h, sigma, dx, ds) -> ufl.Form:
        """The residual on the cells (``dx``) and the boundary facets (``ds``) of one degree."""

    @abc.abstractmethod
    def interface(self, u_p: Fields, u_m: Fields, v_p: Fields, v_m: Fields, n, h, sigma, dS):
        """The coupling terms on interior facets, between the fields ``u_p``, ``v_p`` on the "+"
        side and ``u_m``, ``v_m`` on the "-" side."""

    def mass(self, u: Fields, v: Fields, dx) -> ufl.Form:
        """The mass matrix of an eigenvalue problem."""
        raise NotImplementedError

    @abc.abstractmethod
    def estimator(self, u: Fields, h, p, gamma, w, dx, ds) -> ufl.Form:
        """The cell and boundary terms of the squared error indicator, tested with the DG-0
        function ``w``."""

    @abc.abstractmethod
    def estimator_interface(self, u_p: Fields, u_m: Fields, n, h, p_p, p_m, gamma, w, dS):
        """The interior facet terms of the squared error indicator."""

    def error_integrands(self, u: Fields) -> dict[str, Expr]:
        """Integrands of the error measures of a solution ``u``."""
        return {}

    @abc.abstractmethod
    def solve(
        self, A: PETSc.Mat, B: PETSc.Mat | PETSc.Vec, nullspace: PETSc.NullSpace | None = None
    ) -> PETSc.Vec:
        """Solve the discrete problem with the stiffness matrix ``A`` and the right hand side
        ``B`` (mass matrix or load vector)."""

    @abc.abstractmethod
    def errors(self, integrals: dict[str, float]) -> tuple[float, dict[str, float]]:
        """The error used to stop the adaptive loop, and the values to report.

        Args:
            integrals: The global integral of each of :meth:`error_integrands`.
        """


def solve_linear(A: PETSc.Mat, b: PETSc.Vec, nullspace: PETSc.NullSpace | None = None) -> PETSc.Vec:
    """Solve A x = b with a direct solver (MUMPS), where ``A`` may have a (constant) nullspace."""
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    if nullspace is not None:
        A.setNullSpace(nullspace)
        nullspace.remove(b)
        # Let MUMPS detect the null pivot, and return a solution
        pc.setFactorSetUpSolverType()
        pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
        pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
    x = A.createVecRight()
    ksp.solve(b, x)
    reason = ksp.getConvergedReason()
    ksp.destroy()
    assert reason > 0, f"Linear solver did not converge ({reason})"
    if nullspace is not None:
        nullspace.remove(x)
    return x


def _sqrt(values: dict[str, float]) -> dict[str, float]:
    return {name: float(np.sqrt(max(value, 0.0))) for name, value in values.items()}


class LaplaceProblem(Problem):
    """-Δu = s(u) in Ω, u = g on ∂Ω, for a scalar u."""

    @abc.abstractmethod
    def source(self, u: Expr) -> Expr:
        """The source term s(u)."""

    def boundary_value(self) -> Expr | float:
        """The Dirichlet data g."""
        return 0.0

    def residual(self, u, v, n, h, sigma, dx, ds):
        u, v = u["u"], v["u"]
        g = self.boundary_value()
        F = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        F += nitsche_boundary(u, v, n, h, sigma, g) * ds
        if not self.is_eigenvalue_problem:
            F -= ufl.inner(self.source(u), v) * dx
        return F

    def interface(self, u_p, u_m, v_p, v_m, n, h, sigma, dS):
        return sipg_interior(u_p["u"], u_m["u"], v_p["u"], v_m["u"], n, h, sigma) * dS

    def mass(self, u, v, dx):
        return ufl.inner(u["u"], v["u"]) * dx

    def estimator(self, u, h, p, gamma, w, dx, ds):
        u = u["u"]
        g = self.boundary_value()
        cell = (h / p) ** 2 * (self.source(u) + ufl.div(ufl.grad(u))) ** 2 * w
        boundary = gamma**2 * p**3 / h * (u - g) ** 2 * w
        return cell * dx + boundary * ds

    def estimator_interface(self, u_p, u_m, n, h, p_p, p_m, gamma, w, dS):
        u_p, u_m = u_p["u"], u_m["u"]
        flux_jump = ufl.dot(ufl.grad(u_p)("+"), n("+")) + ufl.dot(ufl.grad(u_m)("-"), n("-"))
        value_jump = u_p("+") - u_m("-")
        return estimator_jumps(flux_jump, value_jump, h, p_p, p_m, gamma**2, w) * dS


class EigenvalueProblem(LaplaceProblem):
    """The first eigenpair of -Δu = λu in Ω, u = 0 on ∂Ω."""

    is_eigenvalue_problem = True

    def __init__(self, mesh: dolfinx.mesh.Mesh, exact: float = EXACT_EIGENVALUE):
        super().__init__(mesh)
        self.exact = exact
        self.eigenvalue = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

    @property
    def constants(self):
        return [self.eigenvalue]

    def source(self, u):
        return self.eigenvalue * u

    def solve(self, A, B, nullspace=None):
        lam, x = solve_eigenproblem(A, B)
        self.eigenvalue.value = lam
        return x

    def errors(self, integrals):
        lam = float(self.eigenvalue.value)
        return abs(lam - self.exact) / self.exact, {"lambda": lam}


class PoissonProblem(LaplaceProblem):
    """-Δu = f in Ω, u = g on ∂Ω, with f and g from a manufactured solution.

    Args:
        mesh: The mesh.
        exact_solution: The manufactured solution as a function of the spatial coordinate,
            e.g. ``lambda x: ufl.sin(ufl.pi * x[0])``.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, exact_solution: ExactSolution):
        super().__init__(mesh)
        self.u_exact = exact_solution(self.x)
        self.f = -ufl.div(ufl.grad(self.u_exact))

    def source(self, u):
        return self.f

    def boundary_value(self):
        return self.u_exact

    def error_integrands(self, u):
        e = u["u"] - self.u_exact
        return {"L2": e**2, "H1": ufl.inner(ufl.grad(e), ufl.grad(e))}

    def solve(self, A, B, nullspace=None):
        return solve_linear(A, B, nullspace)

    def errors(self, integrals):
        values = _sqrt(integrals)
        return values["H1"], values


class ElasticityProblem(Problem):
    """Linear elasticity, -∇·σ(u) = f in Ω, u = g on ∂Ω, with
    σ(u) = 2με(u) + λ tr(ε(u)) I, and f and g from a manufactured displacement.

    The penalty parameter is scaled with 2μ + λ.

    Args:
        mesh: The mesh.
        exact_solution: The manufactured displacement as a function of the spatial coordinate.
        mu: First Lamé parameter.
        lmbda: Second Lamé parameter.
    """

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        exact_solution: ExactSolution,
        mu: float = 1.0,
        lmbda: float = 1.0,
    ):
        super().__init__(mesh)
        self.fields = (Field("u", (mesh.geometry.dim,)),)
        self.mu = mu
        self.lmbda = lmbda
        self.kappa = 2 * mu + lmbda
        self.u_exact = exact_solution(self.x)
        self.f = -ufl.div(self.stress(self.u_exact))

    def stress(self, u: Expr) -> Expr:
        eps = ufl.sym(ufl.grad(u))
        return 2 * self.mu * eps + self.lmbda * ufl.tr(eps) * ufl.Identity(len(u))

    def residual(self, u, v, n, h, sigma, dx, ds):
        u, v = u["u"], v["u"]
        F = ufl.inner(self.stress(u), ufl.grad(v)) * dx - ufl.inner(self.f, v) * dx
        F += nitsche_boundary(u, v, n, h, self.kappa * sigma, self.u_exact, self.stress) * ds
        return F

    def interface(self, u_p, u_m, v_p, v_m, n, h, sigma, dS):
        terms = sipg_interior(
            u_p["u"], u_m["u"], v_p["u"], v_m["u"], n, h, self.kappa * sigma, self.stress
        )
        return terms * dS

    def estimator(self, u, h, p, gamma, w, dx, ds):
        u = u["u"]
        r = self.f + ufl.div(self.stress(u))
        cell = (h / p) ** 2 * ufl.inner(r, r) * w
        e = u - self.u_exact
        boundary = (self.kappa * gamma) ** 2 * p**3 / h * ufl.inner(e, e) * w
        return cell * dx + boundary * ds

    def estimator_interface(self, u_p, u_m, n, h, p_p, p_m, gamma, w, dS):
        u_p, u_m = u_p["u"], u_m["u"]
        flux_jump = ufl.dot(self.stress(u_p)("+"), n("+")) + ufl.dot(self.stress(u_m)("-"), n("-"))
        value_jump = u_p("+") - u_m("-")
        weight = (self.kappa * gamma) ** 2
        return estimator_jumps(flux_jump, value_jump, h, p_p, p_m, weight, w) * dS

    def error_integrands(self, u):
        e = u["u"] - self.u_exact
        return {"L2": ufl.inner(e, e), "H1": ufl.inner(ufl.grad(e), ufl.grad(e))}

    def solve(self, A, B, nullspace=None):
        return solve_linear(A, B, nullspace)

    def errors(self, integrals):
        values = _sqrt(integrals)
        return values["H1"], values


class StokesProblem(Problem):
    """The Stokes equations, -νΔu + ∇p = f, ∇·u = 0 in Ω, u = g on ∂Ω, with f and g from a
    manufactured solution.

    The velocity has the degree of the cell, and the pressure one degree less. The pressure is
    only determined up to a constant, which is removed from the solution and the error.

    Args:
        mesh: The mesh.
        exact_velocity: The manufactured velocity (divergence free) as a function of the spatial
            coordinate.
        exact_pressure: The manufactured pressure as a function of the spatial coordinate.
        viscosity: The viscosity ν.
    """

    nullspace_fields = ("p",)

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        exact_velocity: ExactSolution,
        exact_pressure: ExactSolution,
        viscosity: float = 1.0,
    ):
        super().__init__(mesh)
        self.fields = (Field("u", (mesh.geometry.dim,)), Field("p", (), -1))
        self.nu = viscosity
        self.u_exact = exact_velocity(self.x)
        self.p_exact = exact_pressure(self.x)
        self.f = -self.nu * ufl.div(ufl.grad(self.u_exact)) + ufl.grad(self.p_exact)

    def residual(self, u, v, n, h, sigma, dx, ds):
        u, p = u["u"], u["p"]
        v, q = v["u"], v["p"]
        g = self.u_exact
        F = self.nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        F += -p * ufl.div(v) * dx - q * ufl.div(u) * dx - ufl.inner(self.f, v) * dx
        F += self.nu * nitsche_boundary(u, v, n, h, sigma, g) * ds
        F += p * ufl.dot(v, n) * ds + q * ufl.dot(u - g, n) * ds
        return F

    def interface(self, u_p, u_m, v_p, v_m, n, h, sigma, dS):
        terms = self.nu * sipg_interior(u_p["u"], u_m["u"], v_p["u"], v_m["u"], n, h, sigma)
        terms += average(u_p["p"], u_m["p"]) * normal_jump(v_p["u"], v_m["u"], n)
        terms += average(v_p["p"], v_m["p"]) * normal_jump(u_p["u"], u_m["u"], n)
        return terms * dS

    def _flux(self, u, p):
        return self.nu * ufl.grad(u) - p * ufl.Identity(len(u))

    def estimator(self, u, h, p, gamma, w, dx, ds):
        u_h, p_h = u["u"], u["p"]
        r = self.f + self.nu * ufl.div(ufl.grad(u_h)) - ufl.grad(p_h)
        cell = ((h / p) ** 2 * ufl.inner(r, r) + ufl.div(u_h) ** 2) * w
        e = u_h - self.u_exact
        boundary = (self.nu * gamma) ** 2 * p**3 / h * ufl.inner(e, e) * w
        return cell * dx + boundary * ds

    def estimator_interface(self, u_p, u_m, n, h, p_p, p_m, gamma, w, dS):
        flux_jump = ufl.dot(self._flux(u_p["u"], u_p["p"])("+"), n("+")) + ufl.dot(
            self._flux(u_m["u"], u_m["p"])("-"), n("-")
        )
        value_jump = u_p["u"]("+") - u_m["u"]("-")
        weight = (self.nu * gamma) ** 2
        return estimator_jumps(flux_jump, value_jump, h, p_p, p_m, weight, w) * dS

    def error_integrands(self, u):
        e_u = u["u"] - self.u_exact
        e_p = u["p"] - self.p_exact
        return {
            "u_H1": ufl.inner(ufl.grad(e_u), ufl.grad(e_u)),
            "u_L2": ufl.inner(e_u, e_u),
            "p_L2": e_p**2,
            "p_mean": e_p,
            "volume": ufl.as_ufl(1.0),
        }

    def solve(self, A, B, nullspace=None):
        return solve_linear(A, B, nullspace)

    def errors(self, integrals):
        # The pressure is determined up to a constant: remove the mean of the pressure error
        p_l2 = integrals["p_L2"] - integrals["p_mean"] ** 2 / integrals["volume"]
        values = _sqrt({"u_H1": integrals["u_H1"], "u_L2": integrals["u_L2"], "p_L2": p_l2})
        return values["u_H1"], values
