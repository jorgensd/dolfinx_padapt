"""The Laplace operator: an eigenvalue problem and the Poisson equation."""

import abc

import dolfinx
import ufl

from ..dg import estimator_jumps, nitsche_boundary, sipg_interior
from ..mesh import EXACT_EIGENVALUE
from ..problems import ExactSolution, Expr, Problem
from ..solvers import solve_eigenproblem
from ._common import _sqrt


class LaplaceProblem(Problem):
    r"""The Laplace operator, :math:`-\Delta u = s(u)` in :math:`\Omega`, :math:`u = g` on
    :math:`\partial\Omega`, for a scalar :math:`u`."""

    @abc.abstractmethod
    def source(self, u: Expr) -> Expr:
        """The source term :math:`s(u)`."""

    def boundary_value(self) -> Expr | float:
        """The Dirichlet data :math:`g`."""
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
    r"""The first eigenpair of :math:`-\Delta u = \lambda u` in :math:`\Omega`, :math:`u = 0` on
    :math:`\partial\Omega`."""

    is_eigenvalue_problem = True

    def __init__(self, mesh: dolfinx.mesh.Mesh, exact: float = EXACT_EIGENVALUE):
        super().__init__(mesh)
        self._exact = exact
        self._eigenvalue = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0))

    @property
    def constants(self):
        return [self._eigenvalue]

    def source(self, u):
        return self._eigenvalue * u

    def solve(self, A, B, nullspace=None):
        lam, x = solve_eigenproblem(A, B)
        self._eigenvalue.value = lam
        return x

    def errors(self, integrals):
        lam = float(self._eigenvalue.value)
        return abs(lam - self._exact) / self._exact, {"lambda": lam}


class PoissonProblem(LaplaceProblem):
    r"""The Poisson equation, :math:`-\Delta u = f` in :math:`\Omega`, :math:`u = g` on
    :math:`\partial\Omega`, with :math:`f` and :math:`g` from a manufactured solution.

    Args:
        mesh: The mesh.
        exact_solution: The manufactured solution as a function of the spatial coordinate,
            e.g. ``lambda x: ufl.sin(ufl.pi * x[0])``.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh, exact_solution: ExactSolution):
        super().__init__(mesh)
        self._u_exact = exact_solution(self._x)
        self._f = -ufl.div(ufl.grad(self._u_exact))

    def source(self, u):
        return self._f

    def boundary_value(self):
        return self._u_exact

    def error_integrands(self, u):
        e = u["u"] - self._u_exact
        return {"L2": e**2, "H1": ufl.inner(ufl.grad(e), ufl.grad(e))}

    def errors(self, integrals):
        values = _sqrt(integrals)
        return values["H1"], values
