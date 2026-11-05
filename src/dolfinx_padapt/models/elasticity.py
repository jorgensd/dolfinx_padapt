"""Linear elasticity."""

import dolfinx
import ufl

from ..dg import estimator_jumps, nitsche_boundary, sipg_interior
from ..problems import ExactSolution, Expr, Field, Problem
from ._common import _sqrt


class ElasticityProblem(Problem):
    r"""Linear elasticity,

    .. math::

        -\nabla \cdot \sigma(u) = f \text{ in } \Omega, \quad u = g \text{ on } \partial\Omega,

    with :math:`\sigma(u) = 2\mu\varepsilon(u) + \lambda \operatorname{tr}(\varepsilon(u)) I`, and
    :math:`f` and :math:`g` from a manufactured displacement.

    The penalty parameter is scaled with :math:`2\mu + \lambda`.

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
        self._fields = (Field("u", (mesh.geometry.dim,)),)
        self._mu = mu
        self._lmbda = lmbda
        self._kappa = 2 * mu + lmbda
        self._u_exact = exact_solution(self._x)
        self._f = -ufl.div(self.stress(self._u_exact))

    def stress(self, u: Expr) -> Expr:
        eps = ufl.sym(ufl.grad(u))
        return 2 * self._mu * eps + self._lmbda * ufl.tr(eps) * ufl.Identity(len(u))

    def residual(self, u, v, n, h, sigma, dx, ds):
        u, v = u["u"], v["u"]
        F = ufl.inner(self.stress(u), ufl.grad(v)) * dx - ufl.inner(self._f, v) * dx
        F += nitsche_boundary(u, v, n, h, self._kappa * sigma, self._u_exact, self.stress) * ds
        return F

    def interface(self, u_p, u_m, v_p, v_m, n, h, sigma, dS):
        terms = sipg_interior(
            u_p["u"], u_m["u"], v_p["u"], v_m["u"], n, h, self._kappa * sigma, self.stress
        )
        return terms * dS

    def estimator(self, u, h, p, gamma, w, dx, ds):
        u = u["u"]
        r = self._f + ufl.div(self.stress(u))
        cell = (h / p) ** 2 * ufl.inner(r, r) * w
        e = u - self._u_exact
        boundary = (self._kappa * gamma) ** 2 * p**3 / h * ufl.inner(e, e) * w
        return cell * dx + boundary * ds

    def estimator_interface(self, u_p, u_m, n, h, p_p, p_m, gamma, w, dS):
        u_p, u_m = u_p["u"], u_m["u"]
        flux_jump = ufl.dot(self.stress(u_p)("+"), n("+")) + ufl.dot(self.stress(u_m)("-"), n("-"))
        value_jump = u_p("+") - u_m("-")
        weight = (self._kappa * gamma) ** 2
        return estimator_jumps(flux_jump, value_jump, h, p_p, p_m, weight, w) * dS

    def error_integrands(self, u):
        e = u["u"] - self._u_exact
        return {"L2": ufl.inner(e, e), "H1": ufl.inner(ufl.grad(e), ufl.grad(e))}

    def errors(self, integrals):
        values = _sqrt(integrals)
        return values["H1"], values
