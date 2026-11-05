"""The Stokes equations."""

import dolfinx
import ufl

from ..dg import average, estimator_jumps, nitsche_boundary, normal_jump, sipg_interior
from ..problems import ExactSolution, Field, Problem
from ._common import _sqrt


class StokesProblem(Problem):
    r"""The Stokes equations,

    .. math::

        -\nu \Delta u + \nabla p = f, \quad \nabla \cdot u = 0 \text{ in } \Omega, \quad
        u = g \text{ on } \partial\Omega,

    with :math:`f` and :math:`g` from a manufactured solution.

    The velocity has the degree of the cell, and the pressure one degree less. The pressure is
    only determined up to a constant, which is removed from the solution and the error.
    The jumps of the pressure are penalised on interior facets by
    :math:`-\gamma_p \frac{h}{\nu} [\![p]\!] [\![q]\!]`, with :math:`\gamma_p` the
    ``pressure_stabilisation`` (Cockburn, Kanschat, Schötzau and Schwab, 2002),
    which removes spurious pressure oscillations when the velocity is under-resolved. As the exact
    pressure has no jumps, the method stays consistent.

    Args:
        mesh: The mesh.
        exact_velocity: The manufactured velocity (divergence free) as a function of the spatial
            coordinate.
        exact_pressure: The manufactured pressure as a function of the spatial coordinate.
        viscosity: The viscosity :math:`\nu`.
        pressure_stabilisation: Parameter of the pressure jump penalty (0 to disable it).
    """

    nullspace_fields = ("p",)

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        exact_velocity: ExactSolution,
        exact_pressure: ExactSolution,
        viscosity: float = 1.0,
        pressure_stabilisation: float = 1.0,
    ):
        super().__init__(mesh)
        self._fields = (Field("u", (mesh.geometry.dim,)), Field("p", (), -1))
        self._nu = viscosity
        self._gamma_p = pressure_stabilisation
        self._u_exact = exact_velocity(self._x)
        self._p_exact = exact_pressure(self._x)
        self._f = -self._nu * ufl.div(ufl.grad(self._u_exact)) + ufl.grad(self._p_exact)

    def residual(self, u, v, n, h, sigma, dx, ds):
        u, p = u["u"], u["p"]
        v, q = v["u"], v["p"]
        g = self._u_exact
        F = self._nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        F += -p * ufl.div(v) * dx - q * ufl.div(u) * dx - ufl.inner(self._f, v) * dx
        F += self._nu * nitsche_boundary(u, v, n, h, sigma, g) * ds
        F += p * ufl.dot(v, n) * ds + q * ufl.dot(u - g, n) * ds
        return F

    def interface(self, u_p, u_m, v_p, v_m, n, h, sigma, dS):
        terms = self._nu * sipg_interior(u_p["u"], u_m["u"], v_p["u"], v_m["u"], n, h, sigma)
        terms += average(u_p["p"], u_m["p"]) * normal_jump(v_p["u"], v_m["u"], n)
        terms += average(v_p["p"], v_m["p"]) * normal_jump(u_p["u"], u_m["u"], n)
        if self._gamma_p > 0:
            p_jump = u_p["p"]("+") - u_m["p"]("-")
            q_jump = v_p["p"]("+") - v_m["p"]("-")
            terms -= self._gamma_p * ufl.avg(h) / self._nu * p_jump * q_jump
        return terms * dS

    def _flux(self, u, p):
        return self._nu * ufl.grad(u) - p * ufl.Identity(len(u))

    def estimator(self, u, h, p, gamma, w, dx, ds):
        u_h, p_h = u["u"], u["p"]
        r = self._f + self._nu * ufl.div(ufl.grad(u_h)) - ufl.grad(p_h)
        cell = ((h / p) ** 2 * ufl.inner(r, r) + ufl.div(u_h) ** 2) * w
        e = u_h - self._u_exact
        boundary = (self._nu * gamma) ** 2 * p**3 / h * ufl.inner(e, e) * w
        return cell * dx + boundary * ds

    def estimator_interface(self, u_p, u_m, n, h, p_p, p_m, gamma, w, dS):
        flux_jump = ufl.dot(self._flux(u_p["u"], u_p["p"])("+"), n("+")) + ufl.dot(
            self._flux(u_m["u"], u_m["p"])("-"), n("-")
        )
        value_jump = u_p["u"]("+") - u_m["u"]("-")
        weight = (self._nu * gamma) ** 2
        eta = estimator_jumps(flux_jump, value_jump, h, p_p, p_m, weight, w)
        if self._gamma_p > 0:
            # The pressure jumps that are penalised by the stabilisation
            p_jump = u_p["p"]("+") - u_m["p"]("-")
            eta += (
                self._gamma_p / self._nu * (p_jump / 2) ** 2 * (h("+") * w("+") + h("-") * w("-"))
            )
        return eta * dS

    def error_integrands(self, u):
        e_u = u["u"] - self._u_exact
        e_p = u["p"] - self._p_exact
        return {
            "u_H1": ufl.inner(ufl.grad(e_u), ufl.grad(e_u)),
            "u_L2": ufl.inner(e_u, e_u),
            "p_L2": e_p**2,
            "p_mean": e_p,
            "volume": ufl.as_ufl(1.0),
        }

    def errors(self, integrals):
        # The pressure is determined up to a constant: remove the mean of the pressure error
        p_l2 = integrals["p_L2"] - integrals["p_mean"] ** 2 / integrals["volume"]
        values = _sqrt({"u_H1": integrals["u_H1"], "u_L2": integrals["u_L2"], "p_L2": p_l2})
        return values["u_H1"], values
