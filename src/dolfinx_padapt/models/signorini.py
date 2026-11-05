"""Linear elasticity with frictionless contact (the Signorini problem)."""

import typing

import dolfinx
import ufl

from ..dg import estimator_jumps, nitsche_boundary, sipg_interior
from ..problems import ExactSolution, Expr, Field, Fields, Problem
from ..proximal import ProximalScheme, proximal_point


class SignoriniProblem(Problem):
    r"""Linear elasticity with frictionless contact against a rigid obstacle (the Signorini
    problem), solved with the latent variable proximal point (LVPP) method of Keith and Surowiec
    (2024), applied to contact by Dokken, Farrell, Keith, Papadopoulos and Surowiec (2025), as
    in the FEniCS workshop.

    The displacement :math:`u` satisfies :math:`u \cdot \hat{n} \leq g` on the contact boundary
    :math:`\Gamma`, where :math:`\hat{n}` is the direction of the obstacle and :math:`g` the gap.
    A latent variable :math:`\psi` on :math:`\Gamma` (a field on the boundary, of the degree of
    the cell of each facet) gives :math:`e^\psi = g - u \cdot \hat{n} > 0`. Every proximal
    iteration :math:`k` solves (with Newton's method)

    .. math::

        \alpha_k (\sigma(u), \varepsilon(v))_\Omega
        - (\psi - \psi_{k-1}, v \cdot \hat{n})_\Gamma &= \alpha_k (f, v)_\Omega,\\
        (u \cdot \hat{n} + e^\psi - g, w)_\Gamma &= 0,

    for all :math:`(v, w)`, with the Dirichlet condition imposed with Nitsche's method. The
    contact pressure is :math:`\lambda = (\psi - \psi_{k-1}) / \alpha_k`. Facets with neither
    marker are traction free.

    Args:
        mesh: The mesh.
        facet_tags: Markers of the exterior facets.
        dirichlet_marker: Marker of the facets with the prescribed displacement.
        contact_marker: Marker of the (potential) contact boundary :math:`\Gamma`.
        displacement: The prescribed displacement as a function of the spatial coordinate.
        gap: The gap :math:`g` between :math:`\Gamma` and the obstacle as a function of the
            spatial coordinate.
        direction: The direction :math:`\hat{n}` of the obstacle.
        E: Young's modulus.
        nu: Poisson's ratio.
        alpha: Initial proximal parameter :math:`\alpha_0`.
        scheme: Update of :math:`\alpha_k` between the proximal iterations.
        tol: Stop the proximal iterations when the l2-norm of the change of the displacement
            (coefficients) is below this tolerance.
        max_iterations: Maximal number of proximal iterations.
    """

    is_nonlinear = True

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        facet_tags: dolfinx.mesh.MeshTags,
        dirichlet_marker: int,
        contact_marker: int,
        displacement: ExactSolution,
        gap: ExactSolution,
        direction: typing.Sequence[float],
        E: float = 2e4,
        nu: float = 0.4,
        alpha: float = 0.1,
        scheme: ProximalScheme | str = ProximalScheme.doubling,
        tol: float = 1e-8,
        max_iterations: int = 100,
    ):
        super().__init__(mesh)
        gdim = mesh.geometry.dim
        self._fields = (Field("u", (gdim,)), Field("psi", (), 0, boundary=contact_marker))
        self._facet_tags = facet_tags
        self._dirichlet = dirichlet_marker
        self._contact = contact_marker
        self._mu = E / (2 * (1 + nu))
        self._lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
        self._kappa = 2 * self._mu + self._lmbda
        self._u_D = displacement(self._x)
        self._g = gap(self._x)
        self._n_hat = ufl.as_vector(direction)
        self._f = ufl.as_vector([0.0] * gdim)
        self._alpha_0 = alpha
        self._alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(alpha))
        self._scheme = ProximalScheme(scheme)
        self._tol = tol
        self._max_iterations = max_iterations
        self._statistics: dict[str, float] = {}

    @property
    def constants(self):
        return [self._alpha]

    @property
    def alpha(self) -> float:
        r"""The current proximal parameter :math:`\alpha_k`."""
        return float(self._alpha.value)

    @property
    def facet_tags(self):
        return self._facet_tags

    @property
    def boundary_markers(self):
        return (self._dirichlet, self._contact)

    def stress(self, u: Expr) -> Expr:
        eps = ufl.sym(ufl.grad(u))
        return 2 * self._mu * eps + self._lmbda * ufl.tr(eps) * ufl.Identity(len(u))

    def residual(self, u, v, n, h, sigma, dx, ds, ds_marker=None, u_prev=None):
        u_h, psi = u["u"], u["psi"]
        v_h, w = v["u"], v["psi"]
        alpha = self._alpha
        F = alpha * ufl.inner(self.stress(u_h), ufl.grad(v_h)) * dx
        F -= alpha * ufl.inner(self._f, v_h) * dx
        F += (
            alpha
            * nitsche_boundary(u_h, v_h, n, h, self._kappa * sigma, self._u_D, self.stress)
            * ds_marker(self._dirichlet)
        )
        gamma = self._contact_measure(ds_marker, psi)
        F -= (psi - u_prev["psi"]) * ufl.dot(v_h, self._n_hat) * gamma
        F += (ufl.dot(u_h, self._n_hat) + ufl.exp(psi) - self._g) * w * gamma
        return F

    def interface(self, u_p, u_m, v_p, v_m, n, h, sigma, dS):
        terms = sipg_interior(
            u_p["u"], u_m["u"], v_p["u"], v_m["u"], n, h, self._kappa * sigma, self.stress
        )
        return self._alpha * terms * dS

    def _contact_measure(self, ds_marker, psi) -> ufl.Measure:
        r"""The measure of :math:`\Gamma` (of the cells of one degree), with quadrature degree
        :math:`2p + 4`. The exponential of the latent variable varies by many orders of magnitude
        over a facet in contact, and the estimated quadrature degree is too low for Newton's
        method to converge for degrees above one."""
        degree = psi.ufl_element().embedded_superdegree
        return ds_marker(self._contact)(degree=2 * degree + 4)

    def contact_pressure(self, u: Fields, u_prev: Fields) -> Expr:
        r"""The contact pressure :math:`\lambda = (\psi - \psi_{k-1}) / \alpha_k` on
        :math:`\Gamma`."""
        return (u["psi"] - u_prev["psi"]) / self._alpha

    def estimator(self, u, h, p, gamma, w, dx, ds, ds_marker=None, u_prev=None):
        u_h = u["u"]
        n = ufl.FacetNormal(self._mesh)
        r = self._f + ufl.div(self.stress(u_h))
        cell = (h / p) ** 2 * ufl.inner(r, r) * w
        e = u_h - self._u_D
        dirichlet = (self._kappa * gamma) ** 2 * p**3 / h * ufl.inner(e, e) * w
        # The traction on the contact boundary is the contact pressure (in the direction of the
        # obstacle)
        t = ufl.dot(self.stress(u_h), n) - self.contact_pressure(u, u_prev) * self._n_hat
        contact = h / p * ufl.inner(t, t) * w
        return (
            cell * dx
            + dirichlet * ds_marker(self._dirichlet)
            + contact * self._contact_measure(ds_marker, u["psi"])
        )

    def estimator_interface(self, u_p, u_m, n, h, p_p, p_m, gamma, w, dS):
        u_p, u_m = u_p["u"], u_m["u"]
        flux_jump = ufl.dot(self.stress(u_p)("+"), n("+")) + ufl.dot(self.stress(u_m)("-"), n("-"))
        value_jump = u_p("+") - u_m("-")
        weight = (self._kappa * gamma) ** 2
        return estimator_jumps(flux_jump, value_jump, h, p_p, p_m, weight, w) * dS

    def solve_nonlinear(self, approach):
        """The proximal point iterations (:func:`dolfinx_padapt.proximal.proximal_point`)."""
        self._statistics = proximal_point(
            approach,
            self._alpha,
            self._alpha_0,
            self._scheme,
            field="u",
            tol=self._tol,
            max_iterations=self._max_iterations,
        )

    def errors(self, integrals):
        return None, dict(self._statistics)
