"""Building blocks of the symmetric interior penalty (SIPG) and Nitsche terms.

On an interior facet, ``a_p`` lives on the "+" side and ``a_m`` on the "-" side, such that
functions from two different spaces (e.g. of two different degrees) can be coupled. The
expressions are restricted inside these functions. The ``flux`` is the operator of the problem,
e.g. :func:`ufl.grad` for the Laplacian, or the stress for linear elasticity.
"""

import typing

import ufl

Flux = typing.Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr]


def jump(a_p, a_m, n):
    r"""The jump :math:`[\![a]\!] = a^+ \otimes n^+ + a^- \otimes n^-` (:math:`a^+ n^+ + a^- n^-`
    for a scalar)."""
    if a_p.ufl_shape == ():
        return a_p("+") * n("+") + a_m("-") * n("-")
    return ufl.outer(a_p("+"), n("+")) + ufl.outer(a_m("-"), n("-"))


def normal_jump(a_p, a_m, n):
    r"""The normal jump :math:`[\![a \cdot n]\!] = a^+ \cdot n^+ + a^- \cdot n^-` of a vector."""
    return ufl.dot(a_p("+"), n("+")) + ufl.dot(a_m("-"), n("-"))


def average(a_p, a_m):
    r"""The average :math:`\{\!\{a\}\!\} = (a^+ + a^-) / 2`."""
    return 0.5 * (a_p("+") + a_m("-"))


def sipg_interior(u_p, u_m, v_p, v_m, n, h, sigma, flux: Flux = ufl.grad):
    r"""SIPG terms on an interior facet, with the penalty :math:`\sigma / \{\!\{h\}\!\}`."""
    return (
        -ufl.inner(average(flux(u_p), flux(u_m)), jump(v_p, v_m, n))
        - ufl.inner(jump(u_p, u_m, n), average(flux(v_p), flux(v_m)))
        + sigma / ufl.avg(h) * ufl.inner(jump(u_p, u_m, n), jump(v_p, v_m, n))
    )


def nitsche_boundary(u, v, n, h, sigma, g=0.0, flux: Flux = ufl.grad):
    r"""Nitsche terms for :math:`u = g` on the boundary, with the penalty :math:`\sigma / h`."""
    return (
        -ufl.inner(ufl.dot(flux(u), n), v)
        - ufl.inner(u - g, ufl.dot(flux(v), n))
        + sigma / h * ufl.inner(u - g, v)
    )


def estimator_jumps(flux_jump, value_jump, h, p_p, p_m, jump_weight, w):
    r"""The facet terms of the error indicator on an interior facet,

    .. math::

        \frac{h_K}{p_K} \left\| \tfrac{1}{2} [\![\text{flux}]\!] \right\|^2
        + c \frac{p_K^3}{h_K} \left\| \tfrac{1}{2} [\![u]\!] \right\|^2,

    with :math:`c` the ``jump_weight``, added to both cells :math:`K` of the facet (each with its
    own :math:`h_K` and :math:`p_K`)."""
    flux_part = ufl.inner(flux_jump, flux_jump) / 4
    value_part = ufl.inner(value_jump, value_jump) / 4
    return flux_part * (
        h("+") / p_p * w("+") + h("-") / p_m * w("-")
    ) + jump_weight * value_part * (p_p**3 / h("+") * w("+") + p_m**3 / h("-") * w("-"))
