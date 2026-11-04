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
    """[a] = a⁺ ⊗ n⁺ + a⁻ ⊗ n⁻ (a⁺n⁺ + a⁻n⁻ for a scalar)."""
    if a_p.ufl_shape == ():
        return a_p("+") * n("+") + a_m("-") * n("-")
    return ufl.outer(a_p("+"), n("+")) + ufl.outer(a_m("-"), n("-"))


def normal_jump(a_p, a_m, n):
    """[a · n] = a⁺ · n⁺ + a⁻ · n⁻ of a vector."""
    return ufl.dot(a_p("+"), n("+")) + ufl.dot(a_m("-"), n("-"))


def average(a_p, a_m):
    """{a} = (a⁺ + a⁻) / 2."""
    return 0.5 * (a_p("+") + a_m("-"))


def sipg_interior(u_p, u_m, v_p, v_m, n, h, sigma, flux: Flux = ufl.grad):
    """SIPG terms on an interior facet, with the penalty ``sigma / {h}``."""
    return (
        -ufl.inner(average(flux(u_p), flux(u_m)), jump(v_p, v_m, n))
        - ufl.inner(jump(u_p, u_m, n), average(flux(v_p), flux(v_m)))
        + sigma / ufl.avg(h) * ufl.inner(jump(u_p, u_m, n), jump(v_p, v_m, n))
    )


def nitsche_boundary(u, v, n, h, sigma, g=0.0, flux: Flux = ufl.grad):
    """Nitsche terms for u = g on the boundary, with the penalty ``sigma / h``."""
    return (
        -ufl.inner(ufl.dot(flux(u), n), v)
        - ufl.inner(u - g, ufl.dot(flux(v), n))
        + sigma / h * ufl.inner(u - g, v)
    )


def estimator_jumps(flux_jump, value_jump, h, p_p, p_m, jump_weight, w):
    """The facet terms of the error indicator on an interior facet,
    ``h_K/p_K ‖[flux]/2‖² + jump_weight p_K³/h_K ‖[u]/2‖²``, added to both cells (each with its
    own ``h_K`` and ``p_K``)."""
    flux_part = ufl.inner(flux_jump, flux_jump) / 4
    value_part = ufl.inner(value_jump, value_jump) / 4
    return flux_part * (
        h("+") / p_p * w("+") + h("-") / p_m * w("-")
    ) + jump_weight * value_part * (p_p**3 / h("+") * w("+") + p_m**3 / h("-") * w("-"))
