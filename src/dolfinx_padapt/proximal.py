r"""The latent variable proximal point (LVPP) method of Keith and Surowiec (2024): a sequence of
nonlinear problems with a proximal parameter :math:`\alpha_k`, each solved with Newton's
method, see :class:`dolfinx_padapt.models.SignoriniProblem`."""

from enum import Enum

import dolfinx


class ProximalScheme(str, Enum):
    r"""Update of the proximal parameter :math:`\alpha_k` between the iterations of
    :func:`proximal_point`."""

    #: :math:`\alpha_k = \alpha_0`
    constant = "constant"
    #: :math:`\alpha_k = 2^k \alpha_0`
    doubling = "doubling"


def proximal_point(
    approach,
    alpha: dolfinx.fem.Constant,
    alpha_0: float,
    scheme: ProximalScheme | str = ProximalScheme.doubling,
    field: str = "u",
    tol: float = 1e-8,
    max_iterations: int = 100,
) -> dict[str, int]:
    """The proximal point iterations, each solved with Newton's method from the previous
    iterate (:meth:`dolfinx_padapt.approaches.BlockApproach.newton`).

    After convergence, the previous iterate is kept (not overwritten by the last iterate), such
    that quantities depending on both (e.g. the contact pressure) can be evaluated.

    Note:
        Collective.

    Args:
        approach: The approach holding the problem and its solution.
        alpha: The proximal parameter in the forms.
        alpha_0: The initial proximal parameter.
        scheme: The update of the proximal parameter between the iterations.
        field: Stop when the change of this field is small.
        tol: Stop when the l2-norm of the change of the coefficients of ``field`` is below this
            tolerance.
        max_iterations: Maximal number of proximal iterations.

    Returns:
        The number of proximal and Newton iterations.

    Raises:
        RuntimeError: If the iterations do not converge.
    """
    scheme = ProximalScheme(scheme)
    alpha.value = alpha_0
    newton_iterations = 0
    for k in range(max_iterations):
        newton_iterations += approach.newton()
        if k > 0 and approach.change(field) < tol:
            return {"proximal_iterations": k + 1, "newton_iterations": newton_iterations}
        approach.store_previous()
        if scheme == ProximalScheme.doubling:
            alpha.value *= 2
    raise RuntimeError(f"The proximal iterations did not converge in {max_iterations} iterations")
