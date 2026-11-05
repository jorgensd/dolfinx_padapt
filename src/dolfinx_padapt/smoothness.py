r"""The local smoothness of a discrete solution, from the decay of its Legendre coefficients
(Mavriplis, 1994; Houston and Süli, 2005), to choose between h- and p-refinement.

On a cell :math:`K` of degree :math:`p_K`, the solution is expanded in the orthonormal polynomials
:math:`P_j` of the reference cell (Dubiner polynomials on simplices, tensor products of Legendre
polynomials on quadrilaterals and hexahedra), :math:`u_h|_K = \sum_j a_j P_j`, and the energy of
the modes of degree :math:`k` is

.. math::

    E_k = \Bigl(\sum_{\deg P_j = k} |a_j|^2\Bigr)^{1/2}.

For a smooth (analytic) function, the energies decay exponentially, :math:`E_k \approx C
e^{-\sigma_K k}`, while they decay slowly near a singularity. The decay rate :math:`\sigma_K` is
the least squares fit of :math:`\log E_k` for :math:`k = 1, \dots, p_K`, and the cell is smooth
if the decay factor :math:`e^{-\sigma_K}` is below a threshold. The mean (:math:`k = 0`) is left
out, as its size is not related to the smoothness. A cell of degree one has a single mode, which
does not give a decay rate, and is considered smooth (it is enriched before it is split).
"""

import basix
import basix.ufl
import dolfinx
import numpy as np
import numpy.typing as npt

from .approaches import Approach
from .approaches.hierarchical import _polynomial_degrees, hierarchical_element


def legendre_energies(approach: Approach, field: str | None = None) -> npt.NDArray[np.float64]:
    r"""The energy :math:`E_k` of the modes of degree :math:`k = 0, \dots, p_{max}` of a field of
    the solution on each owned cell.

    The solution is interpolated into the hierarchical space of degree :math:`p_{max}`, whose
    degrees of freedom are the moments against the orthonormal polynomials, i.e. the
    coefficients :math:`a_j` (see :func:`dolfinx_padapt.approaches.hierarchical_element`).

    Note:
        Collective.

    Args:
        approach: The approach holding the solution.
        field: The field (by default the first).

    Returns:
        The energies, with shape ``(num_owned_cells, pmax + 1)``.
    """
    mesh = approach.mesh
    pmax = approach.params.pmax
    fields = {f.name: f for f in approach.problem.fields}
    field_ = fields[field] if field is not None else approach.problem.fields[0]
    cell = mesh.basix_cell()
    element, _ = hierarchical_element(cell, pmax)
    if field_.shape:
        element = basix.ufl.blocked_element(element, shape=field_.shape)
    V = dolfinx.fem.functionspace(mesh, element)
    u = dolfinx.fem.Function(V)
    approach.interpolate_solution(u, field_.name)

    num_owned = approach.degrees.num_owned_cells
    bs = V.dofmap.bs
    coefficients = u.x.array.reshape(-1, bs)[V.dofmap.list[:num_owned]]
    squares = np.sum(np.abs(coefficients) ** 2, axis=2)
    degree = _polynomial_degrees(cell, pmax)
    energies = np.zeros((num_owned, pmax + 1))
    for k in range(pmax + 1):
        energies[:, k] = np.sqrt(np.sum(squares[:, degree == k], axis=1))
    return energies


def decay_factor(approach: Approach, field: str | None = None) -> npt.NDArray[np.float64]:
    r"""The decay factor :math:`e^{-\sigma_K}` of the Legendre coefficients of a field of the
    solution on each owned cell (see the module documentation). Small values mean a smooth
    solution. Cells of degree one have the decay factor zero.

    Note:
        Collective.

    Args:
        approach: The approach holding the solution.
        field: The field (by default the first).

    Returns:
        The decay factor of each owned cell.
    """
    energies = legendre_energies(approach, field)
    num_owned = energies.shape[0]
    cell_degrees = approach.degrees.cell_degrees[:num_owned]
    # Avoid the logarithm of zero for a (locally) vanishing solution
    floor = 1e-14 * max(float(energies.max(initial=0.0)), np.finfo(np.float64).tiny)
    log_energies = np.log(np.maximum(energies, floor))
    factor = np.zeros(num_owned)
    for p in np.unique(cell_degrees[cell_degrees >= 2]):
        cells = np.flatnonzero(cell_degrees == p)
        k = np.arange(1, p + 1)
        slope = np.polyfit(k, log_energies[cells][:, k].T, 1)[0]
        factor[cells] = np.exp(slope)
    return factor
