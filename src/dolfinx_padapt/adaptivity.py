"""The adaptive loop: solve, estimate, mark and enrich."""

import time
import typing
from dataclasses import dataclass, field
from enum import Enum

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt

from .approaches import Approach
from .degrees import DegreeField


class MarkingStrategy(str, Enum):
    """Strategy for marking the cells to enrich."""

    #: :math:`\eta_K > \theta \max_{K'} \eta_{K'}`, see :func:`dolfinx.mesh.mark_maximum`
    maximum = "maximum"
    #: :math:`\eta_K^2 > \theta^2 \Vert\eta\Vert^2 / N`, see
    #: :func:`dolfinx.mesh.mark_equidistribution`
    equidistribution = "equidistribution"


@dataclass
class IterationResult:
    """Result of one iteration of the adaptive loop."""

    iteration: int
    num_dofs: int
    error: float  # The error measure of the problem, or the error indicator (stops the loop)
    eta: float  # Global error indicator
    degrees: dict[int, int]  # Number of cells of each degree
    #: Values reported by the problem, e.g. the eigenvalue, or the L2 and H1 errors
    values: dict[str, float] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def mark_cells(
    eta: npt.NDArray[np.floating],
    theta: float,
    degrees: DegreeField,
    strategy: MarkingStrategy | str = MarkingStrategy.maximum,
) -> npt.NDArray[np.int32]:
    r"""Mark the cells to enrich, among the cells that are not at the maximal degree.

    See :class:`MarkingStrategy`. The indicator of cells at the
    maximal degree is set to zero, as the error at a singularity would otherwise keep selecting
    the same (saturated) cells.

    Args:
        eta: Error indicator of each owned cell.
        theta: Marking parameter in (0, 1].
        degrees: The degree of each cell.
        strategy: Marking strategy.

    Returns:
        The (local) indices of the marked owned cells.
    """
    strategy = MarkingStrategy(strategy)
    mesh = degrees.mesh
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    values = dolfinx.la.vector(cell_map, dtype=eta.dtype)
    values.array[: cell_map.size_local] = eta if strategy == MarkingStrategy.maximum else eta**2
    values.scatter_forward()
    values.array[degrees.cell_degrees >= degrees.pmax] = 0.0
    if strategy == MarkingStrategy.maximum:
        marked = dolfinx.mesh.mark_maximum(values.array, cell_map, theta)
    else:
        marked = dolfinx.mesh.mark_equidistribution(values.array, cell_map, theta)
    return marked[marked < cell_map.size_local]


def mark_cells_for_coarsening(
    eta: npt.NDArray[np.floating], theta: float, degrees: DegreeField
) -> npt.NDArray[np.int32]:
    r"""Mark the cells to coarsen: the owned cells with :math:`p_K > p_{min}` and
    :math:`\eta_K < \theta \max_{K'} \eta_{K'}`.

    Note:
        Collective. Use a ``theta`` much smaller than the one used for enrichment, such that
        the cells do not alternate between being enriched and coarsened.

    Args:
        eta: Error indicator of each owned cell.
        theta: Coarsening parameter in (0, 1).
        degrees: The degree of each cell.

    Returns:
        The (local) indices of the marked cells.
    """
    eta_max = degrees.mesh.comm.allreduce(eta.max(initial=0.0), op=MPI.MAX)
    candidates = degrees.cell_degrees[: degrees.num_owned_cells] > degrees.pmin
    return np.flatnonzero(candidates & (eta < theta * eta_max)).astype(np.int32)


def degree_histogram(degrees: DegreeField) -> dict[int, int]:
    """Global number of cells of each degree."""
    counts = np.bincount(
        degrees.cell_degrees[: degrees.num_owned_cells], minlength=degrees.pmax + 1
    )
    counts = degrees.mesh.comm.allreduce(counts, op=MPI.SUM)
    return {p: int(c) for p, c in enumerate(counts) if c > 0}


def solve_adaptive(
    approach: Approach,
    theta: float = 0.5,
    strategy: MarkingStrategy | str = MarkingStrategy.maximum,
    max_iterations: int = 15,
    tol: float = 1e-8,
    coarsen_theta: float | None = None,
    callback: typing.Callable[[Approach, IterationResult], None] | None = None,
) -> list[IterationResult]:
    """Run the P-adaptive loop.

    Every iteration rebuilds the space for the current degrees, assembles and solves the problem
    of the approach, estimates the error and increases the degree of the marked cells (and,
    optionally, decreases the degree of the cells with a small error).

    Note:
        Collective.

    Args:
        approach: The representation of the variable degree space (holding the problem).
        theta: Marking parameter, see :func:`mark_cells`.
        strategy: Marking strategy, see :func:`mark_cells`.
        max_iterations: Maximal number of iterations.
        tol: Stop when the error measure of the problem is below this tolerance.
        coarsen_theta: If given, also decrease the degree of the cells marked by
            :func:`mark_cells_for_coarsening` with this parameter (except the cells marked for
            enrichment).
        callback: Called with the approach (holding the solution) and the result of every
            iteration, before the degrees are increased.

    Returns:
        The result of each iteration.
    """
    history = []
    degrees = approach.degrees
    comm = approach.mesh.comm
    problem = approach.problem
    for it in range(max_iterations):
        timings = {}
        t = time.perf_counter()
        approach.update()
        timings["update"] = time.perf_counter() - t

        # Assembly and solve
        t = time.perf_counter()
        approach.solve()
        timings["solve"] = time.perf_counter() - t

        t = time.perf_counter()
        eta = approach.estimate()
        timings["estimate"] = time.perf_counter() - t

        error, values = problem.errors(approach.errors())
        eta_global = float(np.sqrt(comm.allreduce(np.sum(eta**2), op=MPI.SUM)))
        result = IterationResult(
            iteration=it,
            num_dofs=approach.num_dofs,
            # Without an error measure, the error indicator is used
            error=eta_global if error is None else error,
            eta=eta_global,
            degrees=degree_histogram(degrees),
            values=values,
            timings=timings,
        )
        history.append(result)
        if callback is not None:
            callback(approach, result)

        if result.error < tol:
            break
        refine = mark_cells(eta, theta, degrees, strategy)
        coarsen = np.empty(0, dtype=np.int32)
        if coarsen_theta is not None:
            coarsen = np.setdiff1d(mark_cells_for_coarsening(eta, coarsen_theta, degrees), refine)
        if degrees.change(refine, coarsen) == 0:
            break
    return history
