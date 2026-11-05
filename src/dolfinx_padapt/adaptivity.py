"""The adaptive loop: solve, estimate, mark, and enrich (p) or refine (h) the marked cells."""

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
from .refinement import BoundaryProjection
from .smoothness import decay_factor


class MarkingStrategy(str, Enum):
    """Strategy for marking the cells to enrich."""

    #: :math:`\eta_K > \theta \max_{K'} \eta_{K'}`, see :func:`dolfinx.mesh.mark_maximum`
    maximum = "maximum"
    #: :math:`\eta_K^2 > \theta^2 \Vert\eta\Vert^2 / N`, see
    #: :func:`dolfinx.mesh.mark_equidistribution`
    equidistribution = "equidistribution"


class Refinement(str, Enum):
    """What happens to a marked cell in the adaptive loop."""

    #: Increase the degree (the mesh is fixed)
    p = "p"
    #: Split the cell (the new cells keep the degree of their parent)
    h = "h"
    #: Increase the degree where the solution is smooth, and split the cell otherwise (or at the
    #: maximal degree), see :mod:`dolfinx_padapt.smoothness`
    hp = "hp"


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
    include_max_degree: bool = False,
) -> npt.NDArray[np.int32]:
    r"""Mark the cells to enrich (or refine).

    See :class:`MarkingStrategy`. Unless ``include_max_degree``, the indicator of cells at the
    maximal degree is set to zero, as the error at a singularity would otherwise keep selecting
    the same (saturated) cells.

    Args:
        eta: Error indicator of each owned cell.
        theta: Marking parameter in (0, 1].
        degrees: The degree of each cell.
        strategy: Marking strategy.
        include_max_degree: Also mark cells at the maximal degree (that can be split).

    Returns:
        The (local) indices of the marked owned cells.
    """
    strategy = MarkingStrategy(strategy)
    mesh = degrees.mesh
    cell_map = mesh.topology.index_map(mesh.topology.dim)
    values = dolfinx.la.vector(cell_map, dtype=eta.dtype)
    values.array[: cell_map.size_local] = eta if strategy == MarkingStrategy.maximum else eta**2
    values.scatter_forward()
    if not include_max_degree:
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
    refinement: Refinement | str = Refinement.p,
    smoothness_threshold: float = 0.2,
    boundary_projection: BoundaryProjection | None = None,
) -> list[IterationResult]:
    """Run the p-, h- or hp-adaptive loop.

    Every iteration rebuilds the space for the current degrees (and mesh), assembles and solves
    the problem of the approach, estimates the error and increases the degree of the marked cells
    (and, optionally, decreases the degree of the cells with a small error). With
    h-refinement, the marked cells are split instead. With hp-refinement, a marked cell is split
    instead of enriched if the solution is not smooth on it
    (:func:`dolfinx_padapt.smoothness.decay_factor` above ``smoothness_threshold``) or if it is at
    the maximal degree. The new cells get the degree of their parent cell
    (:meth:`dolfinx_padapt.approaches.Approach.refine`).

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
        refinement: p-, h- or hp-refinement.
        smoothness_threshold: With hp-refinement, a marked cell is enriched if the decay factor
            of the Legendre coefficients of the (first field of the) solution is below this
            threshold, and split otherwise.
        boundary_projection: With h- and hp-refinement, move the new nodes on the boundary to
            the exact boundary with this map (see
            :func:`dolfinx_padapt.refinement.refine_curved`).

    Returns:
        The result of each iteration.
    """
    refinement = Refinement(refinement)
    history = []
    comm = approach.mesh.comm
    problem = approach.problem
    for it in range(max_iterations):
        timings = {}
        t = time.perf_counter()
        # The degrees (and the mesh) of the current iteration
        degrees = approach.degrees
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
        # Cells at the maximal degree can be split
        splits = refinement in (Refinement.h, Refinement.hp)
        marked = mark_cells(eta, theta, degrees, strategy, include_max_degree=splits)
        split = np.empty(0, dtype=np.int32)
        enrich = np.empty(0, dtype=np.int32)
        if refinement == Refinement.p:
            enrich = marked
        elif refinement == Refinement.h:
            split = marked
        else:
            smooth = decay_factor(approach)[marked] <= smoothness_threshold
            enrich = marked[smooth & (degrees.cell_degrees[marked] < degrees.pmax)]
            split = np.setdiff1d(marked, enrich).astype(np.int32)
        coarsen = np.empty(0, dtype=np.int32)
        if coarsen_theta is not None:
            coarsen = np.setdiff1d(mark_cells_for_coarsening(eta, coarsen_theta, degrees), marked)
        changed = degrees.change(enrich, coarsen)
        if comm.allreduce(len(split), op=MPI.SUM) > 0:
            approach.refine(split, boundary_projection)
        elif changed == 0:
            break
    return history
