"""Reporting and output of the adaptive loop."""

from pathlib import Path

from petsc4py import PETSc

import basix.ufl
import dolfinx

from .adaptivity import IterationResult
from .approaches import Approach

PHASES = ("update", "assemble", "solve", "estimate")


def print_header():
    """Print the header of the table written by :func:`print_iteration`."""
    PETSc.Sys.Print(
        f"{'it':>3} {'dofs':>8} {'error':>10} {'eta':>10} {'update':>7} {'assemble':>8}"
        f" {'solve':>7} {'estim.':>7}  values, degrees"
    )


def print_iteration(approach: Approach, result: IterationResult):
    """Print one row of the convergence table (can be used as a callback of
    :func:`dolfinx_padapt.adaptivity.solve_adaptive`)."""
    t = result.timings
    values = ", ".join(f"{name}={value:.10g}" for name, value in result.values.items())
    PETSc.Sys.Print(
        f"{result.iteration:3d} {result.num_dofs:8d} {result.error:10.3e} {result.eta:10.3e}"
        f" {t['update']:7.3f} {t['assemble']:8.3f} {t['solve']:7.3f} {t['estimate']:7.3f}"
        f"  {values}, {result.degrees}"
    )


def total_timings(history: list[IterationResult]) -> dict[str, float]:
    """Time spent in each phase, summed over all iterations."""
    return {phase: sum(r.timings[phase] for r in history) for phase in PHASES}


def max_relative_difference(
    history: list[IterationResult], reference: list[IterationResult], key: str | None = None
) -> float:
    """Largest relative difference of a reported value of two runs, over their common iterations.

    Args:
        history: The iterations of a run.
        reference: The iterations of the reference run.
        key: The value to compare (e.g. ``"lambda"``). Defaults to the first reported value.
    """
    n = min(len(history), len(reference))
    if n == 0:
        return 0.0
    key = key or next(iter(reference[0].values))
    return max(
        abs(history[i].values[key] - reference[i].values[key]) / abs(reference[i].values[key])
        for i in range(n)
    )


def print_summary(
    histories: dict[str, list[IterationResult]], key: str | None = None, timings: bool = False
):
    """Compare several runs: the number of iterations, the final number of dofs and error, and
    the largest relative difference of a reported value (by default the first) to the first run.

    Args:
        histories: The iterations of each run, labelled by name.
        key: The reported value to compare.
        timings: Also print the time spent in each phase, summed over all iterations.
    """
    header = f"{'approach':>13} {'its':>4} {'dofs':>8} {'error':>10} {'max rel. diff':>13}"
    if timings:
        header += f" {'update':>7} {'assemble':>8} {'solve':>7} {'estim.':>7}"
    PETSc.Sys.Print(header)
    reference = next(iter(histories.values()))
    for name, history in histories.items():
        row = (
            f"{name:>13} {len(history):4d} {history[-1].num_dofs:8d} {history[-1].error:10.3e}"
            f" {max_relative_difference(history, reference, key):13.2e}"
        )
        if timings:
            total = total_timings(history)
            row += (
                f" {total['update']:7.3f} {total['assemble']:8.3f} {total['solve']:7.3f}"
                f" {total['estimate']:7.3f}"
            )
        PETSc.Sys.Print(row)


class SolutionWriter:
    """Write the solution (as a broken function of the maximal degree) and the degree of each
    cell to VTX files, one time step per iteration.

    The mesh does not change in P-adaptivity, so a single file holds all iterations.
    """

    def __init__(self, approach: Approach, folder: Path | str, name: str | None = None):
        mesh = approach.mesh
        name = name or approach.name
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        element = basix.ufl.element(
            "DG", mesh.basix_cell(), approach.degrees.pmax, basix.LagrangeVariant.equispaced
        )
        self.u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element), name="u")
        self._writers = [
            dolfinx.io.VTXWriter(mesh.comm, folder / f"{name}_u.bp", [self.u], engine="BP4"),
            dolfinx.io.VTXWriter(
                mesh.comm, folder / f"{name}_p.bp", [approach.degrees.p], engine="BP4"
            ),
        ]

    def __call__(self, approach: Approach, result: IterationResult):
        """Write the current solution (can be used as a callback of
        :func:`dolfinx_padapt.adaptivity.solve_adaptive`)."""
        approach.interpolate_solution(self.u)
        for writer in self._writers:
            writer.write(float(result.iteration))

    def close(self):
        for writer in self._writers:
            writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
