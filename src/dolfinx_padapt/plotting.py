"""Visualise a P-adaptive solution with PyVista, and the convergence with Matplotlib.

For every active degree ``i``, the solution is interpolated onto a DG-``i`` Lagrange space on a
submesh of the cells with degree ``i``, and converted to a PyVista grid. The grids of every
process are gathered on one process (as in the ``amr.py`` demo of the DOLFINx tutorial) and drawn
in a single plot: the solution with iso-lines, the cell edges, and the nodes of each space (as
spheres coloured by degree), showing where the space has been enriched.
"""

import typing
import weakref

from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.plot
import numpy as np
import pyvista
from matplotlib.colors import to_rgb

from .adaptivity import IterationResult, degree_histogram
from .approaches import Approach
from .degrees import DegreeField
from .postprocessing import print_header, print_iteration

#: Categorical colours, assigned in fixed order (to the degrees from ``pmin``, or to the runs)
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
]
SOLUTION_CMAP = "viridis"
EDGE_COLOR = "#3d3d3a"
TEXT_COLOR = "#1f1f1e"


def degree_color(degree: int, pmin: int) -> str:
    return CATEGORICAL_COLORS[(degree - pmin) % len(CATEGORICAL_COLORS)]


def create_grids(
    approach: Approach,
) -> tuple[dict[int, pyvista.UnstructuredGrid], dict[int, pyvista.PolyData], pyvista.DataSet]:
    """Create the PyVista grids of the owned cells on this process.

    Note:
        Collective.

    Returns:
        For each active degree: the solution on a Lagrange grid of that degree and the nodes of
        that grid. Also returns the (curved) edges of the owned cells.
    """
    mesh = approach.mesh
    degrees = approach.degrees
    tdim = mesh.topology.dim

    # The solution on a broken space of the maximal degree is exact on every cell
    u_parent = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", degrees.pmax)))
    approach.interpolate_solution(u_parent)

    solutions, nodes = {}, {}
    for i in degrees.active_degrees():
        submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(mesh, tdim, degrees.cells(i))
        V = dolfinx.fem.functionspace(submesh, ("DG", i))
        u = dolfinx.fem.Function(V)
        cell_imap = submesh.topology.index_map(tdim)
        sub_cells = np.arange(cell_imap.size_local + cell_imap.num_ghosts, dtype=np.int32)
        parent_cells = cell_map.sub_topology_to_topology(sub_cells, inverse=False)
        u.interpolate(u_parent, cells0=parent_cells, cells1=sub_cells)

        grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V))
        grid.point_data["u"] = u.x.array.real
        solutions[i] = grid
        nodes[i] = pyvista.PolyData(V.tabulate_dof_coordinates())

    # Curved cell edges: the facets of the owned cells, tessellated to line segments
    mesh.topology.create_connectivity(tdim, tdim - 1)
    owned = np.arange(degrees.num_owned_cells, dtype=np.int32)
    facets = dolfinx.mesh.compute_incident_entities(mesh.topology, owned, tdim, tdim - 1)
    edges = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, tdim - 1, facets))
    if edges.n_cells > 0:
        edges = edges.tessellate()
    return solutions, nodes, edges


def _add_legend(plotter: pyvista.Plotter, entries: list[tuple[str, str]]):
    """Legend with coloured discs and labels in the text colour.

    :meth:`pyvista.Plotter.add_legend` draws each label in the colour of its entry. The entries
    are therefore given the text colour, and each disc is coloured by its own RGB scalars.
    """
    legend = plotter.add_legend(
        [[label, TEXT_COLOR] for label, _ in entries],
        face="circle",
        bcolor="white",
        size=(0.1, 0.04 * len(entries)),
        loc="upper right",
    )
    for k, (_, color) in enumerate(entries):
        symbol = pyvista.wrap(legend.GetEntrySymbol(k)).copy()
        rgb = np.round(np.array(to_rgb(color)) * 255).astype(np.uint8)
        symbol.point_data["rgb"] = np.tile(rgb, (symbol.n_points, 1))
        symbol.point_data.active_scalars_name = "rgb"
        legend.SetEntrySymbol(k, symbol)
    legend.ScalarVisibilityOn()


def plot_solution(
    approach: Approach,
    plotter: pyvista.Plotter | None = None,
    root: int = 0,
    *,
    show_nodes: bool = True,
    node_size: float = 8.0,
    num_isolines: int = 10,
    title: str | None = None,
) -> pyvista.Plotter | None:
    """Plot the solution, the cell edges and (optionally) the nodes coloured by degree.

    Note:
        Collective. The grids are gathered and drawn on ``root``, all other processes return
        ``None``.

    Args:
        approach: Approach holding the current solution.
        plotter: Plotter to draw in (on ``root``), which is cleared first, e.g. to write a frame
            of an animation. If ``None``, a new plotter is created.
        root: Process gathering the grids and plotting.
        show_nodes: Draw the nodes of each degree as spheres.
        node_size: Sphere size (in pixels).
        num_isolines: Number of iso-lines of the solution (0 for none).
        title: Text in the upper left corner.

    Returns:
        The plotter on ``root``, ``None`` on the other processes.
    """
    if approach.mesh.geometry.dim != 2:
        raise NotImplementedError(
            "The nodes are drawn in the plane, use plot_degrees and plot_function in 3D"
        )
    if len(approach.problem.fields) != 1 or approach.problem.fields[0].shape != ():
        raise NotImplementedError("The nodes are only drawn for a scalar problem")
    comm = approach.mesh.comm
    assert root < comm.size, f"Cannot gather on process {root} of {comm.size}"
    pieces = comm.gather(create_grids(approach), root=root)
    if comm.rank != root:
        return None
    assert pieces is not None

    solution = pyvista.merge(
        [grid for solutions, _, _ in pieces for grid in solutions.values() if grid.n_cells > 0]
    ).tessellate()
    # The sign of an eigenfunction is arbitrary, choose the one with a positive extremum
    u = solution.point_data["u"]
    if u[np.argmax(np.abs(u))] < 0:
        solution.point_data["u"] = -u

    if plotter is None:
        plotter = pyvista.Plotter(window_size=[1200, 1000])
    else:
        plotter.clear()
        plotter.remove_legend()
    plotter.set_background("white")
    plotter.add_mesh(
        solution,
        scalars="u",
        cmap=SOLUTION_CMAP,
        scalar_bar_args={"title": "u", "color": TEXT_COLOR, "vertical": True},
    )
    if num_isolines > 0:
        isolines = solution.contour(isosurfaces=num_isolines, scalars="u")
        if isolines.n_points > 0:
            plotter.add_mesh(isolines, color=TEXT_COLOR, line_width=1.5)
    edges = [e for _, _, e in pieces if e.n_cells > 0]
    if edges:
        plotter.add_mesh(pyvista.merge(edges), color=EDGE_COLOR, line_width=1.0)

    if show_nodes:
        pmin = approach.degrees.pmin
        entries = []
        for i in sorted({i for _, nodes, _ in pieces for i in nodes}):
            clouds = [n[i] for _, n, _ in pieces if i in n and n[i].n_points > 0]
            if not clouds:
                continue
            color = degree_color(i, pmin)
            cloud = pyvista.merge(clouds)
            # A white halo behind each node separates it from any colour of the solution. The
            # nodes are lifted above the solution by more than their radius, as the depth of a
            # sphere is that of its front.
            for c, size, z in (("white", node_size + 4, 0.02), (color, node_size, 0.05)):
                points = cloud.copy()
                points.points[:, 2] = z
                plotter.add_mesh(points, color=c, render_points_as_spheres=True, point_size=size)
            entries.append((f"P{i}", color))
        _add_legend(plotter, entries)

    if title is not None:
        plotter.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)
    plotter.view_xy()
    plotter.enable_parallel_projection()
    return plotter


def _gather(mesh: dolfinx.mesh.Mesh, grid: pyvista.UnstructuredGrid, root: int):
    """Gather the grids of every process on ``root`` and merge them (``None`` elsewhere)."""
    assert root < mesh.comm.size, f"Cannot gather on process {root} of {mesh.comm.size}"
    pieces = mesh.comm.gather(grid, root=root)
    if pieces is None:
        return None
    return pyvista.merge([piece for piece in pieces if piece.n_cells > 0])


def _new_plotter(plotter: pyvista.Plotter | None) -> pyvista.Plotter:
    if plotter is None:
        plotter = pyvista.Plotter(window_size=[1000, 900])
    else:
        plotter.clear()
        plotter.remove_legend()
    plotter.set_background("white")
    return plotter


def _show_interior(grid: pyvista.UnstructuredGrid, clip: bool) -> pyvista.UnstructuredGrid:
    """In 3D, remove the (whole) cells in the octant of the bounding box that faces the camera
    of the isometric view, to show the interior."""
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    if not clip or zmax - zmin < 1e-12:
        return grid
    centre = np.array([xmin + xmax, ymin + ymax, zmin + zmax]) / 2
    in_octant = np.all(grid.cell_centers().points > centre, axis=1)
    return grid.extract_cells(np.flatnonzero(~in_octant))


def _set_view(plotter: pyvista.Plotter, gdim: int):
    if gdim == 2:
        plotter.view_xy()
        plotter.enable_parallel_projection()
    else:
        plotter.view_isometric()


def _degree_grid(degrees: DegreeField, root: int) -> pyvista.UnstructuredGrid | None:
    """The owned cells with their degree, gathered on ``root`` (collective)."""
    mesh = degrees.mesh
    owned = np.arange(degrees.num_owned_cells, dtype=np.int32)
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim, owned))
    grid.cell_data["p"] = degrees.cell_degrees[: degrees.num_owned_cells]
    return _gather(mesh, grid, root)


def _function_grid(u: dolfinx.fem.Function, root: int) -> pyvista.UnstructuredGrid | None:
    """The function on the owned cells, gathered on ``root`` (collective)."""
    # The owned cells, with the degrees of freedom on the process as points. A vector is shown by
    # its magnitude.
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u.function_space))
    values = u.x.array.real.reshape(grid.n_points, -1)
    grid.point_data["u"] = values[:, 0] if values.shape[1] == 1 else np.linalg.norm(values, axis=1)
    merged = _gather(u.function_space.mesh, grid, root)
    return None if merged is None else merged.tessellate()


def _draw_degrees(plotter: pyvista.Plotter, grid: pyvista.UnstructuredGrid, pmin: int, clip: bool):
    grid = _show_interior(grid, clip)
    entries = []
    for i in np.unique(grid.cell_data["p"]):
        cells = grid.extract_cells(np.flatnonzero(grid.cell_data["p"] == i))
        color = degree_color(int(i), pmin)
        # Light edges, so that the colour of the cells stays visible where they are small
        plotter.add_mesh(cells, color=color, show_edges=True, edge_color="white", line_width=0.5)
        entries.append((f"P{i}", color))
    _add_legend(plotter, entries)


def _draw_function(
    plotter: pyvista.Plotter,
    grid: pyvista.UnstructuredGrid,
    name: str,
    origin: tuple[float, float, float] | None,
):
    scalar_bar = {"title": name, "color": TEXT_COLOR, "vertical": True, "fmt": "%.2f"}
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    if zmax - zmin > 1e-12:
        x, y, z = origin if origin is not None else grid.center
        slices = grid.slice_orthogonal(x=x, y=y, z=z)
        plotter.add_mesh(slices, scalars="u", cmap=SOLUTION_CMAP, scalar_bar_args=scalar_bar)
        plotter.add_mesh(grid.outline(), color=EDGE_COLOR, line_width=1.5)
    else:
        plotter.add_mesh(grid, scalars="u", cmap=SOLUTION_CMAP, scalar_bar_args=scalar_bar)


def _add_title(plotter: pyvista.Plotter, title: str | None):
    if title is not None:
        plotter.add_text(title, position="upper_left", font_size=12, color=TEXT_COLOR)


#: The bar chart of the degrees of each plotter. A chart is drawn in place at every frame, as
#: PyVista does not draw new charts in a renderer whose last chart has been removed.
_degree_charts: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _draw_degree_histogram(
    plotter: pyvista.Plotter, counts: dict[int, int], pmin: int, pmax: int, num_cells: int
):
    """A bar chart of the number of cells of each degree, with a fixed range (all cells)."""
    from matplotlib.figure import Figure

    chart = _degree_charts.get(plotter)
    if chart is None:
        # Not created with pyplot, such that the figure keeps the canvas of the chart
        fig = Figure(figsize=(5, 4))
        fig.add_subplot()
        chart = pyvista.ChartMPL(fig, size=(1, 1), loc=(0, 0))
        plotter.add_chart(chart)
        _degree_charts[plotter] = chart
    fig = chart.figure
    ax = fig.axes[0]
    ax.clear()

    labels = [f"P{p}" for p in range(pmin, pmax + 1)]
    heights = [counts.get(p, 0) for p in range(pmin, pmax + 1)]
    colors = [degree_color(p, pmin) for p in range(pmin, pmax + 1)]
    bars = ax.bar(labels, heights, color=colors, width=0.6)
    ax.bar_label(bars, color=TEXT_COLOR, padding=2)
    ax.set_ylim(0, 1.1 * num_cells)
    ax.set_ylabel("Number of cells", color=TEXT_COLOR)
    ax.set_xlabel("Degree", color=TEXT_COLOR)
    ax.grid(True, axis="y", color="#e0dfda", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    fig.tight_layout()
    fig.canvas.draw()


def plot_degrees(
    degrees: DegreeField,
    plotter: pyvista.Plotter | None = None,
    root: int = 0,
    *,
    clip: bool = True,
    title: str | None = None,
) -> pyvista.Plotter | None:
    """Plot the cells coloured by their polynomial degree.

    Note:
        Collective. The grids are gathered and drawn on ``root``, all other processes return
        ``None``.

    Args:
        degrees: The degree of each cell.
        plotter: Plotter to draw in (on ``root``), which is cleared first. If ``None``, a new
            plotter is created.
        root: Process gathering the grids and plotting.
        clip: In 3D, remove the cells of one octant to show the interior.
        title: Text in the upper left corner.
    """
    grid = _degree_grid(degrees, root)
    if grid is None:
        return None
    plotter = _new_plotter(plotter)
    _draw_degrees(plotter, grid, degrees.pmin, clip)
    _add_title(plotter, title)
    _set_view(plotter, degrees.mesh.geometry.dim)
    return plotter


def plot_function(
    u: dolfinx.fem.Function,
    plotter: pyvista.Plotter | None = None,
    root: int = 0,
    *,
    origin: tuple[float, float, float] | None = None,
    title: str | None = None,
) -> pyvista.Plotter | None:
    """Plot a (scalar, Lagrange) function on the parent mesh, e.g. the solution interpolated with
    :meth:`dolfinx_padapt.approaches.Approach.interpolate_solution`.

    In 3D, the function is shown on three orthogonal slices through ``origin``, together with the
    outline of the domain.

    Note:
        Collective. The grids are gathered and drawn on ``root``, all other processes return
        ``None``.

    Args:
        u: The function.
        plotter: Plotter to draw in (on ``root``), which is cleared first. If ``None``, a new
            plotter is created.
        root: Process gathering the grids and plotting.
        origin: The point the slices pass through in 3D (default: the centre of the domain).
        title: Text in the upper left corner.
    """
    grid = _function_grid(u, root)
    if grid is None:
        return None
    plotter = _new_plotter(plotter)
    _draw_function(plotter, grid, u.name, origin)
    _add_title(plotter, title)
    _set_view(plotter, u.function_space.mesh.geometry.dim)
    return plotter


def plot_iteration(
    approach: Approach,
    plotter: pyvista.Plotter | None = None,
    root: int = 0,
    *,
    origin: tuple[float, float, float] | None = None,
    histogram: bool | None = None,
    field: str | None = None,
    title: str | None = None,
) -> pyvista.Plotter | None:
    """Plot the solution (left) and the degrees (right) side by side, e.g. to write a frame of an
    animation of the adaptive loop.

    The solution is interpolated into a discontinuous space of the maximal degree, and shown on
    three orthogonal slices through ``origin`` in 3D. The degrees are shown as the mesh coloured by
    the degree of each cell, or as a bar chart of the number of cells of each degree.

    Note:
        Collective. The grids are gathered and drawn on ``root``, all other processes return
        ``None``.

    Args:
        approach: Approach holding the current solution.
        plotter: Plotter with ``shape=(1, 2)`` to draw in (on ``root``), which is cleared first.
            If ``None``, a new plotter is created.
        root: Process gathering the grids and plotting.
        origin: The point the slices pass through in 3D (default: the centre of the domain).
        histogram: Show the number of cells of each degree as a bar chart instead of the mesh.
            Defaults to a bar chart in 3D, where the cells inside the domain are hidden.
        field: The field of the solution to plot (by default the first). A vector is shown by its
            magnitude.
        title: Text in the upper left corner.
    """
    mesh = approach.mesh
    fields = {f.name: f for f in approach.problem.fields}
    field = field or next(iter(fields))
    degrees = approach.degrees
    gdim = mesh.geometry.dim
    histogram = gdim == 3 if histogram is None else histogram
    element = basix.ufl.element(
        "DG",
        mesh.basix_cell(),
        max(fields[field].degree(degrees.pmax), 1),
        basix.LagrangeVariant.equispaced,
        shape=fields[field].shape,
    )
    name = f"{field}_h" if fields[field].shape == () else f"|{field}_h|"
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element), name=name)
    approach.interpolate_solution(u, field)
    function_grid = _function_grid(u, root)
    if histogram:
        counts = degree_histogram(degrees)
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
    else:
        degree_grid = _degree_grid(degrees, root)
    if function_grid is None:
        return None

    # The sign of an eigenfunction is arbitrary, choose the one with a positive extremum
    values = function_grid.point_data["u"]
    if values[np.argmax(np.abs(values))] < 0:
        function_grid.point_data["u"] = -values

    if plotter is None:
        plotter = pyvista.Plotter(shape=(1, 2), border=False, window_size=[1600, 700])
    for k in range(2):
        plotter.subplot(0, k)
        # The bar chart is drawn in place, and clearing the actors would remove the actor that
        # draws it
        # Plotter.clear_actors clears all subplots
        if not plotter.renderer.has_charts:
            plotter.renderer.clear_actors()
            plotter.renderer.remove_legend()
        plotter.set_background("white")
    plotter.subplot(0, 0)
    _draw_function(plotter, function_grid, u.name, origin)
    _add_title(plotter, title)
    _set_view(plotter, gdim)
    plotter.subplot(0, 1)
    if histogram:
        _draw_degree_histogram(plotter, counts, degrees.pmin, degrees.pmax, num_cells)
        # The image of the chart is updated during a render, after the frame has been drawn
        plotter.render()
    else:
        _draw_degrees(plotter, degree_grid, degrees.pmin, clip=True)
        _set_view(plotter, gdim)
        plotter.link_views()
    return plotter


#: Marker of each run in the line plots. Runs that converge identically lie on top of each other,
#: so each run has its own marker, drawn smaller than the one before, such that coinciding curves
#: show as nested markers.
_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]


def _line_style(k: int) -> dict:
    return {
        "linestyle": "-",
        "marker": _MARKERS[k % len(_MARKERS)],
        "color": CATEGORICAL_COLORS[k % len(CATEGORICAL_COLORS)],
        "linewidth": 2.0 if k == 0 else 1.0,
        "markersize": max(10 - 3 * k, 3),
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
    }


def _style_axes(ax):
    ax.grid(True, which="major", color="#e0dfda", linewidth=0.8)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")


class Animation:
    """A GIF of the adaptive loop, with the solution and the degrees at each iteration (see
    :func:`plot_iteration`). It is a callback of
    :func:`dolfinx_padapt.adaptivity.solve_adaptive`, which also prints the progress::

        with Animation(mesh.comm, "solution.gif") as animation:
            solve_adaptive(approach, callback=animation)

    Note:
        Collective. The plotter only exists on ``root``.

    Args:
        comm: The communicator of the mesh.
        filename: The GIF file.
        root: Process gathering the grids and plotting.
        fps: Frames (iterations) per second.
        window_size: Size of each frame in pixels.
        verbose: Print the result of each iteration
            (:func:`dolfinx_padapt.postprocessing.print_iteration`).
        kwargs: Keyword arguments to :func:`plot_iteration`.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        filename: str,
        root: int = 0,
        *,
        fps: int = 1,
        window_size: tuple[int, int] = (1400, 600),
        verbose: bool = True,
        **kwargs,
    ):
        self.root = root
        self.verbose = verbose
        self.kwargs = kwargs
        self.plotter = None
        if comm.rank == root:
            self.plotter = pyvista.Plotter(
                shape=(1, 2), border=False, window_size=list(window_size)
            )
            self.plotter.open_gif(filename, fps=fps)
        if verbose:
            print_header()

    def __call__(self, approach: Approach, result: IterationResult):
        if self.verbose:
            print_iteration(approach, result)
        plotter = plot_iteration(
            approach, self.plotter, self.root, title=f"Iteration {result.iteration}", **self.kwargs
        )
        if plotter is not None:
            plotter.write_frame()

    def close(self):
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def plot_convergence(
    histories: dict[str, list[IterationResult]],
    filename: str | None = None,
    ylabel: str = "Error",
    annotate: typing.Sequence[str] = (),
):
    """The error measure of the problem against the number of degrees of freedom.

    Args:
        histories: The iterations of each run, labelled by name.
        filename: If given, save the figure to this file.
        ylabel: Label of the error axis.
        annotate: Names of the runs whose points are labelled with the iteration number, e.g. a
            run with coarsening, where the number of degrees of freedom can decrease.

    Returns:
        The Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, (name, history) in enumerate(histories.items()):
        dofs = [r.num_dofs for r in history]
        errors = [r.error for r in history]
        ax.loglog(dofs, errors, label=name, **_line_style(k))
        if name in annotate:
            for r in history:
                ax.annotate(
                    str(r.iteration),
                    (r.num_dofs, r.error),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color=TEXT_COLOR,
                )
    ax.set_xlabel("Number of degrees of freedom", color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    _style_axes(ax)
    if len(histories) > 1:
        ax.legend(frameon=False, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=150)
    return fig


def plot_history(
    histories: dict[str, list[IterationResult]],
    filename: str | None = None,
    ylabel: str = "Error",
):
    """The error measure (top) and the number of degrees of freedom (bottom) at each iteration.

    Args:
        histories: The iterations of each run, labelled by name.
        filename: If given, save the figure to this file.
        ylabel: Label of the error axis.

    Returns:
        The Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, (ax_error, ax_dofs) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    for k, (name, history) in enumerate(histories.items()):
        iterations = [r.iteration for r in history]
        ax_error.semilogy(iterations, [r.error for r in history], label=name, **_line_style(k))
        ax_dofs.plot(iterations, [r.num_dofs for r in history], label=name, **_line_style(k))
    ax_error.set_ylabel(ylabel, color=TEXT_COLOR)
    ax_dofs.set_ylabel("Number of degrees of freedom", color=TEXT_COLOR)
    ax_dofs.set_xlabel("Iteration", color=TEXT_COLOR)
    ax_dofs.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in (ax_error, ax_dofs):
        _style_axes(ax)
    if len(histories) > 1:
        ax_error.legend(frameon=False, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=150)
    return fig
