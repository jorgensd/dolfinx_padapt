"""Refinement of curved meshes, transfer of functions to the refined mesh, the smoothness indicator
and hp-adaptivity."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl
from conftest import all_ranks

from dolfinx_padapt import EigenvalueProblem, PoissonProblem, create_pacman
from dolfinx_padapt.adaptivity import Refinement, solve_adaptive
from dolfinx_padapt.approaches import APPROACHES, SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.mesh import (
    HALF_DISK_CURVED,
    HALF_DISK_FLAT,
    create_half_disk,
    pacman_boundary_projection,
)
from dolfinx_padapt.refinement import interpolate_from_parent, refine_curved
from dolfinx_padapt.smoothness import decay_factor

QDEG = 8


@pytest.fixture(scope="module")
def mesh():
    return create_pacman(MPI.COMM_WORLD, h=0.4)


def integrate(form) -> float:
    return MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)


def every_third_cell(mesh):
    return np.arange(0, mesh.topology.index_map(mesh.topology.dim).size_local, 3, dtype=np.int32)


@pytest.mark.parametrize("local", [False, True])
def test_refine_curved_is_nested(mesh, local):
    """The refined cells are nested in the curved cells: the area and the flux of x through the
    (curved) boundary are unchanged, and every cell is inside its parent."""
    cells = every_third_cell(mesh) if local else None
    refined = refine_curved(mesh, cells)
    fine = refined.mesh
    tdim = fine.topology.dim
    assert fine.geometry.cmaps[0].degree == mesh.geometry.cmaps[0].degree
    assert all_ranks(len(refined.parent_cells) == fine.topology.index_map(tdim).size_local)
    num_cells = [m.topology.index_map(tdim).size_global for m in (mesh, fine)]
    assert num_cells[1] > num_cells[0]
    if not local:
        assert num_cells[1] == 4 * num_cells[0]

    md = {"quadrature_degree": QDEG}
    for measure in ("dx", "ds"):
        values = []
        for m in (mesh, fine):
            x, n = ufl.SpatialCoordinate(m), ufl.FacetNormal(m)
            dm = ufl.Measure(measure, domain=m, metadata=md)
            values.append(integrate((1.0 if measure == "dx" else ufl.dot(x, n)) * dm))
        assert np.isclose(values[0], values[1], rtol=1e-12)

    # The midpoint of each cell is inside its parent
    midpoints = dolfinx.mesh.compute_midpoints(
        fine, tdim, np.arange(len(refined.parent_cells), dtype=np.int32)
    )
    tree = dolfinx.geometry.bb_tree(mesh, tdim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, midpoints)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, midpoints)
    inside = all(parent in colliding.links(i) for i, parent in enumerate(refined.parent_cells))
    assert all_ranks(inside)


def test_refine_transfers_facet_tags():
    mesh, facet_tags = create_half_disk(MPI.COMM_WORLD, h=0.15)
    refined = refine_curved(mesh, every_third_cell(mesh), facet_tags)
    lengths = []
    for m, tags in ((mesh, facet_tags), (refined.mesh, refined.facet_tags)):
        ds = ufl.Measure("ds", domain=m, subdomain_data=tags, metadata={"quadrature_degree": 8})
        lengths.append(
            [integrate(1.0 * ds(marker)) for marker in (HALF_DISK_FLAT, HALF_DISK_CURVED)]
        )
    assert np.isclose(lengths[1][0], 1.0)
    # The curved boundary is the (second order) boundary of the coarse mesh, up to the quadrature
    # of the arc length
    assert np.allclose(lengths[0], lengths[1], rtol=1e-6)


def test_interpolate_from_parent_discontinuous(mesh):
    """A discontinuous function is transferred exactly, also on the facets of the parent cells."""
    refined = refine_curved(mesh)
    element = ("DG", 2)
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, element))
    imap = u.function_space.dofmap.index_map
    rng = np.random.default_rng(imap.local_range[0])
    u.x.array[:] = rng.random(u.x.array.size)
    u.x.scatter_forward()
    u_fine = dolfinx.fem.Function(dolfinx.fem.functionspace(refined.mesh, element))
    interpolate_from_parent(u, u_fine, refined.parent_cells)
    md = {"quadrature_degree": QDEG}
    for f, g in ((u, u_fine),):
        for integrand in (lambda v: v, lambda v: v**2):
            coarse = integrate(integrand(f) * ufl.dx(domain=mesh, metadata=md))
            fine = integrate(integrand(g) * ufl.dx(domain=refined.mesh, metadata=md))
            assert np.isclose(coarse, fine, rtol=1e-10)


def quadratic(x):
    return 1 + x[0] ** 2 - x[0] * x[1]


def test_decay_factor_smooth_and_degree_one():
    """With a uniform degree three, the quadratic solution is exact and has no modes of degree
    three, so the decay is fast. Cells of degree one have the decay factor zero."""
    square = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    params = Parameters(pmin=1, pmax=3, quadrature_degree=QDEG)
    results = {}
    for name in APPROACHES:
        approach = APPROACHES[name](PoissonProblem(square, quadratic), params)
        num_owned = approach.degrees.num_owned_cells
        approach.degrees.set_degrees(np.full(num_owned, 3, dtype=np.int32))
        approach.update()
        approach.solve()
        assert all_ranks(np.all(decay_factor(approach) < 1e-3))

        degrees = np.where(np.arange(num_owned) % 2 == 0, 3, 1).astype(np.int32)
        approach.degrees.set_degrees(degrees)
        approach.update()
        approach.solve()
        factor = decay_factor(approach)
        assert all_ranks(np.all(factor[degrees == 1] == 0))
        assert all_ranks(np.all(factor[degrees == 3] > 0))
        results[name] = factor
    for factor in results.values():
        assert all_ranks(np.allclose(factor, results["submesh"], rtol=1e-6, atol=1e-12))


@pytest.mark.parametrize("name", APPROACHES.keys())
def test_refined_approach_matches_new_approach(mesh, name):
    """The forms compiled for the original mesh give the same problem on the refined mesh as an
    approach created on the refined mesh."""
    params = Parameters(pmin=1, pmax=3, quadrature_degree=QDEG, penalty="global")
    approach = APPROACHES[name](EigenvalueProblem(mesh), params)
    approach.degrees.set_degrees(np.full(approach.degrees.num_owned_cells, 2, np.int32))
    refined = approach.refine(every_third_cell(mesh))
    assert approach.mesh is refined.mesh
    assert all_ranks(np.all(approach.degrees.cell_degrees[: approach.degrees.num_owned_cells] == 2))
    approach.update()
    approach.solve()
    lam = approach.problem.errors(approach.errors())[1]["lambda"]

    reference = APPROACHES[name](EigenvalueProblem(refined.mesh), params)
    reference.degrees.set_degrees(np.full(reference.degrees.num_owned_cells, 2, np.int32))
    reference.update()
    reference.solve()
    lam_ref = reference.problem.errors(reference.errors())[1]["lambda"]
    assert np.isclose(lam, lam_ref, rtol=1e-10)
    assert approach.num_dofs == reference.num_dofs


def test_hp_adaptive_loop(mesh):
    approach = SubmeshApproach(EigenvalueProblem(mesh), Parameters(pmin=1, pmax=4))
    history = solve_adaptive(approach, max_iterations=6, refinement=Refinement.hp)
    cells = [sum(r.degrees.values()) for r in history]
    assert cells[-1] > cells[0]
    assert history[-1].error < 0.2 * history[0].error
    assert all(1 <= p <= 4 for p in history[-1].degrees)


def test_h_adaptive_loop_keeps_degrees(mesh):
    approach = SubmeshApproach(EigenvalueProblem(mesh), Parameters(pmin=2, pmax=4))
    history = solve_adaptive(approach, max_iterations=5, refinement=Refinement.h)
    assert all(set(r.degrees) == {2} for r in history)
    assert sum(history[-1].degrees.values()) > sum(history[0].degrees.values())
    assert history[-1].error < history[0].error


def test_refine_curved_boundary_projection(mesh):
    """With a projection, the new boundary nodes are on the unit circle, and the area is closer to
    the exact area."""
    refined = refine_curved(mesh, boundary_projection=pacman_boundary_projection)
    fine = refined.mesh
    tdim = fine.topology.dim
    facets = dolfinx.mesh.exterior_facet_indices(fine.topology)
    nodes = np.unique(dolfinx.mesh.entities_to_geometry(fine, tdim - 1, facets))
    x = fine.geometry.x[nodes]
    # The straight edges (including the corner at the origin, up to round-off)
    tol = 1e-12
    straight = (np.isclose(x[:, 1], 0) & (x[:, 0] >= -tol)) | (
        np.isclose(x[:, 0], 0) & (x[:, 1] <= tol)
    )
    assert all_ranks(np.allclose(np.linalg.norm(x[~straight, :2], axis=1), 1.0))

    exact = 3 * np.pi / 4
    areas = [
        integrate(1.0 * ufl.dx(domain=m, metadata={"quadrature_degree": QDEG}))
        for m in (mesh, fine)
    ]
    assert abs(areas[1] - exact) < 0.1 * abs(areas[0] - exact)
