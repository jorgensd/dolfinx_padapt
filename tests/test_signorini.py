"""The Signorini problem solved with the latent variable proximal point method."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl

from dolfinx_padapt.approaches import APPROACHES, SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.mesh import HALF_DISK_CURVED, HALF_DISK_FLAT, create_half_disk
from dolfinx_padapt.models import SignoriniProblem

U_D = 0.72
PLANE = -0.13


@pytest.fixture(scope="module")
def half_disk():
    return create_half_disk(MPI.COMM_WORLD, h=0.12)


def signorini(mesh, facet_tags):
    return SignoriniProblem(
        mesh,
        facet_tags,
        dirichlet_marker=HALF_DISK_FLAT,
        contact_marker=HALF_DISK_CURVED,
        displacement=lambda x: ufl.as_vector((0.0, -U_D)),
        gap=lambda x: x[1] - PLANE,
        direction=(0.0, -1.0),
    )


def mixed_degrees(mesh, pmin, pmax):
    """Degrees that depend on the global cell index only."""
    imap = mesh.topology.index_map(mesh.topology.dim)
    gidx = imap.local_to_global(np.arange(imap.size_local, dtype=np.int32))
    return (pmin + (gidx * 7919) % (pmax - pmin + 1)).astype(np.int32)


@pytest.mark.parametrize("uniform", [1, 2, None])
def test_signorini_contact(half_disk, uniform):
    """No penetration, a compressive contact pressure, and no pressure outside the contact
    region, for a uniform and a mixed degree."""
    mesh, facet_tags = half_disk
    problem = signorini(mesh, facet_tags)
    params = Parameters(pmin=1, pmax=2, continuous=True)
    approach = SubmeshApproach(problem, params)
    num_cells = approach.degrees.num_owned_cells
    if uniform is None:
        approach.degrees.set_degrees(mixed_degrees(mesh, params.pmin, params.pmax))
    else:
        approach.degrees.set_degrees(np.full(num_cells, uniform, dtype=np.int32))
    approach.update()
    approach.solve()
    _, statistics = problem.errors(approach.errors())
    assert statistics["proximal_iterations"] > 1

    # Penetration of the plane by the displacement
    u = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ("DG", params.pmax, (2,))))
    approach.interpolate_solution(u)
    x = ufl.SpatialCoordinate(mesh)
    ds = ufl.Measure(
        "ds", domain=mesh, subdomain_data=facet_tags, metadata={"quadrature_degree": 8}
    )
    gap = x[1] - PLANE
    penetration = ufl.max_value(-u[1] - gap, 0) * ds(HALF_DISK_CURVED)
    contact_length = ufl.conditional(ufl.lt(gap + u[1], 1e-3), 1, 0) * ds(HALF_DISK_CURVED)
    penetration, contact_length = (
        mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)
        for form in (penetration, contact_length)
    )
    assert contact_length > 0.1
    # The constraint holds weakly (tested with the latent space), so the displacement penetrates
    # the plane by the order of the discretisation error: on average less than 0.1% of u_D
    assert penetration < 1e-3 * U_D * contact_length

    # The contact pressure at the nodes of the latent variable
    pressures, positions = [], []
    force = 0.0
    for (i, name), psi in approach.solution.items():
        if name != "psi":
            continue
        Q = psi.function_space
        num_owned = Q.dofmap.index_map.size_local
        pressure = dolfinx.fem.Function(Q)
        pressure.x.array[:] = psi.x.array - approach.previous_solution[(i, name)].x.array
        pressure.x.array[:] /= problem.alpha
        force += dolfinx.fem.assemble_scalar(dolfinx.fem.form(pressure * ufl.dx(domain=Q.mesh)))
        pressures.append(pressure.x.array[:num_owned])
        positions.append(Q.tabulate_dof_coordinates()[:num_owned, 0])
    pressures, positions = np.concatenate(pressures), np.concatenate(positions)
    force = mesh.comm.allreduce(force, op=MPI.SUM)
    scale = mesh.comm.allreduce(np.max(np.abs(pressures), initial=0.0), op=MPI.MAX)
    assert force < 0
    assert np.all(pressures <= 1e-6 * scale)
    # Far from the plane (near the corners of the half disk), there is no contact
    assert np.all(np.abs(pressures[np.abs(positions) > 0.45]) < 1e-6 * scale)


@pytest.mark.parametrize("name", ["restricted", "hierarchical"])
def test_signorini_requires_submesh(half_disk, name):
    mesh, facet_tags = half_disk
    with pytest.raises(NotImplementedError):
        APPROACHES[name](signorini(mesh, facet_tags), Parameters(pmin=1, pmax=2, continuous=True))
