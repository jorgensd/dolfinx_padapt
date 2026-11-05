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

    # The contact pressure. Its values at the nodes are not sign preserving where the degree of
    # the facets alternates, so the sign is checked in the integral sense: the positive part is
    # small compared to the total pressure.
    integrals = {"force": 0.0, "positive": 0.0, "total": 0.0, "far": 0.0}
    for (i, name), psi in approach.solution.items():
        if name != "psi":
            continue
        Q = psi.function_space
        pressure = dolfinx.fem.Function(Q)
        pressure.x.array[:] = psi.x.array - approach.previous_solution[(i, name)].x.array
        pressure.x.array[:] /= problem.alpha
        dx = ufl.Measure("dx", domain=Q.mesh, metadata={"quadrature_degree": 8})
        y = ufl.SpatialCoordinate(Q.mesh)
        corners = ufl.conditional(ufl.gt(abs(y[0]), 0.45), 1, 0)
        integrands = {
            "force": pressure,
            "positive": ufl.max_value(pressure, 0),
            "total": abs(pressure),
            "far": corners * abs(pressure),
        }
        for key, integrand in integrands.items():
            integrals[key] += dolfinx.fem.assemble_scalar(dolfinx.fem.form(integrand * dx))
    force, positive, total, far = (
        mesh.comm.allreduce(value, op=MPI.SUM) for value in integrals.values()
    )
    assert force < 0
    assert positive < 1e-2 * total
    # Far from the plane (near the corners of the half disk), there is no contact
    assert far < 1e-8 * total


@pytest.mark.parametrize("name", ["restricted", "hierarchical"])
def test_signorini_requires_submesh(half_disk, name):
    mesh, facet_tags = half_disk
    with pytest.raises(NotImplementedError):
        APPROACHES[name](signorini(mesh, facet_tags), Parameters(pmin=1, pmax=2, continuous=True))
