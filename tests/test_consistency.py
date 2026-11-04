"""Consistency of the three P-adaptive approaches with each other and with a reference
implementation on a single space (uniform degree)."""

from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pytest
import ufl

from dolfinx_padapt import DegreeField, PoissonProblem, create_pacman
from dolfinx_padapt.adaptivity import MarkingStrategy, mark_cells, mark_cells_for_coarsening
from dolfinx_padapt.approaches import APPROACHES
from dolfinx_padapt.eigen import solve_eigenproblem
from dolfinx_padapt.forms import (
    Parameters,
    cell_size,
    nitsche_boundary,
    sipg_interior,
)
from dolfinx_padapt.problems import ElasticityProblem, StokesProblem

QDEG = 8
PMAX = 3


@pytest.fixture(scope="module")
def mesh():
    return create_pacman(MPI.COMM_WORLD, h=0.35)


def solve(name, mesh, params, owned_degrees, problem=None):
    """Solve on the given degrees, and return the first reported value (the eigenvalue by
    default), the global error indicator and the number of dofs."""
    values, eta, ndofs = solve_all(name, mesh, params, owned_degrees, problem)
    return next(iter(values.values())), eta, ndofs


def solve_all(name, mesh, params, owned_degrees, problem=None):
    degrees = DegreeField(mesh, params.pmin, params.pmax)
    degrees.set_degrees(owned_degrees)
    approach = APPROACHES[name](degrees, params, problem)
    approach.update()
    A, B = approach.assemble()
    x = approach.problem.solve(A, B, approach.nullspace())
    approach.set_solution(x)
    eta = approach.estimate()
    eta_global = np.sqrt(mesh.comm.allreduce(np.sum(eta**2), op=MPI.SUM))
    _, values = approach.problem.errors(approach.errors())
    return values, eta_global, approach.num_dofs


def reference(mesh, k, alpha, continuous):
    """SIPG (or CG + Nitsche) on a single space of degree k, with standard forms."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange" if continuous else "DG", k))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    h = cell_size(mesh)
    n = ufl.FacetNormal(mesh)
    md = {"quadrature_degree": QDEG}
    dx, ds, dS = (ufl.Measure(m, domain=mesh, metadata=md) for m in ("dx", "ds", "dS"))
    sigma = alpha * k**2
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx + nitsche_boundary(u, v, n, h, sigma) * ds
    if not continuous:
        a += sipg_interior(u, u, v, v, n, h, sigma) * dS
    m = ufl.inner(u, v) * dx
    A = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a))
    A.assemble()
    M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(m))
    M.assemble()
    lam, _ = solve_eigenproblem(A, M)
    return lam, V.dofmap.index_map.size_global


def owned_cells(mesh):
    return mesh.topology.index_map(mesh.topology.dim).size_local


def pseudo_random_degrees(mesh, pmin, pmax):
    """Degrees that depend on the global cell index only."""
    imap = mesh.topology.index_map(mesh.topology.dim)
    gidx = imap.local_to_global(np.arange(imap.size_local, dtype=np.int32))
    return (pmin + (gidx * 7919) % (pmax - pmin + 1)).astype(np.int32)


@pytest.mark.parametrize("name", APPROACHES.keys())
@pytest.mark.parametrize("k", [1, 2, 3])
def test_uniform_degree_dg(mesh, name, k):
    params = Parameters(pmin=1, pmax=PMAX, quadrature_degree=QDEG)
    lam, _, ndofs = solve(name, mesh, params, np.full(owned_cells(mesh), k, dtype=np.int32))
    lam_ref, ndofs_ref = reference(mesh, k, params.alpha, continuous=False)
    assert ndofs == ndofs_ref
    assert np.isclose(lam, lam_ref, rtol=1e-9)


@pytest.mark.parametrize("name", ["restricted", "submesh"])
@pytest.mark.parametrize("k", [1, 2])
def test_uniform_degree_cg(mesh, name, k):
    params = Parameters(pmin=1, pmax=PMAX, quadrature_degree=QDEG, continuous=True)
    lam, _, ndofs = solve(name, mesh, params, np.full(owned_cells(mesh), k, dtype=np.int32))
    lam_ref, ndofs_ref = reference(mesh, k, params.alpha, continuous=True)
    assert ndofs == ndofs_ref
    assert np.isclose(lam, lam_ref, rtol=1e-9)


def test_hierarchical_rejects_cg(mesh):
    params = Parameters(pmin=1, pmax=PMAX, continuous=True)
    with pytest.raises(NotImplementedError):
        APPROACHES["hierarchical"](DegreeField(mesh, 1, PMAX), params)


@pytest.mark.parametrize("penalty", ["local", "global"])
def test_mixed_degrees_dg(mesh, penalty):
    params = Parameters(pmin=1, pmax=PMAX, penalty=penalty, quadrature_degree=QDEG)
    degrees = pseudo_random_degrees(mesh, params.pmin, params.pmax)
    results = {name: solve(name, mesh, params, degrees) for name in APPROACHES}
    lam0, eta0, ndofs0 = results["hierarchical"]
    for name, (lam, eta, ndofs) in results.items():
        assert ndofs == ndofs0, name
        assert np.isclose(lam, lam0, rtol=1e-9), name
        assert np.isclose(eta, eta0, rtol=1e-8), name


def test_mixed_degrees_cg(mesh):
    params = Parameters(pmin=1, pmax=PMAX, quadrature_degree=QDEG, continuous=True)
    degrees = pseudo_random_degrees(mesh, params.pmin, params.pmax)
    lam_r, eta_r, ndofs_r = solve("restricted", mesh, params, degrees)
    lam_s, eta_s, ndofs_s = solve("submesh", mesh, params, degrees)
    assert ndofs_r == ndofs_s
    assert np.isclose(lam_r, lam_s, rtol=1e-9)
    assert np.isclose(eta_r, eta_s, rtol=1e-8)


@pytest.mark.parametrize("continuous", [False, True])
def test_degree_on_single_rank(mesh, continuous):
    """The highest degree only lives on rank 0, i.e. other ranks have no cells of that degree."""
    params = Parameters(pmin=1, pmax=PMAX, quadrature_degree=QDEG, continuous=continuous)
    degrees = np.ones(owned_cells(mesh), dtype=np.int32)
    if mesh.comm.rank == 0:
        degrees[: len(degrees) // 2] = PMAX
    names = ["restricted", "submesh"] + ([] if continuous else ["hierarchical"])
    results = [solve(name, mesh, params, degrees) for name in names]
    for lam, eta, ndofs in results[1:]:
        assert ndofs == results[0][2]
        assert np.isclose(lam, results[0][0], rtol=1e-9)
        assert np.isclose(eta, results[0][1], rtol=1e-8)


def test_refine(mesh):
    degrees = DegreeField(mesh, 1, 2)
    num_owned = owned_cells(mesh)
    cells = np.arange(min(5, num_owned), dtype=np.int32)
    changed = degrees.refine(cells)
    assert changed == mesh.comm.allreduce(len(cells), op=MPI.SUM)
    assert np.all(degrees.cell_degrees[cells] == 2)
    assert degrees.refine(cells) == 0  # Capped at pmax
    assert 2 in degrees.active_degrees()


@pytest.mark.parametrize("strategy", list(MarkingStrategy))
def test_mark_cells(mesh, strategy):
    """Cells at the maximal degree are never marked, the others by the given strategy."""
    degrees = DegreeField(mesh, 1, 2)
    num_owned = owned_cells(mesh)
    rng = np.random.default_rng(mesh.comm.rank)
    eta = rng.random(num_owned)
    # The cells with the largest indicator are at the maximal degree
    saturated = eta > 0.8
    degrees.set_degrees(np.where(saturated, 2, 1).astype(np.int32))
    marked = mark_cells(eta, 0.5, degrees, strategy)
    assert np.all(marked < num_owned)
    assert not np.any(saturated[marked])
    assert mesh.comm.allreduce(len(marked), op=MPI.SUM) > 0


# Poisson problem on a unit cube


@pytest.fixture(scope="module")
def cube():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2)


def quadratic(x):
    return 1 + x[0] ** 2 + 2 * x[1] * x[2] - x[2] ** 2


def bump(x):
    return ufl.exp(-10 * ((x[0] - 0.3) ** 2 + (x[1] - 0.4) ** 2 + (x[2] - 0.6) ** 2))


@pytest.mark.parametrize("continuous", [False, True])
def test_poisson_exact_quadratic(cube, continuous):
    """A quadratic solution is reproduced on any distribution of degrees >= 2."""
    params = Parameters(pmin=2, pmax=3, quadrature_degree=6, continuous=continuous)
    problem = PoissonProblem(cube, quadratic)
    degrees = pseudo_random_degrees(cube, params.pmin, params.pmax)
    names = ["restricted", "submesh"] + ([] if continuous else ["hierarchical"])
    for name in names:
        values, _, _ = solve_all(name, cube, params, degrees, problem)
        assert values["L2"] < 1e-10, name
        assert values["H1"] < 1e-9, name


def poisson_reference(mesh, k, alpha, u_exact):
    """L2 error of SIPG on a single DG space of degree k, with standard forms."""
    V = dolfinx.fem.functionspace(mesh, ("DG", k))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    h = cell_size(mesh)
    n = ufl.FacetNormal(mesh)
    md = {"quadrature_degree": QDEG}
    dx, ds, dS = (ufl.Measure(m, domain=mesh, metadata=md) for m in ("dx", "ds", "dS"))
    x = ufl.SpatialCoordinate(mesh)
    g = u_exact(x)
    f = -ufl.div(ufl.grad(g))
    sigma = alpha * k**2
    F = (
        ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        - f * v * dx
        + sipg_interior(u, u, v, v, n, h, sigma) * dS
        + nitsche_boundary(u, v, n, h, sigma, g) * ds
    )
    a, L = ufl.lhs(F), ufl.rhs(F)
    problem = dolfinx.fem.petsc.LinearProblem(
        a,
        L,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="poisson_reference_",
    )
    uh = problem.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    error = dolfinx.fem.form((uh - g) ** 2 * dx)
    return np.sqrt(mesh.comm.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM))


@pytest.mark.parametrize("name", APPROACHES.keys())
@pytest.mark.parametrize("k", [1, 2])
def test_poisson_uniform_degree(cube, name, k):
    params = Parameters(pmin=1, pmax=2, quadrature_degree=QDEG)
    problem = PoissonProblem(cube, bump)
    values, _, _ = solve_all(name, cube, params, np.full(owned_cells(cube), k, np.int32), problem)
    assert np.isclose(values["L2"], poisson_reference(cube, k, params.alpha, bump), rtol=1e-8)


def test_poisson_mixed_degrees(cube):
    params = Parameters(pmin=1, pmax=2, quadrature_degree=QDEG)
    problem = PoissonProblem(cube, bump)
    degrees = pseudo_random_degrees(cube, params.pmin, params.pmax)
    results = {name: solve_all(name, cube, params, degrees, problem) for name in APPROACHES}
    values0, eta0, ndofs0 = results["hierarchical"]
    for name, (values, eta, ndofs) in results.items():
        assert ndofs == ndofs0, name
        for key in ("L2", "H1"):
            assert np.isclose(values[key], values0[key], rtol=1e-8), (name, key)
        assert np.isclose(eta, eta0, rtol=1e-8), name


def test_coarsen(mesh):
    """Coarsening lowers the degree by one, but never below pmin."""
    degrees = DegreeField(mesh, 1, 3)
    num_owned = owned_cells(mesh)
    degrees.set_degrees(np.full(num_owned, 2, dtype=np.int32))
    cells = np.arange(num_owned, dtype=np.int32)
    assert degrees.coarsen(cells) == mesh.comm.allreduce(num_owned, op=MPI.SUM)
    assert np.all(degrees.cell_degrees == 1)
    assert degrees.coarsen(cells) == 0  # Bounded below by pmin
    # Enrich and coarsen at the same time
    half = cells[: num_owned // 2]
    degrees.change(half, np.empty(0, dtype=np.int32))
    degrees.change(half, cells[num_owned // 2 :])
    assert np.all(degrees.cell_degrees[half] == 3)
    assert np.all(degrees.cell_degrees[num_owned // 2 : num_owned] == 1)
    assert mark_cells_for_coarsening(np.zeros(num_owned), 0.1, degrees).size == 0


# Linear elasticity and Stokes (submesh approach only)


@pytest.fixture(scope="module")
def square():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)


def quadratic_displacement(x):
    return ufl.as_vector((x[0] ** 2 + x[1], x[0] * x[1] - x[1] ** 2))


def stokes_velocity(x):
    # The curl of the stream function x²y - xy², which is divergence free
    return ufl.as_vector((x[0] ** 2 - 2 * x[0] * x[1], x[1] ** 2 - 2 * x[0] * x[1]))


def stokes_pressure(x):
    return x[0] - x[1]


def test_elasticity_exact_quadratic(square):
    """A quadratic displacement is reproduced on any distribution of degrees >= 2."""
    params = Parameters(pmin=2, pmax=3, quadrature_degree=6)
    problem = ElasticityProblem(square, quadratic_displacement, mu=1.0, lmbda=2.0)
    degrees = pseudo_random_degrees(square, params.pmin, params.pmax)
    values, _, _ = solve_all("submesh", square, params, degrees, problem)
    assert values["L2"] < 1e-10
    assert values["H1"] < 1e-9


@pytest.mark.parametrize("continuous", [False, True])
def test_stokes_exact_quadratic(square, continuous):
    """A quadratic velocity and linear pressure are reproduced on any distribution of degrees
    >= 2 (the pressure has one degree less), with a pressure nullspace."""
    params = Parameters(pmin=2, pmax=3, quadrature_degree=6, continuous=continuous)
    problem = StokesProblem(square, stokes_velocity, stokes_pressure)
    degrees = pseudo_random_degrees(square, params.pmin, params.pmax)
    values, _, _ = solve_all("submesh", square, params, degrees, problem)
    assert values["u_L2"] < 1e-10
    assert values["u_H1"] < 1e-9
    assert values["p_L2"] < 1e-9


def test_stokes_converges(square):
    """The errors decrease with the degree (including degree 1/0)."""
    problem = StokesProblem(
        square,
        lambda x: ufl.as_vector(
            (
                ufl.sin(ufl.pi * x[1]) * ufl.cos(ufl.pi * x[0]),
                -ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
            )
        ),
        lambda x: ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1]),
    )
    params = Parameters(pmin=1, pmax=3, quadrature_degree=8)
    errors = [
        solve_all("submesh", square, params, np.full(owned_cells(square), k, np.int32), problem)[0]
        for k in (1, 2, 3)
    ]
    for key in ("u_H1", "p_L2"):
        assert errors[1][key] < errors[0][key] and errors[2][key] < errors[1][key], key


@pytest.mark.parametrize("name", ["restricted", "hierarchical"])
def test_vector_problems_require_submesh(square, name):
    params = Parameters(pmin=1, pmax=2)
    problem = ElasticityProblem(square, quadratic_displacement)
    with pytest.raises(NotImplementedError):
        APPROACHES[name](DegreeField(square, 1, 2), params, problem)
