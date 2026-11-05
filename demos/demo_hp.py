# # hp-adaptivity
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# The previous demos only change the polynomial degree, on a mesh that is graded towards the
# re-entrant corner of the PacMan domain. Near a singularity, increasing the degree is not
# efficient: the error converges algebraically in the number of degrees of freedom, also for
# high degrees. Splitting the cells (h-refinement) is better at a singularity, and increasing
# the degree (p-refinement) where the solution is smooth. Combining the two, hp-adaptivity
# converges exponentially {cite}`hp-houston2005`.
#
# We solve the Laplace eigenvalue problem of [the submesh demo](./demo_submesh) on a
# *quasi-uniform* mesh, without grading towards the corner, and with third order geometry, with
#
# - p-refinement: the degree of the marked cells is increased;
# - h-refinement: the marked cells are split, and keep their degree;
# - hp-refinement: the degree of a marked cell is increased if the solution is smooth on the
#   cell, and the cell is split otherwise.
#
# ## Refinement of a curved mesh
# The cells are split with the algorithm of {cite}`hp-plaza2000`
# ({py:func}`dolfinx.mesh.refine`), which also gives the parent cell of each new cell.
# It only supports straight cells, while our mesh has second order geometry. The mesh is
# therefore refined in four steps ({py:func}`dolfinx_padapt.refinement.refine_curved`):
#
# 1. The straight mesh with the same topology is created with
#    {py:func}`dolfinx.fem.interpolate_geometry`.
# 2. The straight mesh is refined.
# 3. The refined mesh gets second order geometry with straight cells, again with
#    {py:func}`dolfinx.fem.interpolate_geometry`.
# 4. The coordinates of the original mesh are a vector valued function in its coordinate space
#    (quadratic Lagrange), which is defined on the straight mesh as well (the meshes have the same
#    topology and dofmap). It is interpolated into the coordinate space of the refined mesh, and
#    its values are the new node coordinates.
#
# The interpolation in the last step evaluates the function at points in the cells of another
# mesh. Usually, the cell containing each point, and the process owning it, are found by a search
# with bounding box trees. Here, every point is in the parent cell of the cell it belongs to, on
# the same process, so we create the
# {py:class}`dolfinx.geometry.PointOwnershipData` directly from the parent cells
# ({py:func}`dolfinx_padapt.refinement.interpolate_from_parent`). This is exact for
# discontinuous functions as well, as a point on a facet of a parent cell is evaluated in the
# parent of its own cell. The refined cells are nested in the curved cells of the original mesh.
#
# The refined cells are nested in the curved cells of the original mesh, so the boundary of the
# refined mesh is the boundary of the original mesh, which is an approximation of the unit circle.
# For high degrees, the error of this approximation would dominate, so the new nodes on the boundary
# are moved to the unit circle ({py:func}`dolfinx_padapt.mesh.pacman_boundary_projection`).
# The cells that are only enriched keep the geometry of the original mesh, so we use third order
# geometry.
#
# The new cells get the degree of their parent cell. The forms are compiled once, for the
# original mesh, and are created on each refined mesh.
#
# ## Smoothness indicator
# On a cell $K$ of degree $p_K$, the solution is expanded in the orthonormal (Legendre)
# polynomials $P_j$ of the reference cell, $u_h\vert_K = \sum_j a_j P_j$. The energy of the modes of
# degree $k$,
#
# $$
# E_k = \Bigl(\sum_{\deg P_j = k}\vert a_j\vert^2\Bigr)^{1/2},
# $$
#
# decays exponentially, $E_k\approx C e^{-\sigma_K k}$, if the solution is smooth, and slowly
# near a singularity {cite}`hp-mavriplis1994,hp-houston2005`. The decay rate $\sigma_K$ is the
# least squares fit of $\log E_k$ for $k=1,\dots,p_K$
# ({py:func}`dolfinx_padapt.smoothness.decay_factor`). A marked cell is enriched if
# $e^{-\sigma_K}$ is below a threshold (0.2 by default), and split otherwise, or if it is at the
# maximal degree. A cell of degree one has a single mode, and is always enriched first.

# +
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx_padapt import EigenvalueProblem, create_pacman
from dolfinx_padapt.adaptivity import Refinement, solve_adaptive
from dolfinx_padapt.approaches import SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.mesh import pacman_boundary_projection
from dolfinx_padapt.plotting import Animation, plot_convergence

params = Parameters(pmin=1, pmax=6)
# -

# ## The adaptive loops
# We start every run from the same quasi-uniform mesh. For h-refinement, the degree is fixed to
# two.

# + tags=["scroll-output"]
histories = {}
for refinement in (Refinement.p, Refinement.h, Refinement.hp):
    PETSc.Sys.Print(f"{refinement.value}-refinement")
    mesh = create_pacman(MPI.COMM_WORLD, h=0.3, order=3)
    run_params = Parameters(pmin=2, pmax=2) if refinement == Refinement.h else params
    approach = SubmeshApproach(EigenvalueProblem(mesh), run_params)
    kwargs = dict(
        max_iterations=16,
        refinement=refinement,
        boundary_projection=pacman_boundary_projection,
    )
    if refinement == Refinement.hp:
        with Animation(mesh.comm, "hp.gif") as animation:
            history = solve_adaptive(approach, callback=animation, **kwargs)
    else:
        history = solve_adaptive(approach, **kwargs)
    histories[f"{refinement.value}-refinement"] = history
# -

# <img src="./hp.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# The cells at the corner are split, and the degree is increased away from it.
#
# ## Convergence

if MPI.COMM_WORLD.rank == 0:
    plot_convergence(
        histories,
        "hp_convergence.png",
        ylabel=r"$\vert\lambda_h - \lambda\vert/\lambda$",
    )

# ![convergence](./hp_convergence.png)
#
# With p-refinement, the error stagnates once the cells at the corner have the maximal degree.
# h-refinement converges algebraically. hp-refinement converges fastest, until the error of the
# third order geometry of the cells that were never split, and the lack of a one-sided bound on the
# discrete eigenvalue, show.
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: hp-
# ```
