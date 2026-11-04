# # One submesh per degree
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# In this demo, we solve the Laplace eigenvalue problem on the PacMan domain from the
# [adaptive mesh refinement demo](https://jorgensd.github.io/dolfinx-tutorial/chapter2/amr.html)
# of the DOLFINx tutorial. Instead of refining the mesh, we keep the mesh fixed and increase the
# polynomial degree of the cells with the largest error.
#
# We seek $(u, \lambda)$, $u\neq 0$, such that
#
# $$
# \begin{align}
# -\Delta u &= \lambda u && \text{in } \Omega,\\
# u &= 0 && \text{on } \partial\Omega,
# \end{align}
# $$
#
# where $\Omega$ is the unit disk with the quadrant $x>0, y<0$ removed. The first eigenvalue is
# $\lambda = j_{2/3,1}^2\approx 11.3947$, where $j_{2/3,1}$ is the first root of the Bessel function
# $J_{2/3}$. The eigenfunction behaves like $r^{2/3}$ at the re-entrant corner.
#
# The space has a polynomial degree $p_K$ on each cell $K$. We represent it by one function space
# per degree $i$, each defined on the submesh $\Omega_i$ of the cells with degree $i$.
# The spaces are coupled by the symmetric interior penalty (SIPG) method
# {cite}`submesh-arnold1982,submesh-arnold2002`, with Dirichlet conditions imposed by Nitsche's
# method {cite}`submesh-nitsche1971`.
#
# ## The mesh
# The mesh is generated with GMSH, with second order (curved) cells, and graded towards the
# re-entrant corner.

# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

from dolfinx_padapt import DegreeField, EigenvalueProblem, create_pacman
from dolfinx_padapt.adaptivity import mark_cells, solve_adaptive
from dolfinx_padapt.approaches import SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import Animation, plot_convergence

mesh = create_pacman(MPI.COMM_WORLD, h=0.3, h_corner=0.01)
num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
PETSc.Sys.Print(f"Number of cells: {num_cells}, geometry order: {mesh.geometry.cmaps[0].degree}")
# -

# ## The degree of each cell
# We store the degree of each cell as cell tags, where the value of each tag is the polynomial
# degree of the cell. All cells start with degree $p_{min}=1$, and the degree is capped at
# $p_{max}=6$.

params = Parameters(pmin=1, pmax=6)
degrees = DegreeField(mesh, params.pmin, params.pmax)
PETSc.Sys.Print(f"Degrees of the first cells on process 0: {degrees.ct.values[:10]}")

# ## Discretisation
# Let $V_i$ be the discontinuous Lagrange space of degree $i$ on $\Omega_i$. On the facets between
# a cell of degree $i$ and a cell of degree $j\geq i$, the restriction "+" is the cell with the
# lowest degree, i.e. $u_i$ on the "+" side and $u_j$ on the "-" side.
# The bilinear form is
#
# $$
# \begin{align}
# a(u, v) =& \sum_{i} \int_{\Omega_i} \nabla u_i\cdot\nabla v_i~\mathrm{d}x
# + \sum_{i\leq j}\int_{\mathcal{F}_{ij}} -\{\nabla u\}\cdot[v] - [u]\cdot\{\nabla v\}
# + \frac{\sigma_{ij}}{\{h\}}[u]\cdot[v]~\mathrm{d}s\\
# &+ \sum_i \int_{\partial\Omega_i\cap\partial\Omega} -\nabla u_i\cdot n\, v_i
# - u_i\nabla v_i\cdot n + \frac{\sigma_i}{h}u_iv_i~\mathrm{d}s,
# \end{align}
# $$
#
# where $\mathcal{F}_{ij}$ are the facets between the cells of degree $i$ and $j$,
# $[u] = u^+n^+ + u^-n^-$ and $\{\nabla u\}=\frac{1}{2}(\nabla u^+ + \nabla u^-)$.
# The penalty parameter $\sigma_{ij}=\alpha\max(i, j)^2$ scales with the square of the degree
# {cite}`submesh-epshteyn2007`.
#
# Every integral $\mathrm{d}x(i)$, $\mathrm{d}s(i)$ and $\mathrm{d}S(i, j)$ gets a unique tag.
# All forms, for every combination of degrees, are compiled **once** with
# {py:func}`dolfinx.fem.compile_form`, with the spaces defined on template domains.
# This happens when we create the approach for the eigenvalue problem:

problem = EigenvalueProblem(mesh)
approach = SubmeshApproach(degrees, params, problem)

# ## One iteration, step by step
# At every iteration, the approach extracts the submesh of each active degree with
# {py:func}`dolfinx.mesh.create_submesh`, creates the function spaces on them, and creates the forms
# with {py:func}`dolfinx.fem.create_form`. The integration entities are given in the numbering of
# the parent mesh, and the entity maps of the submeshes map them to the cells of each submesh.
# The interface entities are found with {py:func}`scifem.find_interface` and
# ordered with {py:func}`scifem.compute_interface_data`.

approach.update()
for i, (submesh, _) in approach.submeshes.items():
    n = submesh.topology.index_map(submesh.topology.dim).size_global
    PETSc.Sys.Print(
        f"Degree {i}: {n} cells, {approach.spaces[(i, 'u')].dofmap.index_map.size_global} dofs"
    )

# We assemble the blocked matrices and solve the eigenvalue problem with SLEPc, using a
# shift-and-invert spectral transform.

A, M = approach.assemble()
x = problem.solve(A, M)
approach.set_solution(x)
error, values = problem.errors(approach.errors())
PETSc.Sys.Print(f"lambda_h = {values['lambda']:.10f}, relative error {error:.2e}")

# ## Error indicator and marking
# We use the residual based error indicator of {cite}`submesh-larson2000` (as in the AMR demo of
# the DOLFINx tutorial), with the polynomial weights of the hp-DG indicator of
# {cite}`submesh-houston2007`:
#
# $$
# \begin{align}
# \eta_K^2 =&\ \frac{h_K^2}{p_K^2}\Vert \lambda_h u_h + \Delta u_h\Vert^2_{K}
# + \frac{h_K}{p_K}\left\Vert\frac{[\nabla u_h\cdot n]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}
# + \alpha^2\frac{p_K^3}{h_K}\left(\left\Vert\frac{[u_h]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}
# + \Vert u_h\Vert^2_{\partial K\cap\partial\Omega}\right).
# \end{align}
# $$
#
# The cells to enrich are chosen by maximum marking {cite}`submesh-babuska1978`,
# $\eta_K > \theta\max_{K'}\eta_{K'}$, with {py:func}`dolfinx.mesh.mark_maximum`, among the cells that
# are not yet at the maximal degree.

eta = approach.estimate()
marked = mark_cells(eta, theta=0.5, degrees=degrees)
num_marked = mesh.comm.allreduce(len(marked), op=MPI.SUM)
PETSc.Sys.Print(f"Marked {num_marked} of {num_cells} cells")

# We reset the degrees, and run the full adaptive loop. We track the solution and the degrees in
# a GIF, with the solution on the left and the degree of each cell on the right.

degrees.set_degrees(np.full(degrees.num_owned_cells, params.pmin, dtype=np.int32))

# + tags=["scroll-output"]
with Animation(mesh.comm, "submesh.gif") as animation:
    history = solve_adaptive(approach, max_iterations=15, callback=animation)
# -

# <img src="./submesh.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# The degree is first increased in the bulk of the domain, where the eigenfunction is smooth, while
# the cells at the corner singularity reach high degree only on the graded mesh.
#
# Finally, we plot the convergence of the eigenvalue.

if mesh.comm.rank == 0:
    plot_convergence(
        {"submesh": history}, "submesh_convergence.png", ylabel=r"$|\lambda_h - \lambda| / \lambda$"
    )

# ![convergence](./submesh_convergence.png)
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: submesh-
# ```
