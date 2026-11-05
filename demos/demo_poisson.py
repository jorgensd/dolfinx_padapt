# # The Poisson equation in 3D
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# The previous demos solve an eigenvalue problem. Here, we use the same library for the Poisson
# equation
#
# $$
# \begin{align}
# -\Delta u &= f && \text{in } \Omega = [0, 1]^3,\\
# u &= g && \text{on } \partial\Omega,
# \end{align}
# $$
#
# with a manufactured solution, such that we can measure the error. We choose a Gaussian bump
#
# $$
# u_{ex}(x) = \mathrm{e}^{-a\vert x - x_0\vert^2},\qquad a = 50,\quad x_0 = (0.3, 0.4, 0.6),
# $$
#
# and set $f=-\Delta u_{ex}$ and $g=u_{ex}$. The solution is smooth, but varies rapidly close to $x_0$
# and is almost constant away from it, so a high degree is only needed close to the bump.
#
# ## The problem
# The discretisation is the symmetric interior penalty method {cite}`poisson-arnold2002`, where the
# Dirichlet condition is imposed weakly with Nitsche's method {cite}`poisson-nitsche1971`.
# The residual of the degree $i$ cells is
#
# $$
# \begin{align}
# F_i(u; v) =& \int_{\Omega_i}\nabla u\cdot\nabla v - f v~\mathrm{d}x
# + \int_{\partial\Omega_i\cap\partial\Omega} -\nabla u\cdot n\, v - (u - g)\nabla v\cdot n
# + \frac{\sigma_i}{h}(u - g) v~\mathrm{d}s,
# \end{align}
# $$
#
# and the interface terms couple the degrees as in the [submesh demo](./demo_submesh).
# The bilinear and linear forms are extracted from the residual with {py:func}`ufl.lhs` and
# {py:func}`ufl.rhs`.
# The problem is described by a {py:class}`dolfinx_padapt.models.PoissonProblem`, which takes the
# manufactured solution as a function of the spatial coordinate:

# +
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl

from dolfinx_padapt import PoissonProblem
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import Animation, plot_convergence, plot_history
from dolfinx_padapt.postprocessing import print_header, print_iteration

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)


def u_exact(x):
    return ufl.exp(-50 * ((x[0] - 0.3) ** 2 + (x[1] - 0.4) ** 2 + (x[2] - 0.6) ** 2))


problem = PoissonProblem(mesh, u_exact)
num_cells = mesh.topology.index_map(mesh.topology.dim).size_global
PETSc.Sys.Print(f"Number of cells: {num_cells}")
# -

# The error is measured in the $L^2$-norm and the broken $H^1$-seminorm,
# $\Vert\nabla(u_h-u_{ex})\Vert_{L^2(\Omega)}$, which is the error measure of the adaptive loop.
#
# ## Error indicator
# The error indicator is the one of the eigenvalue demos, with the cell residual $f + \Delta u_h$
# and the boundary term $u_h - g$ {cite}`poisson-houston2007`:
#
# $$
# \begin{align}
# \eta_K^2 =&\ \frac{h_K^2}{p_K^2}\Vert f + \Delta u_h\Vert^2_{K}
# + \frac{h_K}{p_K}\left\Vert\frac{[\nabla u_h\cdot n]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}
# + \alpha^2\frac{p_K^3}{h_K}\left(\left\Vert\frac{[u_h]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}
# + \Vert u_h - g\Vert^2_{\partial K\cap\partial\Omega}\right).
# \end{align}
# $$
#
# ## The adaptive loop
# We use one submesh per degree, with degrees from 1 to 4, and the approach takes the problem as
# an argument. We track the progress in a GIF: the solution on three slices through the centre of
# the bump (left), and the number of cells of each degree (right).

params = Parameters(pmin=1, pmax=4)
approach = SubmeshApproach(problem, params)
degrees = approach.degrees

# + tags=["scroll-output"]
with Animation(mesh.comm, "poisson.gif", origin=(0.3, 0.4, 0.6)) as animation:
    history = solve_adaptive(approach, max_iterations=12, tol=1e-3, callback=animation)
# -

# <img src="./poisson.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# ## Comparison with a uniform degree
# We compare the adaptive solution with the solution of the same degree on every cell.

# + tags=["scroll-output"]
uniform = []
for k in range(params.pmin, params.pmax + 1):
    degrees.set_degrees(np.full(degrees.num_owned_cells, k, dtype=np.int32))
    result = solve_adaptive(approach, max_iterations=1)[0]
    PETSc.Sys.Print(f"P{k}: {result.num_dofs} dofs, H1 error {result.error:.3e}")
    uniform.append(result)
# -

# ## Coarsening
# The degree can also be lowered where the error is small. To see the effect of coarsening, we
# start two runs from degree 3 on every cell: one that only increases the degree, and one that at
# every iteration also lowers the degree (never below $p_{min}$) of the cells with
# $\eta_K < \theta_c \max_{K'}\eta_{K'}$, with $\theta_c = 0.05$. The cells marked for
# enrichment are not coarsened, and $\theta_c$ is much smaller than the marking parameter for
# enrichment, such that cells do not alternate between the two.

# + tags=["scroll-output"]
degrees.set_degrees(np.full(degrees.num_owned_cells, 3, dtype=np.int32))
print_header()
enriched = solve_adaptive(approach, max_iterations=8, tol=1e-3, callback=print_iteration)

degrees.set_degrees(np.full(degrees.num_owned_cells, 3, dtype=np.int32))
print_header()
coarsened = solve_adaptive(
    approach, max_iterations=8, tol=1e-3, coarsen_theta=0.05, callback=print_iteration
)
# -

# We plot the error and the number of degrees of freedom at each iteration of the two runs.

if mesh.comm.rank == 0:
    plot_history(
        {"from P3": enriched, "from P3, with coarsening": coarsened},
        "poisson_history.png",
        ylabel=r"$\Vert\nabla(u_h - u_{ex})\Vert_{L^2(\Omega)}$",
    )

# ![history](./poisson_history.png)
#
# Both runs enrich the cells close to the bump, and reach the same error once these cells have the
# maximal degree, after which the error cannot be reduced further on this mesh. With coarsening,
# the degree is also lowered away from the bump: the error decreases slightly slower in the first
# iterations, but with less than half of the degrees of freedom.
#
# Against the number of degrees of freedom (with the iteration number next to each point of the
# run with coarsening), all adaptive runs need far fewer degrees of freedom than a uniform degree
# to reach the same error, and coarsening removes the degrees of freedom that the initial degree
# put away from the bump.

if mesh.comm.rank == 0:
    plot_convergence(
        {
            "from P1": history,
            "from P3": enriched,
            "from P3, with coarsening": coarsened,
            "uniform degree": uniform,
        },
        "poisson_convergence.png",
        ylabel=r"$\Vert\nabla(u_h - u_{ex})\Vert_{L^2(\Omega)}$",
        annotate=["from P3, with coarsening"],
    )

# ![convergence](./poisson_convergence.png)
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: poisson-
# ```
