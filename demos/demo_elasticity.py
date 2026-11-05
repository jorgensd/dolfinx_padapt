# # Linear elasticity
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# We solve the equations of linear elasticity
#
# $$
# \begin{align}
# -\nabla\cdot\sigma(u) &= f && \text{in } \Omega = [0, 1]^2,\\
# u &= g && \text{on } \partial\Omega,\\
# \sigma(u) &= 2\mu\epsilon(u) + \lambda \mathrm{tr}(\epsilon(u))I, &&
# \epsilon(u) = \frac{1}{2}\left(\nabla u + \nabla u^T\right),
# \end{align}
# $$
#
# with a manufactured displacement, which is localised around a point $x_0$:
#
# $$
# u_{ex}(x) = \mathrm{e}^{-a\vert x - x_0\vert^2}\begin{pmatrix}1\\ x_0 - x\end{pmatrix},
# \qquad a = 60,\quad x_0 = (0.35, 0.6).
# $$
#
# The displacement is discontinuous across the interfaces between cells, and the problem is
# discretised with the symmetric interior penalty method {cite}`elasticity-arnold2002` with the
# stress $\sigma(u)$ in place of $\nabla u$, and with the Dirichlet condition imposed with Nitsche's
# method {cite}`elasticity-nitsche1971`. The penalty parameter is scaled by $2\mu + \lambda$.
# Vector valued problems are only implemented for the
# {py:class}`~dolfinx_padapt.approaches.SubmeshApproach`.

# +
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import ufl

from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import Animation, plot_convergence
from dolfinx_padapt.problems import ElasticityProblem

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 12, 12)
x0 = (0.35, 0.6)


def u_exact(x):
    bump = ufl.exp(-60 * ((x[0] - x0[0]) ** 2 + (x[1] - x0[1]) ** 2))
    return bump * ufl.as_vector((1, x0[0] - x[0]))


problem = ElasticityProblem(mesh, u_exact, mu=1.0, lmbda=10.0)
# -

# ## Error indicator
# The error indicator is the one of the other demos, with the stress in place of the gradient
# {cite}`elasticity-houston2007`:
#
# $$
# \begin{align}
# \eta_K^2 =&\ \frac{h_K^2}{p_K^2}\Vert f + \nabla\cdot\sigma(u_h)\Vert^2_{K}
# + \frac{h_K}{p_K}\left\Vert\frac{[\sigma(u_h) n]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}
# + (2\mu+\lambda)^2\alpha^2\frac{p_K^3}{h_K}\left(\left\Vert\frac{[u_h]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}
# + \Vert u_h - g\Vert^2_{\partial K\cap\partial\Omega}\right).
# \end{align}
# $$
#
# ## The adaptive loop
# The error is measured in the broken $H^1$-seminorm. We plot the magnitude of the displacement and
# the degree of each cell at every iteration.

# + tags=["scroll-output"]
params = Parameters(pmin=1, pmax=5)
approach = SubmeshApproach(problem, params)
degrees = approach.degrees
with Animation(mesh.comm, "elasticity.gif") as animation:
    history = solve_adaptive(approach, max_iterations=12, tol=1e-4, callback=animation)
# -

# <img src="./elasticity.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# ## Comparison with a uniform degree

# + tags=["scroll-output"]
uniform = []
for k in range(params.pmin, params.pmax + 1):
    degrees.set_degrees(np.full(degrees.num_owned_cells, k, dtype=np.int32))
    result = solve_adaptive(approach, max_iterations=1)[0]
    PETSc.Sys.Print(f"P{k}: {result.num_dofs} dofs, H1 error {result.error:.3e}")
    uniform.append(result)
# -

if mesh.comm.rank == 0:
    plot_convergence(
        {"adaptive": history, "uniform degree": uniform},
        "elasticity_convergence.png",
        ylabel=r"$\Vert\nabla(u_h - u_{ex})\Vert_{L^2(\Omega)}$",
    )

# ![convergence](./elasticity_convergence.png)
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: elasticity-
# ```
