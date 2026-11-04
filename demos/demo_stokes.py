# # The Stokes equations
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# We solve the Stokes equations
#
# $$
# \begin{align}
# -\nu\Delta u + \nabla p &= f && \text{in } \Omega = [0, 1]^2,\\
# \nabla\cdot u &= 0 && \text{in } \Omega,\\
# u &= g && \text{on } \partial\Omega,
# \end{align}
# $$
#
# with a manufactured solution. The velocity is a vortex, given by the curl of the stream function
# $\psi(x) = \mathrm{e}^{-a\vert x - x_0\vert^2}$ with $a=40$ and $x_0 = (0.4, 0.55)$, such that it
# is divergence free, and the pressure is $p(x) = \sin(\pi x_0)\cos(\pi x_1)$.
#
# ## Discretisation
# Each cell of degree $k$ has a discontinuous velocity of degree $k$ and a discontinuous pressure
# of degree $k-1$. The problem has two fields, and each degree has one space for the velocity and
# one for the pressure on the submesh of its cells. The viscous term is discretised with the
# symmetric interior penalty method, and the velocity and pressure are coupled by
# {cite}`stokes-schotzau2002`
#
# $$
# b(v, q) = -\sum_K\int_K q\nabla\cdot v~\mathrm{d}x + \sum_F\int_F \{q\}[v\cdot n]~\mathrm{d}s
# + \int_{\partial\Omega} q\, v\cdot n~\mathrm{d}s.
# $$
#
# With a Dirichlet condition on the whole boundary, the pressure is only determined up to a
# constant. The approach gives the constant pressure as the nullspace of the matrix
# ({py:meth}`~dolfinx_padapt.approaches.Approach.nullspace`), and the error of the pressure is
# measured after removing its mean value.

# +
from mpi4py import MPI

import dolfinx
import pyvista
import ufl

from dolfinx_padapt import DegreeField
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import Animation, plot_history, plot_iteration
from dolfinx_padapt.problems import StokesProblem

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 12, 12)
x0 = (0.4, 0.55)


def u_exact(x):
    psi = ufl.exp(-40 * ((x[0] - x0[0]) ** 2 + (x[1] - x0[1]) ** 2))
    return ufl.as_vector((psi.dx(1), -psi.dx(0)))


def p_exact(x):
    return ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


problem = StokesProblem(mesh, u_exact, p_exact, viscosity=1.0)
# -

# ## The adaptive loop
# The error indicator has the residual of the momentum equation and of the divergence on each cell,
# and the jumps of the velocity and of the normal flux $(\nu\nabla u - pI)n$ on the facets.
# The adaptive loop uses the error of the velocity in the broken $H^1$-seminorm. We plot the
# magnitude of the velocity and the degree of each cell at every iteration.

# + tags=["scroll-output"]
params = Parameters(pmin=1, pmax=5)
degrees = DegreeField(mesh, params.pmin, params.pmax)
approach = SubmeshApproach(degrees, params, problem)
with Animation(mesh.comm, "stokes.gif") as animation:
    history = solve_adaptive(approach, max_iterations=12, tol=1e-4, callback=animation)
# -

# <img src="./stokes.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# The errors of the velocity and the pressure at each iteration:

if mesh.comm.rank == 0:
    plot_history(
        {"adaptive": history}, "stokes_history.png", ylabel=r"$\Vert\nabla(u_h-u_{ex})\Vert$"
    )
    for result in history[-3:]:
        print(f"{result.iteration}: {result.num_dofs} dofs, {result.values}")

# ![history](./stokes_history.png)
#
# Finally, we plot the pressure of the last iteration, together with the degrees.

plotter = plot_iteration(approach, field="p", title="Pressure")
if plotter is not None:
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot("stokes_pressure.png")

# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: stokes-
# ```
