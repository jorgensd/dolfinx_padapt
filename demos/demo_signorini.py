# # Contact: the Signorini problem
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# We consider a linear elastic half disk $\Omega$ pressed against a rigid plane, as in the
# [FEniCS workshop](https://jsdokken.com/FEniCS-workshop/src/multiphysics/coupling.html).
# The flat top $\Gamma_D$ of the disk is moved downwards by a prescribed displacement $u_D$, and
# the curved boundary $\Gamma$ may come into contact with the plane, which is at a distance $g$
# from $\Gamma$ in the direction $\hat{n} = (0, -1)$. The displacement satisfies
#
# $$
# \begin{align}
# -\nabla\cdot\sigma(u) &= 0 && \text{in } \Omega,\\
# u &= u_D && \text{on } \Gamma_D,\\
# u\cdot\hat{n} \leq g, \quad \sigma_n(u) \leq 0, \quad
# \sigma_n(u)(u\cdot\hat{n} - g) &= 0 && \text{on } \Gamma,
# \end{align}
# $$
#
# with $\sigma(u) = 2\mu\epsilon(u) + \lambda\mathrm{tr}(\epsilon(u))I$ and $\sigma_n(u)$ the normal
# stress on $\Gamma$, which is zero where the disk is not in contact.
#
# ## The latent variable proximal point method
# We impose the inequality constraint with the latent variable proximal point (LVPP) method
# {cite}`signorini-keith2024`, as for contact problems in {cite}`signorini-dokken2025`.
# A latent variable $\psi$ on $\Gamma$ gives the gap $g - u\cdot\hat{n} = e^{\psi}$, which is
# positive for any $\psi$. Starting from $\psi_0 = 0$, every proximal iteration $k$ solves the
# nonlinear problem: find $(u, \psi)$ such that
#
# $$
# \begin{align}
# \alpha_k(\sigma(u), \epsilon(v))_\Omega - (\psi - \psi_{k-1}, v\cdot\hat{n})_\Gamma &= 0,\\
# (u\cdot\hat{n} + e^{\psi} - g, w)_\Gamma &= 0,
# \end{align}
# $$
#
# for all $(v, w)$, with Newton's method, and sets $\psi_k = \psi$. The proximal parameter is
# doubled every iteration, $\alpha_k = 2^k\alpha_0$. The contact pressure is
# $\lambda = (\psi_k - \psi_{k-1})/\alpha_k \leq 0$.
#
# ## Discretisation with a variable degree
# The displacement is continuous within each region $\Omega_i$ of cells of degree $i$, and the
# regions are coupled with Nitsche's method, as in
# [the continuous demo](./demo_continuous.py). The Dirichlet condition is also imposed with
# Nitsche's method. The latent variable lives on a submesh of the facets of $\Gamma$ of each
# degree $i$, with the degree and continuity of the trace of the displacement. The nonlinear
# problems are solved with {py:class}`dolfinx.fem.petsc.NonlinearProblem`.
#
# The integrals over $\Gamma$ use a quadrature degree of $2i + 4$: in the contact region,
# $e^{\psi}$ varies by many orders of magnitude over a facet, and with the estimated quadrature
# degree Newton's method does not converge for degrees above one.

# +
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl

from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.mesh import HALF_DISK_CURVED, HALF_DISK_FLAT, create_half_disk
from dolfinx_padapt.models import SignoriniProblem
from dolfinx_padapt.plotting import Animation

mesh, facet_tags = create_half_disk(MPI.COMM_WORLD, radius=0.5, centre=1.0, h=0.06)
u_D = 0.72
plane = -0.13


def displacement(x):
    return ufl.as_vector((0.0, -u_D))


def gap(x):
    # The distance from Γ to the plane y = plane, in the direction (0, -1)
    return x[1] - plane


problem = SignoriniProblem(
    mesh,
    facet_tags,
    dirichlet_marker=HALF_DISK_FLAT,
    contact_marker=HALF_DISK_CURVED,
    displacement=displacement,
    gap=gap,
    direction=(0.0, -1.0),
    E=2e4,
    nu=0.4,
    alpha=0.1,
)
# -

# ## Error indicator
# The error indicator is the one of the [elasticity demo](./demo_elasticity.py), where the
# traction on $\Gamma$ is replaced by its residual: the difference between the traction and the
# contact pressure,
#
# $$
# \frac{h_K}{p_K}\Vert\sigma(u_h)n - \lambda_h\hat{n}\Vert^2_{\partial K\cap\Gamma}.
# $$
#
# There is no exact solution, so the adaptive loop stops at a given number of iterations, and we
# report the error indicator.
#
# ## The adaptive loop
# Every iteration restarts the proximal iterations from $\psi_0 = 0$ on the new spaces. We plot the
# magnitude of the displacement and the degree of each cell.

# + tags=["scroll-output"]
params = Parameters(pmin=1, pmax=3, continuous=True)
approach = SubmeshApproach(problem, params)
with Animation(mesh.comm, "signorini.gif") as animation:
    history = solve_adaptive(approach, max_iterations=10, callback=animation)
# -

# <img src="./signorini.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# ## The contact pressure
# The contact pressure $\lambda_h = (\psi_k - \psi_{k-1})/\alpha_k$ is in the space of the latent
# variable of each degree (a Lagrange space, so its coefficients are the values at the nodes). We
# plot it along $\Gamma$. The total contact force is the integral of the pressure over $\Gamma$.

# +
force = 0.0
points, pressures, degrees = [], [], []
for (i, name), psi in approach.solution.items():
    if name != "psi":
        continue
    psi_prev = approach.previous_solution[(i, name)]
    Q = psi.function_space
    pressure = dolfinx.fem.Function(Q)
    pressure.x.array[:] = (psi.x.array - psi_prev.x.array) / problem.alpha
    force += dolfinx.fem.assemble_scalar(dolfinx.fem.form(pressure * ufl.dx(domain=Q.mesh)))
    num_owned = Q.dofmap.index_map.size_local
    points.append(Q.tabulate_dof_coordinates()[:num_owned, 0])
    pressures.append(pressure.x.array[:num_owned])
    degrees.append(np.full(num_owned, i))
force = mesh.comm.allreduce(force, op=MPI.SUM)
PETSc.Sys.Print(f"Total contact force: {force:.6g}")

data = [np.concatenate(values) for values in (points, pressures, degrees)]
data = [mesh.comm.gather(values, root=0) for values in data]
if mesh.comm.rank == 0:
    x, p, d = (np.concatenate(values) for values in data)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in np.unique(d):
        ax.plot(x[d == i], p[d == i], "o", markersize=3, label=f"$p = {i}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"Contact pressure $\lambda_h$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("signorini_pressure.png")
# -

# ![pressure](./signorini_pressure.png)
#
# The highest degree is used along $\Gamma$, where the residual of the traction is measured, and
# at the corners between $\Gamma_D$ and $\Gamma$.
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: signorini-
# ```
