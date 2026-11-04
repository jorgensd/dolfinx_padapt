# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.19.1
# ---

# # Comparison of the approaches
#
# Author: Jørgen S. Dokken
#
# We run the three representations of a space with a cell-wise degree on the same problem:
#
# 1. [A hierarchical basis](./demo_hierarchical), restricted algebraically with sub-matrices.
# 2. [Restricted dofmaps](./demo_restricted) of one space per degree on the whole mesh.
# 3. [One submesh per degree](./demo_submesh).
#
# With the same penalty parameter, all three give the same discrete problem, so the eigenvalues
# agree at every iteration up to the quadrature: the hierarchical approach integrates every cell
# with the quadrature of the maximal degree.
# We use the penalty $\sigma = \alpha p_{max}^2$ on every facet, so that the hierarchical
# approach only assembles its matrices once.

# +
from mpi4py import MPI

from dolfinx_padapt import DegreeField, create_pacman
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import APPROACHES
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import plot_convergence
from dolfinx_padapt.postprocessing import print_summary

mesh = create_pacman(MPI.COMM_WORLD, h=0.1, h_corner=0.005)
params = Parameters(pmin=1, pmax=6, penalty="global")

histories = {}
for name, Approach in APPROACHES.items():
    degrees = DegreeField(mesh, params.pmin, params.pmax)
    approach = Approach(degrees, params)
    histories[name] = solve_adaptive(approach, max_iterations=12)
# -

# The summary shows the number of iterations, the final number of degrees of freedom and error, and
# the largest relative difference of the eigenvalue to the first approach over all iterations.
# The problem is too small for the timings to be meaningful, so we do not compare them.

print_summary(histories)

# All approaches follow the same convergence history:

if mesh.comm.rank == 0:
    plot_convergence(
        histories, "comparison_convergence.png", ylabel=r"$|\lambda_h - \lambda| / \lambda$"
    )

# ![convergence](./comparison_convergence.png)
#
# ## Which approach to use
#
# - **Submeshes** use the mixed-domain assembly of DOLFINx: the integrals over the
#   interfaces are taken over the parent mesh and couple the submeshes through their entity maps.
#   They support both discontinuous and continuous spaces with the same code, and the number of
#   degrees of freedom and the quadrature follow the active degrees.
# - **Restricted dofmaps** give the same results, but rely on hand made dofmaps whose rows outside
#   the restriction are not valid, need the neighbour trick for the kernels on the interfaces, and
#   keep the full dofmap of every degree on the whole mesh.
# - **A hierarchical basis** only assembles its matrices once with a fixed penalty, but is limited to
#   discontinuous spaces on simplices, stores the operator of the maximal degree on the whole mesh,
#   and integrates every cell with the quadrature of the maximal degree.
#
# We recommend the submesh approach.
