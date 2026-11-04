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

# # Continuous Galerkin with Nitsche coupling
#
# Author: Jørgen S. Dokken
#
# In the previous demos, the space is discontinuous on every facet. Here we use *continuous*
# Lagrange elements in the region $\Omega_i$ of each degree $i$, and couple the regions weakly
# across the interfaces $\mathcal{F}_{ij}$, $i<j$, with Nitsche's method
# {cite}`continuous-nitsche1971`, using the SIPG terms {cite}`continuous-arnold2002` on the
# interfaces only. On the facets inside a region, the jump of a continuous function is zero, so
# these terms vanish and are not assembled.
#
# Both the [submesh](./demo_submesh) and the [restricted dofmap](./demo_restricted) approaches
# support this: the space $V_i$ is continuous within $\Omega_i$, and its degrees of freedom on an
# interface are separate from those of $V_j$, so the coupling is discontinuous exactly where the
# Nitsche terms are applied.
# The [hierarchical approach](./demo_hierarchical) does not, as selecting rows and columns of a
# continuous space cannot break its continuity at the interfaces, which move at every iteration.

# +
from mpi4py import MPI

import pyvista

from dolfinx_padapt import DegreeField, create_pacman
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import RestrictedApproach, SubmeshApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import plot_iteration
from dolfinx_padapt.postprocessing import print_header, print_iteration, print_summary

mesh = create_pacman(MPI.COMM_WORLD, h=0.1, h_corner=0.005)
params = Parameters(pmin=1, pmax=6, continuous=True)
# -

# ## Submeshes
# We run the adaptive loop with continuous spaces on the submeshes, and keep the final plot.

# + tags=["scroll-output"]
degrees = DegreeField(mesh, params.pmin, params.pmax)
submesh_approach = SubmeshApproach(degrees, params)
print_header()
histories = {
    "submesh": solve_adaptive(submesh_approach, max_iterations=12, callback=print_iteration)
}
# -

# We plot the solution and the degree of each cell. The solution is continuous within each region
# of cells with the same degree, while the regions are coupled weakly.

plotter = plot_iteration(submesh_approach, title="Continuous, submesh")
if plotter is not None:
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot("continuous.png")

# ## Restricted dofmaps
# With the same parameters, the restricted dofmaps give the same eigenvalue at every iteration.
# For continuous spaces, a degree of freedom on the boundary of $\Omega_i$ may be owned by a process
# that has no cell of degree $i$, so {py:func}`dolfinx.common.create_sub_index_map` changes its
# owner.

# + tags=["scroll-output"]
degrees = DegreeField(mesh, params.pmin, params.pmax)
restricted_approach = RestrictedApproach(degrees, params)
print_header()
histories["restricted"] = solve_adaptive(
    restricted_approach, max_iterations=12, callback=print_iteration
)
# -

print_summary(histories)

# ```{note}
# The indicator is dominated by the jump terms on the interfaces between the degree regions.
# Increasing the degree of a cell creates new interfaces, so the enriched region grows one layer of
# cells at a time, and the continuous method needs more iterations than the discontinuous one.
# ```
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: continuous-
# ```
