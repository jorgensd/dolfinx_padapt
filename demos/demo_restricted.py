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

# # Restricted dofmaps
#
# Author: Jørgen S. Dokken
#
# We solve the same Laplace eigenvalue problem as in [the submesh demo](./demo_submesh), but
# instead of creating a submesh for each degree, we create one function space $V_i$ per degree $i$
# on the *whole* mesh, and restrict its dofmap to the cells of degree $i$.
# The discretisation (SIPG {cite}`restricted-arnold2002` with weakly imposed boundary conditions
# {cite}`restricted-nitsche1971`) and the error indicator {cite}`restricted-larson2000` are the same.

# +
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import numpy as np
import pyvista

from dolfinx_padapt import DegreeField, create_pacman
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import RestrictedApproach, restrict_function_space
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import plot_iteration
from dolfinx_padapt.postprocessing import print_header, print_iteration

mesh = create_pacman(MPI.COMM_WORLD, h=0.1, h_corner=0.005)
params = Parameters(pmin=1, pmax=6)
degrees = DegreeField(mesh, params.pmin, params.pmax)
# -

# ## Restricting a dofmap
# Given the cells of degree $i$ (including ghost cells), we keep the degrees of freedom of the
# parent space $V_i$ in the closure of these cells. The parallel layout of the kept degrees of
# freedom is computed with {py:func}`dolfinx.common.create_sub_index_map`, and a new
# {py:class}`dolfinx.cpp.fem.DofMap` is created on the same mesh.
#
# As an illustration, we give the cells in the left half of the domain degree 2, and restrict a
# second order space to them.

V2 = dolfinx.fem.functionspace(mesh, ("DG", 2))
midpoints = dolfinx.mesh.compute_midpoints(
    mesh, mesh.topology.dim, np.arange(degrees.num_owned_cells, dtype=np.int32)
)
degrees.set_degrees(np.where(midpoints[:, 0] < 0, 2, 1).astype(np.int32))
cells = degrees.cells(2, include_ghosts=True)
V2_restricted, sub_to_parent = restrict_function_space(V2, cells)
PETSc.Sys.Print(
    f"Parent space: {V2.dofmap.index_map.size_global} dofs,"
    f" restricted space: {V2_restricted.dofmap.index_map.size_global} dofs"
)

# ```{admonition} Interface integrals see both cells
# :class: warning
# A kernel of an interior facet integral is always tabulated for *both* cells of the facet.
# On the interface between a cell of degree $i$ and a cell of degree $j$, the space $V_i$ only
# lives on one side, but the assembler still looks up the dofmap of the other cell.
# The dofmap of a cell outside $\Omega_i$ that shares a facet with $\Omega_i$ therefore points to
# the degrees of freedom of its neighbour in $\Omega_i$. The contributions to these are
# identically zero, as the restriction of $V_i$ to the outside cell never appears in the form, and
# they are added to an existing block of the matrix, so the sparsity pattern does not change.
# All other cells have the dofmap entries `-1`.
# With submeshes, the entity maps handle this, as a cell that is not part of a submesh is mapped
# to `-1`, and the assembler skips it.
# ```

# On each process, the owned cells of degree 1 next to a cell of degree 2 point to the degrees of
# freedom of their neighbour, the other cells of degree 1 have no degrees of freedom:

outside = np.flatnonzero(degrees.cell_degrees[: degrees.num_owned_cells] != 2)
rows = V2_restricted.dofmap.list[outside]
num_outside = mesh.comm.allreduce(len(outside), op=MPI.SUM)
num_neighbours = mesh.comm.allreduce(int(np.sum(rows[:, 0] >= 0)), op=MPI.SUM)
PETSc.Sys.Print(f"Cells of degree 1: {num_outside}, next to a cell of degree 2: {num_neighbours}")

# ## The adaptive loop
# The forms of every block $(i, j)$ are compiled once. Every iteration restricts the parent spaces
# to the current degrees, and creates the forms with these spaces.

# +
degrees.set_degrees(np.full(degrees.num_owned_cells, params.pmin, dtype=np.int32))
approach = RestrictedApproach(degrees, params)

plotter = None
if mesh.comm.rank == 0:
    plotter = pyvista.Plotter(shape=(1, 2), border=False, window_size=[1400, 600])
    plotter.open_gif("restricted.gif", fps=1)


def write_frame(approach, result):
    print_iteration(approach, result)
    p = plot_iteration(approach, plotter, title=f"Iteration {result.iteration}")
    if p is not None:
        p.write_frame()


# + tags=["scroll-output"]
print_header()
history = solve_adaptive(approach, max_iterations=12, callback=write_frame)
if plotter is not None:
    plotter.close()
# -

# <img src="./restricted.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: restricted-
# ```
