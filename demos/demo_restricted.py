# # Restricted dofmaps
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
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

from dolfinx_padapt import DegreeField, EigenvalueProblem, create_pacman
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import RestrictedApproach, restrict_function_space
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import Animation

mesh = create_pacman(MPI.COMM_WORLD, h=0.3, h_corner=0.01)
params = Parameters(pmin=1, pmax=6)
# -

# ## Restricting a dofmap
# Given the cells of degree $i$ (including ghost cells), we keep the degrees of freedom of the
# parent space $V_i$ in the closure of these cells. The parallel layout of the kept degrees of
# freedom is computed with {py:func}`dolfinx.common.create_sub_index_map`, and a new
# {py:class}`dolfinx.cpp.fem.DofMap` is created on the same mesh.
#
# As an illustration, we give the cells in the left half of the domain degree 2, and restrict a
# second order space to them.

degrees = DegreeField(mesh, 1, 2)
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

approach = RestrictedApproach(EigenvalueProblem(mesh), params)

# + tags=["scroll-output"]
with Animation(mesh.comm, "restricted.gif") as animation:
    history = solve_adaptive(approach, max_iterations=15, callback=animation)
# -

# <img src="./restricted.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: restricted-
# ```
