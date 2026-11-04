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

# # A hierarchical basis
#
# Author: Jørgen S. Dokken
#
# We solve the same Laplace eigenvalue problem as in [the submesh demo](./demo_submesh), with the
# same SIPG discretisation {cite}`hierarchical-arnold2002` and error indicator
# {cite}`hierarchical-larson2000`.
# Here, the space with a cell-wise degree is represented as a *subset* of the degrees of freedom of
# a single discontinuous space of the maximal degree $p_{max}$, which only works with a hierarchical
# basis.
#
# ## A hierarchical basis
# Basix provides an orthonormal (Legendre) variant of the discontinuous Lagrange element.
# On simplices, its basis functions are the orthonormal Dubiner polynomials
# {cite}`hierarchical-dubiner1991`, ordered by degree, so the first $\dim P_k$ basis functions of
# the element of degree $k+1$ span $P_k$.
# We verify this by tabulating the elements at some points.

# +
from mpi4py import MPI
from petsc4py import PETSc

import basix
import numpy as np

from dolfinx_padapt import DegreeField, create_pacman
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import HierarchicalApproach
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import Animation

points = basix.create_lattice(basix.CellType.triangle, 8, basix.LatticeType.equispaced, True)
for k in range(1, 5):
    elements = [
        basix.create_element(
            basix.ElementFamily.P,
            basix.CellType.triangle,
            degree,
            basix.LagrangeVariant.legendre,
            discontinuous=True,
        )
        for degree in (k, k + 1)
    ]
    phi_k, phi_k1 = (e.tabulate(0, points)[0, :, :, 0] for e in elements)
    is_subset = np.allclose(phi_k, phi_k1[:, : phi_k.shape[1]])
    PETSc.Sys.Print(
        f"First {phi_k.shape[1]} basis functions of degree {k + 1} equal degree {k}: {is_subset}"
    )
# -

# ## Restriction to the cell-wise degree
# A cell $K$ of degree $p_K$ keeps the degrees of freedom `dofmap.list[K, :dim(P_pK)]`.
# The parallel layout of the kept degrees of freedom is computed with
# {py:func}`dolfinx.common.create_sub_index_map`, and the matrices of the cell-wise degree are
# extracted from the matrices of degree $p_{max}$ with
# {py:meth}`petsc4py.PETSc.Mat.createSubMatrix`.
#
# The penalty parameter of SIPG depends on the degree on each facet. With
# `penalty="global"`, the penalty $\sigma=\alpha p_{max}^2$ is used on every facet, which is
# stable for every distribution of degrees. Then the matrices of degree $p_{max}$ are only
# assembled **once**, and every iteration only extracts sub-matrices.

mesh = create_pacman(MPI.COMM_WORLD, h=0.3, h_corner=0.01)
params = Parameters(pmin=1, pmax=6, penalty="global")
degrees = DegreeField(mesh, params.pmin, params.pmax)
approach = HierarchicalApproach(degrees, params)

# + tags=["scroll-output"]
with Animation(mesh.comm, "hierarchical.gif") as animation:
    history = solve_adaptive(approach, max_iterations=15, callback=animation)
# -

# <img src="./hierarchical.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# Note that the assembly time of the first iteration includes the assembly of the full matrices.
# The fine matrices hold the degrees of freedom of degree $p_{max}$ on every cell:

num_full = approach.V.dofmap.index_map.size_global
PETSc.Sys.Print(f"Full space: {num_full} dofs, active: {approach.num_dofs} dofs")

# ```{admonition} Limitations
# :class: note
# - The approach is limited to discontinuous spaces: Basix has no hierarchical *continuous*
#   Lagrange basis, and selecting rows and columns of a continuous space cannot break the
#   continuity at the interface between two degrees.
# - The basis is only hierarchical in this ordering on simplices.
# - The matrices of degree $p_{max}$ are assembled and stored, and each cell is integrated with
#   the quadrature of degree $p_{max}$.
# - With a degree dependent penalty (`penalty="local"`), the fine matrices are assembled at every
#   iteration.
# ```
#
# ```{bibliography}
#    :filter: cited
#    :labelprefix:
#    :keyprefix: hierarchical-
# ```
