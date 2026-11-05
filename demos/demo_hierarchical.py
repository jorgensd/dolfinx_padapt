# # A hierarchical basis
#
# Author: Jørgen S. Dokken
#
# SPDX-License-Identifier: MIT
#
# We solve the same Laplace eigenvalue problem as in [the submesh demo](./demo_submesh), with the
# same SIPG discretisation {cite}`hierarchical-arnold2002` and error indicator
# {cite}`hierarchical-larson2000`.
# Here, the space with a cell-wise degree is represented as a *subset* of the degrees of freedom of
# a single discontinuous space of the maximal degree $p_{max}$, which only works with a hierarchical
# basis.
#
# ## A hierarchical basis
# Basix provides orthonormal polynomial sets: the Dubiner polynomials
# {cite}`hierarchical-dubiner1991` on simplices, and tensor products of Legendre polynomials on
# quadrilaterals and hexahedra. {py:func}`dolfinx_padapt.approaches.hierarchical_element` builds a
# discontinuous element from these polynomials, sorted by degree (the total degree on simplices,
# and the largest degree in each direction on quadrilaterals and hexahedra). Its degrees of freedom
# are the moments against the basis functions. The first `dims[k]` basis functions then span the
# space of degree $k$, $P_k$ on simplices and $Q_k$ on quadrilaterals and hexahedra.
#
# We verify this against the Legendre variant of the discontinuous Lagrange element of Basix: as
# both bases are orthonormal, the projection onto the first `dims[k]` basis functions reproduces
# every basis function of degree $k$.

# +
from mpi4py import MPI
from petsc4py import PETSc

import basix
import numpy as np

from dolfinx_padapt import EigenvalueProblem, create_pacman
from dolfinx_padapt.adaptivity import solve_adaptive
from dolfinx_padapt.approaches import HierarchicalApproach, hierarchical_element
from dolfinx_padapt.forms import Parameters
from dolfinx_padapt.plotting import Animation

pmax = 3
for cell in (basix.CellType.triangle, basix.CellType.quadrilateral, basix.CellType.hexahedron):
    element, dims = hierarchical_element(cell, pmax)
    points, weights = basix.make_quadrature(cell, 2 * pmax)
    phi = element.tabulate(0, points)[0]
    spans = []
    for k in range(pmax + 1):
        reference = basix.create_element(
            basix.ElementFamily.P, cell, k, basix.LagrangeVariant.legendre, discontinuous=True
        )
        psi = reference.tabulate(0, points)[0, :, :, 0]
        coefficients = phi[:, : dims[k]].T @ (weights[:, None] * psi)
        spans.append(bool(np.allclose(phi[:, : dims[k]] @ coefficients, psi)))
    PETSc.Sys.Print(f"{cell.name}: dims = {dims.tolist()}, first dims[k] span degree k: {spans}")
# -

# ## Restriction to the cell-wise degree
# A cell $K$ of degree $p_K$ keeps the degrees of freedom `dofmap.list[K, :dims[p_K]]`.
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
approach = HierarchicalApproach(EigenvalueProblem(mesh), params)

# + tags=["scroll-output"]
with Animation(mesh.comm, "hierarchical.gif") as animation:
    history = solve_adaptive(approach, max_iterations=15, callback=animation)
# -

# <img src="./hierarchical.gif" alt="gif" class="bg-primary mb-1" width="100%">
#
# Note that the assembly time of the first iteration includes the assembly of the full matrices.
# The fine matrices hold the degrees of freedom of degree $p_{max}$ on every cell:

num_full = approach.function_space.dofmap.index_map.size_global
PETSc.Sys.Print(f"Full space: {num_full} dofs, active: {approach.num_dofs} dofs")

# ```{admonition} Limitations
# :class: note
# - The approach is limited to discontinuous spaces: Basix has no hierarchical *continuous*
#   Lagrange basis, and selecting rows and columns of a continuous space cannot break the
#   continuity at the interface between two degrees.
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
