# The approaches

The library provides three representations of a space with a cell-wise polynomial degree. All of
them assemble the same discrete problem, and give the same solution up to the quadrature.

| Approach | Space | Continuous spaces | Demo |
|---|---|---|---|
| {py:class}`~dolfinx_padapt.approaches.SubmeshApproach` | One space per degree on the submesh of its cells, coupled through entity maps | Yes | [Submesh](demos/demo_submesh) |
| {py:class}`~dolfinx_padapt.approaches.RestrictedApproach` | One space per degree on the whole mesh, with its dofmap restricted to the cells of that degree | Yes | [Restricted](demos/demo_restricted) |
| {py:class}`~dolfinx_padapt.approaches.HierarchicalApproach` | Sub-matrices of a Legendre DG space of degree $p_{max}$, which is hierarchical on simplices {cite}`approaches-dubiner1991` | No | [Hierarchical](demos/demo_hierarchical) |

## Submeshes (recommended)
The forms are compiled with the space of each degree on a template domain. Every iteration
extracts the submesh of each active degree with {py:func}`dolfinx.mesh.create_submesh`, and
creates the forms over the parent mesh with the entity maps of the submeshes. The integrals over
the interfaces between the degrees use the mixed-domain assembly of DOLFINx: a cell that is not
part of a submesh is mapped to `-1`, and the assembler skips it.

## Restricted dofmaps
Every iteration keeps the degrees of freedom of the parent space $V_i$ in the closure of the cells
of degree $i$, computes their parallel layout with {py:func}`dolfinx.common.create_sub_index_map`,
and creates a new dofmap on the same mesh. The results are the same as with submeshes, but the
dofmaps are made by hand:

- A kernel of an interior facet integral always tabulates both cells of the facet. A cell outside
  the restriction that neighbours it must therefore point to the degrees of freedom of its
  neighbour, which receive zero contributions.
- The dofmap of every degree is kept on the whole mesh.
- Anything beyond assembly over the restricted cells (output, interpolation on all cells, locating
  degrees of freedom) has to be done with care.

## A hierarchical basis
A cell of degree $p_K$ keeps the first $\dim P_{p_K}$ degrees of freedom of a Legendre DG space of
degree $p_{max}$, and the matrices are extracted with {py:meth}`petsc4py.PETSc.Mat.createSubMatrix`.
With a fixed penalty $\sigma=\alpha p_{max}^2$, the matrices are only assembled once. However:

- it is limited to discontinuous spaces on simplices,
- the operator of the maximal degree is assembled and stored on every cell,
- every cell is integrated with the quadrature of the maximal degree,
- with a degree dependent penalty, the matrices are assembled at every iteration.

## Continuous spaces with Nitsche coupling
With `Parameters(continuous=True)`, the space of each degree is continuous within its region, and
the regions are coupled weakly with Nitsche's method on the interfaces
{cite}`approaches-nitsche1971`. The submesh and the restricted dofmap approaches support this: the
degrees of freedom of $V_i$ on an interface are separate from those of $V_j$, so the coupling is
discontinuous exactly where the Nitsche terms are applied. A hierarchical basis cannot do this, as
selecting rows and columns of a continuous space cannot break its continuity at the interfaces,
which move at every iteration. See the [continuous demo](demos/demo_continuous).

```{bibliography}
   :filter: cited
   :labelprefix:
   :keyprefix: approaches-
```
