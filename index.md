# P-adaptivity in DOLFINx

Author: Jørgen S. Dokken

`dolfinx_padapt` solves problems with a polynomial degree that varies from cell to cell in
[DOLFINx](https://github.com/FEniCS/dolfinx), and adapts the degrees with an error indicator,
i.e. P-adaptivity.
It supports discontinuous Galerkin spaces, and continuous Galerkin spaces within each region of
cells with the same degree. Spaces of different degrees are coupled by interior penalty (Nitsche)
terms on the interfaces between them.

The library provides three representations of a space with a cell-wise degree:

- `SubmeshApproach`: one function space per degree, on a submesh of the cells with that degree,
- `RestrictedApproach`: one function space per degree on the whole mesh, with its dofmap
  restricted to the cells with that degree,
- `HierarchicalApproach`: sub-matrices of a single hierarchical (Legendre) DG space of the
  maximal degree.

**We recommend the submesh approach.** It uses the mixed-domain assembly of DOLFINx, supports
discontinuous and continuous spaces with the same code, and both the number of degrees of freedom
and the quadrature follow the active degrees. See [the approaches](approaches.md) for the
trade-offs.

The main model problem is the Laplace eigenvalue problem on the PacMan domain of the
[adaptive mesh refinement demo](https://jorgensd.github.io/dolfinx-tutorial/chapter2/amr.html)
of the DOLFINx tutorial. Instead of refining the mesh, which has second order geometry and is
generated with GMSH, we increase the degree of the cells with the largest error.
The library also solves the Poisson equation (in 2D and 3D), linear elasticity and the Stokes
equations, see the [Poisson](demos/demo_poisson), [elasticity](demos/demo_elasticity) and
[Stokes](demos/demo_stokes) demos.

- [Getting started](installation): installation, and running the demos and the tests.
- [The method](method): the adaptive loop, the discretisation and the error indicator.
- [The approaches](approaches.md): how a space with a cell-wise degree is represented.
- The demos, starting with [one submesh per degree](demos/demo_submesh).
- The [API reference](api).

The code is distributed under the MIT license.
