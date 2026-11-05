# P-adaptivity in DOLFINx

`dolfinx_padapt` implements P-adaptive continuous and discontinuous Galerkin methods in
[DOLFINx](https://github.com/FEniCS/dolfinx), where the polynomial degree varies from cell to
cell and is increased where an error indicator is large. Spaces of different degrees are coupled
with interior penalty (Nitsche) terms on the interfaces between them.

It provides three representations of such a space:

- `SubmeshApproach` (recommended): one space per degree on a submesh of its cells
- `RestrictedApproach`: one space per degree on the whole mesh, with a restricted dofmap
- `HierarchicalApproach`: sub-matrices of one hierarchical (Legendre) DG space of the maximal degree

It solves the Laplace eigenvalue problem, the Poisson equation, linear elasticity and the Stokes
equations, in 2D and 3D, and the Signorini contact problem with the latent variable proximal point
method. Besides p-refinement, it supports h- and hp-refinement of meshes with curved geometry.

The documentation, with the method, the demos and a comparison of the approaches, is at
<https://jorgensd.github.io/dolfinx_padapt/>.

## Installation

Requires DOLFINx (with PETSc and SLEPc) and GMSH.

```bash
python3 -m pip install --no-build-isolation .[docs,test]
```

## Usage

```bash
cd demos
mpirun -n 2 python3 demo_submesh.py
```

The tests are run with `python3 -m pytest tests`.

## License

MIT, see [LICENSE](LICENSE).
