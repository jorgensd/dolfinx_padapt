# P-adaptivity in DOLFINx

`dolfinx_padapt` implements P-adaptive continuous and discontinuous Galerkin methods in
[DOLFINx](https://github.com/FEniCS/dolfinx), where the polynomial degree varies from cell to
cell and is increased where an error indicator is large. Spaces of different degrees are coupled
with interior penalty (Nitsche) terms on the interfaces between them.

It provides three representations of such a space:

- `SubmeshApproach` (recommended): one space per degree on a submesh of its cells
- `RestrictedApproach`: one space per degree on the whole mesh, with a restricted dofmap
- `HierarchicalApproach`: sub-matrices of one hierarchical (Legendre) DG space of the maximal degree

Problems are given as a source term and Dirichlet data for $-\Delta$: the Laplace eigenvalue problem
(`EigenvalueProblem`) and the Poisson equation with a manufactured solution (`PoissonProblem`), in
2D and 3D.

The documentation, with the method, the demos and a comparison of the approaches, is at
<https://jorgensd.github.io/dolfinx-padapt/>.

## Installation

Requires DOLFINx and [scifem](https://github.com/scientificcomputing/scifem) (>= 0.27).

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
