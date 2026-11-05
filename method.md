# The method

## Model problems
A problem ({py:class}`dolfinx_padapt.problems.Problem`) has one or more fields, each with a value
shape and a degree relative to the degree of the cell. It provides the residual on the cells and
boundary of one degree, the coupling terms on the interfaces between two degrees, the error
indicator and the error measures. The library has the following problems:

- {py:class}`dolfinx_padapt.models.EigenvalueProblem`: $-\Delta u = \lambda u$, $u=0$ on
  $\partial\Omega$. The default domain is the PacMan domain of the
  [adaptive mesh refinement demo](https://jorgensd.github.io/dolfinx-tutorial/chapter2/amr.html)
  of the DOLFINx tutorial, with the first eigenvalue $\lambda = j_{2/3,1}^2 \approx 11.3947$.
  The eigenfunction behaves like $r^{2/3}$ at the re-entrant corner, so the mesh is graded
  towards it.
- {py:class}`dolfinx_padapt.models.PoissonProblem`: $-\Delta u = f$, $u = g$ on $\partial\Omega$.
- {py:class}`dolfinx_padapt.models.ElasticityProblem`: $-\nabla\cdot\sigma(u) = f$, $u = g$ on
  $\partial\Omega$, for a vector valued displacement.
- {py:class}`dolfinx_padapt.models.StokesProblem`: $-\nu\Delta u + \nabla p = f$,
  $\nabla\cdot u = 0$, $u = g$ on $\partial\Omega$, with a velocity of the degree of the cell and
  a pressure of one degree less {cite}`method-schotzau2002`, and a penalty on the jumps of the
  pressure {cite}`method-cockburn2002`.
- {py:class}`dolfinx_padapt.models.SignoriniProblem`: linear elasticity with frictionless contact
  against a rigid plane, $u\cdot\hat{n}\leq g$ on the contact boundary $\Gamma$. The constraint is
  imposed with the latent variable proximal point (LVPP) method
  {cite}`method-keith2024,method-dokken2025`, with a latent variable on $\Gamma$, see
  [the Signorini demo](demos/demo_signorini).

The source problems use a manufactured solution $u_{ex}$, with $f$ and $g$ computed from it, such
that the error can be measured. The problems with vector valued or several fields are only
implemented for the submesh approach.

## Discretisation
Discontinuous Galerkin spaces are coupled with the symmetric interior penalty (SIPG) method
{cite}`method-arnold1982,method-arnold2002`, and the Dirichlet condition is imposed weakly with
Nitsche's method {cite}`method-nitsche1971`.
Continuous spaces are only coupled by these terms on the interfaces between cells of different
degrees. A field on the boundary (the latent variable of the Signorini problem) lives on a submesh
of the facets of each degree, with the continuity of the trace of the fields on the cells.
For the Laplacian, the residual is written once, with the boundary terms in $u - g$,

$$
\begin{align}
F(u; v) =&\ \sum_K \int_K \nabla u\cdot\nabla v - s v~\mathrm{d}x
+ \sum_{F\in\mathcal{F}_i} \int_F -\{\nabla u\}\cdot[v] - [u]\cdot\{\nabla v\}
+ \frac{\sigma}{\{h\}}[u]\cdot[v]~\mathrm{d}s\\
&+ \int_{\partial\Omega} -\nabla u\cdot n\, v - (u - g)\nabla v\cdot n
+ \frac{\sigma}{h}(u - g) v~\mathrm{d}s,
\end{align}
$$

where $[u] = u^+n^+ + u^-n^-$ and $\{\nabla u\}=\frac{1}{2}(\nabla u^+ + \nabla u^-)$.
The stiffness matrix and the load vector are extracted with {py:func}`ufl.lhs` and
{py:func}`ufl.rhs`. The eigenvalue problem has $g=0$ and the mass matrix as its right hand side.
For linear elasticity, the stress replaces the gradient, and for the Stokes equations the
velocity and pressure are also coupled through $\{p\}[v\cdot n]$ on the facets. The building
blocks are in {py:mod}`dolfinx_padapt.dg`.

The penalty parameter is $\sigma = \alpha p^2$ {cite}`method-epshteyn2007`, where $p$ is
$\max(p^+, p^-)$ on an interior facet and the degree of the cell on the boundary
({py:attr}`dolfinx_padapt.forms.Penalty.local`), or $p_{max}$ on every facet
({py:attr}`dolfinx_padapt.forms.Penalty.global_`).

## The adaptive loop
Starting from degree $p_{min}$ on every cell, {py:func}`dolfinx_padapt.adaptivity.solve_adaptive`

1. creates the space for the current degrees,
2. assembles and solves the problem: an eigenvalue problem with SLEPc, a linear problem with
   {py:class}`dolfinx.fem.petsc.LinearProblem`, and a nonlinear problem with
   {py:class}`dolfinx.fem.petsc.NonlinearProblem` (Newton's method), inside the outer iteration
   of the problem (e.g. the proximal point iterations of the Signorini problem,
   {py:func}`dolfinx_padapt.proximal.proximal_point`),
3. computes the error indicator,
4. marks cells, and increases their degree by one (p-refinement), splits them (h-refinement), or
   chooses between the two from the smoothness of the solution (hp-refinement), see below.

The degree of each cell is owned by the approach, within the range $[p_{min}, p_{max}]$ of its
{py:class}`~dolfinx_padapt.forms.Parameters`, and stored as cell tags
({py:class}`dolfinx_padapt.degrees.DegreeField`),
where the value of a tag is the degree. Every integral gets a unique tag: $\mathrm{d}x(i)$ and
$\mathrm{d}s(i)$ for the cells of degree $i$, and $\mathrm{d}S(i, j)$ for the facets between the
cells of degree $i \leq j$. The interior facets are computed once with
{py:func}`dolfinx.fem.compute_integration_domains`, and grouped by the degrees of their two cells,
where the cell with the lowest degree is the "+" restriction.

All forms, for every combination of degrees, are compiled once with
{py:func}`dolfinx.fem.compile_form`, and every iteration only calls
{py:func}`dolfinx.fem.create_form` with the new spaces and integration entities.

### Error indicator
We use the per-cell residual indicator of {cite}`method-larson2000` (see also
{cite}`method-duran2003`), as in the AMR demo of the DOLFINx tutorial, with the polynomial weights
of the hp-DG indicator of {cite}`method-houston2007`:

$$
\begin{align}
\eta_K^2 =&\ \frac{h_K^2}{p_K^2}\Vert s(u_h) + \Delta u_h\Vert^2_{K}
+ \frac{h_K}{p_K}\left\Vert\frac{[\nabla u_h\cdot n]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}\\
&+ \alpha^2\frac{p_K^3}{h_K}\left(\left\Vert\frac{[u_h]}{2}\right\Vert^2_{\partial K\setminus\partial\Omega}
+ \Vert u_h - g\Vert^2_{\partial K\cap\partial\Omega}\right).
\end{align}
$$

An interior facet contributes to both of its cells, each with its own $h_K$ and $p_K$.
The flux jumps are included on every interior facet, also for continuous spaces, where the jumps
$[u_h]$ vanish inside a region of one degree, but the flux jumps do not.

### Marking
The cells with $\eta_K > \theta \max_{K'}\eta_{K'}$ {cite}`method-babuska1978` are marked with
{py:func}`dolfinx.mesh.mark_maximum` (or with {py:func}`dolfinx.mesh.mark_equidistribution`),
among the cells with $p_K<p_{max}$. Cells at the maximal degree are left out, as the error at a
singularity would otherwise keep selecting the same cells.

Optionally, the degree of the cells with $\eta_K < \theta_c \max_{K'}\eta_{K'}$ and
$p_K > p_{min}$ is lowered by one
({py:func}`dolfinx_padapt.adaptivity.mark_cells_for_coarsening`), but never below $p_{min}$.
The cells marked for enrichment are not coarsened, and $\theta_c$ should be much smaller than
$\theta$, such that the cells do not alternate between being enriched and coarsened.
As the spaces and forms are created from the cell tags at every iteration, lowering a degree is no
different from increasing it.

### h- and hp-refinement
With h-refinement ({py:attr}`dolfinx_padapt.adaptivity.Refinement.h`), the marked cells are split
with the algorithm of {cite}`method-plaza2000`, and the new cells get the degree of their parent
cell. Meshes with curved (higher order) geometry are refined through the straight mesh with the
same topology, and the nodes of the refined mesh are placed with the coordinates of the original
mesh, interpolated with the known parent cells
({py:func}`dolfinx_padapt.refinement.refine_curved`). Optionally, the new nodes on the boundary
are moved to the exact boundary. The forms are compiled once, for the initial mesh, and are only
created on the refined meshes.

With hp-refinement ({py:attr}`dolfinx_padapt.adaptivity.Refinement.hp`), the degree of a marked
cell is increased if the solution is smooth on it, and the cell is split otherwise (or at the
maximal degree). The smoothness is estimated from the decay of the Legendre coefficients of the
solution on the cell {cite}`method-mavriplis1994,method-houston2005`
({py:mod}`dolfinx_padapt.smoothness`). See [the hp demo](demos/demo_hp).

```{bibliography}
   :filter: cited
   :labelprefix:
   :keyprefix: method-
```
