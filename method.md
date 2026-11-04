# The method

## Model problems
The operator is always $-\Delta$, while the source term $s(u)$ and the Dirichlet data $g$ are given
by a problem:

$$
\begin{align}
-\Delta u &= s(u) && \text{in } \Omega,\\
u &= g && \text{on } \partial\Omega.
\end{align}
$$

- {py:class}`dolfinx_padapt.problems.EigenvalueProblem`: $s(u)=\lambda u$ and $g=0$. The default
  domain is the PacMan domain of the
  [adaptive mesh refinement demo](https://jorgensd.github.io/dolfinx-tutorial/chapter2/amr.html)
  of the DOLFINx tutorial, with the first eigenvalue $\lambda = j_{2/3,1}^2 \approx 11.3947$.
  The eigenfunction behaves like $r^{2/3}$ at the re-entrant corner, so the mesh is graded
  towards it.
- {py:class}`dolfinx_padapt.problems.PoissonProblem`: $s(u)=f=-\Delta u_{ex}$ and $g=u_{ex}$ for a
  manufactured solution $u_{ex}$, such that the error can be measured.

## Discretisation
Discontinuous Galerkin spaces are coupled with the symmetric interior penalty (SIPG) method
{cite}`method-arnold1982,method-arnold2002`, and the Dirichlet condition is imposed weakly with
Nitsche's method {cite}`method-nitsche1971`.
Continuous spaces are only coupled by these terms on the interfaces between cells of different
degrees.
The residual is written once, with the boundary terms in $u - g$,

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

The penalty parameter is $\sigma = \alpha p^2$ {cite}`method-epshteyn2007`, where $p$ is
$\max(p^+, p^-)$ on an interior facet and the degree of the cell on the boundary
({py:attr}`dolfinx_padapt.forms.Penalty.local`), or $p_{max}$ on every facet
({py:attr}`dolfinx_padapt.forms.Penalty.global_`).

## The adaptive loop
Starting from degree $p_{min}$ on every cell, {py:func}`dolfinx_padapt.adaptivity.solve_adaptive`

1. creates the space for the current degrees,
2. assembles and solves the problem,
3. computes the error indicator,
4. marks cells, and increases their degree by one.

The degree of each cell is stored as cell tags ({py:class}`dolfinx_padapt.degrees.DegreeField`),
where the value of a tag is the degree. Every integral gets a unique tag: $\mathrm{d}x(i)$ and
$\mathrm{d}s(i)$ for the cells of degree $i$, and $\mathrm{d}S(i, j)$ for the facets between the
cells of degree $i \leq j$. The interface facets are found with {py:func}`scifem.find_interface`, and
{py:func}`scifem.compute_interface_data` orders each facet such that the cell with the lowest degree is the
"+" restriction.

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
different from increasing it. The mesh itself is never changed.

```{bibliography}
   :filter: cited
   :labelprefix:
   :keyprefix: method-
```
