"""Compilation of the forms of a problem (see :mod:`dolfinx_padapt.problems`) for a space with a
cell-wise polynomial degree.

Two flavours are provided:

- *Block forms* (submeshes and restricted dofmaps), where every polynomial degree ``i`` has its
  own space for each field. Volume integrals ``dx(i)`` and boundary integrals ``ds(i)`` only see
  the spaces of degree ``i``, while the interface integral ``dS(pair_tag(i, j))`` couples the
  spaces of degree ``i`` (``"+"``, lowest degree) and ``j`` (``"-"``). The penalty on each
  integral is a compile time constant.
- *Single forms* (hierarchical basis), where one space of degree ``pmax`` is used on all cells,
  and the degree enters through a DG-0 coefficient.
"""

from dataclasses import dataclass, field
from enum import Enum

from mpi4py import MPI

import dolfinx
import numpy as np
import ufl
from dolfinx.fem import compile_form
from dolfinx.fem.forms import CompiledForm

from .degrees import boundary_tag, degree_pairs, pair_tag
from .dg import nitsche_boundary, sipg_interior  # noqa: F401
from .problems import Problem


class Penalty(str, Enum):
    r"""Polynomial degree :math:`p` in the penalty parameter :math:`\sigma = \alpha p^2 / h`."""

    #: :math:`p = \max(p^+, p^-)` on an interior facet, and the degree of the cell on the boundary
    local = "local"
    #: :math:`p = p_{max}` everywhere
    global_ = "global"


@dataclass(frozen=True)
class Parameters:
    pmin: int = 1
    pmax: int = 6
    alpha: float = 10.0  # Penalty parameter: sigma = alpha p^2 / h
    penalty: Penalty | str = Penalty.local
    continuous: bool = False  # Continuous Lagrange within each degree region
    quadrature_degree: int | None = None  # Fixed quadrature degree (None: estimated)
    form_compiler_options: dict = field(
        default_factory=lambda: {"scalar_type": dolfinx.default_scalar_type}
    )

    def __post_init__(self):
        object.__setattr__(self, "penalty", Penalty(self.penalty))

    @property
    def degrees(self) -> range:
        return range(self.pmin, self.pmax + 1)

    def penalty_degree(self, i: int, j: int | None = None) -> int:
        return (max(i, j) if j is not None else i) if self.penalty == Penalty.local else self.pmax

    def measures(self, domain: ufl.AbstractDomain):
        md = {} if self.quadrature_degree is None else {"quadrature_degree": self.quadrature_degree}
        return (
            ufl.Measure("dx", domain=domain, metadata=md),
            ufl.Measure("ds", domain=domain, metadata=md),
            ufl.Measure("dS", domain=domain, metadata=md),
        )


BlockKey = tuple[int, str]  # (degree of the cells, name of the field)


@dataclass
class BlockForms:
    """Compiled block forms. The blocks are ordered by degree, and within a degree by the fields
    of the problem, see :attr:`~dolfinx_padapt.forms.BlockForms.keys`."""

    #: The ``(degree, field)`` of each block
    keys: list[BlockKey]
    #: The bilinear form of each pair of blocks (``None`` for a zero block)
    a: list[list[CompiledForm | None]]
    #: The right hand side of each block, i.e. the diagonal blocks of the mass matrix (eigenvalue
    #: problem) or the linear forms (source problem)
    rhs: list[CompiledForm]
    #: The error indicator, split into ``(i,)`` for ``dx(i) + ds(i)``, and ``(i, j)`` for
    #: ``dS(pair_tag(i, j))``
    eta: dict[tuple[int, ...], CompiledForm]
    #: The error functionals of each degree, by name
    errors: dict[int, dict[str, CompiledForm]]
    #: Placeholder for the solution of each block
    u: dict[BlockKey, ufl.Coefficient]
    #: Placeholder for a previous iterate of each block (nonlinear problems)
    u_prev: dict[BlockKey, ufl.Coefficient]
    #: The cell size in the forms
    h: dolfinx.fem.Function
    #: Constants added to the forms (e.g. zero right hand sides)
    constants: list[dolfinx.fem.Constant]

    def index(self, key: BlockKey) -> int:
        return self.keys.index(key)


def compile_block_forms(
    mesh: dolfinx.mesh.Mesh,
    spaces: dict[BlockKey, ufl.FunctionSpace],
    h: dolfinx.fem.Function,
    params: Parameters,
    problem: Problem,
) -> BlockForms:
    """Compile all forms that can appear for any distribution of degrees.

    Args:
        mesh: The parent (integration) mesh.
        spaces: UFL function space of each field for each degree, with keys ``(degree, field)``.
            These can live on another domain than ``mesh`` (submeshes), as long as an entity map
            is supplied when creating the forms.
        h: Cell size (DG-0 on ``mesh``).
        params: Discretisation parameters.
        problem: The problem to solve.
    """
    degrees = list(params.degrees)
    names = [f.name for f in problem.fields]
    keys = [(i, name) for i in degrees for name in names]
    assert sorted(spaces.keys()) == sorted(keys)
    W = ufl.MixedFunctionSpace(*[spaces[key] for key in keys])
    trial = dict(zip(keys, ufl.TrialFunctions(W)))
    test = dict(zip(keys, ufl.TestFunctions(W)))
    u_coeffs = {key: ufl.Coefficient(spaces[key]) for key in keys}
    u_prev = {key: ufl.Coefficient(spaces[key]) for key in keys} if problem.is_nonlinear else {}

    def fields(functions, i):
        return {name: functions[(i, name)] for name in names}

    domain = mesh.ufl_domain()
    dx, ds, dS = params.measures(domain)
    n = ufl.FacetNormal(domain)

    def extra(i):
        """The keyword arguments of a problem with marked boundaries, or a nonlinear problem."""
        kwargs = {}
        if problem.boundary_markers:
            kwargs["ds_marker"] = lambda marker: ds(boundary_tag(marker, i, params.pmax))
        if problem.is_nonlinear:
            kwargs["u_prev"] = fields(u_prev, i)
        return kwargs

    # The residual F(u; v), where u is the trial function of a linear problem, and the current
    # iterate of a nonlinear problem.
    unknowns = u_coeffs if problem.is_nonlinear else trial
    F = 0
    for i in degrees:
        sigma = params.alpha * params.penalty_degree(i) ** 2
        F += problem.residual(
            fields(unknowns, i), fields(test, i), n, h, sigma, dx(i), ds(i), **extra(i)
        )
    for i, j in degree_pairs(params.pmin, params.pmax, params.continuous):
        sigma = params.alpha * params.penalty_degree(i, j) ** 2
        F += problem.interface(
            fields(unknowns, i),
            fields(unknowns, j),
            fields(test, i),
            fields(test, j),
            n,
            h,
            sigma,
            dS(pair_tag(i, j, params.pmax)),
        )
    # The left hand side is the stiffness matrix (linear problem) or the Jacobian (nonlinear
    # problem). The right hand side is the load vector, the mass matrix of an eigenvalue problem,
    # or the residual of a nonlinear problem.
    zero_constants = []
    if problem.is_nonlinear:
        a = sum(ufl.derivative(F, u_coeffs[key], trial[key]) for key in keys)
        rhs_blocks = list(ufl.extract_blocks(F))
    else:
        a = ufl.lhs(F)
        if problem.is_eigenvalue_problem:
            rhs = sum(problem.mass(fields(trial, i), fields(test, i), dx(i)) for i in degrees)
            rhs_blocks = [ufl.extract_blocks(rhs)[k][k] for k in range(len(keys))]
        else:
            rhs_blocks = list(ufl.extract_blocks(ufl.rhs(F)))
    if not problem.is_eigenvalue_problem:
        # Every block needs a linear form, also where the right hand side is zero
        for k, (i, name) in enumerate(keys):
            if rhs_blocks[k] is None:
                zero = dolfinx.fem.Constant(mesh, np.zeros(test[(i, name)].ufl_shape))
                zero_constants.append(zero)
                rhs_blocks[k] = ufl.inner(zero, test[(i, name)]) * dx(i)

    w = ufl.TestFunction(h.function_space)
    eta = {}
    for i in degrees:
        eta[(i,)] = problem.estimator(
            fields(u_coeffs, i), h, i, params.alpha, w, dx(i), ds(i), **extra(i)
        )
    # The flux jumps also on the facets inside a degree region of continuous spaces
    for i, j in degree_pairs(params.pmin, params.pmax):
        form = problem.estimator_interface(
            fields(u_coeffs, i),
            fields(u_coeffs, j),
            n,
            h,
            i,
            j,
            params.alpha,
            w,
            dS(pair_tag(i, j, params.pmax)),
        )
        if i == j:
            eta[(i,)] += form
        else:
            eta[(i, j)] = form

    comm = mesh.comm
    opts = params.form_compiler_options
    errors = {
        i: {
            name: _compile(comm, integrand * dx(i), opts)
            for name, integrand in problem.error_integrands(fields(u_coeffs, i)).items()
        }
        for i in degrees
    }
    return BlockForms(
        keys=keys,
        a=[[_compile(comm, a_ij, opts) for a_ij in row] for row in ufl.extract_blocks(a)],
        rhs=[_compile(comm, form, opts) for form in rhs_blocks],
        eta={key: _compile(comm, form, opts) for key, form in eta.items()},
        errors=errors,
        u=u_coeffs,
        u_prev=u_prev,
        h=h,
        constants=zero_constants,
    )


@dataclass
class SingleForms:
    """Compiled forms for a single (hierarchical) space on the whole mesh."""

    a: CompiledForm
    rhs: CompiledForm  # Mass matrix or linear form
    eta: CompiledForm
    errors: dict[str, CompiledForm]
    u: ufl.Coefficient


def compile_single_forms(
    V: dolfinx.fem.FunctionSpace,
    h: dolfinx.fem.Function,
    p: dolfinx.fem.Function,
    params: Parameters,
    problem: Problem,
) -> SingleForms:
    """Compile forms on a single space where the cell-wise degree is given by ``p``.

    The problem must have a single field.
    """
    assert len(problem.fields) == 1
    name = problem.fields[0].name
    mesh = V.mesh
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dx, ds, dS = params.measures(mesh.ufl_domain())
    n = ufl.FacetNormal(mesh)

    if params.penalty == Penalty.local:
        p_facet = ufl.max_value(p("+"), p("-"))
        p_bnd = p
    else:
        p_facet = p_bnd = params.pmax
    F = problem.residual({name: u}, {name: v}, n, h, params.alpha * p_bnd**2, dx, ds)
    F += problem.interface(
        {name: u}, {name: u}, {name: v}, {name: v}, n, h, params.alpha * p_facet**2, dS
    )
    a = ufl.lhs(F)
    rhs = problem.mass({name: u}, {name: v}, dx) if problem.is_eigenvalue_problem else ufl.rhs(F)

    w = ufl.TestFunction(h.function_space)
    uh = ufl.Coefficient(V)
    eta = problem.estimator({name: uh}, h, p, params.alpha, w, dx, ds)
    eta += problem.estimator_interface(
        {name: uh}, {name: uh}, n, h, p("+"), p("-"), params.alpha, w, dS
    )
    comm = mesh.comm
    opts = params.form_compiler_options
    return SingleForms(
        a=_compile(comm, a, opts),
        rhs=_compile(comm, rhs, opts),
        eta=_compile(comm, eta, opts),
        errors={
            key: _compile(comm, integrand * dx, opts)
            for key, integrand in problem.error_integrands({name: uh}).items()
        },
        u=uh,
    )


def _compile(comm: MPI.Comm, form: ufl.Form | None, options: dict) -> CompiledForm | None:
    if form is None:
        return None
    return compile_form(comm, form, form_compiler_options=options)


def cell_size(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.Function:
    """Cell diameter as a DG-0 function (owned and ghost cells)."""
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
    h = dolfinx.fem.Function(Q, name="h")
    num_cells = len(Q.dofmap.list)
    h.x.array[Q.dofmap.list[:, 0]] = mesh.h(mesh.topology.dim, np.arange(num_cells, dtype=np.int32))
    return h
