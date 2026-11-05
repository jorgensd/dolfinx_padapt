r"""Model problems (see :class:`dolfinx_padapt.problems.Problem`):

- :class:`EigenvalueProblem`: find :math:`(u, \lambda)` with :math:`-\Delta u = \lambda u` in
  :math:`\Omega` and :math:`u = 0` on :math:`\partial\Omega`.
- :class:`PoissonProblem`: :math:`-\Delta u = f` in :math:`\Omega`, :math:`u = g` on
  :math:`\partial\Omega`.
- :class:`ElasticityProblem`: :math:`-\nabla\cdot\sigma(u) = f` in :math:`\Omega`,
  :math:`u = g` on :math:`\partial\Omega`.
- :class:`StokesProblem`: :math:`-\nu\Delta u + \nabla p = f`, :math:`\nabla\cdot u = 0`
  in :math:`\Omega`, :math:`u = g` on :math:`\partial\Omega`.
- :class:`SignoriniProblem`: linear elasticity with frictionless contact (a nonlinear problem).

The source problems use a manufactured solution, such that the error can be measured.
"""

from .elasticity import ElasticityProblem
from .laplace import EigenvalueProblem, LaplaceProblem, PoissonProblem
from .signorini import SignoriniProblem
from .stokes import StokesProblem

__all__ = [
    "EigenvalueProblem",
    "ElasticityProblem",
    "LaplaceProblem",
    "PoissonProblem",
    "SignoriniProblem",
    "StokesProblem",
]
