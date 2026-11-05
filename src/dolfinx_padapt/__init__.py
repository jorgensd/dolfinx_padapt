"""P-adaptive discontinuous (and Nitsche-coupled continuous) Galerkin methods in DOLFINx."""

from .degrees import DegreeField, pair_tag
from .mesh import EXACT_EIGENVALUE, create_pacman
from .models import EigenvalueProblem, PoissonProblem
from .problems import Problem

__all__ = [
    "EXACT_EIGENVALUE",
    "DegreeField",
    "EigenvalueProblem",
    "PoissonProblem",
    "Problem",
    "create_pacman",
    "pair_tag",
]
