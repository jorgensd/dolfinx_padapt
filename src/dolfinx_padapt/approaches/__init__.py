from .base import Approach
from .hierarchical import HierarchicalApproach
from .restricted import RestrictedApproach, restrict_function_space
from .submesh import SubmeshApproach

APPROACHES: dict[str, type[Approach]] = {
    cls.name: cls for cls in (HierarchicalApproach, RestrictedApproach, SubmeshApproach)
}

__all__ = [
    "APPROACHES",
    "Approach",
    "HierarchicalApproach",
    "RestrictedApproach",
    "SubmeshApproach",
    "restrict_function_space",
]
