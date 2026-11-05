from .base import Approach, BlockApproach
from .hierarchical import HierarchicalApproach, hierarchical_element
from .restricted import RestrictedApproach, restrict_function_space
from .submesh import SubmeshApproach

APPROACHES: dict[str, type[Approach]] = {
    cls.name: cls for cls in (HierarchicalApproach, RestrictedApproach, SubmeshApproach)
}

__all__ = [
    "APPROACHES",
    "Approach",
    "BlockApproach",
    "HierarchicalApproach",
    "RestrictedApproach",
    "SubmeshApproach",
    "hierarchical_element",
    "restrict_function_space",
]
