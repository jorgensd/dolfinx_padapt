import numpy as np


def _sqrt(values: dict[str, float]) -> dict[str, float]:
    return {name: float(np.sqrt(max(value, 0.0))) for name, value in values.items()}
