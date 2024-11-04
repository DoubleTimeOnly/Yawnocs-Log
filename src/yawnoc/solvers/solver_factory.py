from enum import Enum
from typing import Callable

import numpy as np

from yawnoc.solvers import gradient_descent, pattern_search


class SolverType(Enum):
    GRADIENT_DESCENT = 1
    PATTERN = 2


def get_backward_fn(
    solver_type: SolverType
) -> Callable[[np.ndarray], np.ndarray]:
    if solver_type is SolverType.GRADIENT_DESCENT:
        return gradient_descent.backward
    if solver_type is SolverType.PATTERN:
        return pattern_search.backward
    return None
