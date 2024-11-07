from enum import Enum
from typing import Callable

import numpy as np

from yawnoc.solvers import deep_solver, gradient_descent, pattern_search


class SolverType(Enum):
    GRADIENT_DESCENT = 1
    PATTERN = 2
    DEEP = 3


def get_backward_fn(
    solver_type: SolverType
) -> Callable[[np.ndarray], np.ndarray]:
    if solver_type is SolverType.GRADIENT_DESCENT:
        return gradient_descent.backward
    elif solver_type is SolverType.PATTERN:
        return pattern_search.backward
    elif solver_type is SolverType.DEEP:
        return deep_solver.create_backward()
    else:
        raise ValueError(f"Did not recognize solver type {solver_type}")
