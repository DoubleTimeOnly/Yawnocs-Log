from enum import Enum

from yawnoc.solvers import brute_force_search
from yawnoc.solvers import pattern_search


class SolverType(Enum):
    BRUTE = 1
    PATTERN = 2

def get_backward_fn(solver_type: SolverType):
    if solver_type is SolverType.BRUTE:
        return brute_force_search.backward
    if solver_type is SolverType.PATTERN:
        return pattern_search.backward
    return None
