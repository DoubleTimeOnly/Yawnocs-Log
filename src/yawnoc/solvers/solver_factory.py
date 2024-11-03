from enum import Enum

from yawnoc.solvers import brute_force_search


class SolverType(Enum):
    BRUTE = 1

def get_backward_fn(solver_type: SolverType):
    if solver_type is SolverType.BRUTE:
        return brute_force_search.backward
    return None