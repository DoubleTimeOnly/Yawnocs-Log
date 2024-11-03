import itertools
import math
from typing import Tuple
import numpy as np
from tqdm import tqdm

from yawnoc.forward import forward


def backward(board: np.ndarray) -> np.ndarray:
    # rows, cols
    idxs = np.array(np.nonzero(board)).T
    idxs = tuple(map(tuple, idxs))
    all_idxs = itertools.permutations(idxs)
    state = np.zeros_like(board)
    for idxs in tqdm(all_idxs, total=math.factorial(len(idxs))):
        prev = backward_search(
            state=state,
            prev=state.copy(),
            idxs=idxs,
            step=0,
        )
        if prev is not None:
            return prev
    raise ValueError(f"Could not find a valid solution")


def backward_search(
    state: np.ndarray, prev: np.ndarray, idxs: np.ndarray, step: int
):
    if step == len(idxs):
        return prev
    
    new_state = state.copy()
    loc = tuple(idxs[step])
    new_state[loc] = 1

    possible_p = get_all_p(new_state, prev, loc=loc)

    for candidate_p in possible_p:
        updated_prev = backward_search(
            new_state, candidate_p, idxs=idxs, step=step+1
        )
        if updated_prev is not None:
            return updated_prev
    return None


def get_all_p(
    state: np.ndarray,
    prev: np.ndarray,
    loc: Tuple[int, int],
) -> np.ndarray:
    pat = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ])
    candidates = []
    for num_rots in range(4):
        new_prev = prev.copy()
        row_slice = slice(loc[0]-1, loc[0]+2)
        col_slice = slice(loc[1]-1, loc[1]+2)

        rotated_pattern = np.rot90(m=pat, k=num_rots)
        new_prev[row_slice, col_slice] = rotated_pattern

        legal = np.all(state == forward(new_prev))

        if legal:
            candidates.append(new_prev)
    return candidates
