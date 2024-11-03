import itertools
import math
from typing import Tuple
import cv2
import numpy as np
from tqdm import tqdm


def forward(board: np.ndarray) -> np.ndarray:
    new_board = board.copy()

    kernel = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1],
    ])

    border_board = cv2.copyMakeBorder(src=new_board, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_WRAP)
    neighbor_count = cv2.filter2D(src=border_board, kernel=kernel, ddepth=cv2.CV_8U)[1:-1,1:-1]
    new_board[neighbor_count == 3] = 1
    new_board[(neighbor_count > 3) | (neighbor_count < 2)] = 0

    return new_board


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


SEEDS = {
    "nice": np.array([[1]]),
    "nice-1+1": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]),
    "3x3": np.ones((3, 3)),
}


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


def init_board(seed: np.ndarray, board: np.ndarray) -> np.ndarray:
    seed_rows, seed_cols = seed.shape[:2]
    board_rows, board_cols = board.shape[:2]
    row_start, col_start = board_rows // 2 - seed_rows // 2, board_cols //2 - seed_cols // 2
    row_slice = slice(row_start, row_start + seed_rows)
    col_slice = slice(col_start, col_start + seed_cols)
    board[row_slice, col_slice] = seed
    return board


if __name__ == "__main__":
    board_size = (50, 50)
    board = np.zeros(board_size, dtype=np.uint8)
    seed = SEEDS["3x3"]
    board = init_board(seed=seed, board=board)

    delay = 0
    while True:
        # visualization
        viz_board = cv2.resize(board*255, dsize=(500, 500), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imshow("Board", viz_board)
        key = cv2.waitKey(delay=delay)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        elif key == ord("."):
            # next
            delay = 0
            board = forward(board)
        elif key == ord(","):
            # prev
            delay = 0
            board = backward(board)


