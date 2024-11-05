from typing import Tuple

import numpy as np
from tqdm import tqdm

from yawnoc.forward import forward


def backward(board: np.ndarray) -> np.ndarray:
    guess = board.copy()
    restarts = 0
    with tqdm() as pbar:
        while True:
            best_find, best_err = backward_search(board, guess, pbar)
            if best_err > 0:
                restarts += 1
                guess = np.random.randint(2, size=board.shape, dtype=np.uint8)
            else:
                return best_find


def backward_search(
    board: np.ndarray,
    guess: np.ndarray,
    pbar: tqdm,
) -> Tuple[np.ndarray, float]:
    current_board = guess
    current_error = get_err(board, current_board)

    while current_error > 0:
        next_step, next_error = get_best_step(board, current_board)
        pbar.update()
        if next_error < current_error:
            current_board = next_step
            current_error = next_error
            pbar.set_description_str(f"Current Error: {current_error:.4f}  ")
        else:
            return current_board, current_error

    return current_board, current_error


def get_best_step(
    board: np.ndarray,
    current_prev: np.ndarray,
) -> Tuple[np.ndarray, float]:
    best_step = current_prev
    best_err = get_err(board, current_prev)

    it = np.nditer(current_prev, flags=['multi_index'])
    for _ in it:
        next_step = current_prev.copy()
        next_step[it.multi_index] = int(not next_step[it.multi_index])
        err = get_err(board, next_step)
        if err < best_err:
            best_err = err
            best_step = next_step
    return best_step, best_err


def get_err(target: np.ndarray, prev_guess: np.ndarray) -> float:
    result = forward(prev_guess)
    diff = np.subtract(target, result, dtype=np.int16)
    err = np.mean(np.square(diff))
    return err
