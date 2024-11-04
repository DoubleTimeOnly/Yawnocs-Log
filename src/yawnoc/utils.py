from typing import Tuple

import numpy as np


def init_board(seed: np.ndarray, board_size: Tuple[int, int]) -> np.ndarray:
    board = np.zeros(board_size, dtype=np.uint8)
    seed_rows, seed_cols = seed.shape[:2]
    board_rows, board_cols = board.shape[:2]
    row_start, col_start = board_rows // 2 - seed_rows // 2, board_cols //2 - seed_cols // 2
    row_slice = slice(row_start, row_start + seed_rows)
    col_slice = slice(col_start, col_start + seed_cols)
    board[row_slice, col_slice] = seed
    return board
