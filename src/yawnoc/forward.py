import cv2
import numpy as np


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
