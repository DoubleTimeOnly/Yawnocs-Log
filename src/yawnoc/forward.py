import cv2
import numpy as np
from torch import Tensor, nn
import torch


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


def batch_forward(board: Tensor) -> Tensor:
    device = board.device
    # board shape: batch_size, row, col
    weight = torch.tensor([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ], dtype=torch.float32)[None, None]
    conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding="same",
        padding_mode="circular",
        bias=False,
        device=device,
    )
    conv.weight = torch.nn.Parameter(weight.to(device))

    neighbor_count = conv(board.to(torch.float32).unsqueeze(1)).squeeze(1)
    new_board = board.clone()
    new_board[neighbor_count == 3] = 1
    new_board[(neighbor_count > 3) | (neighbor_count < 2)] = 0
    return new_board


