from typing import Union

import numpy as np
import torch
import torch.nn as nn

ArrayLike = Union[np.ndarray, torch.Tensor]

class DeepSolver(nn.Module):
    """
    MLP for predicting previous board state
    """
    def __init__(self, board_size: tuple[int, int], num_features: int):
        """
        Args:
            board_size (tuple[int, int]):
                size of board in rows, cols
            num_featres (int):
                hidden feature size in MLP
        """
        super().__init__()
        self._board_size = board_size
        input_size = board_size[0] * board_size[1]

        # TODO: Investigate better models
        self._model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=num_features),
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(),
            nn.Linear(in_features=num_features, out_features=num_features),
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(),
            nn.Linear(in_features=num_features, out_features=input_size),
            # remove for cross entropy with logits?
            nn.Sigmoid(),
        )
    
    def forward(self, board: ArrayLike) -> torch.Tensor:
        """
        Predict a previous generation given a current board state

        Args:
            board (ArrayLike):
                A 2D array/tensor where 0/1 represent alive/dead cells.
                Shape (batch_size, rows, cols)
        Returns:
            A probability map of the same shape as board. Each pixel contains
            the probability that pixel is alive in the previous generation.
            To get the binary board state, run DeepSolver.decode()
        """
        if isinstance(board, np.ndarray):
            board = torch.from_numpy(board)

        # flatten input for linear layer
        board = torch.flatten(board, start_dim=1, end_dim=-1)

        # normalize
        # we assume that the board contains uniform values from 0-1
        # this has mean: 0.5 and stdev 0.5
        # (not necessarily true but close enough)
        board = (board.to(torch.float32) - 0.5) / 0.5

        prev = self._model(board)
        
        # unflatten the output
        prev = prev.reshape((prev.shape[0], *self._board_size))

        return prev
    
    def decode(self, probability_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probability_map (Tensor):
                A probability map of the same shape as board. Each pixel contains
                the probability that pixel is alive in the previous generation.
        Returns:
            A binary board state where 0/1 represents a dead/alive cell

        """
        return torch.where(probability_map >= 0.5, 1.0, 0.0)


class DeepEstimator(nn.Module):
    """
    MLP for predicting board closeness
    """
    def __init__(self, board_size: tuple[int, int], num_features: int):
        """
        Args:
            board_size (tuple[int, int]):
                size of board in rows, cols
            num_featres (int):
                hidden feature size in MLP
        """
        super().__init__()
        self._board_size = board_size
        input_size = 2 * board_size[0] * board_size[1]

        # TODO: Investigate better models
        self._model = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=num_features),
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=num_features, out_features=num_features),
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=num_features, out_features=1),
        )
    
    def forward(self, board: ArrayLike, target_board: ArrayLike) -> torch.Tensor:
        """
        Predict a previous generation given a current board state

        Args:
            board (ArrayLike):
                A 2D array/tensor where 0/1 represent alive/dead cells.
                Shape (batch_size, rows, cols)
        Returns:
            A probability map of the same shape as board. Each pixel contains
            the probability that pixel is alive in the previous generation.
            To get the binary board state, run DeepSolver.decode()
        """
        def preprocess_board(b: np.ndarray) -> np.ndarray:
            if isinstance(b, np.ndarray):
                b = torch.from_numpy(b)
            # flatten input for linear layer
            b = torch.flatten(b, start_dim=1, end_dim=-1)

            # normalize
            # we assume that the board contains uniform values from 0-1
            # this has mean: 0.5 and stdev 0.5
            # (not necessarily true but close enough)
            b = (b.to(torch.float32) - 0.5) / 0.5

            return b
        
        board = preprocess_board(board)
        target_board = preprocess_board(target_board)
        input_tensor = torch.concat([board, target_board], axis=-1)

        closeness = self._model(input_tensor)

        return closeness
