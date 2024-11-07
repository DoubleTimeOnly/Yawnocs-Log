from pathlib import Path
from typing import Tuple

import cv2
import lightning as L
import numpy as np
import torch
from torch import Tensor, nn, optim

from yawnoc.deep_solver.model import DeepEstimator, DeepSolver
from yawnoc.deep_solver.utils import draw_idx_from_batch
from yawnoc.forward import batch_forward


class LitSolver(L.LightningModule):
    """
    Wrapper around solver model for training
    """
    def __init__(
        self, 
        board_size: Tuple[int, int], 
        inter_features: int
    ):
        """
        Args:
            board_size (Tuple[int, int]):
                board size in rows, cols
            inter_features (int):
                hidden feature size in MLP
        """
        super().__init__()
        self._model = DeepSolver(
            board_size=board_size, 
            num_features=inter_features
        )
        self.save_hyperparameters()
        self._loss_fn = nn.functional.binary_cross_entropy

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch (Tensor):
                A batch of board states to predict previous generation of
                Shape: (batch size, rows, cols)
        Returns:
            - If training: probability of each pixel being on
            - If eval: predicted binary board state of previous generation
        """
        output = self._model(batch)
        if not self.training:
            output = self._model.decode(output)
        return output

    def training_step(self, batch, batch_idx):
        """
        """
        # Given a board state, compute its next generation
        next_batch = batch_forward(batch, device=batch.device)

        # Try to predict the original board state from the next generation
        prev = self._model(next_batch)
        loss = self._loss_fn(target=batch, input=prev)
        
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        curr_batch = batch_forward(batch, device=batch.device)
        prev = self._model(curr_batch)
        loss = self._loss_fn(target=batch, input=prev)

        decoded_prev = self._model.decode(prev)
        prev_num_errors = torch.count_nonzero(decoded_prev != batch) / batch.shape[0]

        curr_hat = batch_forward(decoded_prev, device=decoded_prev.device)
        curr_num_errors = torch.count_nonzero(curr_hat != curr_batch) / batch.shape[0]


        self.log("test/loss", loss, on_epoch=True, on_step=False)
        self.log("test/prev_num_errors", prev_num_errors, on_epoch=True, on_step=False)
        self.log("test/curr_num_errors", curr_num_errors, on_epoch=True, on_step=False)

        # save images
        save_path = Path(self.logger.log_dir) / "diags"
        save_path.mkdir(exist_ok=True)

        # save what the predicted and ground truth previous generations look like
        prev_diag = draw_idx_from_batch(left=decoded_prev, right=batch, idx=0, short_edge=500)
        cv2.imwrite(str(save_path / f"{batch_idx}_prev.png"), prev_diag)

        # save what the predicted and ground truth next generations look like
        curr_diag = draw_idx_from_batch(left=curr_hat, right=curr_batch, idx=0, short_edge=500)
        cv2.imwrite(str(save_path / f"{batch_idx}_next.png"), curr_diag)

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
        

class LitEstimator(L.LightningModule):
    """
    Wrapper around solver model for training
    """
    def __init__(
        self, 
        board_size: Tuple[int, int], 
        inter_features: int,
        series_length: int,
    ):
        """
        Args:
            board_size (Tuple[int, int]):
                board size in rows, cols
            inter_features (int):
                hidden feature size in MLP
        """
        super().__init__()
        self._model = DeepEstimator(
            board_size=board_size, 
            num_features=inter_features
        )
        self.save_hyperparameters()
        self._loss_fn = nn.functional.smooth_l1_loss
        self._series_length = series_length

    def forward(self, board_batch: torch.Tensor, target_board_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch (Tensor):
                A batch of board states to predict previous generation of
                Shape: (batch size, rows, cols)
        Returns:
            - If training: probability of each pixel being on
            - If eval: predicted binary board state of previous generation
        """
        if isinstance(board_batch, np.ndarray):
            board_batch = torch.from_numpy(board_batch).to(self.device)
        if isinstance(target_board_batch, np.ndarray):
            target_board_batch = torch.from_numpy(target_board_batch).to(self.device)

        return self._model(board=board_batch, target_board=target_board_batch)

    @staticmethod
    def _create_batch(board_batch, series_length: int) -> Tuple[Tensor, Tensor, Tensor]:
        all_boards, steps = [], []
        for step in range(series_length):
            if step == 0:
                boards = board_batch
            else:
                boards = batch_forward(boards)
            all_boards.append(boards)
            steps.append(
                torch.full(
                    (boards.shape[0],), 
                    series_length - step, 
                    device=board_batch.device,
                    dtype=torch.float32
                )
            )
        target_boards = batch_forward(all_boards[-1]).repeat((series_length, 1, 1))
        all_boards = torch.concat(all_boards, dim=0)
        steps = torch.concat(steps, dim=0).unsqueeze(-1)

        return all_boards, steps, target_boards


    def training_step(self, board_batch, batch_idx):
        """
        """
        if isinstance(self._series_length, (tuple, list)):
            low, high = self._series_length
            series_length = np.random.randint(low=low, high=high)
        else:
            series_length = self._series_length

        all_boards, steps, target_boards = LitEstimator._create_batch(
            board_batch, series_length=series_length,
        )

        batch_size = all_boards.shape[0]
        random_idxs = np.random.choice(batch_size, size=batch_size//2, replace=False)
        all_boards = all_boards[random_idxs]
        steps = steps[random_idxs]
        target_boards = target_boards[random_idxs]

        # Try to predict the original board state from the next generation
        closeness = self._model(board=all_boards, target_board=target_boards)
        loss = self._loss_fn(target=steps, input=closeness)
        
        mean_abs_error = (closeness - steps).abs().mean()
        self.log("train/loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train/mean_abs_error", mean_abs_error, on_epoch=True, on_step=False)

        return loss

    def test_step(self, board_batch, batch_idx):
        all_boards, steps, target_boards = LitEstimator._create_batch(
            board_batch, series_length=10
        )

        closeness = self._model(board=all_boards, target_board=target_boards)
        mean_abs_error = (closeness - steps).abs().mean()
        loss = self._loss_fn(target=steps, input=closeness)

        self.log("test/loss", loss, on_epoch=True, on_step=False)
        self.log("test/mean_abs_error", mean_abs_error, on_epoch=True, on_step=False)

    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.01)
        config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer, 
                    T_max=self.trainer.estimated_stepping_batches,
                ),
                "frequency": 1,
                "interval": "step",
            }
        }

        return config
               