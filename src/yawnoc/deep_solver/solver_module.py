from pathlib import Path
from typing import Tuple

import cv2
import lightning as L
import torch
from torch import nn, optim

from yawnoc.deep_solver.model import DeepSolver
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
        