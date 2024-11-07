from typing import Callable, Optional

import numpy as np
import torch

from yawnoc.deep_solver import WEIGHTS_FOLDER
from yawnoc.deep_solver.solver_module import LitSolver


def create_backward(ckpt: Optional[str] = None) -> Callable[[np.ndarray], np.ndarray]:
    ckpt = ckpt if ckpt is not None else "epoch=999-step=1000.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitSolver.load_from_checkpoint(
        WEIGHTS_FOLDER / ckpt
    ).eval().to(device)

    
    @torch.no_grad()
    def backward(board: np.ndarray) -> np.ndarray:
        assert board.shape == model._model._board_size, f"Model expects a board size of {model._model._board_size} but got board size of {board.shape}"
        board_tensor = torch.from_numpy(board)[None].to(device)
        prev_generation = model(board_tensor)[0].to("cpu").numpy().astype(np.uint8)
        return prev_generation

    return backward