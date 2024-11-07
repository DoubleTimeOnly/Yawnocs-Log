from enum import Enum
from functools import partial
from typing import Callable

import numpy as np

from yawnoc.forward import forward
from yawnoc.solvers import deep_solver, gradient_descent, pattern_search


class SolverType(Enum):
    GRADIENT_DESCENT = 1
    PATTERN = 2
    DEEP = 3
    DEEP_GRADIENT_DESCENT = 4


def get_backward_fn(
    solver_type: SolverType
) -> Callable[[np.ndarray], np.ndarray]:
    if solver_type is SolverType.GRADIENT_DESCENT:
        err_fn = gradient_descent.mse_bit_error
        return partial(gradient_descent.backward, err_fn=err_fn)
    elif solver_type is SolverType.DEEP_GRADIENT_DESCENT:
        err_fn = _get_deep_estimator_err_fn()
        return partial(gradient_descent.backward, err_fn=err_fn)
    elif solver_type is SolverType.PATTERN:
        return pattern_search.backward
    elif solver_type is SolverType.DEEP:
        return deep_solver.create_backward()
    else:
        raise ValueError(f"Did not recognize solver type {solver_type}")


def _get_deep_estimator_err_fn():
    import torch
    from yawnoc.deep_solver.solver_module import LitEstimator
    from yawnoc.deep_solver import WEIGHTS_FOLDER

    # ckpt = WEIGHTS_FOLDER / "deep_estimator_5x5_epoch=999-step=1000.ckpt"
    ckpt = r"C:\Users\Victor\Documents\Projects\conway_game\experiment_results\train_deep_estimator\version_34\checkpoints\epoch=9999-step=10000.ckpt"
    # ckpt = r"C:\Users\Victor\Documents\Projects\conway_game\experiment_results\train_deep_estimator\version_3\checkpoints\epoch=999-step=1000.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LitEstimator.load_from_checkpoint(ckpt).eval().to(device)

    @torch.no_grad()
    def err_fn(target: np.ndarray, guess: np.ndarray) -> np.ndarray:
        assert target.shape == tuple(model._model._board_size), f"Model expects a board size of {model._model._board_size} but got board size of {target.shape}"

        next_gen = guess
        for i in range(3):
            next_gen = forward(next_gen)
            if (next_gen == target).all():
                return 0

        # board_tensor = torch.from_numpy(target)[None].to(device)
        closeness = model(
            board_batch=guess[None],
            target_board_batch=target[None]
        )[0].to("cpu").numpy().item()

        return closeness

    return err_fn

