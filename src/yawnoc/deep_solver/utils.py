from pathlib import Path

import cv2
import hydra
import lightning as L
import numpy as np
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader

CFG_DIR = Path(__file__).parent  / "configs"


def get_hydra_cfg(config: str, overrides: list = None) -> OmegaConf:
    """
    Convenience function to load a hydra config
    """
    with hydra.initialize_config_dir(str(CFG_DIR), version_base="1.3"):
        cfg = hydra.compose(config, overrides=overrides)
    return cfg


def get_dataloader_from_cfg(config: OmegaConf) -> DataLoader:
    dataset = hydra.utils.instantiate(config.dataset, _convert_="partial")
    dataloader = hydra.utils.instantiate(
        config.dataloader,
        dataset=dataset,
    )
    return dataloader


def draw_idx_from_batch(left: Tensor, right: Tensor, idx: int, short_edge: int) -> np.ndarray:
    """
    Given two batches of board states, draw the a single board state from each batch
    Useful for drawing diagnostics.

    Args:
        left, right (Tensor):
            batch of board states of shape (batch size, rows, cols)
        idx (int):
            which index of batch to draw from left / right
        short_edge (int):
            resize short edge of the diag image to this size in pixels
    Returns:
        An ndarray of the board states specified by idx concatenated horizontally
    """
    assert left.shape == right.shape
    left = left[idx].to("cpu").numpy()
    right = right[idx].to("cpu").numpy()

    # grayscale bar to separate left and right board states
    separator = 0.5 * np.ones((left.shape[0], 1), dtype=np.uint8)

    image = np.hstack([left, separator, right])

    # resize board to have short_edge number of pixels on the smallest size
    min_size = min(image.shape)
    alpha = short_edge // min_size
    image = cv2.resize(255 * image, dsize=(0, 0), fx=alpha, fy=alpha, interpolation=cv2.INTER_NEAREST_EXACT)
    
    return image