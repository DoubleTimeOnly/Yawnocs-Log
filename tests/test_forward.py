import torch
from yawnoc.forward import batch_forward, forward
from yawnoc.utils import init_board
import numpy as np


def test_batch_forward_is_same_as_forward():
    batch = [np.random.randint(low=0, high=2, size=(10, 10), dtype=np.uint8) for i in range(10)]

    result1 = np.stack([forward(b) for b in batch], axis=0)
    batch = np.stack(batch, axis=0)
    result2 = batch_forward(torch.from_numpy(batch)).numpy()

    assert np.all(result1 == result2)
    