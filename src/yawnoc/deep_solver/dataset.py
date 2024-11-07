import torch
from torch import Tensor


class ConwayDataset:
    """
    Dataset that generates random board states with each __getitem__ call
    """
    def __init__(self, board_size: tuple[int, int], length: int):
        """
        board_size: 
            size of board in rows, cols

        length (int):
            Used by pytorch dataloader to compute number of batches in the dataloader.
            Samples are generated on the fly so there technically isn't a dataset length.
            Instead it's hardcoded
        """
        self._board_size = board_size
        self._length = length

    def __getitem__(self, idx) -> Tensor:
        return torch.randint(low=0, high=2, size=self._board_size, dtype=torch.float32)

    def __len__(self) -> int:
        return self._length
