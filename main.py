from typing import Callable
import cv2
import numpy as np

from yawnoc import get_backward_fn, SolverType
from yawnoc.forward import forward
from yawnoc.utils import init_board



SEEDS = {
    "nice": np.array([[1]]),
    "nice-1+1": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 0]]),
    "3x3": np.ones((3, 3)),
}


def run(board: np.ndarray, backward_fn: Callable[[np.ndarray], np.ndarray]) -> None:
    delay = 0
    while True:
        # visualization
        viz_board = cv2.resize(board*255, dsize=(500, 500), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imshow("Board", viz_board)
        key = cv2.waitKey(delay=delay)
        if key == ord("q"):
            cv2.destroyAllWindows()
            break
        elif key == ord("."):
            # next
            board = forward(board)
        elif key == ord(","):
            # prev
            board = backward_fn(board)


def main():
    board_size = (50, 50)
    seed = SEEDS["nice"]
    board = init_board(seed=seed, board_size=board_size)

    backward_fn = get_backward_fn(SolverType.BRUTE)
    run(board=board, backward_fn=backward_fn)


if __name__ == "__main__":
    main()
