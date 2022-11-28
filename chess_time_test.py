#!/usr/bin/env python3

import time
import numpy as np
from search import minmax
from chessboard import (
    ChessBoard,
)
from chess import eval_chess_board


b = ChessBoard()
b.board = np.array(
    (
        "r . . . k . . r".split(),
        "p . p p q p b .".split(),
        "b n . . p n p .".split(),
        ". . . P N . . .".split(),
        ". p . . P . . .".split(),
        ". . N . . Q . p".split(),
        "P P P B B P P P".split(),
        "R . . . K . . R".split(),
    )
)
b._sync_board_to_piece_set()

t0 = time.time()
_, move = minmax(b, eval_chess_board, 4)
t1 = time.time()
print(t1 - t0)