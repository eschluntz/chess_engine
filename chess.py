#!/usr/bin/env python3

import time

import numpy as np

WIN_SCORE = 1000


class ChessBoard(object):
    TURNS = ["white", "black"]

    def __init__(self):
        self.board = self.board = np.full(shape=(8, 8), fill_value=".", dtype="<U1")
        self.past_moves = []
        self.turn = "white"
        self.set_pieces()

    def set_pieces(self):
        """Places all the pieces on the board.
        white pieces: UPPER-CASE
        black pieces: lower-case

        king:   k
        queen:  q
        bishop: b
        knight: n
        rook:   r
        """
        back_row = ["♖", "♘", "♗", "♛", "♔", "♗", "♘", "♖"]
        front_row = ["♙"] * 8

        self.board[0] = np.array(back_row)
        self.board[1] = np.array(front_row)
        self.board[6] = np.array([p.upper() for p in front_row])
        self.board[7] = np.array([p.upper() for p in back_row])

    def __str__(self):
        """Displays the chess board"""
        out = ""
        for row in self.board:
            out += (" ".join(row) + "\n")
        return out

b = ChessBoard()
print(b)
