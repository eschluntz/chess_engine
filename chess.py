#!/usr/bin/env python3

import time
from typing import Dict, List, Tuple

import numpy as np

WIN_SCORE = 1000
SIZE = 8


class ChessBoard(object):
    TURNS = ["white", "black"]

    def __init__(self):
        self.board = self.board = np.full(shape=(SIZE, SIZE), fill_value=".", dtype="<U1")
        self.past_moves = []
        self.turn = "white"
        # TODO: store info to assess whether castling is still allowed, and en passant is still allowed
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
        back_row = ["r", "n", "b", "q", "k", "b", "n", "r"]
        front_row = ["p"] * 8

        self.board[0] = np.array(back_row)
        self.board[1] = np.array(front_row)
        self.board[6] = np.array([p.upper() for p in front_row])
        self.board[7] = np.array([p.upper() for p in back_row])

    def find_my_pieces(self):
        """Returns a list of all the current player's pieces and their locations.
        [(piece, row, column),]"""
        pieces = []

        if self.turn == "white":
            my_piece = str.islower
        else:
            my_piece = str.isupper

        for r in range(SIZE):
            for c in range(SIZE):
                if my_piece(self.board[r, c]):
                    pieces.append((self.board[r, c], r, c))

        return pieces

    def rotate_coords(self, r : int, c : int) -> Tuple[int, int]:
        """Flips coordinates into the other player's point of view"""
        return (7 - r, 7 - c)

    def get_possible_moves(self, piece : str, r : int, c : int) -> List[Tuple[int, int]]:
        """Given a particular piece, generates all possible on board moves for it,
        even if some are illegal (i.e. or ontop of other pieces, or would put us in check)
        Returns [(r,c),...]. """

        # view all pieces are now from their player's point of view
        rotated = False
        if piece.islower():  # black piece, look from their perspective
            rotated = True
            r, c = self.rotate_coords(r, c)
        else:
            piece = piece.lower()

        # define these so they're reusable for the queen moves
        def get_rook_moves(r, c):
            return [ (r, i) for i in range(SIZE) if i != c ] + [ (i, c) for i in range(SIZE) if i != r]

        def get_bishop_moves(r, c):
            rel_move_sublists = [[(di, di), (di, -di), (-di, di), (-di, -di)] for di in range(1, SIZE + 1)]
            return sum(rel_move_sublists, [])

        if piece == "p":  # TODO does not handle en passant
            if r == 6:  # double move forward
                moves = [(r - 1, c), (r - 2, c)]
            else:
                moves = [(r - 1, c)]

        elif piece == "r":  # careful to avoid out noop moves
            moves = get_rook_moves(r, c)

        elif piece == "n":
            rel_moves = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
            moves = [ (r + dr, c + dc) for dr, dc in rel_moves ]

        elif piece == 'b': # careful to filter out noop moves
            moves = get_bishop_moves(r, c)

        elif piece == 'q':
            moves = get_bishop_moves(r, c) + get_rook_moves(r, c)

        elif piece == 'k':  # TODO does not handle castling
            rel_moves = [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]
            moves = [ (r + dr, c + dc) for dr, dc in rel_moves ]

        else:
            raise ValueError("Unknown piece! {}".format(piece))

        if rotated:  # put back into board frame
            moves = [ self.rotate_coords(r, c) for r, c in moves ]

        # filter off board moves
        final_moves = []
        for r, c in moves:
            if 0 <= r <= 7 and 0 <= c <= 7:
                final_moves.append((r, c))

        return final_moves

    def moves(self):
        """returns a list of all possible moves given the current board state and turn"""

        # 1 find all my pieces. TODO: better to just maintain this as a second datastore?
        pieces = self.find_my_pieces()

        # generate possible moves
        for row, column, piece in pieces:
            possible_moves = self.get_possible_moves(piece, row, column)


    def __str__(self):
        """Displays the chess board"""
        out = ""
        for row in self.board:
            out += (" ".join(row) + "\n")
        return out

b = ChessBoard()
print(b)
