#!/usr/bin/env python3

import time
from typing import Dict, List, Tuple, Sequence

import numpy as np

WIN_SCORE = 1000
SIZE = 8


class ChessBoard(object):
    TURNS = ["white", "black"]

    def __init__(self):
        self.board = np.full(shape=(SIZE, SIZE), fill_value=".", dtype="<U1")
        self.past_moves = []
        self.turn = "white"
        # TODO: store info to assess whether castling is still allowed, and en passant is still allowed
        self.set_pieces()

    def clear_pieces(self):
        """Remove all pieces from the board"""
        self.board = np.full(shape=(SIZE, SIZE), fill_value=".", dtype="<U1")

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

    def find_my_pieces(self) -> Sequence[Tuple[str, int, int]]:
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

    def get_possible_moves(self, r : int, c : int, piece=None) -> Sequence[Tuple[int, int]]:
        """Given a particular piece, generates all possible on board moves for it.
        piece: optional param to override piece at board location
        TODO: filter moves that would put us in check.
        Returns [(r,c),...]. """

        if piece is None:
            piece : str = self.board[r, c]
        if piece.islower():
            my_piece = str.islower
            other_piece = str.isupper
        else:
            my_piece = str.isupper
            other_piece = str.islower

        def inbound(r, c):
            """Checks if coords are in the board"""
            return 0 <= r <= 7 and 0 <= c <= 7

        def get_sliding_moves(steps : Sequence[Tuple[int, int]], max_steps=SIZE) -> Sequence[Tuple[int, int]]:
            """Expand a list of "step" directions into a list of possible moves for the piece"""
            moves = []
            for dr, dc in steps:
                for i in range(1, max_steps + 1):
                    r2, c2 = r + i * dr, c + i * dc  # slide along step direction
                    if not inbound(r2, c2):
                        break
                    elif my_piece(self.board[r2, c2]):
                        break
                    elif other_piece(self.board[r2, c2]):
                        moves.append((r2, c2))
                        break
                    else:  # empty square, don't break the search
                        moves.append((r2, c2))
            return moves

        def get_jumping_moves(jumps : Sequence[Tuple[int, int]]) -> Sequence[Tuple[int, int]]:
            """Filter a list of jumping moves and return the valid ones"""
            moves = []
            print("-----------------------")
            print(self.board)
            for dr, dc in jumps:
                r2, c2 = r + dr, c + dc
                print("looking at: {}".format((r2, c2)))
                if inbound(r2, c2):
                    print("inbound")
                    print("contents: {}".format(self.board[r2, c2]))
                    if not my_piece(self.board[r2, c2]):
                        moves.append((r2, c2))
                        print("added!")
            return moves

        # piece movements
        knight_jumps = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
        king_jumps = [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]
        rook_steps = [(1,0), (0,1), (-1,0), (0,-1)]
        bishop_steps = [(1,1), (1,-1), (-1,1), (-1,-1)]
        queen_steps = rook_steps + bishop_steps

        piece_type = piece.lower()
        if piece_type == "p":
            # pawns are both asymmetric, and moves depend on their position D:
            if piece.isupper():  # white
                pawn_steps = [(-1, 0)]
                max_steps = 2 if r == 6 else 1  # double jump from starting row
            else:  # black
                pawn_steps = [(1, 0)]
                max_steps = 2 if r == 1 else 1  # double jump from starting row
            moves = get_sliding_moves(pawn_steps, max_steps)
        elif piece_type == "r":
            moves = get_sliding_moves(rook_steps)
        elif piece_type == "n":
            moves = get_jumping_moves(knight_jumps)
        elif piece_type == 'b':
            moves = get_sliding_moves(bishop_steps)
        elif piece_type == 'q':
            moves = get_sliding_moves(queen_steps)
        elif piece_type == "k":
            moves = get_jumping_moves(king_jumps)
        else:
            raise ValueError("Unknown piece! {}".format(piece))

        return moves

    def filter_legal_moves(self, piece: str, moves : Sequence[Tuple[int,int]]):
        """Filters a list of moves to only include legal ones.
        1. is not blocked by any other pieces
        2. piece does not land on any of our other pieces.
        3. TODO: does not put us in check
        """

    def moves_to_array(self, moves : Sequence[Tuple[int,int]]) -> np.array:
        """For visualization purposes, draw all locations of moves onto a board"""
        board = np.full(shape=(SIZE, SIZE), fill_value=0)
        for r, c in moves:
            board[r, c] = 1
        return board

    # def moves(self):
    #     """returns a list of all possible moves given the current board state and turn"""

    #     # 1 find all my pieces. TODO: better to just maintain this as a second datastore?
    #     pieces = self.find_my_pieces()

    #     # generate possible moves
    #     for piece, row, column in pieces:
    #         pos_moves = self.get_possible_moves(piece, row, column)

    #         def allowed(r, c):
    #             if self.turn == "white":
    #                 return not self.board[r, c].isupper()  # can go anywhere except on our own pieces
    #             else:

    #         moves = [ (r, c) for r,c in pos_moves if self.board[r,c] ]


    def __str__(self):
        """Displays the chess board"""
        out = ""
        for row in self.board:
            out += (" ".join(row) + "\n")
        return out
