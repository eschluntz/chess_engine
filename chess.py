#!/usr/bin/env python3

import time
from typing import Dict, List, Tuple, Sequence
from copy import deepcopy
from termcolor import colored

import numpy as np

WIN_SCORE = 1000
SIZE = 8
class Move(object):
    """Class to represent a move.
    TODO: handle castles and promotions"""

    def __init__(self, r_from : int, c_from : int, r_to : int, c_to : int, piece=None) -> None:
        self.r_from = r_from
        self.c_from = c_from
        self.r_to = r_to
        self.c_to = c_to
        self.piece = piece

        self.special = False

    def __str__(self) -> str:
        return "Move {} ({}, {}) -> ({}, {})".format(self.piece, self.r_from, self.c_from, self.r_to, self.c_to)

    def __eq__(self, other) -> bool:
        return self.__dict__ == other.__dict__

class ChessBoard(object):
    TURNS = ["white", "black"]

    def __init__(self):
        self.board = np.full(shape=(SIZE, SIZE), fill_value=".", dtype="<U1")
        self.past_moves = []
        self.turn = "white"
        # TODO: store info to assess whether castling is still allowed, and en passant is still allowed
        self.set_pieces()

    def next_turn(self) -> str:
        """Returns "white" or "black whichever is not our current turn"""
        if self.turn == "white":
            return "black"
        else:
            return "white"

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
            my_piece = str.isupper
        else:
            my_piece = str.islower

        for r in range(SIZE):
            for c in range(SIZE):
                if my_piece(self.board[r, c]):
                    pieces.append((self.board[r, c], r, c))

        return pieces

    def get_dests_for_piece(self, r : int, c : int, piece=None) -> Sequence[Tuple[int, int]]:
        """Given a particular piece, generates all possible destinatinos for it to move to.
        piece: optional param to override piece at board location
        TODO: filter destinations that would put us in check.
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

        def get_sliding_dests(steps : Sequence[Tuple[int, int]], max_steps=SIZE) -> Sequence[Tuple[int, int]]:
            """Expand a list of "step" directions into a list of possible destinations for the piece"""
            dests = []
            for dr, dc in steps:
                for i in range(1, max_steps + 1):
                    r2, c2 = r + i * dr, c + i * dc  # slide along step direction
                    if not inbound(r2, c2):
                        break
                    elif my_piece(self.board[r2, c2]):
                        break
                    elif other_piece(self.board[r2, c2]):
                        dests.append((r2, c2))
                        break
                    else:  # empty square, don't break the search
                        dests.append((r2, c2))
            return dests

        def get_jumping_dests(jumps : Sequence[Tuple[int, int]]) -> Sequence[Tuple[int, int]]:
            """Filter a list of jumping destinations and return the valid ones"""
            dests = []
            for dr, dc in jumps:
                r2, c2 = r + dr, c + dc
                if inbound(r2, c2):
                    if not my_piece(self.board[r2, c2]):
                        dests.append((r2, c2))
            return dests

        def get_pawn_dests():
            """pawns are actually the most complex pieces on the board! Their moves:
            1. are asymmetric, 2. depend on their position, 3. depends on opponents 4. moves do not equal captures
            TODO: implement promoting
            TODO: clean up to be less repetitive?"""
            pawn_jumps = []
            if piece.isupper():  # white
                if self.board[r - 1, c] == ".":  # jump forward if clear
                    pawn_jumps.append((-1, 0))
                    if r == 6 and self.board[r - 2, c] == ".":  # double jump if not blocked and on home row
                        pawn_jumps.append((-2, 0))
                for dc in [-1, 1]:  # captures
                    r2, c2 = r - 1, c + dc
                    if inbound(r2, c2) and self.board[r2, c2].islower():
                        pawn_jumps.append((-1, dc))
            else:  # black
                if self.board[r + 1, c] == ".":  # jump forward if clear
                    pawn_jumps.append((1, 0))
                    if r == 1 and self.board[r + 2, c] == ".":  # double jump if not blocked and on home row
                        pawn_jumps.append((2, 0))
                for dc in [-1, 1]:  # captures
                    r2, c2 = r - 1, c + dc
                    if inbound(r2, c2) and self.board[r2, c2].isupper():
                        pawn_jumps.append((1, dc))
            return get_jumping_dests(pawn_jumps)

        # piece delta movements
        knight_jumps = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
        king_jumps = [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]
        rook_steps = [(1,0), (0,1), (-1,0), (0,-1)]
        bishop_steps = [(1,1), (1,-1), (-1,1), (-1,-1)]
        queen_steps = rook_steps + bishop_steps

        piece_type = piece.lower()
        if piece_type == "p":
            destinations = get_pawn_dests()
        elif piece_type == "r":
            destinations = get_sliding_dests(rook_steps)
        elif piece_type == "n":
            destinations = get_jumping_dests(knight_jumps)
        elif piece_type == 'b':
            destinations = get_sliding_dests(bishop_steps)
        elif piece_type == 'q':
            destinations = get_sliding_dests(queen_steps)
        elif piece_type == "k":
            destinations = get_jumping_dests(king_jumps)
        else:
            raise ValueError("Unknown piece! {}".format(piece))

        return destinations

    def dests_to_array(self, dests : Sequence[Tuple[int,int]]) -> np.array:
        """For visualization purposes, draw all locations of destinations onto a board"""
        board = np.full(shape=(SIZE, SIZE), fill_value=0)
        for r, c in dests:
            board[r, c] = 1
        return board

    def moves(self) -> Sequence[Move]:
        """returns a list of all possible moves given the current board state and turn.
        Move: ((r_from, c_from), (r_to, c_to))"""

        # 1 find all my pieces. TODO: better to just maintain this as a second datastore?
        pieces = self.find_my_pieces()

        # generate possible moves
        all_moves = []
        for piece, r_from, c_from in pieces:
            for r_to, c_to in self.get_dests_for_piece(r_from, c_from):
                move = Move(r_from, c_from, r_to, c_to, piece=piece)
                all_moves.append(move)

        # TODO order these nicely to improve alpha beta pruning
        return all_moves

    def do_move(self, move: Move):
        """Do a move on the chessboard"""
        piece = self.board[move.r_from, move.c_from]
        self.board[move.r_from, move.c_from] = "."
        self.board[move.r_to, move.c_to] = piece
        self.turn = self.next_turn()

    def print_move(self, move: Move):
        """Graphically represents a move"""
        print(move)
        board = deepcopy(self.board)
        board = board.astype("<U20")  # to allow for color strings
        board[move.r_from, move.c_from] = colored(board[move.r_from, move.c_from], "red")
        board[move.r_to, move.c_to] = colored(board[move.r_to, move.c_to], "green")
        out = ""
        for row in board:
            out += (" ".join(row) + "\n")
        print(out)


    def __str__(self):
        """Displays the chess board"""
        out = ""
        for row in self.board:
            out += (" ".join(row) + "\n")
        return out
