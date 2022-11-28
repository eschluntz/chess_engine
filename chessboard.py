#!/usr/bin/env python3

"""Contains objects to represent a Chessboard and a Move.
Handles all the logic of the rules of chess, generating valid moves, etc.

Does not contain any 'AI' to control the actual selection or evaluation of moves"""


import time
from typing import Dict, List, Tuple, Sequence, Set, Callable, TypeVar, Optional
from copy import deepcopy
from termcolor import colored
import functools

import numpy as np

from search import minmax, iterative_deepening

SIZE = 8
WHITE_PIECES = ["P", "R", "N", "B", "K", "Q"]
BLACK_PIECES = [p.lower() for p in WHITE_PIECES]
ALL_PIECES = WHITE_PIECES + BLACK_PIECES


def inbound(r, c):
    """Checks if coords are in the board"""
    return 0 <= r < SIZE and 0 <= c < SIZE


class Move(object):
    """Class to represent a move.
    Special moves: represented by the 'special' char code.
        c: castle
        e: en passant
        q,n,b,r: promotion to that piece
    Tracking for special moves:
        castle_changed: True if this move changed some ability to castle. (important for undo_move)
        castling: use the movement of the king as the to/from fields to distinguish which castle
    """

    def __init__(self, r_from: int, c_from: int, r_to: int, c_to: int,
            piece : Optional[str] = None,
            captured : Optional[str] = None,
            special : Optional[str] = None) -> None:
        self.r_from = r_from
        self.c_from = c_from
        self.r_to = r_to
        self.c_to = c_to
        self.special = special
        self.old_flags = {}  # filled in by do_move, for undoing later

        # optional and are filled in by the board when doing a move
        self.piece = piece
        self.captured = captured


    def __str__(self) -> str:
        if self.special is None:
            special = ""
        else:
            special = " *{}*".format(self.special)

        if self.captured is None:
            capt = ""
        else:
            capt = " x {}".format(self.captured)

        return "Move {} ({}, {}) -> ({}, {}) {} {}".format(
            self.piece, self.r_from, self.c_from, self.r_to, self.c_to, capt, special
        )

    def __eq__(self, other) -> bool:
        """Note: only compares to and from positions, not piece or capture"""
        return (self.r_from, self.c_from, self.r_to, self.c_to) == (other.r_from, other.c_from, other.r_to, other.c_to)


class ChessBoard(object):
    """Class to represent a chessboard.
    Board is represented by a 2D array + a piece set, which must be kept in sync.
    Several extra flags store information for special moves that require info about the past, i.e. castling
    """
    TURNS = ["white", "black"]

    def __init__(self):
        self.board = np.full(shape=(SIZE, SIZE), fill_value=".", dtype="<U1")
        self.piece_set: Set[Tuple[str, int, int]] = set()  # caches pieces for speedup. (piece, row, column?)

        # some special moves require past info of board state
        self.flags = dict(
            w_castle_left=True,
            w_castle_right=True,
            b_castle_left=True,
            b_castle_right=True,
            en_passant_spot=None # destination of en passant in the most recent move. # TODO: do i need this in the board?
        )

        self.past_moves: Sequence[Tuple[Move, str]] = []
        self.turn = "white"
        self.set_starting_pieces()

    def next_turn(self) -> str:
        """Returns "white" or "black whichever is not our current turn"""
        if self.turn == "white":
            return "black"
        else:
            return "white"

    def _sync_board_to_piece_set(self) -> None:
        """Sets the piece list from the ground truth of the board"""
        self.piece_set = set()
        for r in range(SIZE):
            for c in range(SIZE):
                p = self.board[r, c]
                if p != ".":
                    self.piece_set.add((p, r, c))
                        
    def clear_pieces(self) -> None:
        """Remove all pieces from the board"""
        self.board = np.full(shape=(SIZE, SIZE), fill_value=".", dtype="<U1")
        self._sync_board_to_piece_set()

    def set_starting_pieces(self) -> None:
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

        self._sync_board_to_piece_set()

    def find_my_pieces(self, turn=None) -> Sequence[Tuple[str, int, int]]:
        """Returns a list of all the current player's pieces and their locations.
        turn: override the current turn
        [(piece, row, column),]"""

        if turn is None:
            turn = self.turn
        if turn == "white":
            my_piece = str.isupper
        else:
            my_piece = str.islower

        return [x for x in self.piece_set if my_piece(x[0])]


    def _get_sliding_dests(
        self, r: int, c: int, player: str, steps: Sequence[Tuple[int, int]], max_steps=SIZE
    ) -> Sequence[Tuple[int, int]]:
        """Expand a list of "step" directions into a list of possible destinations for the piece"""
        if player == "black":
            my_piece = str.islower
            other_piece = str.isupper
        else:
            my_piece = str.isupper
            other_piece = str.islower

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

    def _get_jumping_dests(
        self, r: int, c: int, player: str, jumps: Sequence[Tuple[int, int]]
    ) -> Sequence[Tuple[int, int]]:
        """Filter a list of jumping destinations and return the valid ones"""
        if player == "black":
            my_piece = str.islower
            other_piece = str.isupper
        else:
            my_piece = str.isupper
            other_piece = str.islower

        dests = []
        for dr, dc in jumps:
            r2, c2 = r + dr, c + dc
            if inbound(r2, c2):
                if not my_piece(self.board[r2, c2]):
                    dests.append((r2, c2))
        return dests

    def _get_pawn_dests(self, r: int, c: int, player: str) -> Sequence[Tuple[int, int]]:
        """pawns are actually the most complex pieces on the board! Their moves:
        1. are asymmetric, 2. depend on their position, 3. depends on opponents 4. moves do not equal captures
        TODO: implement promoting"""

        pawn_jumps = []
        if player == "white":
            r2, c2 = r - 1, c
            if inbound(r2, c2) and self.board[r2, c2] == ".":  # jump forward if clear
                pawn_jumps.append((-1, 0))
                if r == 6 and self.board[r - 2, c] == ".":  # double jump if not blocked and on home row
                    pawn_jumps.append((-2, 0))
            for dc in [-1, 1]:  # captures
                r2, c2 = r - 1, c + dc
                if inbound(r2, c2) and self.board[r2, c2].islower():
                    pawn_jumps.append((-1, dc))
        else:  # black
            r2, c2 = r + 1, c
            if inbound(r2, c2) and self.board[r2, c2] == ".":  # jump forward if clear
                pawn_jumps.append((1, 0))
                if r == 1 and self.board[r + 2, c] == ".":  # double jump if not blocked and on home row
                    pawn_jumps.append((2, 0))
            for dc in [-1, 1]:  # captures
                r2, c2 = r + 1, c + dc
                if inbound(r2, c2) and self.board[r2, c2].isupper():
                    pawn_jumps.append((1, dc))
        return self._get_jumping_dests(r, c, player, pawn_jumps)

    def _get_castle_dests(self, player : str) -> Sequence[Tuple[int, int]]:
        """Returns any castle moves available to the current player"""
        pass

    def get_dests_for_piece(self, r: int, c: int, piece=None) -> Sequence[Tuple[int, int]]:
        """Given a particular piece, generates all possible destinations for it to move to.
        piece: optional param to override piece at board location
        TODO: filter destinations that would put us in check.
        Returns [(r,c),...]. """

        if piece is None:
            piece = self.board[r, c]
        if piece.islower():
            player = "black"
        else:
            player = "white"

        # piece delta movements
        knight_jumps = [(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)]
        king_jumps = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
        rook_steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        bishop_steps = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        queen_steps = rook_steps + bishop_steps

        piece_type = piece.lower()
        if piece_type == "p":
            destinations = self._get_pawn_dests(r, c, player)
        elif piece_type == "r":
            destinations = self._get_sliding_dests(r, c, player, rook_steps)
        elif piece_type == "n":
            destinations = self._get_jumping_dests(r, c, player, knight_jumps)
        elif piece_type == "b":
            destinations = self._get_sliding_dests(r, c, player, bishop_steps)
        elif piece_type == "q":
            destinations = self._get_sliding_dests(r, c, player, queen_steps)
        elif piece_type == "k":
            destinations = self._get_jumping_dests(r, c, player, king_jumps)
        else:
            raise ValueError("Unknown piece! {}".format(piece))

        return destinations

    def dests_to_array(self, dests: Sequence[Tuple[int, int]]) -> np.array:
        """For visualization purposes, draw all locations of destinations onto a board"""
        board = np.full(shape=(SIZE, SIZE), fill_value=0)
        for r, c in dests:
            board[r, c] = 1
        return board

    def moves(self, turn=None) -> Sequence[Move]:
        """returns a list of all possible moves given the current board state and turn.
        turn: "white" or "black" or None, to use the current turn
        Returns a list of Move Objects."""

        # 1 find all my pieces. TODO: better to just maintain this as a second datastore?
        pieces = self.find_my_pieces(turn)

        # generate possible normal moves
        all_moves = []
        for piece, r_from, c_from in pieces:
            for r_to, c_to in self.get_dests_for_piece(r_from, c_from):
                move = Move(r_from, c_from, r_to, c_to, piece=piece)
                all_moves.append(move)

        # add special moves

        return all_moves

    def do_move(self, move: Move):
        """Do a move on the chessboard"""
        piece = self.board[move.r_from, move.c_from]  # type: str
        captured = self.board[move.r_to, move.c_to]  # type: str
        self.board[move.r_from, move.c_from] = "."
        self.board[move.r_to, move.c_to] = piece

        # save current state of flags for undoing later
        move.old_flags = deepcopy(self.flags)

        # record info for future En Passant
        if piece.lower() == "p" and abs(move.r_from - move.r_to) == 2:  # detect a double jump to enable en passant
            self.flags["en_passant_spot"] = (move.r_to, move.c_to)
        else:
            self.flags["en_passant_spot"] = None  # clear

        # record info for future Castling. TODO nicer way to encode this logic?
        if piece == "k":
            self.flags["b_castle_left"], self.flags["b_castle_right"] = False, False
        elif piece == "K":
            self.flags["w_castle_left"], self.flags["w_castle_right"] = False, False
        elif piece == "r":
            if move.c_from == 0 and move.r_from == 0:  # move from upper left
                self.flags["b_castle_left"] = False
            elif move.c_from == 7 and move.r_from == 0:  # move from upper right
                self.flags["b_castle_right"] = False
        elif piece == "R":
            if move.c_from == 0 and move.r_from == 7:  # move from bottom left
                self.flags["w_castle_left"] = False
            elif move.c_from == 7 and move.r_from == 7:  # move from bottom right
                self.flags["w_castle_right"] = False

        # implement special moves
        if move.special == "c":  # castle
            pass
        elif move.special == "e":  # en passant
            pass
        elif move.special in ["q", "n", "b", "r"]:  # promotion
            if piece.isupper():
                piece = move.special.upper()
            else:
                piece = move.special
            self.board[move.r_to, move.c_to] = piece

        self.turn = self.next_turn()

        # update piece set datastructure
        self.piece_set.remove((piece, move.r_from, move.c_from))
        if captured != ".":
            self.piece_set.remove((captured, move.r_to, move.c_to))
        self.piece_set.add((piece, move.r_to, move.c_to))
            

        # save move
        move.captured = captured
        move.piece = piece
        self.past_moves.append(move)

    def undo_move(self):
        """Undo the most recent move"""
        move = self.past_moves.pop()

        piece = move.piece
        captured = move.captured
        self.board[move.r_from, move.c_from] = piece
        self.board[move.r_to, move.c_to] = captured

        # undo recorded info needed for special moves.
        self.flags = deepcopy(move.old_flags)

        # special moves
        if move.special == "c":  # castle
            pass
        elif move.special == "e":  # en passant
            pass
        elif move.special in ["q", "n", "b", "r"]:  # promotion
            # demote back to a pawn :p
            # NOTE that processing a promotion forward saves move.piece as the new piece
            if piece.isupper():
                piece = "P"
            else:
                piece = "p"
        self.board[move.r_from, move.c_from] = piece

        self.turn = self.next_turn()

        # update piece set datastructure
        self.piece_set.add((piece, move.r_from, move.c_from))
        if captured != ".":
            self.piece_set.add((captured, move.r_to, move.c_to))
        self.piece_set.remove((piece, move.r_to, move.c_to))
        

    def print_move(self, move: Move):
        """Graphically represents a move"""
        print(move)
        board = deepcopy(self.board)
        board = board.astype("<U20")  # to allow for color strings
        board[move.r_from, move.c_from] = colored(board[move.r_from, move.c_from], "red")
        board[move.r_to, move.c_to] = colored(board[move.r_to, move.c_to], "green")
        out = ""
        for row in board:
            out += " ".join(row) + "\n"
        print(out)

    def __str__(self):
        """Displays the chess board"""
        out = ""
        for row in self.board:
            out += " ".join(row) + "\n"
        return out
