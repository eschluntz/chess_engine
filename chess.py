#!/usr/bin/env python3

import time
from typing import Dict, List, Tuple, Sequence, Set, Callable, TypeVar
from copy import deepcopy
from termcolor import colored
import functools

import numpy as np

from search import minmax, iterative_deepening

WIN_SCORE = 1000
SIZE = 8
WHITE_PIECES = ["P", "R", "N", "B", "K", "Q"]
BLACK_PIECES = [p.lower() for p in WHITE_PIECES]
ALL_PIECES = WHITE_PIECES + BLACK_PIECES


def inbound(r, c):
    """Checks if coords are in the board"""
    return 0 <= r < SIZE and 0 <= c < SIZE


class Move(object):
    """Class to represent a move.
    TODO: handle castles and promotions"""

    def __init__(self, r_from: int, c_from: int, r_to: int, c_to: int, piece=None, captured=None) -> None:
        self.r_from = r_from
        self.c_from = c_from
        self.r_to = r_to
        self.c_to = c_to

        # optional and are filled in by the board when doing a move
        self.piece = piece
        self.captured = captured

        self.special = False

    def __str__(self) -> str:
        if self.captured is None:
            capt = ""
        else:
            capt = " x {}".format(self.captured)
        return "Move {} ({}, {}) -> ({}, {}) {}".format(
            self.piece, self.r_from, self.c_from, self.r_to, self.c_to, capt
        )

    def __eq__(self, other) -> bool:
        """Note: only compares to and from positions, not piece or capture"""
        return (self.r_from, self.c_from, self.r_to, self.c_to) == (other.r_from, other.c_from, other.r_to, other.c_to)


class ChessBoard(object):
    TURNS = ["white", "black"]

    def __init__(self):
        self.board = np.full(shape=(SIZE, SIZE), fill_value=".", dtype="<U1")
        self.piece_set: Set[Tuple[str, int, int]] = set()
        self.past_moves: Sequence[Tuple[Move, str]] = []
        self.turn = "white"
        # TODO: store info to assess whether castling is still allowed, and en passant is still allowed
        self.set_pieces()

    def next_turn(self) -> str:
        """Returns "white" or "black whichever is not our current turn"""
        if self.turn == "white":
            return "black"
        else:
            return "white"

    def _reset_piece_set(self) -> None:
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
        self._reset_piece_set()

    def set_pieces(self) -> None:
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

        self._reset_piece_set()

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

        # pieces = []

        # for r in range(SIZE):
        #     for c in range(SIZE):
        #         if my_piece(self.board[r, c]):
        #             pieces.append((self.board[r, c], r, c))

        # return pieces

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

    def _get_pawn_dests(self, r: int, c: int, player: str):
        """pawns are actually the most complex pieces on the board! Their moves:
        1. are asymmetric, 2. depend on their position, 3. depends on opponents 4. moves do not equal captures
        TODO: implement promoting"""

        if player == "black":
            my_piece = str.islower
            other_piece = str.isupper
        else:
            my_piece = str.isupper
            other_piece = str.islower

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

    def get_dests_for_piece(self, r: int, c: int, piece=None) -> Sequence[Tuple[int, int]]:
        """Given a particular piece, generates all possible destinatinos for it to move to.
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

        # generate possible moves
        all_moves = []
        for piece, r_from, c_from in pieces:
            for r_to, c_to in self.get_dests_for_piece(r_from, c_from):
                move = Move(r_from, c_from, r_to, c_to, piece=piece)
                all_moves.append(move)

        return all_moves

    def do_move(self, move: Move):
        """Do a move on the chessboard"""
        piece = self.board[move.r_from, move.c_from]
        captured = self.board[move.r_to, move.c_to]
        self.board[move.r_from, move.c_from] = "."
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


_PIECE_TABLE = None  # cache

PIECE_VALUES = {
    "K": 20000,
    "k": -20000,
    "Q": 900,
    "q": -900,
    "R": 500,
    "r": -500,
    "B": 330,
    "b": -330,
    "N": 320,
    "n": -320,
    "P": 100,
    "p": -100,
    ".": 0,
}


def _get_piece_tables() -> Dict:
    """Returns piece tables for the eval function.
    source: https://www.chessprogramming.org/Simplified_Evaluation_Function"""
    global _PIECE_TABLE
    if _PIECE_TABLE is not None:
        return _PIECE_TABLE

    piece_table = {}
    piece_table["P"] = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (50, 50, 50, 50, 50, 50, 50, 50),
            (10, 10, 20, 30, 30, 20, 10, 10),
            (5, 5, 10, 25, 25, 10, 5, 5),
            (0, 0, 0, 20, 20, 0, 0, 0),
            (5, -5, -10, 0, 0, -10, -5, 5),
            (5, 10, 10, -20, -20, 10, 10, 5),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )
    piece_table["N"] = np.array(
        (
            (-50, -40, -30, -30, -30, -30, -40, -50),
            (-40, -20, 0, 0, 0, 0, -20, -40),
            (-30, 0, 10, 15, 15, 10, 0, -30),
            (-30, 5, 15, 20, 20, 15, 5, -30),
            (-30, 0, 15, 20, 20, 15, 0, -30),
            (-30, 5, 10, 15, 15, 10, 5, -30),
            (-40, -20, 0, 5, 5, 0, -20, -40),
            (-50, -40, -30, -30, -30, -30, -40, -50),
        )
    )
    piece_table["B"] = np.array(
        (
            (-20, -10, -10, -10, -10, -10, -10, -20),
            (-10, 0, 0, 0, 0, 0, 0, -10),
            (-10, 0, 5, 10, 10, 5, 0, -10),
            (-10, 5, 5, 10, 10, 5, 5, -10),
            (-10, 0, 10, 10, 10, 10, 0, -10),
            (-10, 10, 10, 10, 10, 10, 10, -10),
            (-10, 5, 0, 0, 0, 0, 5, -10),
            (-20, -10, -10, -10, -10, -10, -10, -20),
        )
    )
    piece_table["R"] = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (5, 10, 10, 10, 10, 10, 10, 5),
            (-5, 0, 0, 0, 0, 0, 0, -5),
            (-5, 0, 0, 0, 0, 0, 0, -5),
            (-5, 0, 0, 0, 0, 0, 0, -5),
            (-5, 0, 0, 0, 0, 0, 0, -5),
            (-5, 0, 0, 0, 0, 0, 0, -5),
            (0, 0, 0, 5, 5, 0, 0, 0),
        )
    )
    piece_table["Q"] = np.array(
        (
            (-20, -10, -10, -5, -5, -10, -10, -20),
            (-10, 0, 0, 0, 0, 0, 0, -10),
            (-10, 0, 5, 5, 5, 5, 0, -10),
            (-5, 0, 5, 5, 5, 5, 0, -5),
            (0, 0, 5, 5, 5, 5, 0, -5),
            (-10, 5, 5, 5, 5, 5, 0, -10),
            (-10, 0, 5, 0, 0, 0, 0, -10),
            (-20, -10, -10, -5, -5, -10, -10, -20),
        )
    )
    piece_table["K"] = np.array(
        (
            (-30, -40, -40, -50, -50, -40, -40, -30),
            (-30, -40, -40, -50, -50, -40, -40, -30),
            (-30, -40, -40, -50, -50, -40, -40, -30),
            (-30, -40, -40, -50, -50, -40, -40, -30),
            (-20, -30, -30, -40, -40, -30, -30, -20),
            (-10, -20, -20, -20, -20, -20, -20, -10),
            (20, 20, 0, 0, 0, 0, 20, 20),
            (20, 30, 10, 0, 0, 10, 30, 20),
        )
    )
    piece_table["."] = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    # fill in black piece table. Flip and negate values
    for p in list(piece_table.keys()):  # cast to list to allow iterating over original keys
        piece_table[p.lower()] = -np.flip(piece_table[p])

    # add base piece values
    # add piece values
    for p, v in PIECE_VALUES.items():
        piece_table[p] += v

    _PIECE_TABLE = piece_table
    return piece_table


def eval_game_over(board: ChessBoard) -> Tuple[int, bool]:
    """Returns (score, game_over).
    white win -> positive."""

    # look for 3 fold repetition tie
    if len(board.past_moves) >= 10:
        if board.past_moves[-1] == board.past_moves[-5] == board.past_moves[-9] and \
            board.past_moves[-2] == board.past_moves[-6] == board.past_moves[-10]:
            return 0, True

    # look for win condition
    if "k" not in board.board:
        return WIN_SCORE, True
    if "K" not in board.board:
        return -WIN_SCORE, True

    # look for max turn limit
    if len(board.past_moves) > 200:
        return 0, True

    return 0, False


def eval_chess_board(board: ChessBoard) -> Tuple[int, bool]:
    """Evaluates a ChessBoard.
    "white" winning -> positive
    "black" winning -> negative.
    game over -> +/- 200,000.
    return (score, game_over)
    Scores are roughly in "millipawns" pawn / 100

    Tons of good heuristics here: https://www.chessprogramming.org/Evaluation
    """

    # check if game is over
    end_score, game_over = eval_game_over(board)
    if game_over:
        return end_score, game_over

    # piece table score + base values
    piece_table = _get_piece_tables()
    score = sum(piece_table[p][r, c] for p, r, c in board.piece_set)

    return score, False


def eval_chess_material_only(board: ChessBoard) -> Tuple[int, bool]:
    """Evaluate a chess board only based on material value"""

    # check if game is over
    end_score, game_over = eval_game_over(board)
    if game_over:
        return end_score, game_over

    # just sum material value
    score = sum(PIECE_VALUES[p] for p, _, _ in board.piece_set)


def eval_chess_mobility(board: ChessBoard) -> Tuple[int, bool]:
    """Evaluate a chess board based on material and mobility value"""

    # check if game is over
    end_score, game_over = eval_game_over(board)
    if game_over:
        return end_score, game_over

    # just sum material value
    score = sum(PIECE_VALUES[p] for p, _, _ in board.piece_set)

    # get number of moves
    moves = len(board.moves(turn="white")) - len(board.moves(turn="black"))
    score += 10 * moves

    return score, False


def file_to_column(f: str) -> int:
    """Translates a letter 'file' to column index"""
    c = ord(f.lower()) - ord("a")
    if c < 0 or c > 7:
        raise ValueError("File is out of bounds: {}".format(f))
    return c


def rank_to_row(rank: str) -> int:
    """Translates a number Rank to row index"""
    r = 8 - int(rank)

    if r < 0 or r > 7:
        raise ValueError("Rank is out of bounds: {}".format(rank))
    return r

##################
# Chess Players
Player = TypeVar('Player', bound=Callable[[ChessBoard], Move])

def player_human(board: ChessBoard) -> Move:
    """Gets CLI input for the next move.
    allow_illegal: don't check that move was a legal one"""
    print("Turn: {}".format(board.turn))

    possible_moves = board.moves()

    move = None
    while move is None:
        uci_str = input("UCI move: ")
        uci_str = uci_str.replace(" ", "")  # strip spaces
        uci_str = uci_str.replace(",", "")  # strip commas

        # parse input
        try:
            c_from = file_to_column(uci_str[0])
            r_from = rank_to_row(uci_str[1])
            c_to = file_to_column(uci_str[2])
            r_to = rank_to_row(uci_str[3])
        except (ValueError, TypeError):
            print("Invalid move input")
            continue

        # see if engine agrees it was a valid move
        piece = board.board[r_from, c_from]
        captured = board.board[r_to, c_to]
        move = Move(r_from, c_from, r_to, c_to, piece, captured)
        if move not in possible_moves:
            print("Illegal move!")
            board.print_move(move)
            move = None
            continue

    return move


def player_depth_5(board: ChessBoard) -> Move:
    """Minmax depth 4, standard heuristics"""
    score, move = minmax(board, eval_chess_board, 5)
    print("expected score: {}".format(score))
    return move

def player_depth_4(board: ChessBoard) -> Move:
    """Minmax depth 4, standard heuristics"""
    score, move = minmax(board, eval_chess_board, 4)
    print("expected score: {}".format(score))
    return move

def player_depth_3(board: ChessBoard) -> Move:
    """Minmax depth 4, standard heuristics"""
    score, move = minmax(board, eval_chess_board, 3)
    print("expected score: {}".format(score))
    return move

def player_depth_2(board: ChessBoard) -> Move:
    """Minmax depth 4, standard heuristics"""
    score, move = minmax(board, eval_chess_board, 2)
    print("expected score: {}".format(score))
    return move

def player_depth_1(board: ChessBoard) -> Move:
    """Minmax depth 4, standard heuristics"""
    _, move = minmax(board, eval_chess_board, 1)
    return move


def play(p_white : Player, p_black : Player):
    """Play a game of chess. both player functions should have (Board)->Move signature"""
    b = ChessBoard()

    players = {"white": p_white, "black": p_black}
    print(b)
    over = False
    while not over:
        print("-----")
        print("Turn: {}".format(b.turn))
        move = players[b.turn](b)
        b.do_move(move)
        b.print_move(move)
        score, over = eval_chess_board(b)
        print("Current Score: {}".format(score))


def time_test():
    """Time a move search"""
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
    b._reset_piece_set()
    import time

    t0 = time.time()
    _, move = minmax(b, eval_chess_board, 4)
    t1 = time.time()
    print(t1 - t0)


if __name__ == "__main__":

    time_test()
    # play(p_white=player_depth_5, p_black=player_depth_5)
    # draw()
