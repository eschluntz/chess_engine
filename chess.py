#!/usr/bin/env python3

import time
from typing import Dict, List, Tuple, Sequence, Set, Callable, TypeVar, Optional
from copy import deepcopy
from termcolor import colored
import functools

import numpy as np

from search import minmax, iterative_deepening
from chessboard import Move, ChessBoard, SIZE, ALL_PIECES


##################
# Chess Evaluation

WIN_SCORE = 1000
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

    # # add base piece values
    # # add piece values
    # for p, v in PIECE_VALUES.items():
    #     piece_table[p] += v

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


def eval_chess_board(board: ChessBoard, params : Dict = {}) -> Tuple[int, bool]:
    """Evaluates a ChessBoard.
    "white" winning -> positive
    "black" winning -> negative.
    return (score, game_over)
    Scores are roughly in "millipawns" pawn / 100.

    params dict:
        piece_tables: bool to include piece_tables in the score
        material: bool to include material in the score
        mobility: bool to include mobility in the score

    Tons of good heuristics here: https://www.chessprogramming.org/Evaluation
    """

    # check if game is over
    end_score, game_over = eval_game_over(board)
    if game_over:
        return end_score, game_over

    score = 0

    # get material score
    if params.get("material", True):
        score += sum(PIECE_VALUES[p] for p, _, _ in board.piece_set)

    # piece table score
    if params.get("piece_table", True):
        piece_table = _get_piece_tables()
        score += sum(piece_table[p][r, c] for p, r, c in board.piece_set)

    # mobility
    if params.get("mobility", False):
        moves = len(board.moves(turn="white")) - len(board.moves(turn="black"))
        score += 10 * moves

    return score, False



##################
# Chess Players
Player = TypeVar('Player', bound=Callable[[ChessBoard, Optional[Dict]], Move])

def human_player(board: ChessBoard) -> Move:
    """Gets CLI input for the next move"""

    def _file_to_column(f: str) -> int:
        """Translates a letter 'file' to column index"""
        c = ord(f.lower()) - ord("a")
        if c < 0 or c > 7:
            raise ValueError("File is out of bounds: {}".format(f))
        return c


    def _rank_to_row(rank: str) -> int:
        """Translates a number Rank to row index"""
        r = 8 - int(rank)

        if r < 0 or r > 7:
            raise ValueError("Rank is out of bounds: {}".format(rank))
        return r

    possible_moves = board.moves()

    move = None
    while move is None:
        uci_str = input("UCI move: ")
        uci_str = uci_str.replace(" ", "")  # strip spaces
        uci_str = uci_str.replace(",", "")  # strip commas

        # parse input
        try:
            c_from = _file_to_column(uci_str[0])
            r_from = _rank_to_row(uci_str[1])
            c_to = _file_to_column(uci_str[2])
            r_to = _rank_to_row(uci_str[3])
        except (ValueError, TypeError):
            print("Invalid move input")
            print("Please enter moves in UCI format: rank_from file_from rank_to file_to")
            print("i.e. e2 e4")
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


def computer_player(board: ChessBoard, params: Dict = {}) -> Move:
    """Wrapper for minmax and eval board options.
    The param dict gets passed down to minmax and the eval_fn.
    Full list of possible params:
        search:
            depth: original max_depth passed to minmax
            time_discount: how much to discount each turn
            explore_ratio: fraction of possible moves to explore
            min_branches: overrides explore_ratio in case there are few branches
        eval:
            piece_tables: bool to include piece_tables in the score
            material: bool to include material in the score
            mobility: bool to include mobility in the score
    """

    depth = params.get("depth", 5)
    _, move = minmax(board, eval_chess_board, depth)
    return move

# def get_all_players() -> Sequence[Player]:
#     """Returns a list of all combinations of different player settings dicts"""
#     all_players = []
#     for branch_ratio in [1.0, .8, .6, .4, .2]:
#         for depth in [3, 4, 5, 6, 7, 8]:
#             # speed control:
#             if depth == 6 and branch_ratio > .6:
#                 continue
#             if depth == 7 and branch_ratio > .4:
#                 continue
#             if depth == 8 and branch_ratio > .2:
#                 continue

#             # # construct a player function
#             # def player(board : ChessBoard) -> Move:
#             #     print((eval_fn, branch_ratio, depth))
#             #     _, move = minmax(board, eval_fn, depth, explore_ratio=branch_ratio, min_branches=10)
#             #     return move

#             # all_players.append(player)

#     return all_players


def play_game(white_params={}, black_params={}, human=None, display=True):
    """Have the computer play itself.
    white_params / black_params: Optional dictionaries passed to those AIs.
    human: optional str 'white' or 'black' to have a human play one of those sides. """
    board = ChessBoard()

    params = {"white": white_params, "black": black_params}

    # show first move if first player is human
    if human == "white":
        print(board)

    over = False
    while not over:
        if display:
            print("-----")
            print("Turn: {}".format(board.turn))

        if board.turn == human:
            move = human_player(board)
        else:
            move = computer_player(board, params[board.turn])
        board.do_move(move)
        score, over = eval_chess_board(board)

        if display:
            board.print_move(move)
            print("Current Score: {}".format(score))

    return score, board


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

    # time_test()
    play_game(human="white")
    # play_game(white_params={"depth":3})
