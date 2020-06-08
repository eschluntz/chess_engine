#!/usr/bin/env python3

import numpy as np
import copy


class TicTacToeBoard(object):
    TURNS = ["x", "o"]

    def __init__(self, turn="x"):
        # 3x3 array of chars. ["", "x", "o"]
        self.board = np.full(shape=(3, 3), fill_value=" ", dtype="<U1")
        self.turn = turn  # "x" or "o"

    def __str__(self):
        output = ""
        for i, row in enumerate(self.board):

            # create underlines to draw the board more compactly
            if i != len(self.board) - 1:
                prefix = "\033[4m"
                postfix = "\033[0m"
            else:
                prefix, postfix = "", ""

            row = prefix + "|".join(row) + postfix + "\n"
            output += row
        return output

    def moves(self):
        """Returns list of available moves, as (r,c) tuples.
        Note: this is actually the same no matter who's turn it is"""
        rs, cs = np.where(self.board == " ")  # get empty spaces
        return list(zip(rs, cs))  # turn two lists into list of tuples of spots

    def next_turn(self):
        """Returns the "x" or "o", whichever is not our current turn"""
        if self.turn == "x":
            return "o"
        else:
            return "x"

    def do_move(self, move):
        """Creates a new game board object with the new move taken
        and the next player's turn set.
        move: (r, c) tuple of ints.
        returns: TicTacToe object"""

        b = copy.deepcopy(self)  # lol does this work?
        assert b.board[move] == " ", "Invalid move!"
        b.board[move] = self.turn
        b.turn = self.next_turn()
        return b


def eval_tictactoe(board):
    """Evaluates a tictactoe board.
    "x" winning -> positive
    "o" winning -> negative.
    game over -> +/- 1000.
    return (score, game_over)
    """
    size = len(board.board)
    win_score = 1000
    for team, team_direction in [("x", 1), ("o", -1)]:

        # down or across
        for i in range(size):
            if np.all(board.board[i, :] == team) or np.all(board.board[:, i] == team):
                return team_direction * win_score, True

        # diags
        diag1 = np.array([ board.board[i, i] for i in range(size) ])
        diag2 = np.array([ board.board[i, size - i - 1] for i in range(size) ])

        if np.all(diag1 == team) or np.all(diag2 == team):
            return team_direction * win_score, True

    # todo add more intermediate rewards to help test alpha beta pruning
    return 0, False


def minmax(board : TicTacToeBoard, eval_fn, current_player, max_depth):
    """Finds the best move using the minmax algorithm.
    board: board representation with this interface:
        [...] = board.moves(player?)
        board2 = board.update(move)
    eval_fn: a function that transforms a board into a score
        score = eval_fn(board, player?)
    player?: how should I represent who the current player is?
    max_depth: how many more layers to search.

    returns: (score, move) the expected score down that path.
    """

    # base case
    if max_depth == 0:
        return eval_fn(board)[0], None

    # someone won
    # TODO: make it prefer victories that are sooner, or defeats that are later.
    score, done = eval_fn(board)
    if done:
        return score, None

    # ties
    moves = board.moves()
    if len(moves) == 0:
        return eval_fn(board)[0], None

    # loop!
    best_move = None
    if board.turn == current_player:  # max
        best_score = -99999
    else:
        best_score = 99999

    for move in moves:
        b2 = board.do_move(move)
        score, _ = minmax(b2, eval_fn, current_player, max_depth - 1)

        if board.turn == current_player:  # max
            best = score > best_score
        else:  # min
            best = score < best_score

        if best:
            best_score = score
            best_move = move

    return best_score, best_move



if __name__ == "__main__":
    b = TicTacToeBoard()
    # b.board[0,0] = "x"
    # b.board[1,1] = "x"
    # score, move = minmax(b, eval_tictactoe, "x", 6)
    # print(move)
    while True:
        print(b)
        rc_str = input("row, column: ")
        move = eval(rc_str)
        b = b.do_move(move)

        print(b)
        score, move = minmax(b, eval_tictactoe, "x", 6)
        b = b.do_move(move)
