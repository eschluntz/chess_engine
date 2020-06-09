#!/usr/bin/env python3

import numpy as np
import copy

WIN_SCORE = 1000


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
    for team, team_direction in [("x", 1), ("o", -1)]:

        # down or across
        for i in range(size):
            if np.all(board.board[i, :] == team) or np.all(board.board[:, i] == team):
                return team_direction * WIN_SCORE, True

        # diags
        diag1 = np.array([ board.board[i, i] for i in range(size) ])
        diag2 = np.array([ board.board[i, size - i - 1] for i in range(size) ])

        if np.all(diag1 == team) or np.all(diag2 == team):
            return team_direction * WIN_SCORE, True

    # check if it's a tie game
    open_positions = np.sum(board.board == " ")
    tie_game = open_positions == 0
    return 0, tie_game


def minmax(board : TicTacToeBoard, eval_fn, max_depth):
    """Finds the best move using the minmax algorithm.
    board: board representation with this interface:
        [...] = board.moves(player?)
        board2 = board.update(move)
    eval_fn: a function that transforms a board into a score
        score = eval_fn(board, player?)
    max_depth: how many more layers to search.


    TODO: make it prefer victories that are sooner, or defeats that are later.

    returns: (score, move) the expected score down that path.
    """

    # base case - hit max depth
    if max_depth == 0:
        return eval_fn(board)[0], None

    # base case - game over
    score, done = eval_fn(board)
    if done:
        return score, None

    # are we maxing or mining?
    direction = 1.0 if board.turn == "x" else -1.0

    # loop!
    best_move = None
    best_score = -np.inf * direction

    for move in board.moves():
        b2 = board.do_move(move)
        score, _ = minmax(b2, eval_fn, max_depth - 1)

        if score * direction > best_score * direction:
            best_score = score
            best_move = move

    return best_score, best_move



if __name__ == "__main__":
    b = TicTacToeBoard()
    b.board[0,0] = "o"
    # b.board[1,1] = "x"
    import time
    for depth in range(11,15):
        b = TicTacToeBoard()
        b.board[0,0] = "o"
        t0 = time.time()
        score, move = minmax(b, eval_tictactoe, depth)
        t1 = time.time()
        print((depth, t1 - t0))
    # print(move)
    # while True:
    #     print(b)
    #     rc_str = input("row, column: ")
    #     move = eval(rc_str)
    #     b = b.do_move(move)

    #     print(b)
    #     score, move = minmax(b, eval_tictactoe, 6)
    #     b = b.do_move(move)
