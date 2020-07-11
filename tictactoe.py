#!/usr/bin/env python3

import time
from typing import Tuple

import numpy as np

from search import minmax

WIN_SCORE = 1000


class TicTacToeBoard(object):
    TURNS = ["x", "o"]

    def __init__(self, turn="x"):
        # 3x3 array of chars. ["", "x", "o"]
        self.board = np.full(shape=(3, 3), fill_value=" ", dtype="<U1")
        self.past_moves = []  # used to pop off and undo moves
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
        """Updates the game board object with the new move taken
        and the next player's turn set.
        move: (r, c) tuple of ints."""

        self.board[move] = self.turn
        self.turn = self.next_turn()
        self.past_moves.append(move)

    def undo_move(self):
        """Pops the last move off the stack"""
        last_move = self.past_moves.pop()
        self.board[last_move] = " "
        self.turn = self.next_turn()


def eval_tictactoe(board) -> Tuple[int, bool]:
    """Evaluates a tictactoe board.
    "x" winning -> positive
    "o" winning -> negative.
    game over -> +/- 1000.
    return (score, game_over)
    """
    for team, team_direction in [("x", 1), ("o", -1)]:

        # I'm not clever enough to come up wth this myself
        # https://stackoverflow.com/questions/46802651/check-for-winner-in-tic-tac-toe-numpy-python
        mask = board.board == team
        win = (
            mask.all(0).any()
            or mask.all(1).any()  # columns
            or np.diag(mask).all()  # rows
            or np.diag(mask[:, ::-1]).all()  # down right  # up right
        )

        if win:
            return team_direction * WIN_SCORE, True

    # check if it's a tie game
    open_positions = np.sum(board.board == " ")
    tie_game = open_positions == 0
    return 0, tie_game


def play_game():
    """Play a friendly game of tic tac toe against the AI"""
    b = TicTacToeBoard()
    while True:
        rc_str = input("row, column: ")
        move = eval(rc_str)  # careful with your input :p
        b.do_move(move)
        print(b)
        score, over = eval_tictactoe(b)
        if over:
            if score == WIN_SCORE:
                print("You win!")
                return WIN_SCORE
            else:
                print("Tie!")
                return 0

        score, move = minmax(b, eval_tictactoe, 9)
        b.do_move(move)
        print(b)
        score, over = eval_tictactoe(b)
        if over:
            if score == -WIN_SCORE:
                print("You lose!")
                return -WIN_SCORE
            else:
                print("Tie!")
                return 0


def time_game():
    """Generate a list of execution times for different search depths"""
    for depth in range(15):
        b = TicTacToeBoard()
        b.board[0, 0] = "o"

        t0 = time.time()
        minmax(b, eval_tictactoe, depth)
        t1 = time.time()
        print((depth, t1 - t0))


if __name__ == "__main__":
    play_game()
