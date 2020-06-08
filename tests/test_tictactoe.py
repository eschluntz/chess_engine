#!/usr/bin/env python3

import pytest
import numpy as np

from games.tictactoe import TicTacToeBoard, eval_tictactoe, minmax

# from tictactoe import eval_tictactoe, TicTacToeBoard


def test_display():
    b = TicTacToeBoard()
    b.board[0, 0] = "x"
    b.board[0, 2] = "x"
    b.board[1, 1] = "o"
    b.board[2, 1] = "o"
    print(b)

    assert b.__str__() == "\x1b[4mx| |x\x1b[0m\n\x1b[4m |o| \x1b[0m\n |o| \n"


def test_moves():
    b = TicTacToeBoard()
    b.board[0, 0] = "x"
    b.board[0, 2] = "x"
    b.board[1, 1] = "o"
    b.board[2, 1] = "o"

    moves = b.moves()
    print(moves)
    expected = [(0, 1), (1, 0), (1, 2), (2, 0), (2, 2)]
    assert set(moves) == set(expected)


def test_do_move():
    b = TicTacToeBoard(turn="x")
    b.board[0, 0] = "x"
    b.board[0, 2] = "x"
    b.board[1, 1] = "o"
    b.board[2, 1] = "o"

    b2 = b.do_move((2, 2))
    assert b2.board[2, 2] == "x"
    assert b.board[2, 2] == " "  # didn't change original board
    assert b2.turn == "o"
    assert b.turn == "x"  # didn't change original turn


def test_eval_tictactoe():
    b = TicTacToeBoard()

    tests = []  # [(score, board), ...]
    tests.append((0, np.array(((" ", " ", " "), (" ", " ", " "), (" ", " ", " ")))))
    tests.append((0, np.array((("x", "x", "o"), ("x", "o", "o"), (" ", "x", "x")))))
    tests.append((1000, np.array((("x", "x", "x"), (" ", " ", " "), (" ", " ", " ")))))
    tests.append((-1000, np.array(((" ", " ", " "), ("o", "o", "o"), (" ", " ", " ")))))
    tests.append((1000, np.array((("o", "x", " "), (" ", "x", "o"), (" ", "x", "o")))))
    tests.append((-1000, np.array((("o", " ", " "), (" ", "o", " "), (" ", " ", "o")))))
    tests.append((-1000, np.array(((" ", " ", "o"), (" ", "o", "x"), ("o", " ", " ")))))

    for expected, board in tests:
        b.board = board
        score, _ = eval_tictactoe(b)
        assert score == expected, "board: \n{}".format(b)


def test_minmax():
    # depth = 0
    b = TicTacToeBoard(turn="x")
    b.board = np.array((("x", "x", "x"), (" ", " ", " "), (" ", " ", " ")))
    score, _ = minmax(b, eval_tictactoe, "x", 0)
    assert score == 1000

    # depth = 1, offense
    b.board = np.array((("x", " ", " "), (" ", "x", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, "x", 1)
    assert score == 1000
    assert move == (2, 2)

    # depth = 2, defense
    b.board = np.array((("o", " ", " "), (" ", "o", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, "x", 2)
    assert score == 0
    assert move == (2, 2)

    # can stop a force win
    b.board = np.array((("o", " ", " "), (" ", " ", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, "x", 6)
    assert score == 0
    assert move == (1, 1)

    # can do a force win
    b.board = np.array((("x", " ", " "), ("o", " ", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, "x", 6)
    assert score == 1000
    assert move == (0, 2) or move == (0, 1)  # there are many other force victories
