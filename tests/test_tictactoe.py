#!/usr/bin/env python3

import numpy as np
import copy
from games.tictactoe import TicTacToeBoard, eval_tictactoe, minmax, WIN_SCORE, play_game
from games import tictactoe

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

    b.do_move((2, 2))
    assert b.board[2, 2] == "x"
    assert b.turn == "o"


def test_eval_tictactoe():
    b = TicTacToeBoard()

    # [(score, done, board), ...]
    tests = [
        (0,             False,  np.array(((" ", " ", " "), (" ", " ", " "), (" ", " ", " ")))),
        (WIN_SCORE,     True,   np.array((("x", "x", "x"), (" ", " ", " "), (" ", " ", " ")))),
        (-WIN_SCORE,    True,   np.array(((" ", " ", " "), ("o", "o", "o"), (" ", " ", " ")))),
        (WIN_SCORE,     True,   np.array((("o", "x", " "), (" ", "x", "o"), (" ", "x", "o")))),
        (-WIN_SCORE,    True,   np.array((("o", " ", " "), (" ", "o", " "), (" ", " ", "o")))),
        (-WIN_SCORE,    True,   np.array(((" ", " ", "o"), (" ", "o", "x"), ("o", " ", " ")))),
        (0,             True,   np.array((("o", "x", "x"), ("x", "o", "o"), ("x", "o", "x")))),
    ]

    for e_score, e_done, board in tests:
        b.board = board
        score, done = eval_tictactoe(b)
        assert score == e_score, "board: \n{}".format(b)
        assert done == e_done, "board: \n{}".format(b)


def test_minmax():
    # depth = 0
    b = TicTacToeBoard(turn="x")
    b.board = np.array((("x", "x", "x"), (" ", " ", " "), (" ", " ", " ")))
    score, _ = minmax(b, eval_tictactoe, 0)
    assert score == WIN_SCORE

    b = TicTacToeBoard(turn="o")
    b.board = np.array((("o", " ", " "), (" ", "o", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, 1)
    assert score == -WIN_SCORE
    assert move == (2,2)

    # depth = 1, offense
    b = TicTacToeBoard(turn="x")
    b.board = np.array((("x", " ", " "), (" ", "x", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, 1)
    assert score == WIN_SCORE
    assert move == (2, 2)

    # depth = 2, defense
    b = TicTacToeBoard(turn="x")
    b.board = np.array((("o", " ", " "), (" ", "o", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, 2)
    assert score == 0
    assert move == (2, 2)

def test_minmax_deep():
    b = TicTacToeBoard(turn="x")
    # can stop a force win
    b.board = np.array((("o", " ", " "), (" ", " ", " "), (" ", " ", " ")))
    score, move = minmax(b, eval_tictactoe, 6)
    assert score == 0
    assert move == (1, 1)

    # can do a force win
    b = TicTacToeBoard(turn="o")
    start = np.array((("o", " ", " "), ("x", " ", " "), (" ", " ", " ")))
    b.board = copy.deepcopy(start)
    score, move = minmax(b, eval_tictactoe, 6)
    assert score == -WIN_SCORE
    assert move == (0, 2) or move == (0, 1)  # there are many other force victories

    # check board is unchanged after call to eval
    assert np.all(b.board == start)
    assert b.past_moves == []

def test_play_game():
    tictactoe.input = lambda text : "0, 0"  # gross, but this is barely worth testing
    assert play_game() == -WIN_SCORE
