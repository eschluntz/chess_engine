#!/usr/bin/env python3

import pytest
from games.tictactoe import TicTacToeBoard


def test_display():
    b = TicTacToeBoard()
    b.board[0,0] = "x"
    b.board[0,2] = "x"
    b.board[1,1] = "o"
    b.board[2,1] = "o"
    print(b)

    assert b.__str__() == '\x1b[4mx| |x\x1b[0m\n\x1b[4m |o| \x1b[0m\n |o| \n'

def test_moves():
    b = TicTacToeBoard()
    b.board[0,0] = "x"
    b.board[0,2] = "x"
    b.board[1,1] = "o"
    b.board[2,1] = "o"

    moves = b.moves()
    print(moves)
    expected = [(0, 1), (1, 0), (1, 2), (2, 0), (2, 2)]
    assert set(moves) == set(expected)

def test_do_move():
    b = TicTacToeBoard(turn="x")
    b.board[0,0] = "x"
    b.board[0,2] = "x"
    b.board[1,1] = "o"
    b.board[2,1] = "o"

    b2 = b.do_move((2,2))
    assert b2.board[2,2] == "x"
    assert b.board[2,2] == " "  # didn't change original board
    assert b2.turn == "o"
    assert b.turn == "x"  # didn't change original turn
