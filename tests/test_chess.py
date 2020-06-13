#!/usr/bin/env python3

import numpy as np
import copy
from games.chess import ChessBoard

def test_setup_and_print():
    b = ChessBoard()
    out = str(b)
    expected = ('r n b q k b n r\np p p p p p p p\n'
    '. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n'
    'P P P P P P P P\nR N B Q K B N R\n')
    assert out == expected

def test_get_pawn_moves():
    b = ChessBoard()

    for c in range(8):
        assert {(5,c), (4,c)} == set(b.get_possible_moves("P", 6, c))
        assert {(2,c)} == set(b.get_possible_moves("P", 3, c))

        # flipped
        assert {(2,c), (3,c)} == set(b.get_possible_moves("p", 1, c))
        assert {(4,c)} == set(b.get_possible_moves("p", 3, c))

def test_get_king_moves():
    b = ChessBoard()

    for k in ["k", "K"]:  # player should not affect moves
        start1 = (3, 5)
        expected1 = np.array((
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 1, 1, 0),
            (0, 0, 0, 0, 1, 0, 1, 0),
            (0, 0, 0, 0, 1, 1, 1, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        ))

        start2 = (0, 0)
        expected2 = np.array((
            (0, 1, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        ))

        start3 = (1, 7)
        expected3 = np.array((
            (0, 0, 0, 0, 0, 0, 1, 1),
            (0, 0, 0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 1, 1),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        ))

        tests = [(start1, expected1), (start2, expected2), (start3, expected3)]
        for (r, c), expected in tests:
            moves = b.get_possible_moves(k, r, c)
            moves_board = b.moves_to_array(moves)
            assert np.all(moves_board == expected)

def test_get_rook_moves():
    b = ChessBoard()

    for p in ["r", "R"]:  # player should not affect moves
        start1 = (3, 5)
        expected1 = np.array((
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (1, 1, 1, 1, 1, 0, 1, 1),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
        ))

        start2 = (0, 0)
        expected2 = np.array((
            (0, 1, 1, 1, 1, 1, 1, 1),
            (1, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 0),
        ))

        tests = [(start1, expected1), (start2, expected2)]
        for (r, c), expected in tests:
            moves = b.get_possible_moves(p, r, c)
            moves_board = b.moves_to_array(moves)
            assert np.all(moves_board == expected)

def test_get_knight_moves():
    b = ChessBoard()

    for p in ["n", "N"]:  # player should not affect moves
        start1 = (3, 5)
        expected1 = np.array((
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 1, 0),
            (0, 0, 0, 1, 0, 0, 0, 1),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 1),
            (0, 0, 0, 0, 1, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        ))

        start2 = (0, 0)
        expected2 = np.array((
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        ))

        tests = [(start1, expected1), (start2, expected2)]
        for (r, c), expected in tests:
            moves = b.get_possible_moves(p, r, c)
            moves_board = b.moves_to_array(moves)
            assert np.all(moves_board == expected)

def test_get_bishop_moves():
    b = ChessBoard()

    for p in ["b", "B"]:  # player should not affect moves
        start1 = (3, 5)
        expected1 = np.array((
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 1),
            (0, 0, 0, 0, 1, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 1, 0),
            (0, 0, 0, 1, 0, 0, 0, 1),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0, 0),
        ))

        start2 = (0, 0)
        expected2 = np.array((
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 0, 1),
        ))

        tests = [(start1, expected1), (start2, expected2)]
        for (r, c), expected in tests:
            moves = b.get_possible_moves(p, r, c)
            moves_board = b.moves_to_array(moves)
            print("----- {}".format((p, r, c)))
            print(moves_board)
            assert np.all(moves_board == expected)

def test_get_queen_moves():
    b = ChessBoard()

    for p in ["q", "Q"]:  # player should not affect moves
        start1 = (3, 5)
        expected1 = np.array((
            (0, 0, 1, 0, 0, 1, 0, 0),
            (0, 0, 0, 1, 0, 1, 0, 1),
            (0, 0, 0, 0, 1, 1, 1, 0),
            (1, 1, 1, 1, 1, 0, 1, 1),
            (0, 0, 0, 0, 1, 1, 1, 0),
            (0, 0, 0, 1, 0, 1, 0, 1),
            (0, 0, 1, 0, 0, 1, 0, 0),
            (0, 1, 0, 0, 0, 1, 0, 0),
        ))

        start2 = (0, 0)
        expected2 = np.array((
            (0, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 0, 0, 0, 0, 0, 0),
            (1, 0, 1, 0, 0, 0, 0, 0),
            (1, 0, 0, 1, 0, 0, 0, 0),
            (1, 0, 0, 0, 1, 0, 0, 0),
            (1, 0, 0, 0, 0, 1, 0, 0),
            (1, 0, 0, 0, 0, 0, 1, 0),
            (1, 0, 0, 0, 0, 0, 0, 1),
        ))

        tests = [(start1, expected1), (start2, expected2)]
        for (r, c), expected in tests:
            moves = b.get_possible_moves(p, r, c)
            moves_board = b.moves_to_array(moves)
            assert np.all(moves_board == expected)

test_get_pawn_moves()
