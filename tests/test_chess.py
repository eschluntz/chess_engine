#!/usr/bin/env python3

from typing import Set
import numpy as np
import copy
from games.chess import ChessBoard, SIZE

def test_setup_and_print():
    b = ChessBoard()
    out = str(b)
    expected = ('r n b q k b n r\np p p p p p p p\n'
    '. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n'
    'P P P P P P P P\nR N B Q K B N R\n')
    assert out == expected

def test_empty_pawn_moves():
    b = ChessBoard()
    b.clear_pieces()

    for c in range(8):
        assert {(5,c), (4,c)} == set(b.get_possible_moves(6, c, piece="P"))
        assert {(2,c)} == set(b.get_possible_moves(3, c, piece="P"))

        # flipped
        assert {(2,c), (3,c)} == set(b.get_possible_moves(1, c, piece="p"))
        assert {(4,c)} == set(b.get_possible_moves(3, c, piece="p"))


def test_starting_moves():
    b = ChessBoard()

    # test block sliding pieces
    for p in ["p", "r", "b", "q", "k"]:
        for c in range(SIZE):
            # black
            moves = b.get_possible_moves(0, c, piece=p)  # overwrite the piece that's there
            assert len(moves) == 0  # no moves available in back row for sliding pieces
            # white
            moves = b.get_possible_moves(7, c, piece=p.upper())  # overwrite the piece that's there
            assert len(moves) == 0  # no moves available in back row for sliding pieces

    # test knights
    moves = b.get_possible_moves(0, 1, piece="n")
    expected = {(2,0), (2,2)}
    assert set(moves) == expected

    moves = b.get_possible_moves(7, 6, piece="N")
    expected = {(5,7), (5,5)}
    assert set(moves) == expected


def test_surrounded():
    b = ChessBoard()
    b.board = np.array((
        ". . . . . . . .".split(),
        ". . p p . . . .".split(),
        ". . p . p p . .".split(),
        ". . p . Q p . .".split(),
        ". . p . . p . .".split(),
        ". . p p p . . .".split(),
        ". . . . . . . .".split(),
        ". . . . . . . .".split(),
    ))

    set1 = (3, 4, "P")
    expected1 = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ))

    set2 = (3, 4, "R")
    expected2 = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ))

    set3 = (3, 4, "N")
    expected3 = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 1, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 1, 0),
        (0, 0, 0, 1, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ))

    set4 = (3, 4, "B")
    expected4 = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 0, 1, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ))

    set5 = (3, 4, "Q")
    expected5 = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 0, 0),
        (0, 0, 1, 1, 0, 1, 0, 0),
        (0, 0, 0, 1, 1, 1, 0, 0),
        (0, 0, 1, 0, 1, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ))

    set6 = (3, 4, "K")
    expected6 = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 0, 0),
        (0, 0, 0, 1, 0, 1, 0, 0),
        (0, 0, 0, 1, 1, 1, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ))

    tests = [(set1, expected1), (set2, expected2), (set3, expected3),
        (set4, expected4), (set5, expected5), (set6, expected6), ]
    for (r, c, p), expected in tests:
        moves = b.get_possible_moves(r, c, p)
        moves_board = b.moves_to_array(moves)
        assert np.all(moves_board == expected), "Test: {}".format((r, c, p))



def test_empty_king_moves():
    b = ChessBoard()
    b.clear_pieces()

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
            moves = b.get_possible_moves(r, c, piece=k)
            moves_board = b.moves_to_array(moves)
            print("----- {}".format((k, r, c)))
            print(moves_board)
            assert np.all(moves_board == expected)

def test_empty_rook_moves():
    b = ChessBoard()
    b.clear_pieces()

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
            moves = b.get_possible_moves(r, c, piece=p)
            moves_board = b.moves_to_array(moves)
            assert np.all(moves_board == expected)

def test_empty_knight_moves():
    b = ChessBoard()
    b.clear_pieces()

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
            moves = b.get_possible_moves(r, c, piece=p)
            moves_board = b.moves_to_array(moves)
            assert np.all(moves_board == expected)

def test_empty_bishop_moves():
    b = ChessBoard()
    b.clear_pieces()

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
            moves = b.get_possible_moves(r, c, piece=p)
            moves_board = b.moves_to_array(moves)
            print("----- {}".format((p, r, c)))
            print(moves_board)
            assert np.all(moves_board == expected)

def test_empty_queen_moves():
    b = ChessBoard()
    b.clear_pieces()

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
            moves = b.get_possible_moves(r, c, piece=p)
            moves_board = b.moves_to_array(moves)
            assert np.all(moves_board == expected)
