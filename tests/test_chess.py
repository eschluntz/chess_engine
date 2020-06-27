#!/usr/bin/env python3

from typing import Set
import numpy as np
import copy
from games.chess import (
    ChessBoard,
    SIZE,
    Move,
    _get_material_score,
    _get_mobility_score,
    _get_piece_table_score,
    eval_chess_board,
)
from games.tictactoe import minmax


def test_setup_and_print():
    b = ChessBoard()
    out = str(b)
    expected = ('r n b q k b n r\np p p p p p p p\n'
    '. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n'
    'P P P P P P P P\nR N B Q K B N R\n')
    assert out == expected

def test_empty_pawn_dests():
    b = ChessBoard()
    b.clear_pieces()

    for c in range(8):
        assert {(5,c), (4,c)} == set(b.get_dests_for_piece(6, c, piece="P"))
        assert {(2,c)} == set(b.get_dests_for_piece(3, c, piece="P"))

        # flipped
        assert {(2,c), (3,c)} == set(b.get_dests_for_piece(1, c, piece="p"))
        assert {(4,c)} == set(b.get_dests_for_piece(3, c, piece="p"))


def test_starting_dests():
    b = ChessBoard()

    # test block sliding pieces
    for p in ["p", "r", "b", "q", "k"]:
        for c in range(SIZE):
            # black
            dests = b.get_dests_for_piece(0, c, piece=p)  # overwrite the piece that's there
            assert len(dests) == 0  # no dests available in back row for sliding pieces
            # white
            dests = b.get_dests_for_piece(7, c, piece=p.upper())  # overwrite the piece that's there
            assert len(dests) == 0  # no dests available in back row for sliding pieces

    # test knights
    dests = b.get_dests_for_piece(0, 1, piece="n")
    expected = {(2,0), (2,2)}
    assert set(dests) == expected

    dests = b.get_dests_for_piece(7, 6, piece="N")
    expected = {(5,7), (5,5)}
    assert set(dests) == expected


def test_dest_bugs():
    b = ChessBoard()
    b.board = np.array((
        "r . . . k . . r".split(),
        "p . p p q p b .".split(),
        "b n . . p n p .".split(),
        ". . . P N . . .".split(),
        ". p . . P . . .".split(),
        ". . N . . Q . p".split(),
        "P P P B B P P P".split(),
        "R . . . K . . R".split(),
    ))
    dests = b.get_dests_for_piece(4, 1)
    expected = np.array((
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 1, 1, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ))
    dests_board = b.dests_to_array(dests)
    assert np.all(dests_board == expected), "Test: Pawn bug from perft 1"


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
        dests = b.get_dests_for_piece(r, c, p)
        dests_board = b.dests_to_array(dests)
        assert np.all(dests_board == expected), "Test: {}".format((r, c, p))


def test_empty_king_dests():
    b = ChessBoard()
    b.clear_pieces()

    for k in ["k", "K"]:  # player should not affect dests
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
            dests = b.get_dests_for_piece(r, c, piece=k)
            dests_board = b.dests_to_array(dests)
            print("----- {}".format((k, r, c)))
            print(dests_board)
            assert np.all(dests_board == expected)


def test_empty_rook_dests():
    b = ChessBoard()
    b.clear_pieces()

    for p in ["r", "R"]:  # player should not affect dests
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
            dests = b.get_dests_for_piece(r, c, piece=p)
            dests_board = b.dests_to_array(dests)
            assert np.all(dests_board == expected)


def test_empty_knight_dests():
    b = ChessBoard()
    b.clear_pieces()

    for p in ["n", "N"]:  # player should not affect dests
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
            dests = b.get_dests_for_piece(r, c, piece=p)
            dests_board = b.dests_to_array(dests)
            assert np.all(dests_board == expected)


def test_empty_bishop_dests():
    b = ChessBoard()
    b.clear_pieces()

    for p in ["b", "B"]:  # player should not affect dests
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
            dests = b.get_dests_for_piece(r, c, piece=p)
            dests_board = b.dests_to_array(dests)
            print("----- {}".format((p, r, c)))
            print(dests_board)
            assert np.all(dests_board == expected)


def test_empty_queen_dests():
    b = ChessBoard()
    b.clear_pieces()

    for p in ["q", "Q"]:  # player should not affect dests
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
            dests = b.get_dests_for_piece(r, c, piece=p)
            dests_board = b.dests_to_array(dests)
            assert np.all(dests_board == expected)


def test_find_my_pieces():
    b = ChessBoard()

    # white
    pieces = b.find_my_pieces()
    coords = set([ (r, c) for _, r, c in pieces])
    expected_coords = set([ (r, c) for r in [6, 7] for c in range(SIZE) ])
    assert coords == expected_coords

    # black
    b.turn = "black"
    pieces = b.find_my_pieces()
    coords = set([ (r, c) for _, r, c in pieces])
    expected_coords = set([ (r, c) for r in [0, 1] for c in range(SIZE) ])
    assert coords == expected_coords


def test_moves_simple():
    b = ChessBoard()
    b.clear_pieces()
    b.board[3,3] = "P"

    moves = b.moves()
    print(moves[0])
    assert moves == [Move(3, 3, 2, 3, piece="P")]


def test_perft_moves():
    """https://www.chessprogramming.org/Perft_Results
    Tests the .moves() function by counting the number of moves it returns for various boards"""

    b = ChessBoard()
    b.turn = "white"
    moves = b.moves()
    assert len(moves) == 20
    b.turn = "black"
    moves = b.moves()
    assert len(moves) == 20

    b.clear_pieces()
    b.turn = "white"
    moves = b.moves()
    assert len(moves) == 0
    b.turn = "black"
    moves = b.moves()
    assert len(moves) == 0

    # perft position 2
    b.turn = "white"
    b.board = np.array((
        "r . . . k . . r".split(),
        "p . p p q p b .".split(),
        "b n . . p n p .".split(),
        ". . . P N . . .".split(),
        ". p . . P . . .".split(),
        ". . N . . Q . p".split(),
        "P P P B B P P P".split(),
        "R . . . K . . R".split(),
    ))
    moves = b.moves()
    # for move in moves:
    #     b.print_move(move)
    assert len(moves) == 46  # TODO this should be 48 once castling is supported

    b.turn = "black"
    moves = b.moves()
    # for move in moves:
    #     b.print_move(move)
    assert len(moves) == 41  # TODO this should be 48 once castling is supported


def test_material_score():
    b = ChessBoard()
    assert 0 == _get_material_score(b), "start board test"

    b.clear_pieces()
    assert 0 == _get_material_score(b), "empty board test"

    b.board[0,0] = "K"
    b.board[0,1] = "q"
    b.board[0,2] = "p"
    assert 20000 - 900 - 100 == _get_material_score(b), "a few pieces"


def test_mobility_score():
    b = ChessBoard()
    assert 0 == _get_mobility_score(b), "start board test"

    b.board[0:2,:] = np.full(shape=(2, SIZE), fill_value=".", dtype="<U1")
    assert 20 * 0.1 == _get_mobility_score(b), "one side test"

    b.clear_pieces()
    assert 0 == _get_mobility_score(b), "empty board test"

    b.set_pieces()
    b.board[6,4] = "."  # advance king's pawn
    b.board[5,4] = "P"
    # - 1 for pawn, +9 for bish and queen + 1 for N, +1 K
    assert (30 - 20) * 0.1 == _get_mobility_score(b), "first move"


def test_pice_table_score():
    b = ChessBoard()
    assert 0 == _get_piece_table_score(b), "start board test"

    b.clear_pieces()
    assert 0 == _get_piece_table_score(b), "empty board test"

    b.board[0,0] = "N"  # -50
    b.board[0,1] = "q"  # +10
    assert -40 == _get_piece_table_score(b), "simple piece test"


def test_eval_chess_board():
    b = ChessBoard()
    assert (0.0, False) == eval_chess_board(b), "start board test"

    b.clear_pieces()
    assert (0.0, True) == eval_chess_board(b), "empty board test"

    b.set_pieces()
    b.board[6,4] = "."  # advance king's pawn
    b.board[5,4] = "P"
    score, over = eval_chess_board(b)
    assert over == False
    assert score > 0, "first move should increase score"


def test_minmax():
    b = ChessBoard()
    score, best_move = minmax(b, eval_chess_board, 4)
    print(score)
    b.print_move(best_move)
    # assert False
