#!/usr/bin/env python3

from typing import Set
import copy

import numpy as np

from chessboard import (
    B_CASTLE_RIGHT,
    EN_PASSANT_SPOT,
    ChessBoard,
    SIZE,
    Move,
    W_CASTLE_LEFT,
    W_CASTLE_RIGHT,
    B_CASTLE_LEFT,
    B_CASTLE_RIGHT,
)
from chess import eval_chess_board, play_game
from search import minmax



def assert_row(b, row, expected):
    actual = "".join(b.board[row])
    assert(actual == expected)


def test_setup_and_print():
    b = ChessBoard()
    out = str(b)
    expected = (
        "r n b q k b n r\np p p p p p p p\n"
        ". . . . . . . .\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n"
        "P P P P P P P P\nR N B Q K B N R\n"
    )
    assert out == expected


def test_empty_pawn_dests():
    b = ChessBoard()
    b.clear_pieces()

    for c in range(8):
        assert {(5, c), (4, c)} == set(b.get_dests_for_piece(6, c, piece="P"))
        assert {(2, c)} == set(b.get_dests_for_piece(3, c, piece="P"))

        # flipped
        assert {(2, c), (3, c)} == set(b.get_dests_for_piece(1, c, piece="p"))
        assert {(4, c)} == set(b.get_dests_for_piece(3, c, piece="p"))


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
    expected = {(2, 0), (2, 2)}
    assert set(dests) == expected

    dests = b.get_dests_for_piece(7, 6, piece="N")
    expected = {(5, 7), (5, 5)}
    assert set(dests) == expected


def test_dest_bugs():
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
    b._sync_board_to_piece_set()
    dests = b.get_dests_for_piece(4, 1)
    expected = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )
    dests_board = b.dests_to_array(dests)
    assert np.all(dests_board == expected), "Test: Pawn bug from perft 1"


def test_surrounded():
    b = ChessBoard()
    b.board = np.array(
        (
            ". . . . . . . .".split(),
            ". . p p . . . .".split(),
            ". . p . p p . .".split(),
            ". . p . Q p . .".split(),
            ". . p . . p . .".split(),
            ". . p p p . . .".split(),
            ". . . . . . . .".split(),
            ". . . . . . . .".split(),
        )
    )

    set1 = (3, 4, "P")
    expected1 = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    set2 = (3, 4, "R")
    expected2 = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 0, 0),
            (0, 0, 1, 1, 0, 1, 0, 0),
            (0, 0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    set3 = (3, 4, "N")
    expected3 = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 0, 0),
            (0, 0, 1, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 1, 0),
            (0, 0, 0, 1, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    set4 = (3, 4, "B")
    expected4 = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    set5 = (3, 4, "Q")
    expected5 = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 1, 1, 0, 0),
            (0, 0, 1, 1, 0, 1, 0, 0),
            (0, 0, 0, 1, 1, 1, 0, 0),
            (0, 0, 1, 0, 1, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    set6 = (3, 4, "K")
    expected6 = np.array(
        (
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 1, 1, 0, 0),
            (0, 0, 0, 1, 0, 1, 0, 0),
            (0, 0, 0, 1, 1, 1, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
    )

    tests = [
        (set1, expected1),
        (set2, expected2),
        (set3, expected3),
        (set4, expected4),
        (set5, expected5),
        (set6, expected6),
    ]
    for (r, c, p), expected in tests:
        dests = b.get_dests_for_piece(r, c, p)
        dests_board = b.dests_to_array(dests)
        assert np.all(dests_board == expected), "Test: {}".format((r, c, p))


def test_empty_king_dests():
    b = ChessBoard()
    b.clear_pieces()

    for k in ["k", "K"]:  # player should not affect dests
        start1 = (3, 5)
        expected1 = np.array(
            (
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 1, 1, 1, 0),
                (0, 0, 0, 0, 1, 0, 1, 0),
                (0, 0, 0, 0, 1, 1, 1, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
            )
        )

        start2 = (0, 0)
        expected2 = np.array(
            (
                (0, 1, 0, 0, 0, 0, 0, 0),
                (1, 1, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
            )
        )

        start3 = (1, 7)
        expected3 = np.array(
            (
                (0, 0, 0, 0, 0, 0, 1, 1),
                (0, 0, 0, 0, 0, 0, 1, 0),
                (0, 0, 0, 0, 0, 0, 1, 1),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
            )
        )

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
        expected1 = np.array(
            (
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (1, 1, 1, 1, 1, 0, 1, 1),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
            )
        )

        start2 = (0, 0)
        expected2 = np.array(
            (
                (0, 1, 1, 1, 1, 1, 1, 1),
                (1, 0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0, 0, 0),
            )
        )

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
        expected1 = np.array(
            (
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 1, 0, 1, 0),
                (0, 0, 0, 1, 0, 0, 0, 1),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0, 0, 1),
                (0, 0, 0, 0, 1, 0, 1, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
            )
        )

        start2 = (0, 0)
        expected2 = np.array(
            (
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
            )
        )

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
        expected1 = np.array(
            (
                (0, 0, 1, 0, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0, 0, 1),
                (0, 0, 0, 0, 1, 0, 1, 0),
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 1, 0, 1, 0),
                (0, 0, 0, 1, 0, 0, 0, 1),
                (0, 0, 1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0, 0, 0),
            )
        )

        start2 = (0, 0)
        expected2 = np.array(
            (
                (0, 0, 0, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0, 0, 0),
                (0, 0, 0, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 1, 0, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 0, 1, 0),
                (0, 0, 0, 0, 0, 0, 0, 1),
            )
        )

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
        expected1 = np.array(
            (
                (0, 0, 1, 0, 0, 1, 0, 0),
                (0, 0, 0, 1, 0, 1, 0, 1),
                (0, 0, 0, 0, 1, 1, 1, 0),
                (1, 1, 1, 1, 1, 0, 1, 1),
                (0, 0, 0, 0, 1, 1, 1, 0),
                (0, 0, 0, 1, 0, 1, 0, 1),
                (0, 0, 1, 0, 0, 1, 0, 0),
                (0, 1, 0, 0, 0, 1, 0, 0),
            )
        )

        start2 = (0, 0)
        expected2 = np.array(
            (
                (0, 1, 1, 1, 1, 1, 1, 1),
                (1, 1, 0, 0, 0, 0, 0, 0),
                (1, 0, 1, 0, 0, 0, 0, 0),
                (1, 0, 0, 1, 0, 0, 0, 0),
                (1, 0, 0, 0, 1, 0, 0, 0),
                (1, 0, 0, 0, 0, 1, 0, 0),
                (1, 0, 0, 0, 0, 0, 1, 0),
                (1, 0, 0, 0, 0, 0, 0, 1),
            )
        )

        tests = [(start1, expected1), (start2, expected2)]
        for (r, c), expected in tests:
            dests = b.get_dests_for_piece(r, c, piece=p)
            dests_board = b.dests_to_array(dests)
            assert np.all(dests_board == expected)


def test_find_my_pieces():
    b = ChessBoard()

    # white
    pieces = b.find_my_pieces()
    coords = set([(r, c) for _, r, c in pieces])
    expected_coords = set([(r, c) for r in [6, 7] for c in range(SIZE)])
    assert coords == expected_coords

    # black
    b.turn = "black"
    pieces = b.find_my_pieces()
    coords = set([(r, c) for _, r, c in pieces])
    expected_coords = set([(r, c) for r in [0, 1] for c in range(SIZE)])
    assert coords == expected_coords


def test_moves_simple():
    b = ChessBoard()
    b.clear_pieces()
    b.board[3, 3] = "P"
    b._sync_board_to_piece_set()

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
    b._sync_board_to_piece_set()
    moves = b.moves()
    # for move in moves:
    #     b.print_move(move)
    assert len(moves) == 48

    b.turn = "black"
    moves = b.moves()
    # for move in moves:
    #     b.print_move(move)
    assert len(moves) == 43
    

def test_castling_moves():
    """Tests castling move generation.
    NOTE: casting does not yet support checking for the king in check"""
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
    b._sync_board_to_piece_set()
    moves = b._get_castle_moves("white")
    expected = [
        Move(7, 4, 7, 2, "K", special="c"),
        Move(7, 4, 7, 6, "K", special="c")
    ]
    assert(moves == expected)  # NOTE: this could be fragile to order changes
    
    moves = b._get_castle_moves("black")
    expected = [
        Move(0, 4, 0, 2, "k", special="c"),
        Move(0, 4, 0, 6, "k", special="c")
    ]
    assert(moves == expected)  # NOTE: this could be fragile to order changes
    
    # set has_moved flags
    b.flags[W_CASTLE_LEFT] = False
    b.flags[W_CASTLE_RIGHT] = False
    b.flags[B_CASTLE_LEFT] = False
    b.flags[B_CASTLE_RIGHT] = False
    
    assert(len(b._get_castle_moves("white")) == 0)
    assert(len(b._get_castle_moves("black")) == 0)
    
    # board 2 - blocked for various reasons
    b = ChessBoard()
    b.board = np.array(
        (
            "r . . k . . . r".split(),
            "p p p p p p p p".split(),
            ". . . . . . . .".split(),
            ". . . . . . . .".split(),
            ". . . . . . . .".split(),
            ". . . . . . . .".split(),
            "P P P P P P P P".split(),
            ". . . . K . N R".split(),
        )
    )
    b._sync_board_to_piece_set()
    assert(len(b._get_castle_moves("white")) == 0)
    assert(len(b._get_castle_moves("black")) == 0)
    

def test_en_passant_moves():
    b = ChessBoard()
    b.board = np.array(
        (
            ". . . k . . . .".split(),
            ". . p . p . p p".split(),
            ". . . . . . . .".split(),
            "p p . . . . . P".split(),
            ". . P p P p . .".split(),
            ". . . . . . . .".split(),
            "P P . P . P P .".split(),
            ". . . . K . . .".split(),
        )
    )
    b._sync_board_to_piece_set()
    moves = b._get_en_passant_moves("white")
    assert len(moves) == 0, "no en passant without flags set"
    
    b.flags[EN_PASSANT_SPOT] = (3, 0)
    moves = b._get_en_passant_moves("white")
    assert len(moves) == 0, "no available pieces"
    
    b.flags[EN_PASSANT_SPOT] = (4, 2)
    moves = b._get_en_passant_moves("black")
    expected = [Move(4, 3, 5, 2, "p", "P", "e")]
    assert moves == expected, "ep available to left"
    
    b.flags[EN_PASSANT_SPOT] = (4, 4)
    moves = b._get_en_passant_moves("black")
    expected = [  # NOTE: fragile to move order. fix hashability so i can use set
        Move(4, 5, 5, 4, "p", "P", "e"),
        Move(4, 3, 5, 4, "p", "P", "e"),
    ]
    assert moves == expected, "two eps available"
    

def test_eval_chess_board():
    b = ChessBoard()
    assert (0.0, False) == eval_chess_board(b), "start board test"

    b.clear_pieces()
    _, over = eval_chess_board(b)
    assert over, "empty board test"

    b.set_starting_pieces()
    b.board[6, 4] = "."  # advance king's pawn
    b.board[5, 4] = "P"
    b._sync_board_to_piece_set()
    score, over = eval_chess_board(b)
    assert over == False
    assert score > 0, "first move should increase score"


def test_minmax_1():
    # sanity check best eval move
    b = ChessBoard()
    b.clear_pieces()
    b.turn = "black"
    b.board[1, 4] = "k"
    b.board[2, 4] = "Q"  # best move for black is to take the queen
    b.board[7, 4] = "K"
    b._sync_board_to_piece_set()

    _, move = minmax(b, eval_chess_board, 1)
    expected = Move(1, 4, 2, 4, piece="k", captured="Q")
    b.print_move(move)
    assert move == expected

    _, move = minmax(b, eval_chess_board, 4)
    expected = Move(1, 4, 2, 4, piece="k", captured="Q")
    b.print_move(move)
    assert move == expected


def test_minmax_2():
    b = ChessBoard()
    b.turn = "white"
    b.board = np.array(
        (
            "r . b . k b . r".split(),
            "p . p p . p p p".split(),
            "n . . . p . . n".split(),
            ". P . . . . q .".split(),  # take the vulnerable queen
            ". . . P P . . .".split(),
            "N . . . . . . .".split(),
            "P P . . . P P P".split(),
            "R . B Q K B N R".split(),
        )
    )
    b._sync_board_to_piece_set()

    _, move = minmax(b, eval_chess_board, 1)
    expected = Move(7, 2, 3, 6)
    assert move == expected

    _, move = minmax(b, eval_chess_board, 2)
    expected = Move(7, 2, 3, 6)
    assert move == expected


def test_minmax_3():
    b = ChessBoard()
    b.turn = "white"
    # https://chesspuzzlesonline.com/solution/ps248/
    b.board = np.array(
        (
            ". . . . . . . k".split(),
            ". . r . n . p .".split(),
            ". . . . B p . .".split(),
            ". . . P . . . .".split(),
            ". . . . . K p .".split(),
            ". . . . . . P .".split(),
            ". . p . . P P .".split(),
            "R . . . . . . .".split(),  # rook to H1, mate
        )
    )
    b._sync_board_to_piece_set()

    _, move = minmax(b, eval_chess_board, 3)
    b.print_move(move)
    expected = Move(7, 0, 7, 7)
    assert move == expected

    # forked!
    b.turn = "black"
    b.board = np.array(
        (
            "r . b . k b . r".split(),
            "p p p p n p p p".split(),
            ". . . . . . . .".split(),
            ". . . P . . . .".split(),
            ". . . n . . . .".split(),
            ". . . . . . . N".split(),
            "P P P . . P P P".split(),  # fork the king and rook!
            "R N B . K B . R".split(),
        )
    )
    b._sync_board_to_piece_set()

    _, move = minmax(b, eval_chess_board, 3)
    b.print_move(move)
    expected = Move(4, 3, 6, 2)
    assert move == expected

def test_play():
    # just make sure this doesn't error
    play_game(white_params={"depth":2}, black_params={"depth":2})


def test_en_passant_flags():
    b = ChessBoard()
    m = Move(r_from=1, c_from=2, r_to=3, c_to=3)  # black pawn forward 2
    b.do_move(m)
    assert b.flags["en_passant_spot"] == (3,3), "should be available"

    m = Move(r_from=1, c_from=3, r_to=2, c_to=3)  # black pawn forward 1
    b.do_move(m)
    assert b.flags["en_passant_spot"] == None, "should not be available"

    m = Move(r_from=6, c_from=4, r_to=4, c_to=4)  # white pawn forward 2
    b.do_move(m)
    assert b.flags["en_passant_spot"] == (4,4), "should be available"

    b.undo_move()
    assert b.flags["en_passant_spot"] == None, "should not available after undo"

def test_castle_flags():
    b = ChessBoard()
    b.board = np.array((
        "r . . . k . . r".split(),
        ". . . . p . . .".split(),
        ". . . . . . . .".split(),
        ". . . . . . . .".split(),
        ". . . . . . . .".split(),
        ". . . . . . . .".split(),
        ". . . . P . . .".split(),
        "R . . . K . . R".split(),
    ))
    b._sync_board_to_piece_set()

    m = Move(r_from=6, c_from=4, r_to=4, c_to=4)  # just a pawn move
    b.do_move(m)
    assert all((
        b.flags["b_castle_right"],
        b.flags["b_castle_left"],
        b.flags["w_castle_right"],
        b.flags["w_castle_left"])), "all should start as true"

    m = Move(r_from=0, c_from=0, r_to=0, c_to=1)  # move left black rook
    b.do_move(m)
    assert all((
        b.flags["b_castle_right"],
        not b.flags["b_castle_left"],
        b.flags["w_castle_right"],
        b.flags["w_castle_left"])), "black can't castle left"

    m = Move(r_from=7, c_from=4, r_to=7, c_to=3)  # move white king
    b.do_move(m)
    assert all((
        b.flags["b_castle_right"],
        not b.flags["b_castle_left"],
        not b.flags["w_castle_right"],
        not b.flags["w_castle_left"])), "white can't castle at all now"

    b.undo_move()
    assert all((
        b.flags["b_castle_right"],
        not b.flags["b_castle_left"],
        b.flags["w_castle_right"],
        b.flags["w_castle_left"])), "back to normal after undo move"


def test_do_and_undo_castle():
    b = ChessBoard()
    b.board = np.array((
        "r . . . k . . r".split(),
        ". . . . p . . .".split(),
        ". . . . . . . .".split(),
        ". . . . . . . .".split(),
        ". . . . . . . .".split(),
        ". . . . . . . .".split(),
        ". . . . P . . .".split(),
        "R . . . K . . R".split(),
    ))
    b._sync_board_to_piece_set()
    
    m = Move(7, 4, 7, 6, special=W_CASTLE_RIGHT)
    b.do_move(m)
    assert_row(b, 0, "r...k..r")
    assert_row(b, 7, "R....RK.")
    b.undo_move()
    assert_row(b, 0, "r...k..r")
    assert_row(b, 7, "R...K..R")
    
    m = Move(7, 4, 7, 2, special=W_CASTLE_LEFT)
    b.do_move(m)
    assert_row(b, 0, "r...k..r")
    assert_row(b, 7, "..KR...R")
    b.undo_move()
    assert_row(b, 0, "r...k..r")
    assert_row(b, 7, "R...K..R")
    
    m = Move(0, 4, 0, 6, special=B_CASTLE_RIGHT)
    b.do_move(m)
    assert_row(b, 0, "r....rk.")
    assert_row(b, 7, "R...K..R")
    b.undo_move()
    assert_row(b, 0, "r...k..r")
    assert_row(b, 7, "R...K..R")
    
    m = Move(0, 4, 0, 2, special=B_CASTLE_LEFT)
    b.do_move(m)
    assert_row(b, 0, "..kr...r")
    assert_row(b, 7, "R...K..R")
    b.undo_move()
    assert_row(b, 0, "r...k..r")
    assert_row(b, 7, "R...K..R")
    

def test_do_and_undo_en_passant():
    b = ChessBoard()
    b.board = np.array((
        ". . . k . . . .".split(),
        ". . p . p . . p".split(),
        ". . . . . . . .".split(),
        "p p . . . . p P".split(),
        ". . P p P p . .".split(),
        ". . . . . . . .".split(),
        "P P . P . P P .".split(),
        ". . . . K . . .".split(),
    ))
    b._sync_board_to_piece_set()
    b.turn = "black"
    
    # black takes with d4
    b.flags[EN_PASSANT_SPOT] = (4, 2)
    m = Move(4, 3, 5, 2, special="e")
    b.do_move(m)
    assert_row(b, 4, "....Pp..")
    assert_row(b, 5, "..p.....")
    assert b.flags[EN_PASSANT_SPOT] is None, "flags cleared"
    b.undo_move()
    assert_row(b, 4, "..PpPp..")
    assert_row(b, 5, "........")
    assert b.flags[EN_PASSANT_SPOT] == (4, 2), "flags reset"
    
    # black takes right with d4
    b.turn = "black"
    b.flags[EN_PASSANT_SPOT] = (4, 4)
    m = Move(4, 3, 5, 4, special="e")
    b.do_move(m)
    assert_row(b, 4, "..P..p..")
    assert_row(b, 5, "....p...")
    assert b.flags[EN_PASSANT_SPOT] is None, "flags cleared"
    b.undo_move()
    assert_row(b, 4, "..PpPp..")
    assert_row(b, 5, "........")
    assert b.flags[EN_PASSANT_SPOT] == (4, 4), "flags reset"

    # white takes with h5
    b.turn = "white"
    b.flags[EN_PASSANT_SPOT] = (3, 6)
    m = Move(3, 7, 2, 6, special="e")
    b.do_move(m)
    assert_row(b, 2, "......P.")
    assert_row(b, 3, "pp......")
    assert b.flags[EN_PASSANT_SPOT] is None, "flags cleared"
    b.undo_move()
    assert_row(b, 4, "..PpPp..")
    assert_row(b, 5, "........")
    assert b.flags[EN_PASSANT_SPOT] == (3, 6), "flags reset"

test_do_and_undo_en_passant()