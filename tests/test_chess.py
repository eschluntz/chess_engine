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

def test_rotate_coords():
    b = ChessBoard()

    assert b.rotate_coords(0,0) == (7,7)
    assert b.rotate_coords(0,7) == (7,0)
