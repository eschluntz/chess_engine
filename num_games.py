#!/usr/bin/env python3

from copy import deepcopy
import numpy as np

# calculate the number of possible ttt games.
# 1. start from emtpy board.
# 2. recursively explore all moves.
# 3. check if a game is an end condition. if so, return 1 up.
# 4. if not terminal, return the sum of all the children boards.

# datastructure
# board is 3x3 array.
# each cell is 0, 1, 2 (empty, x, o)

empty_board = np.zeros((3,3))


def check_if_game_over(board):
    """return True if the game is over"""
    for i in range(3):
        if all(board[i,:] == [1,1,1]) or all(board[i,:] == [2,2,2]):
            return True
        if all(board[:,i] == [1,1,1]) or all(board[:,i] == [2,2,2]):
            return True
    # check diags
    diag1 = [ board[i, i] for i in range(3) ]
    diag2 = [ board[2 - i, i] for i in range(3) ]
    if diag1 == [1,1,1] or diag1 == [2,2,2] or diag2 == [1,1,1] or diag2 == [2,2,2]:
        return True

    # check if board is full
    if sum(sum(board == 0)) == 0:
        return True

    # print(board)
    return False


def count_ttt_games(board: np.array = empty_board, move : int = 1) -> int:
    """board: the ttt board so far.
    move: 1/2 for x or o. current move to be made
    returns how many games are possible from this position, inclusive.
    """

    # check if this is terminal
    if check_if_game_over(board):
        return 1

    # loop over all moves. (all empty squares)
    move_count = 0
    for i in range(3):
        for j in range(3):
            if board[i,j] == 0:  # empty, possible move
                # create next situation
                new_board = deepcopy(board)
                new_board[i,j] = move
                new_move = 3 - move  # flips 2 and 1 # todo: cleanup move as enum
                move_count += count_ttt_games(new_board, new_move)

    # return sum
    return move_count


def test_check_game_over():
    b = empty_board
    assert check_if_game_over(b) == False, "empty board"

    b = deepcopy(empty_board)
    for i in range(3):
        b[0,i] = 1
    assert check_if_game_over(b) == True

    b = deepcopy(empty_board)
    for i in range(3):
        b[i, 1] = 1
    assert check_if_game_over(b) == True

    b = deepcopy(empty_board)
    for i in range(3):
        b[i,i] = 2
    assert check_if_game_over(b) == True

    b = deepcopy(empty_board)
    for i in range(3):
        b[0,i] = i
    assert check_if_game_over(b) == False

if __name__ == "__main__":
    # poor man's unit tests
    test_check_game_over()

    print("number of games:")
    num_moves = count_ttt_games()
    print(num_moves)
