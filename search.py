#!/usr/bin/env python3

import numpy as np

TRANSPOSITION_TABLE = {}
# Maps (board+depth) -> score to avoid repeated work and improve move ordering
# key: str(board.board.flatten()) + depth


def iterative_deepening(board, eval_fn, max_depth, max_t=10.0):
    """Iteratively calls minmax with higher depths.
    1. this allows us to gracefully add a time limit.
    2. NOTE: should fill up the transposition table for better move ordering,
    but using that for move ordering doesn't seem to be better than the straight eval_fn

    Returns: (score, move)
    """
    t0 = time.time()
    score, move = None, None
    for depth in range(max_depth + 1):
        tot_t = time.time() - t0
        if tot_t > max_t:
            break
        score, move = minmax(board, eval_fn, depth)
    return score, move


def minmax(board, eval_fn, max_depth, alpha=-np.inf, beta=np.inf):
    """Finds the best move using MinMax and AlphaBeta pruning.
    Hopefully this function can be used across many different games!

    board: board representation with this interface:
        [...] = board.moves()
        board.do_move(move)
        board.undo_move()
    eval_fn: a function that transforms a board into a score
        score, over = eval_fn(board)
    max_depth: how many more layers to search.
    alpha:  worst possible score for "x" = -inf
    beta:   worst possible score for "o" = +inf


    TODO: make it prefer victories that are sooner, or defeats that are later.
    TODO: sort possible moves by the eval_fn heuristic to improve pruning

    returns: (score, move) the expected score down that path.
    """

    # base cases
    score, done = eval_fn(board)
    if done or max_depth == 0:
        return score, None

    # are we maxing or mining?
    direction = 1.0 if board.turn in ["x", "white"] else -1.0  # TODO: make turn binary?

    # loop!
    best_move = None
    best_score = -np.inf * direction

    all_moves = board.moves()

    # order these nicely to improve alpha beta pruning
    def score_move_heuristic(move):
        board.do_move(move)
        score, _ = eval_fn(board)
        board.undo_move()
        return score

    all_moves.sort(key=score_move_heuristic, reverse=(board.turn in ["white", "x"]))  # TODO: fix white / x

    # we've already sorted
    if max_depth == 1:
        move = all_moves[0]
        board.do_move(move)
        score, _ = eval_fn(board)
        board.undo_move()
        return score, move

    # search the tree!
    for move in all_moves:
        board.do_move(move)
        # add to transposition table
        key = "".join(board.board.flatten()) + str(max_depth - 1)

        if key in TRANSPOSITION_TABLE:
            score = TRANSPOSITION_TABLE[key]
        else:
            score, _ = minmax(board, eval_fn, max_depth - 1, alpha, beta)
            TRANSPOSITION_TABLE[key] = score

        board.undo_move()

        if score * direction > best_score * direction:
            best_score = score
            best_move = move

        # update heuristics
        if direction > 0:
            alpha = max(alpha, score)  # only if max
        else:
            beta = min(beta, score)  # only if min
        if beta <= alpha:  # we know the parent won't choose us. abandon the search!
            break

    return best_score, best_move
