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


def minmax(board, eval_fn, max_depth, alpha=-np.inf, beta=np.inf, params={}):
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
    params: optional dict for controlling search and eval:
        time_discount: how much to discount each turn
        explore_ratio: fraction of possible moves to explore
        min_branches: overrides explore_ratio in case there are few branches
        ... others passed on to eval_fn

    returns: (score, move) the expected score down that path.
    """

    TIME_DISCOUNT = params.get("time_discount", 0.95)

    # base cases
    score, done = eval_fn(board, params)
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
        score, _ = eval_fn(board, params)
        board.undo_move()
        return score
    all_moves.sort(key=score_move_heuristic, reverse=(board.turn in ["white", "x"]))  # TODO: fix white / x

    # we've already sorted, just return now (10% speedup)
    if max_depth == 1:
        move = all_moves[0]
        board.do_move(move)
        score, _ = eval_fn(board)
        board.undo_move()
        return int(score * TIME_DISCOUNT), move

    # search the tree!
    explore_ratio = params.get("explore_ratio", 1.0)
    min_branches = params.get("min_branches", 10)
    num_to_explore = max(int(len(all_moves) * explore_ratio), min_branches)

    for move in all_moves[:num_to_explore]:
        board.do_move(move)
        # add to transposition table
        key = "".join(board.board.flatten()) + str(max_depth - 1)

        if key in TRANSPOSITION_TABLE:
            score = TRANSPOSITION_TABLE[key]
        else:
            score, _ = minmax(board, eval_fn, max_depth - 1, alpha, beta, params)
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

    return int(best_score * TIME_DISCOUNT), best_move
