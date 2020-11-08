#!/usr/bin/env python3

import time
from typing import Dict, List, Tuple, Sequence, Set, Callable, TypeVar, Optional
from copy import deepcopy
from termcolor import colored
import functools
from multiprocessing import Pool, cpu_count
import pickle

from tqdm import tqdm
import numpy as np

from search import minmax, iterative_deepening
from chessboard import Move, ChessBoard, SIZE, ALL_PIECES
from chess import play_game, computer_player, Player


def get_all_players() -> Sequence[Player]:
    """Returns a list of all combinations of different player settings dicts"""
    all_params = []
    for explore_ratio in [1.0]:
        for depth in [3]:
            for piece_table in [True, False]:
                for mobility in [True, False]:
                    # speed control:
                    if depth == 6 and explore_ratio > .6:
                        continue
                    if depth == 7 and explore_ratio > .4:
                        continue
                    if depth == 8 and explore_ratio > .2:
                        continue

                    # construct player params dict
                    params = dict(
                        depth=depth,
                        explore_ratio=explore_ratio,
                        min_branches=10,
                        mobility=mobility,
                        piece_table=piece_table
                    )

                    all_params.append(params)

    return all_params


# TODO extract job server stuff into a nice library between projects
def single_run(cfg):
    """run a set for a particular config"""
    white_params = cfg["white_params"]
    black_params = cfg["black_params"]
    score, game = play_game(white_params, black_params, display=True)
    return cfg, score, game


def run_job_server(func, experiments, save_file, resume=True, num_experiments=None, n_cores=None):
    """Runs a job server to run the given func over a list of many input args (experiments).
    Runs with a multiprocess pool, saves all results to a picke file, can resume if cancelled,
    and displays a progress bar.
    func: the function the job server will be calling.
    experiments: list or generator of args to pass to func. If a generator, also pass in num_experiments.
    save_file: str file name of where to save our picked results.
    resume: whether to resume a cancelled run.
    num_experiments: None or int. required if experiments is a generator expression.
    n_cores: None or int. overrides using all they systems cores.
    """

    def _count_pickles(f_name):
        """Count the number of objects that has been stored in a pickle file"""
        count = 0
        try:
            with open(f_name, "rb") as f:
                while True:
                    try:
                        pickle.load(f)
                        count += 1
                    except EOFError:
                        return count
        except IOError:  # handle case of no file there yet
            return 0

    if n_cores is None:
        n_cores = cpu_count()

    if hasattr(experiments, "__len__"):
        num_experiments = len(experiments)

    # resume a previous run?
    start_index = 0
    if resume:
        start_index = _count_pickles(save_file)
        if start_index != 0:
            print("Resuming from index: {}".format(start_index))
        # burn off that many experiments
        for _ in range(start_index):
            next(experiments)

    with Pool(n_cores) as p:
        for result in tqdm(
                            p.imap(single_run, experiments),
                            total=num_experiments,
                            initial=start_index):
            # write results one at a time to a pickle object, so we never need to store them in mem
            with open(save_file, "ab") as f:
                pickle.dump(result, f)


if __name__ == '__main__':
    # build experiments
    players = get_all_players()

    matches = []
    for white in players:
        for black in players:
            cfg = {"white_params": white, "black_params": black}
            matches.append(cfg)

    # run!
    run_job_server(
        single_run,
        matches,
        "heuristics.p",
        resume=True)
