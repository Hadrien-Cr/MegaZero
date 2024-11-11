import pyximport; pyximport.install()
from torch import multiprocessing as mp
from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.connect4d.connect4d import Game, NUM_BOARDS
from alphazero.GenericPlayers import RandomPlayer, RawMCTSPlayer, RawOSLA
from alphazero.utils import dotdict
from alphazero.envs.connect4d.config import CONFIG_MCTS_VANILLA, CONFIG_EMCTS_VANILLA, CONFIG_MCTS_BB, CONFIG_EMCTS_BB

import argparse
parser = argparse.ArgumentParser(description="Run training for Connect4D with different MCTS configurations.")
parser.add_argument("--algorithm", type=str, default="emcts-vanilla",
                    choices=["mcts-vanilla", "emcts-vanilla", "mcts-bb", "emcts-bb"],
                    help="Specify which MCTS algorithm to use: 'mcts-vanilla', 'emcts-vanilla', 'mcts-bb', or 'emcts-bb'")
args_parsed = parser.parse_args()
name_to_config = {
    "mcts-vanilla": CONFIG_MCTS_VANILLA,
    "emcts-vanilla": CONFIG_EMCTS_VANILLA,
    "mcts-bb": CONFIG_MCTS_BB,
    "emcts-bb": CONFIG_EMCTS_BB
}
config = name_to_config[args_parsed.algorithm]

args = get_args(dotdict({
    'baselineTester': [(RawOSLA, None)],
    'workers': (mp.cpu_count()-1),
    'startIter': 1,
    'numIters': 50,
    'numWarmupIters': 1,
    'process_batch_size': 64,
    'train_batch_size': 1024,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 2*64*(mp.cpu_count()-1),
    'symmetricSamples': True,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 10*(mp.cpu_count()-1),
    'arenaCompare': 10*(mp.cpu_count()-1),
    'arena_batch_size': 10,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'arenaBatched': True,
    'baselineCompareFreq': 2,
    'compareWithPast': True,
    'pastCompareFreq': 2,
    'cpuct': 2,
    'fpu_reduction': 0.1,
    'load_model': True,
}),
    model_gating=True,
    max_gating_iters=None,
    max_moves=24,

    lr=0.01,
    num_channels=128,
    depth=6,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[1024, 128],
    policy_dense_layers=[1024]
)
args.scheduler_args.milestones = [75, 150]
args.update(config)

if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
