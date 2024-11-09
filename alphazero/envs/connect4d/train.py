import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.connect4d.connect4d import Game, NUM_BOARDS
from alphazero.GenericPlayers import RawMCTSPlayer, RawOSLA
from alphazero.utils import dotdict

args = get_args(dotdict({
    'run_name': 'connect4d_fpu',
    'emcts_horizon': 2*NUM_BOARDS,
    'emcts_bb_phases': 10,
    'self_play_mode': 'mcts', #'emcts', 'mcts',
    'self_play_strategy': 'vanilla', #'bridge-burning','mode'
    'baselineTester': RawOSLA,
    'workers': mp.cpu_count(),
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 16,
    'train_batch_size': 1024,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 16*mp.cpu_count(),
    'symmetricSamples': True,
    'numMCTSSims': 1000,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 16*mp.cpu_count(),
    'arenaCompare': 16*mp.cpu_count(),
    'arena_batch_size': 16,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'arenaBatched': True,
    'baselineCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'cpuct': 4,
    'fpu_reduction': 0.4,
    'load_model': True,
}),
    model_gating=True,
    max_gating_iters=None,
    max_moves=42,

    lr=0.01,
    num_channels=128,
    depth=4,
    value_head_channels=32,
    policy_head_channels=32,
    value_dense_layers=[1024, 256],
    policy_dense_layers=[1024]
)
args.scheduler_args.milestones = [75, 150]


if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
