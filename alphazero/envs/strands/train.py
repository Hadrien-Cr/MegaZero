import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.strands.strands import Game, rules_strands
from alphazero.envs.strands.heuristic import StrandsHeuristicMCTS,StrandsHeuristicOSLA
from alphazero.utils import dotdict

args = get_args(dotdict({
    'run_name': 'strands_fpu',
    'emcts_horizon': 12,
    'emcts_bb_phases': 5,
    'self_play_mode': 'mcts', #'emcts', 'mcts',
    'self_play_strategy': 'vanilla', #'bridge-burning','mode'
    'baselineTester': StrandsHeuristicOSLA,
    'workers': mp.cpu_count()-1,
    'symmetricSamples': True,
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 128,
    'train_batch_size': 1024,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 10*128*(mp.cpu_count() -1),
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'numMCTSSims': 1000,
    'numFastSims': 1000,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 16*(mp.cpu_count() -1),
    'arenaCompare': 16*(mp.cpu_count() -1),
    'arena_batch_size': 16,
    'arenaTemp': 1,
    'arenaMCTS': False,
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
    max_moves=rules_strands['MAX_TURNS'],

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
