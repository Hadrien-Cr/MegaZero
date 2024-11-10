import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.strands.strands import Game, rules_strands
from alphazero.envs.strands.heuristic import StrandsHeuristicMCTS,StrandsHeuristicOSLA
from alphazero.utils import dotdict
from alphazero.envs.strands.config import CONFIG_MCTS_VANILLA, CONFIG_EMCTS_VANILLA, CONFIG_MCTS_BB, CONFIG_EMCTS_BB

config = CONFIG_MCTS_VANILLA
# config = CONFIG_EMCTS_VANILLA
# config = CONFIG_MCTS_BB
# config = CONFIG_EMCTS_BB

args = get_args(dotdict({
    'baselineTester': StrandsHeuristicOSLA,
    'workers': mp.cpu_count(),
    'symmetricSamples': True,
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 128,
    'train_batch_size': 1024,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 10*128*mp.cpu_count(),
    'symmetricSamples': True,
    'skipSelfPlayIters': None,
    'selfPlayModelIter': None,
    'arenaCompareBaseline': 16*mp.cpu_count() ,
    'arenaCompare': 16*mp.cpu_count() ,
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
args.update(config)

if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
