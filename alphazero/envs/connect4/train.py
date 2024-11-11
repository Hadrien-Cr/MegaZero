import pyximport; pyximport.install()

from torch import multiprocessing as mp

from alphazero.Coach import Coach, get_args
from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.envs.connect4d.connect4d import Game
from alphazero.GenericPlayers import RandomPlayer, RawMCTSPlayer, RawOSLA
from alphazero.utils import dotdict
from alphazero.envs.connect4.config import CONFIG_MCTS_VANILLA, CONFIG_EMCTS_VANILLA, CONFIG_MCTS_BB, CONFIG_EMCTS_BB

# config = CONFIG_MCTS_VANILLA
config = CONFIG_EMCTS_VANILLA
# config = CONFIG_MCTS_BB
# config = CONFIG_EMCTS_BB

args = get_args(dotdict({
    'baselineTester': [(RawOSLA, None), (RawMCTSPlayer, "vanilla")],
    'workers': (mp.cpu_count()-1),
    'startIter': 1,
    'numIters': 1000,
    'numWarmupIters': 1,
    'process_batch_size': 128,
    'train_batch_size': 1024,
    # should preferably be a multiple of process_batch_size and workers
    'gamesPerIteration': 128*(mp.cpu_count()-1),
    'symmetricSamples': True,
    'compareWithBaseline': True,
    'arenaCompareBaseline': 16*(mp.cpu_count()-1),
    'arenaCompare': 16*(mp.cpu_count()-1),
    'arena_batch_size': 16,
    'arenaTemp': 1,
    'arenaMCTS': True,
    'arenaBatched': True,
    'baselineCompareFreq': 1,
    'compareWithPast': True,
    'pastCompareFreq': 1,
    'cpuct': 2,
    'fpu_reduction': 0.1,
    'load_model': True,
}),
    model_gating=True,
    max_gating_iters=None,
    max_moves=42,

    lr=0.01,
    num_channels=128,
    depth=8,
    value_head_channels=16,
    policy_head_channels=16,
    value_dense_layers=[256, 128],
    policy_dense_layers=[256]
)
args.scheduler_args.milestones = [75, 150]
args.update(config)

if __name__ == "__main__":
    nnet = nn(Game, args)
    c = Coach(Game, nnet, args)
    c.learn()
