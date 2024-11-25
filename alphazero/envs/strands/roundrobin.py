import pyximport
pyximport.install()

from alphazero.NNetWrapper import NNetWrapper as nn
from alphazero.GenericPlayers import *
from alphazero.Arena import Arena
from pathlib import Path
from glob import glob
from alphazero.envs.strands.train import args, Game
from alphazero.envs.strands.pit_config import *
import numpy as np
import pprint
import choix
import multiprocessing as mp

if __name__ == '__main__':
    
    args['arenaBatched'] = True
    args['arenaMCTS'] = True
    args['arenaTemp'] = 0.25
    args['arena_batch_size'] = 96
    args['arenaCompare'] = 96*(mp.cpu_count()-1)
    args['_num_players'] = 2
    args['compareWithBaseline'] = False


    assert Path('experiments/strands').exists()
    print('Beginning round robin')
    networks = sorted(glob('experiments/strands/*/*.pkl'), reverse=True)
    model_count = len(networks) + int(args.compareWithBaseline)
    print(networks, model_count)


    total_games = 0
    for i in range(model_count):
        total_games += i
    total_games *= args.arenaCompare
    print(
        f'Comparing {model_count} different models in {total_games} total games')
    win_matrix = np.zeros((model_count, model_count))



    for i in range(model_count - 1):
        for j in range(i + 1, model_count):
            file1 = networks[i]
            file2 = networks[j]
            players = []
            for file in [file1, file2]:
                nnet = nn(Game, args)
                nnet.load_checkpoint(folder= '', filename=networks[0])
                pl_args = args.copy()
                if "emcts-vanilla" in file:
                    pl_args.update(PIT_CONFIG_EMCTS_VANILLA)
                    player = EMCTSPlayer('vanilla', nnet, Game, pl_args)
                elif "emcts-bb" in file:
                    pl_args.update(PIT_CONFIG_EMCTS_BB)
                    player = EMCTSPlayer('bridge-burning', nnet, Game, pl_args)
                elif "mcts-vanilla" in file:
                    pl_args.update(PIT_CONFIG_MCTS_VANILLA)
                    player = MCTSPlayer('vanilla', nnet, Game, pl_args)
                elif "mcts-bb" in file:
                    pl_args.update(PIT_CONFIG_MCTS_BB)
                    player = MCTSPlayer('bridge-burning', nnet, Game, pl_args)
                else:
                    raise ValueError
                
                players.append(player)

            print(f'{players[0]} vs {players[1]}')
            arena = Arena(players, Game, use_batched_mcts=args.arenaBatched, args=args)
            wins, draws, winrates = arena.play_games(args.arenaCompare)
            win_matrix[i, j] = wins[0] + 0.5 * draws
            win_matrix[j, i] = wins[1] + 0.5 * draws
            print(f'wins: {wins[0]}, ties: {draws}, losses:{wins[1]}\n')

    print("\nWin Matrix(row beat column):")
    print(win_matrix)
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            params = choix.ilsr_pairwise_dense(win_matrix)
        print("\nRankings:")
        for i, player in enumerate(np.argsort(params)[::-1]):
            name = 'random' if args.compareWithBaseline and player == model_count - \
                               1 else networks[player]
            print(f"{i + 1}. {name} with {params[player]:0.2f} rating")
        print(
            "\n(Rating Diff, Winrate) -> (0.5, 62%), (1, 73%), (2, 88%), (3, 95%), (5, 99%)")
    except Exception:
        print("\nNot Enough data to calculate rankings")
