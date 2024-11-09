
from alphazero.envs.connect4d.connect4d import Game, NUM_BOARDS 
from alphazero.GenericPlayers import RawMCTSPlayer, RandomPlayer, RawEMCTSPlayer, RawOSLA
from alphazero.Arena import Arena
import alphazero.Coach as c

def matchup():
    from alphazero.envs.connect4d.train import args
    args = c.get_args(args)
    args['_num_players'] = 2
    args['arenaTemp'] = 0.5
    args['numMCTSSims'] = 3000
    args['emcts_bb_phases'] = 10
    strategy = "vanilla"
    agents = [
                RawOSLA(Game, args),
                RawEMCTSPlayer(strategy, Game, args),
                #RawMCTSPlayer(strategy, Game, args),
                #RandomPlayer(Game),
            ]
    players = [agents[0], agents[1]]
    print(players[0].__class__.__name__, "vs", players[1].__class__.__name__)
    arena = Arena(players, Game, use_batched_mcts=args.arenaBatched, args=args)
    arena.play_games(args.arenaCompare)



if __name__ == "__main__":
    matchup()
