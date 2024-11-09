from alphazero.envs.strands.strands import Game, rules_strands, STRANDS_MODE
from alphazero.Arena import Arena
from alphazero.envs.strands.heuristic import StrandsHeuristicEMCTS, StrandsHeuristicMCTS,StrandsHeuristicOSLA
from alphazero.GenericPlayers import RandomPlayer

import alphazero.Coach as c

def matchup():
    from alphazero.envs.strands.train import args
    args = c.get_args(args)
    args['_num_players'] = 2
    args['arenaTemp'] = 0.5
    strategy = "vanilla"
    agents = [
                StrandsHeuristicEMCTS(strategy,Game, args),
                StrandsHeuristicMCTS(strategy, Game, args),
                StrandsHeuristicOSLA(strategy, Game, args),
                RandomPlayer(Game),
            ]
    players = [agents[0], agents[1]]
    print(players[0].__class__.__name__, "vs", players[1].__class__.__name__)
    arena = Arena(players, Game, use_batched_mcts=args.arenaBatched, args=args)
    arena.play_games(args.arenaCompare)

if __name__ == "__main__":
    matchup()
