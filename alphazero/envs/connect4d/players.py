from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer
from alphazero.MCTS import MCTS
import numpy as np


class HumanConnect4dPlayer(BasePlayer):
    @staticmethod
    def is_human() -> bool:
        return True

    def play(self, state: GameState) -> int:
        valid_moves = state.valid_moves()
        print('\nMoves:', [i for (i, valid)
                           in enumerate(valid_moves) if valid])

        while True:
            move = int(input())
            if valid_moves[move]:
                break
            else:
                print('Invalid move')
        return move


class OneStepLookaheadConnect4dPlayer(BasePlayer):
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""

    def __init__(self, game_cls, args, verbose=False):
        self.verbose = verbose
        self.args = args
    
    def update(self, state: GameState, action: int) -> None:
        self.mcts.update_root(state, action)

    def reset(self):
        self.mcts = MCTS(self.args)

    def play(self, state) -> int:
        self.mcts.search(state, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        policy = self.mcts.probs(state, self.temp)

    def play(self, state: GameState) -> int:
        valid_moves = state.valid_moves()
        win_move_set = set()
        fallback_move_set = set()
        stop_loss_move_set = set()

        for move, valid in enumerate(valid_moves):
            if not valid: continue

            new_state = state.clone()
            new_state.play_action(move)
            ws = new_state.win_state()
            if ws[state.player]:
                win_move_set.add(move)
            elif ws[new_state.player]:
                stop_loss_move_set.add(move)
            else:
                fallback_move_set.add(move)

        if len(win_move_set) > 0:
            ret_move = np.random.choice(list(win_move_set))
            if self.verbose:
                print('Playing winning action %s from %s' %
                      (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set))
            if self.verbose:
                print('Playing loss stopping action %s from %s' %
                      (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set))
            if self.verbose:
                print('Playing random action %s from %s' %
                      (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % state)

        return ret_move
