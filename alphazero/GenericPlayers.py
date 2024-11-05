from alphazero.MCTS import MCTS
from alphazero.EMCTS import EMCTS
from alphazero.Game import GameState
from alphazero.NNetWrapper import NNetWrapper
from alphazero.utils import dotdict, plot_mcts_tree

from abc import ABC, abstractmethod

import numpy as np
import torch


class BasePlayer(ABC):
    def __init__(self, game_cls: GameState = None, args: dotdict = None, verbose: bool = False):
        self.game_cls = game_cls
        self.args = args
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        return self.play(*args, **kwargs)

    @staticmethod
    def supports_process() -> bool:
        return False

    @staticmethod
    def requires_model() -> bool:
        return False

    @staticmethod
    def is_human() -> bool:
        return False

    def update(self, state: GameState, action: int) -> None:
        pass

    def reset(self):
        pass

    @abstractmethod
    def play(self, state: GameState) -> int:
        pass

    def process(self, batch):
        raise NotImplementedError


class RandomPlayer(BasePlayer):
    def play(self, state):
        turn = []
        current_player = state._player
        while state._player == current_player and not state.win_state().any():
            valids = state.valid_moves()
            valids = valids / np.sum(valids)
            action = np.random.choice(state.action_size(), p=valids)
            turn.append(action)
            state.play_action(action)
        return(turn)

class NNPlayer(BasePlayer):
    def __init__(self, nn: NNetWrapper, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn
        self.temp = self.args.startTemp
        self.mode = "osla"
    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return True

    def play(self, state) -> int:
        turn = []
        current_player = state._player
        while state._player == current_player and not state.win_state().any():
            policy, _ = self.nn.predict(state.observation())
            valids = state.valid_moves()
            options = policy * valids
            self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
            if self.temp == 0:
                bestA = np.argmax(options)
                probs = [0] * len(options)
                probs[bestA] = 1
            else:
                probs = [x ** (1. / self.temp) for x in options]
                probs /= np.sum(probs)

            a = np.random.choice(
                np.arange(state.action_size()), p=probs
            ).item()

            state.play_action(a)
            turn.append(a)
        return turn

    def process(self, *args, **kwargs):
        return self.nn.process(*args, **kwargs)


############### MCTSPlayer ####################

class MCTSPlayer(BasePlayer):
    def __init__(self, strategy, nn: NNetWrapper, *args, print_policy=False,
                 average_value=False, draw_mcts=False, draw_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn
        self.strategy = strategy
        self.mode = "mcts"
        self.__class__.__name__ = f"MCTSPlayer(strategy = {strategy})"
        self.temp = self.args.startTemp
        self.print_policy = print_policy
        self.average_value = average_value
        self.draw_mcts = draw_mcts
        self.draw_depth = draw_depth
        self.mcts = MCTS(self.args)

        if self.verbose:
            self.mcts.search(
                self.game_cls(), self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp
            )
            value = self.mcts.value(self.average_value)
            self.__rel_val_split = value if value > 0.5 else 1 - value
            print('initial value:', self.__rel_val_split)

    @staticmethod
    def supports_process() -> bool:
        return False

    @staticmethod
    def requires_model() -> bool:
        return False

    def reset(self):
        self.mcts.reset()

    def play(self, state) -> int:
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        
        if self.strategy == "vanilla":
            self.reset()
            self.mcts.search(state, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            while not self.mcts.turn_completed:
                self.mcts.update_turn(state, self.temp)

        elif self.strategy == "bridge-burning":
            self.reset()
            while not self.mcts.turn_completed:
                self.mcts.search(state, self.nn, self.args.numMCTSSims/state.avg_atomic_actions(), self.args.add_root_noise, self.args.add_root_temp)
                if not self.mcts.turn_completed: self.mcts.update_turn(state, self.temp)

        turn, pi, state_history = self.mcts.get_results(state)
        return turn
    
    def process(self, *args, **kwargs):
        return self.nn.process(*args, **kwargs)



############### EMCTSPlayer ####################


class EMCTSPlayer(BasePlayer):
    def __init__(self, strategy, nn: NNetWrapper, *args, print_policy=False,
                 average_value=False, draw_mcts=False, draw_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn
        self.strategy = strategy
        self.mode = "emcts"
        self.__class__.__name__ = f"EMCTSPlayer(strategy = {strategy})"
        self.temp = self.args.startTemp
        self.print_policy = print_policy
        self.average_value = average_value
        self.draw_mcts = draw_mcts
        self.draw_depth = draw_depth
        self.emcts = EMCTS(self.args)

    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return True

    def reset(self):
        self.emcts.reset()

    def play(self, state) -> int:
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())

        if self.strategy == "vanilla":
            self.reset()
            self.emcts.search(state, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            assert not self.emcts.turn_completed, self.phase
            while not self.emcts.turn_completed:
                self.emcts.update_turn(state, self.temp)

        elif self.strategy == "bridge-burning":
            self.reset()
            for _ in range(self.args.emcts_bb_phases):
                self.emcts.search(state, self.nn, self.args.numMCTSSims/self.args.emcts_bb_phases, self.args.add_root_noise, self.args.add_root_temp)
                if not self.emcts.turn_completed: self.emcts.update_turn(state, self.temp)
        else:
            raise ValueError
        
        turn, pi, state_history = self.emcts.get_results(state)
        return turn
    
    def process(self, *args, **kwargs):
        return self.nn.process(*args, **kwargs)


############### RawMCTSPlayer ####################

class RawMCTSPlayer(MCTSPlayer):
    def __init__(self, strategy = "vanilla", *args, **kwargs):
        super().__init__(strategy, None, *args, **kwargs)
        self.strategy = strategy
        self.mode = "mcts"
        self.__class__.__name__ = f"RawMCTSPlayer(strategy = {strategy})"
        self._POLICY_SIZE = self.game_cls.action_size()
        self._POLICY_FILL_VALUE = 1 / self._POLICY_SIZE
        self._VALUE_SIZE = self.game_cls.num_players() + 1
    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return False

    def play(self, state) -> int:
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())  

        if self.strategy == "vanilla":
            self.reset()
            self.mcts.raw_search(state, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            while not self.mcts.turn_completed:
                self.mcts.update_turn(state, self.temp)

        elif self.strategy == "bridge-burning":
            self.reset()
            while not self.mcts.turn_completed:
                self.mcts.raw_search(state, self.args.numMCTSSims/state.avg_atomic_actions(), self.args.add_root_noise, self.args.add_root_temp)
                if not self.mcts.turn_completed: self.mcts.update_turn(state, self.temp)
        
        turn, pi, state_history = self.mcts.get_results(state)
        return turn
    
    def process(self, batch: torch.Tensor):
        return torch.full((batch.shape[0], self._POLICY_SIZE), self._POLICY_FILL_VALUE).to(batch.device), \
               torch.zeros(batch.shape[0], self._VALUE_SIZE).to(batch.device)



############### RawEMCTSPlayer ####################

class RawEMCTSPlayer(MCTSPlayer):
    def __init__(self, strategy = "vanilla", *args, **kwargs):
        super().__init__(strategy, None, *args, **kwargs)
        self.strategy = strategy
        self.mode = "emcts"
        self.__class__.__name__ = f"RawEMCTSPlayer(strategy = {strategy})"
        self._POLICY_SIZE = self.game_cls.action_size()
        self._POLICY_FILL_VALUE = 1 / self._POLICY_SIZE
        self._VALUE_SIZE = self.game_cls.num_players() + 1
    @staticmethod
    def supports_process() -> bool:
        return False

    @staticmethod
    def requires_model() -> bool:
        return False

    def update(self, state: GameState, action: int) -> None:
        self.emcts.update_root(state, action)

    def play(self, state) -> int:
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        if self.strategy == "vanilla":
            self.reset()
            self.emcts.raw_search(state, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            while not self.emcts.turn_completed:
                self.emcts.update_turn(state, self.temp)

        elif self.strategy == "bridge-burning":
            self.reset()
            for _ in range(self.args.emcts_bb_phases):
                self.emcts.raw_search(state, self.args.numMCTSSims/self.args.emcts_bb_phases, self.args.add_root_noise, self.args.add_root_temp)
                if not self.emcts.turn_completed: self.emcts.update_turn(state, self.temp)

        turn, pi, state_history = self.emcts.get_results(state)
        return turn

    def process(self, batch: torch.Tensor):
        return torch.full((batch.shape[0], self._POLICY_SIZE), self._POLICY_FILL_VALUE).to(batch.device), \
               torch.zeros(batch.shape[0], self._VALUE_SIZE).to(batch.device)
    



############### RawOSLA ####################
class RawOSLA(BasePlayer):
    """Simple player who always takes a win if presented, or blocks a loss if obvious, otherwise is random."""

    def __init__(self, game_cls, args, verbose=False):
        self.verbose = verbose
        self.args = args
        self.mode = "osla"
    def update(self, state: GameState, action: int) -> None:
        self.mcts.update_root(state, action)

    def reset(self):
        self.mcts = MCTS(self.args)
    
    @staticmethod
    def supports_process() -> bool:
        return False

    @staticmethod
    def requires_model() -> bool:
        return False
    
    def play(self, state: GameState) -> int:
        current_player = state._player
        turn = []
        while state._player == current_player and not state.win_state().any():
            action = self.play_atomic_action(state)
            state.play_action(action)
            turn.append(action)
        return turn
    
    def play_atomic_action(self, state: GameState) -> int:
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
            ret_move = np.random.choice(list(win_move_set)).item()
            if self.verbose:
                print('Playing winning action %s from %s' %
                      (ret_move, win_move_set))
        elif len(stop_loss_move_set) > 0:
            ret_move = np.random.choice(list(stop_loss_move_set)).item()
            if self.verbose:
                print('Playing loss stopping action %s from %s' %
                      (ret_move, stop_loss_move_set))
        elif len(fallback_move_set) > 0:
            ret_move = np.random.choice(list(fallback_move_set)).item()
            if self.verbose:
                print('Playing random action %s from %s' %
                      (ret_move, fallback_move_set))
        else:
            raise Exception('No valid moves remaining: %s' % state)

        return ret_move
