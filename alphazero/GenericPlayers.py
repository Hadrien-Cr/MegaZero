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
        while state._player == current_player:
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
    def __init__(self, nn: NNetWrapper, *args, print_policy=False,
                 average_value=False, draw_mcts=False, draw_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn
        self.temp = self.args.startTemp
        self.print_policy = print_policy
        self.average_value = average_value
        self.draw_mcts = draw_mcts
        self.draw_depth = draw_depth
        self.reset()
        if self.verbose:
            self.mcts.search(
                self.game_cls(), self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp
            )
            value = self.mcts.value(self.average_value)
            self.__rel_val_split = value if value > 0.5 else 1 - value
            print('initial value:', self.__rel_val_split)

    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return True

    def update(self, state: GameState, action: int) -> None:
        self.mcts.update_root(state, action)

    def reset(self):
        self.mcts = MCTS(self.args)

    def play(self, state) -> int:
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        
        if self.args.baseline_search_strategy == "VANILLA-MCTS":
            self.reset()
            self.mcts.search(state, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            while not self.mcts.turn_completed:
                self.mcts.search(state, self.nn, 1, self.args.add_root_noise, self.args.add_root_temp)
                self.mcts.update_turn(state, self.temp)

        elif self.args.baseline_search_strategy == "BB-MCTS":
            self.reset()
            while not self.mcts.turn_completed:
                self.mcts.search(state, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
                self.mcts.update_turn(state, self.temp)
        return self.mcts.action_history
    
    def process(self, *args, **kwargs):
        return self.nn.process(*args, **kwargs)



############### EMCTSPlayer ####################


class EMCTSPlayer(BasePlayer):
    def __init__(self, nn: NNetWrapper, *args, print_policy=False,
                 average_value=False, draw_mcts=False, draw_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = nn
        self.temp = self.args.startTemp
        self.print_policy = print_policy
        self.average_value = average_value
        self.draw_mcts = draw_mcts
        self.draw_depth = draw_depth
        self.reset()

    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return True

    def update(self, state: GameState, action: int) -> None:
        self.emcts.update_root(state, action)

    def reset(self):
        self.emcts = EMCTS(self.args)

    def play(self, state) -> int:
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())

        if self.args.baseline_search_strategy == "VANILLA-MCTS":
            self.reset()
            self.emcts.search(gs, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            current_player = gs._player

            for a in self.emcts.action_history:
                if current_player != gs.player or gs.win_state().any(): break
                gs.play_action(a)

        elif self.args.baseline_search_strategy == "BB-MCTS":
            gs = state.clone()
            self.reset()
            for _ in range(self.args.emcts_bb_phases):
                self.emcts.search(gs, self.nn, self.args.numMCTSSims/self.args.emcts_bb_phases, self.args.add_root_noise, self.args.add_root_temp)
                m = self.emcts._root.best_child(0, 0)
                self.emcts.update_root(gs, m)
            current_player = state._player

            for a in self.emcts.action_history:
                if current_player != gs.player or gs.win_state().any(): break
                gs.play_action(a)

        return self.emcts.action_history
    
    def process(self, *args, **kwargs):
        return self.nn.process(*args, **kwargs)


############### RawMCTSPlayer ####################

class RawMCTSPlayer(MCTSPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
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
        gs = state.clone()   

        if self.args.baseline_search_strategy == "VANILLA-MCTS":
            self.reset()
            self.mcts.raw_search(state, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            while not self.mcts.turn_completed:
                self.mcts.raw_search(state, 1, self.args.add_root_noise, self.args.add_root_temp)
                self.mcts.update_turn(state, self.temp)

        elif self.args.baseline_search_strategy == "BB-MCTS":
            self.reset()
            while not self.mcts.turn_completed:
                self.mcts.raw_search(state, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
                self.mcts.update_turn(state, self.temp)
        return self.mcts.action_history
    
    def process(self, batch: torch.Tensor):
        return torch.full((batch.shape[0], self._POLICY_SIZE), self._POLICY_FILL_VALUE).to(batch.device), \
               torch.zeros(batch.shape[0], self._VALUE_SIZE).to(batch.device)


class RawEMCTSPlayer(MCTSPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self._POLICY_SIZE = self.game_cls.action_size()
        self._POLICY_FILL_VALUE = 1 / self._POLICY_SIZE
        self._VALUE_SIZE = self.game_cls.num_players() + 1
    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return False

    def update(self, state: GameState, action: int) -> None:
        self.emcts.update_root(state, action)

    def reset(self):
        self.emcts = EMCTS(self.args)

    def play(self, state) -> int:
        self.temp = self.args.temp_scaling_fn(self.temp, state.turns, state.max_turns())
        gs = state.clone()
        
        if self.args.baseline_search_strategy == "VANILLA-MCTS":
            self.reset()
            self.emcts.raw_search(gs, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
            current_player = gs._player

            for a in self.emcts.action_history:
                if current_player != gs.player or gs.win_state().any(): break
                gs.play_action(a)

        elif self.args.baseline_search_strategy == "BB-MCTS":
            self.reset()
            for _ in range(self.args.emcts_bb_phases):
                self.emcts.raw_search(gs, self.args.numMCTSSims/self.args.emcts_bb_phases, self.args.add_root_noise, self.args.add_root_temp)
                m = self.emcts._root.best_child(0, 0)
                self.emcts.update_root(gs, m)
            current_player = state._player

            for a in self.emcts.action_history:
                if current_player != gs.player or gs.win_state().any(): break
                gs.play_action(a)

        return self.emcts.action_history
    def process(self, batch: torch.Tensor):
        return torch.full((batch.shape[0], self._POLICY_SIZE), self._POLICY_FILL_VALUE).to(batch.device), \
               torch.zeros(batch.shape[0], self._VALUE_SIZE).to(batch.device)