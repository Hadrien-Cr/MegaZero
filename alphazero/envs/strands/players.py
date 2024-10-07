from alphazero.MCTS import MCTS
from alphazero.Game import GameState
from alphazero.GenericPlayers import BasePlayer
import numpy as np
from scipy import signal
from alphazero.NNetWrapper import BaseWrapper

class Heuristics(BaseWrapper):
    def __init__(self, game_cls, args):
        super().__init__(game_cls, args)

    def predict(self, obs: np.ndarray):
        state = self.game_cls()
        state._board.hexes = np.copy(obs[0,:])
        # Value estimate
        # heuritic for value estimation: (black - white) / (1+ empty)
        areas_black, areas_white, areas_empty = state._board.compute_areas()
        advantage = areas_black[0] - areas_white[0]
        value_black = (areas_black[0] - areas_white[0])/(1+np.sum(areas_empty))
        value_white = (areas_white[0] - areas_black[0])/(1+np.sum(areas_empty))
        values = np.array([value_black, value_white], dtype=np.float32)
        

        # Policy estimate
        # The hex probability is proportionnal to the number of neighbouring hexes of the correct color
        # This performed by computing the convolution between the board "black"(assuming the current player is the player black) 
        # and the filter ([0,1,1],[1,0,1],[1,1,0])
        pi = np.zeros(state.action_size(), dtype=np.float32)

        color = [1,-1][state._player] 
        tiles = np.where(np.asarray(state._board.hexes, dtype = np.float32) == color, 1, 0) 
        neighbour_filter = (1/6)*np.array([[0,1,1],
                                    [1,0,1],
                                    [1,1,0]], dtype=np.float32)
        
        pi_hexes = 0.1*np.asarray(state._board.hexes_to_labels) + signal.convolve2d(tiles, neighbour_filter, mode='same')
        pi = pi_hexes.flatten()
        
        return np.asarray(pi, dtype=np.float32), values
    
    def load_checkpoint(self, folder, filename):
        pass
    def save_checkpoint(self, folder, filename):
        pass
    def train(self, *args, **kwargs):
        pass
    
class MCTSPlayerWithHeuristics(BasePlayer):
    def __init__(self,  *args, print_policy=False,
                 average_value=False, draw_mcts=False, draw_depth=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp = self.args.startTemp
        self.print_policy = print_policy
        self.average_value = average_value
        self.draw_mcts = draw_mcts
        self.draw_depth = draw_depth
        self.nn = Heuristics(self.game_cls, self.args)
        self.reset()
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
            
            macro_action = []
            current_player = state._player
            gs = state.clone()

            self.mcts.search(gs, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)

            while gs._player == current_player:
                try:
                    policy = self.mcts.probs(gs, self.temp)
                    action = np.random.choice(len(policy), p=policy)
                    macro_action.append(action)
                    gs.play_action(action)
                    self.update(gs, action)
                except:
                    p, v = self.nn(gs.observation())
                    policy = gs.valid_moves()*p
                    policy/=np.sum(policy)
                    action = np.random.choice(len(policy), p=policy)
                    macro_action.append(action)
                    gs.play_action(action)
            return macro_action

        elif self.args.baseline_search_strategy == "BB-MCTS":
            self.reset()

            macro_action = []
            current_player = state._player
            gs = state.clone()

            while gs._player == current_player:
                self.mcts.search(gs, self.nn, self.args.numMCTSSims, self.args.add_root_noise, self.args.add_root_temp)
                policy = self.mcts.probs(gs, self.temp)
                action = np.random.choice(len(policy), p=policy)
                macro_action.append(action)
                gs.play_action(action)
                self.update(gs, action)
            return macro_action


class OneStepLookAheadPlayerWithHeuristics(BasePlayer):
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = Heuristics(self.game_cls, self.args)
        self.reset()
        
    @staticmethod
    def supports_process() -> bool:
        return True

    @staticmethod
    def requires_model() -> bool:
        return True
    
    def one_step_look_ahead(self, state): 
        best_action, best_value = 0, -np.inf
        for action,is_valid in enumerate(state.valid_moves()):
            if is_valid:
                gs = state.clone()
                gs.play_action(action)
                pi, value = self.nn(state.observation())
                if value[state._player] > best_value:
                    best_action, best_value = action, value[state._player]
        
        return best_action, np.zeros(state.action_size(), dtype=np.float32)
    
    def play(self, state) -> int:
        if not self.args.macro_act:
            action, pi = self.one_step_look_ahead(state)
            return action

        elif self.args.macro_act:
            macro_action = []
            current_player = state._player
            gs = state.clone()
            while gs._player == current_player:
                action, pi = self.one_step_look_ahead(gs)
                macro_action.append(action)
                gs.play_action(action)
            return macro_action

