from alphazero.MCTS import MCTS
from alphazero.Game import GameState
import numpy as np
from scipy import signal
from alphazero.NNetWrapper import BaseWrapper
from alphazero.GenericPlayers import NNPlayer, MCTSPlayer, EMCTSPlayer
import torch
class StrandsHeuristic(BaseWrapper):
    def __init__(self, game_cls, *args):
        super().__init__(game_cls, *args)

    def predict(self, obs: np.ndarray):
        state = self.game_cls()
        state._board.hexes = np.copy(obs[0,:])
        # Value estimate
        # heuritic for value estimation: (0.5 + 0.5*advantage/(1+sum(areas_empty))
        areas_black, areas_white, areas_empty = state._board.compute_areas()
        advantage = areas_black[0] - areas_white[0]
        value_black = 0.5 + (+ 0.5*advantage/(1+np.sum(areas_empty)))
        value_white = 0.5 +(- 0.5*advantage/(1+np.sum(areas_empty)))
        values = np.array([value_black, value_white, 0], dtype=np.float32)
        

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
        
        pi_hexes = signal.convolve2d(tiles, neighbour_filter, mode='same')
        pi = pi_hexes.flatten() + np.full(state.action_size(), 0.1)

        return np.asarray(pi, dtype=np.float32), values
    
    def load_checkpoint(self, folder, filename):
        pass
    def save_checkpoint(self, folder, filename):
        pass
    def train(self, *args, **kwargs):
        pass
    def process(self, batch: torch.Tensor):
        p = torch.zeros(batch.shape[0], self.game_cls.action_size()).to(batch.device)
        v = torch.zeros(batch.shape[0], 3).to(batch.device)
        for i in range(batch.shape[0]):
            p_i, v_i = self.predict(batch[i])
            p[i] = torch.Tensor(p_i)
            v[i] = torch.Tensor(v_i)
        return (p,v)
    
class StrandsHeuristicMCTS(MCTSPlayer):
    def __init__(self, strategy = "vanilla", *args, **kwargs):
        super().__init__(strategy, None, *args, **kwargs)
        self.nn = StrandsHeuristic(self.game_cls, args)
    def __repr__(self):
        return f"StrandsHeuristicMCTS(strategy = {self.strategy})"
    def supports_process(self):
        return False
class StrandsHeuristicEMCTS(EMCTSPlayer):
    def __init__(self, strategy = "vanilla", *args, **kwargs):
        super().__init__(strategy, None, *args, **kwargs)
        self.nn = StrandsHeuristic(self.game_cls, args)
    def __repr__(self):
        return f"StrandsHeuristicEMCTS(strategy = {self.strategy})"
    def supports_process(self):
        return False
class StrandsHeuristicOSLA(NNPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nn = StrandsHeuristic(self.game_cls, args)
    def supports_process(self):
        return False
    def __repr__(self):
        return "StrandsHeuristicOSLA"