from alphazero.MCTS import MCTS
from alphazero.Game import GameState
import numpy as np
from scipy import signal
from alphazero.NNetWrapper import BaseWrapper
from alphazero.GenericPlayers import MCTSPlayer, NNPlayer
import torch
class StrandsHeuristics(BaseWrapper):
    def __init__(self, game_cls, *args):
        super().__init__(game_cls, *args)

    def predict(self, obs: np.ndarray):
        state = self.game_cls()
        state._board.hexes = np.copy(obs[0,:])
        # Value estimate
        # heuritic for value estimation: (black - white) / (1+ empty)
        areas_black, areas_white, areas_empty = state._board.compute_areas()
        advantage = areas_black[0] - areas_white[0]
        value_black = (areas_black[0] - areas_white[0])/(1+np.sum(areas_empty))
        value_white = (areas_white[0] - areas_black[0])/(1+np.sum(areas_empty))
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
        
        pi_hexes = 0.1*np.asarray(state._board.hexes_to_labels) + signal.convolve2d(tiles, neighbour_filter, mode='same')
        pi = pi_hexes.flatten()
        
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
    
class StrandsHeursiticMCTS(MCTSPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.nn = StrandsHeuristics(self.game_cls, args)
    
class StrandsHeursiticOSLA(NNPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.nn = StrandsHeuristics(self.game_cls, args)
