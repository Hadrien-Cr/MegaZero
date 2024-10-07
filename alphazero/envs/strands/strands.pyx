# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True

from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.strands.StrandsLogic import Board

import numpy as np

STRANDS_MODE = 'STRANDS_6'

if STRANDS_MODE == 'STRANDS_6':
    DEFAULT_WIDTH, DEFAULT_HEIGHT = 11, 11
    MAX_TURNS = 34
    NUM_PLAYERS = 2
    MULTI_PLANE_OBSERVATION, NUM_CHANNELS = True, 9
    DEFAULT_HEXES_TO_LABELS = np.array([[0, 0, 0, 0, 0, 6, 5, 5, 5, 5, 6],
                                        [0, 0, 0, 0, 5, 3, 3, 3, 3, 3, 5],
                                        [0, 0, 0, 5, 3, 2, 2, 2, 2, 3, 5],
                                        [0, 0, 5, 3, 2, 2, 2, 2, 2, 3, 5],
                                        [0, 5, 3, 2, 2, 2, 2, 2, 2, 3, 5],
                                        [6, 3, 2, 2, 2, 1, 2, 2, 2, 3, 6],
                                        [5, 3, 2, 2, 2, 2, 2, 2, 3, 5, 0],
                                        [5, 3, 2, 2, 2, 2, 2, 3, 5, 0, 0],
                                        [5, 3, 2, 2, 2, 2, 3, 5, 0, 0, 0],
                                        [5, 3, 3, 3, 3, 3, 5, 0, 0, 0, 0],
                                        [6, 5, 5, 5, 5, 6, 0, 0, 0, 0, 0]],
                                        dtype = np.intc)
    DEFAULT_HEXES_AVAILABLE = np.array([0, 1, 36, 24, 0, 24, 6],dtype = np.intc)

elif STRANDS_MODE == 'STRANDS_5':
    DEFAULT_WIDTH, DEFAULT_HEIGHT = 9, 9
    MAX_TURNS = 23
    NUM_PLAYERS = 2
    MULTI_PLANE_OBSERVATION, NUM_CHANNELS = True, 9
    DEFAULT_HEXES_TO_LABELS = np.array([[0, 0, 0, 0, 6, 4, 4, 4, 6],
                                        [0, 0, 0, 4, 3, 3, 3, 3, 4],
                                        [0, 0, 4, 3, 2, 2, 2, 3, 4],
                                        [0, 4, 3, 2, 2, 2, 2, 3, 4],
                                        [6, 3, 2, 2, 1, 2, 2, 3, 6],
                                        [4, 3, 2, 2, 2, 2, 3, 4, 0],
                                        [4, 3, 2, 2, 2, 3, 4, 0, 0],
                                        [4, 3, 3, 3, 3, 4, 0, 0, 0],
                                        [6, 4, 4, 4, 6, 0, 0, 0, 0]],
                                        dtype = np.intc)
    DEFAULT_HEXES_AVAILABLE = np.array([0, 1, 18, 18, 18, 0, 6],dtype = np.intc)

class Game(GameState):
    """
    Game class for the game of strands
    
    Please check the rules here https://boardgamegeek.com/boardgame/364343/strands
    
    The rules can be summarize as: each turn, 
        placing d tiles labeled d on the empty hexes corresponding to the digit chosen 
    (if only p<d such hexes are available, place d tiles; p is denoted "rest" in the code)

    """
    def __init__(self):
        super().__init__(board=self._get_board())
        self.micro_step = 1

    @staticmethod
    def _get_board():
        return Board(width=DEFAULT_WIDTH, 
                    default_hexes_to_labels = np.copy(DEFAULT_HEXES_TO_LABELS),
                    default_hexes_available = np.copy(DEFAULT_HEXES_AVAILABLE))

    def __hash__(self):
        return hash(self._board.hexes.tobytes() + bytes([self._turns]) + bytes([self._player]))

    def __eq__(self, other):
        return (self._player == other._player
                and self._turns == other._turns
                and self.micro_step == other.micro_step
                and np.array_equal(self._board.hexes, other._board.hexes)
                and self._board.digit_chosen == other._board.digit_chosen
                and self._board.rest == other._board.rest)

    def clone(self) -> 'Game':
        game = Game()
        game._board.hexes = np.copy(np.asarray(self._board.hexes))
        game._board.digit_chosen = self._board.digit_chosen
        game._board.rest = self._board.rest
        game._board.hexes_available = np.copy(np.asarray(self._board.hexes_available))

        game._player = self._player
        game._turns = self._turns
        game.micro_step = self.micro_step
        game.last_action = self.last_action

        return game

    @staticmethod
    def max_turns() -> int:
        return MAX_TURNS

    @staticmethod
    def has_draw() -> bool:
        return True

    @staticmethod
    def num_players() -> int:
        return NUM_PLAYERS

    @staticmethod
    def action_size() -> int:
        return DEFAULT_WIDTH ** 2

    @staticmethod
    def observation_size() -> tuple[int, int, int]:
        return NUM_CHANNELS, DEFAULT_HEIGHT, DEFAULT_WIDTH

    def valid_moves(self):
        valid = np.zeros((self.action_size(),), dtype=np.intc)
        if self._board.rest > 0:
            unoccupied_hexes =  np.where(np.asarray(self._board.hexes) == 0, 1, 0)
            valid_hexes = np.where(np.asarray(self._board.hexes_to_labels) == self._board.digit_chosen, 1, 0)
            valid = np.logical_and(valid_hexes, unoccupied_hexes).flatten()
        
        elif self._board.rest == 0:
            unoccupied_hexes =  np.where(np.asarray(self._board.hexes) == 0, 1, 0)
            valid_hexes = np.where(np.asarray(self._board.hexes_to_labels) != 0, 1, 0)
            valid = np.logical_and(valid_hexes, unoccupied_hexes).flatten()
        
        return valid

    def play_action(self, action: int) -> None:
        super().play_action(action)
        x, y = self.HexIndexingToXYIndexing(action)
        self._board.add_tile(x = x, y = y, target=(1, -1)[self.player])
        self.micro_step+= 1 
        
        if self._board.rest == 0:
            self._update_turn()

    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            hexes = np.asarray(self._board.hexes)
            player1 = np.where(hexes == 1, 1, 0)
            player2 = np.where(hexes == -1, 1, 0)
            empty = np.where(hexes == 0, 1, 0)

            colour = np.full_like(hexes, self.player)
            digit_chosen = np.full_like(hexes, self._board.digit_chosen, dtype=np.intc)
            rest = np.full_like(hexes, self._board.rest, dtype=np.intc)
            turn = np.full_like(hexes, self._turns / MAX_TURNS, dtype=np.intc)
            micro_step = np.full_like(hexes, self.micro_step, dtype=np.intc)

            return np.array([hexes, player1, player2, empty, 
                        digit_chosen, rest, colour, turn, micro_step], dtype=np.intc)

        else:
            return np.expand_dims(np.asarray(self._board.hexes), axis=0)

    def win_state(self) -> np.ndarray:
        if self._turns < MAX_TURNS:
            return np.array([False] * 3, dtype=np.uint8)

        areas_black, areas_white, areas_empty = self._board.compute_areas()

        for i in range(min(len(areas_black),len(areas_white))):
            if areas_white[i] < areas_black[i]:
                return np.array([True, False, False], dtype=np.uint8)  # win for player1 (black)
            elif areas_white[i] > areas_black[i]:
                return np.array([False, True, False], dtype=np.uint8)  # win for player2 (white)
        
        if len(areas_black)>len(areas_white):
            return np.array([True, False, False], dtype=np.uint8)  # win for player1 (black)
        elif len(areas_black)<len(areas_white):
            return np.array([False, True, False], dtype=np.uint8)  # win for player2 (white)
                
        return np.array([False, False, True], dtype=np.uint8)

    def XYIndexingToHexIndexing(self, x: int, y: int) -> int:
        return x * DEFAULT_WIDTH + y
    
    def HexIndexingToXYIndexing(self, hex: int):
        return hex//DEFAULT_WIDTH, hex%DEFAULT_WIDTH

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        
        # We use 4 out of the 6 axes of symmetries for simplicty
        data = [(self.clone(), pi)]

        # (top left / bot right) diagonal symmetry
        new_state = self.clone()
        new_state._board.hexes = np.transpose(self._board.hexes)
        new_pi = np.transpose(np.reshape(pi,(DEFAULT_WIDTH,DEFAULT_WIDTH))).flatten()
        data.append((new_state, new_pi))
        
        # (top right / bot left) diagonal symmetry
        new_state = self.clone()
        new_state._board.hexes = np.fliplr(np.transpose(np.fliplr(self._board.hexes)))
        new_pi = np.copy(pi)
        new_pi = np.fliplr(np.transpose(np.fliplr(np.reshape(pi, (DEFAULT_WIDTH, DEFAULT_WIDTH))))).flatten()
        data.append((new_state, new_pi))
        
        # center rotation
        new_state = self.clone()
        new_state._board.hexes = np.transpose(np.fliplr(np.transpose(np.fliplr(self._board.hexes))))
        new_pi = np.copy(pi)
        new_pi = np.transpose(np.fliplr(np.transpose(np.fliplr(np.reshape(pi, (DEFAULT_WIDTH, DEFAULT_WIDTH)))))).flatten()
        data.append((new_state, new_pi))

        return data

