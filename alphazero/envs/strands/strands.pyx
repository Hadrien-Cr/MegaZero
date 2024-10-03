# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True

from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.strands.StrandsLogic import Board

import numpy as np

DEFAULT_WIDTH = 11
DEFAULT_HEIGHT = 11
MAX_TURNS = 34
NUM_PLAYERS = 2
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 9
class Game(GameState):
    """
    Game class for the game of strands
    
    Please check the rules here https://boardgamegeek.com/boardgame/364343/strands
    
    A macro action is made of 2 phases:
    - picking a digit d (from 1 to 6)
    - placing d tiles labeled d on the empty hexes corresponding to the digit chosen 
    (if only p<d such hexes are available, place d tiles; p is denoted "rest" in the code)

    """
    def __init__(self):
        super().__init__(board=self._get_board())
        self.micro_step = 1

    @staticmethod
    def _get_board():
        return Board(width=DEFAULT_WIDTH)

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
        return 7 + DEFAULT_WIDTH ** 2

    @staticmethod
    def observation_size() -> tuple[int, int, int]:
        return NUM_CHANNELS, DEFAULT_HEIGHT, DEFAULT_WIDTH

    def valid_moves(self):
        valid = np.zeros((7 + DEFAULT_WIDTH**2), dtype=np.intc)

        if self.micro_step == 0:  # Have to Select A Digit
            if self._board.digit_chosen != 0:
                raise ValueError(f"Cannot perform digit picking, digit_chosen already set to {self._board.digit_chosen}")
            valid_digits = np.where(np.asarray(self._board.hexes_available) > 0, 1, 0)
            valid[0:7] = valid_digits

        elif self.micro_step < 7:
            if self._board.rest == 0:
                valid[0] = 1
            else:
                unoccupied_hexes =  np.where(np.asarray(self._board.hexes) == 0, 1, 0)
                valid_hexes = np.where(np.asarray(self._board.hexes_to_labels) == self._board.digit_chosen, 1, 0)
                valid[7::] = np.logical_and(valid_hexes, unoccupied_hexes).flatten()
        else:
            raise ValueError(f"Invalid micro_step, got {self.micro_step}, should be reset to 0")

        return valid
    def play_action(self, action: int) -> None:
        super().play_action(action)

        # Skip Action
        if action == 0:
            if self._board.digit_chosen != 0:
                raise ValueError("Cannot perform skip if the macro-action is already completed")

        # Select A Digit
        elif action <= 6:
            if self.micro_step != 0:
                raise ValueError(f"Cannot perform digit picking other than micro-step 0, got micro-step {self.micro_step}")
            if self._board.digit_chosen != 0:
                raise ValueError("Cannot overwrite digit chosen")

            self._board.update_digit_chosen(new_digit=action)
            self.micro_step+= 1 

        # Place Tiles
        elif action >= 7:
            hex = action - 7
            x, y = self.HexIndexingToXYIndexing(hex)
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
        new_pi = np.copy(pi)
        new_pi[7:] = np.transpose(np.reshape(new_pi[7:],(DEFAULT_WIDTH,DEFAULT_WIDTH))).flatten()
        data.append((new_state, new_pi))
        
        # (top right / bot left) diagonal symmetry
        new_state = self.clone()
        new_state._board.hexes = np.fliplr(np.transpose(np.fliplr(self._board.hexes)))
        new_pi = np.copy(pi)
        new_pi[7:] = np.fliplr(np.transpose(np.fliplr(np.reshape(new_pi[7:], (DEFAULT_WIDTH, DEFAULT_WIDTH))))).flatten()
        data.append((new_state, new_pi))
        
        # center rotation
        new_state = self.clone()
        new_state._board.hexes = np.transpose(np.fliplr(np.transpose(np.fliplr(self._board.hexes))))
        new_pi = np.copy(pi)
        new_pi[7:] = np.transpose(np.fliplr(np.transpose(np.fliplr(np.reshape(new_pi[7:], (DEFAULT_WIDTH, DEFAULT_WIDTH)))))).flatten()
        data.append((new_state, new_pi))

        return data

