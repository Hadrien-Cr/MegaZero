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
NUM_CHANNELS = 8

assert(DEFAULT_HEIGHT == DEFAULT_WIDTH and DEFAULT_HEIGHT % 2 == 1)

class Game(GameState):
    def __init__(self):
        super().__init__(board=self._get_board(), d=7)
        self.micro_step = 1

    @staticmethod
    def _get_board():
        return Board(width=DEFAULT_WIDTH)

    def __str__(self):
        return str(self.tiles)

    def __hash__(self):
        return hash(self.hexes.tobytes())

    def __eq__(self, other):
        return (self._player == other._player
                and self._turns == other._turns
                and self.micro_step == other.micro_step
                and self._board.hexes == other._board.hexes
                and self._board.digit_chosen == other._board.digit_chosen)

    def clone(self) -> 'Game':
        game = Game()
        game._board.hexes = np.copy(np.asarray(self._board.hexes))
        game._player = self._player
        game._turns = self._turns
        game.micro_step = self.micro_step
        game.last_action = self.last_action
        game._board.digit_chosen = self._board.digit_chosen
        game._board.tiles_left_to_place = self._board.tiles_left_to_place
        game._board.hexes_available = np.copy(self._board.hexes_available)
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
        if self.micro_step == 0:  # Have to Select A Digit
            if self._board.digit_chosen != 0:
                raise ValueError(f"Cannot perform digit picking, digit_chosen already set to {self._board.digit_chosen}")
            return self._board.get_digits_available()

        elif self.micro_step < 7:
            if self._board.tiles_left_to_place == 0:
                return self._board.get_skip_action_only()
            else:
                return self._board.get_hexes_available()

        else:
            raise ValueError(f"Invalid micro_step, got {self.micro_step}, should be reset to 0")

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

        # Place Tiles
        elif action >= 7:
            hex = action - 7
            self._board.add_tile(hex=hex, target=(1, -1)[self.player])

        self._update_turn()

    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            tiles = np.asarray(self._board.hexes)
            player1 = np.where(tiles == 1, 1, 0)
            player2 = np.where(tiles == -1, 1, 0)
            empty = np.where(tiles == 0, 1, 0)

            colour = np.full_like(tiles, self.player)
            digit_chosen = np.full_like(tiles, self._board.digit_chosen, dtype=np.intc)
            digit_left_to_place = np.full_like(tiles, self._board.tiles_left_to_place, dtype=np.intc)
            turn = np.full_like(tiles, self._turns / MAX_TURNS, dtype=np.intc)
            micro_step = np.full_like(tiles, self.micro_step, dtype=np.intc)

            return np.array([tiles, player1, player2, empty, digit_chosen, digit_left_to_place, colour, turn, micro_step], dtype=np.intc)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def win_state(self) -> np.ndarray:
        if self._turns < MAX_TURNS:
            return np.array([False] * 3, dtype=np.intc)

        areas_white = self._board.compute_areas(target=-1)
        areas_black = self._board.compute_areas(target=1)

        for i in range(len(areas_white)):
            if areas_white[i] < areas_black[i]:
                return np.array([True, False, False], dtype=np.intc)  # win for player1 (black)
            elif areas_white[i] > areas_black[i]:
                return np.array([False, True, False], dtype=np.intc)  # win for player2 (white)

        return np.array([False, False, True], dtype=np.intc)

    def XYIndexingToHexIndexing(self, x: int, y: int) -> int:
        return x * self._board.width + y
