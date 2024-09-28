# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any

from alphazero.Game import GameState
from alphazero.envs.connect4d.Connect4dLogic import Board

import numpy as np

NUM_BOARDS = 7 # Number of boards in the stacked environment, also equals to d
DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4
NUM_PLAYERS = 2
MAX_TURNS = 42
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 3 + 3*NUM_BOARDS if MULTI_PLANE_OBSERVATION else NUM_BOARDS


class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board(), d=NUM_BOARDS)

    @staticmethod
    def _get_board():
        return Board(height = DEFAULT_HEIGHT, 
                              width = DEFAULT_WIDTH, 
                              win_length = DEFAULT_WIN_LENGTH, 
                              num_boards = NUM_BOARDS)

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self._turns]) + bytes([self._player]))

    def __eq__(self, other: 'Game') -> bool:
        return (self._player == other._player 
                and self._turns == other._turns 
                and self.micro_step == other.micro_step 
                and np.array_equal(self._board.pieces,other._board.pieces))


    def clone(self) -> 'Game':
        game = Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
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
        return DEFAULT_WIDTH

    @staticmethod
    def observation_size() -> Tuple[int, int, int]:
        return NUM_CHANNELS, DEFAULT_HEIGHT, DEFAULT_WIDTH

    def valid_moves(self):
        return np.asarray(self._board.get_valid_moves(sub_board=self.micro_step))

    def play_action(self, action: int) -> None:
        super().play_action(action)
        self._board.add_stone(action, (1, -1)[self.player], sub_board=self.micro_step)
        self._update_turn()

    def win_state(self) -> np.ndarray:
        for i in range(NUM_BOARDS):
            game_over, player = self._board.get_win_state(sub_board=self.micro_step)

            if game_over and player!= 0:
                result = [False] * 3
                index = -1
                if player == 1:
                    index = 0
                elif player == -1:
                    index = 1
                result[index] = True
                return np.array(result, dtype=np.uint8)
            
            elif game_over and i == NUM_BOARDS - 1:
                result = [False, False, True] 
                return np.array(result, dtype=np.uint8)

        return np.array([False] * 3, dtype=np.uint8)
    
    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            player1 = np.where(pieces == 1, 1, 0)
            player2 = np.where(pieces == -1, 1, 0)

            colour = np.full_like(pieces[0], self.player)
            turn = np.full_like(pieces[0], self._turns / MAX_TURNS, dtype=np.intc)
            micro_step = np.full_like(pieces[0], self.micro_step, dtype=np.intc)

            colour = np.expand_dims(colour, axis = 0)
            turn = np.expand_dims(turn, axis = 0)
            micro_step = np.expand_dims(micro_step, axis = 0)

            return np.concatenate([pieces, player1, player2, colour, turn, micro_step], axis = 0)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        new_state = self.clone()
        new_state._board.pieces = self._board.pieces[:, :, ::-1]
        return [(self.clone(), pi), (new_state, pi[::-1])]


def display(board, action=None):
    if action:
        print(f'Action: {action}, Move: {action + 1}')
    print(" -----------------------")
    #print(' '.join(map(str, range(len(board[0])))))
    print(board)
    print(" -----------------------")