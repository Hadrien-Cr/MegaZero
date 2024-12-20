# cython: language_level=3
# cython: auto_pickle=True
# cython: profile=True
from typing import List, Tuple, Any
from alphazero.EMCTS import Mutation
from alphazero.Game import GameState
from alphazero.envs.connect4d.Connect4dLogic import Board

import numpy as np

NUM_BOARDS = 5 # Number of boards in the stacked environment, also equals to d
DEFAULT_HEIGHT = 6
DEFAULT_WIDTH = 7
DEFAULT_WIN_LENGTH = 4
NUM_PLAYERS = 2
MAX_TURNS = 42
MULTI_PLANE_OBSERVATION = True
NUM_CHANNELS = 3 + 3*NUM_BOARDS if MULTI_PLANE_OBSERVATION else NUM_BOARDS


class Game(GameState):
    def __init__(self):
        super().__init__(self._get_board(), d = NUM_BOARDS)

    @staticmethod
    def _get_board():
        return Board(height = DEFAULT_HEIGHT, 
                    width = DEFAULT_WIDTH, 
                    win_length = DEFAULT_WIN_LENGTH, 
                    num_boards = NUM_BOARDS)

    def __hash__(self) -> int:
        return hash(self._board.pieces.tobytes() + bytes([self._turns, self._player, self.winner]))

    def __eq__(self, other: 'Game') -> bool:
        return (self._player == other._player 
                and self._turns == other._turns 
                and self._board == other._board)


    def clone(self) -> 'Game':
        game = Game()
        game._board.pieces = np.copy(np.asarray(self._board.pieces))
        game._board.winner = self._board.winner
        game._player = self._player
        game._turns = self._turns
        game.micro_step = self.micro_step
        game.last_action = self.last_action
        
        return game
    
    @staticmethod
    def avg_atomic_actions() -> int:
        return NUM_BOARDS

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
        
        self.micro_step+= 1 
        if self.micro_step == NUM_BOARDS:
            self._update_turn()

    def win_state(self) -> np.ndarray:
        game_over, player = self._board.get_win_state(sub_board=self.micro_step)

        if game_over:
            result = [False] * 3
            index = -1
            if player == 1:
                index = 0
            elif player == -1:
                index = 1
            elif player == 0:
                index = 2
            result[index] = True
            return np.array(result, dtype=np.uint8)

        return np.array([False] * 3, dtype=np.uint8)
    
    def observation(self):
        if MULTI_PLANE_OBSERVATION:
            pieces = np.asarray(self._board.pieces)
            player1 = np.where(pieces == 1, 1, 0)
            player2 = np.where(pieces == -1, 1, 0)

            colour = np.full_like(pieces[0], (1, -1)[self.player])
            turn = np.full_like(pieces[0], self._turns / MAX_TURNS, dtype=np.intc)
            micro_step = np.full_like(pieces[0], self.micro_step, dtype=np.intc)

            colour = np.expand_dims(colour, axis = 0)
            turn = np.expand_dims(turn, axis = 0)
            micro_step = np.expand_dims(micro_step, axis = 0)

            return np.concatenate([pieces, player1, player2, 
                                colour, turn, micro_step], axis = 0)

        else:
            return np.expand_dims(np.asarray(self._board.pieces), axis=0)

    def symmetries(self, pi) -> List[Tuple[Any, int]]:
        new_state = self.clone()
        new_state._board.pieces = self._board.pieces[:, :, ::-1]
        return [(self.clone(), pi), (new_state, pi[::-1])]


    def __str__(self):
        s = f"Player: {self._player} Turn: {self.turns} \n"
        for sub_board in range(NUM_BOARDS):
            for r in range(DEFAULT_HEIGHT):
                s += " ".join([['_', '0', '1' ][self._board.pieces[sub_board][r][c]] for c in range(DEFAULT_WIDTH)])
                if sub_board == self.micro_step and r == DEFAULT_HEIGHT - 1:
                    s+= "  <--"
                s+= "\n"
            s+= "\n"
        return(s)