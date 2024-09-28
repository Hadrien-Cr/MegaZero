import pytest
import numpy as np
from alphazero.envs.strands.strands import Game, DEFAULT_WIDTH, MAX_TURNS
import dill as pickle
'''
Run this test: 
python3 -m alphazero.envs.strands.test_strands
'''

# Helper function for initializing the board with a series of moves
def init_board_from_moves(moves):
    game = Game()
    for move in moves:
        game.play_action(move)
    return game

# Helper function for initializing the board from a given 2D array
def init_board_from_array(board_array):
    game = Game()
    game._board.hexes = np.array(board_array, dtype=np.intc).flatten()
    return game

# Test 1: Simple dynamics check
def test_simple_moves():
    game = Game()
    expected = np.zeros((DEFAULT_WIDTH,DEFAULT_WIDTH))
    
    assert np.array_equal(game._board.hexes , expected.flatten())


    game = init_board_from_moves([7 + 5*11 + 4]) # plays an hex on the spot (x = 5, y 4)
    expected = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
    assert np.array_equal(game._board.hexes , expected.flatten())

# Test 2: Overlapping Tiles and catching an error
def test_overlap_tiles():
    game = init_board_from_moves([   7 + 5*11 + 4, 0, 0, 0, 0, 0, 
                                  1, 7 + 5*11 + 5, 0, 0, 0, 0, 0,])

    with pytest.raises(ValueError):
        game.play_action(2) 
        game.play_action(7 + 5*11 + 4)  # Should raise error when overlap

# Test 3: Valid moves
def test_get_valid_moves():
    game = Game()
    valid_moves = game.valid_moves()
    expected_valid_moves = np.array([False for _ in range(7)] + [game._board.hexes_to_labels[hex] == 2 for hex in range(11*11)], dtype=np.intc)
    assert np.array_equal(valid_moves, expected_valid_moves)

# Test 4: Symmetries of the board
def test_symmetries():
    pass


# Test 5: Game end detection
def test_game_ended():
    # Initial State
    game = Game()
    assert np.array_equal(game.win_state(), np.array([False, False, False], dtype = np.intc)) # Player 1 wins with a diagonal

    # Winning terminal state for white (player2)
    reference = np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1,-1,-1,-1,-1,-1, 1],
        [0, 0, 0, 1,-1,-1,-1,-1,-1,-1, 1],
        [0, 0, 1,-1,-1,-1,-1,-1,-1,-1, 1],
        [0, 1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 1],
        [1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 0],
        [1,-1,-1,-1,-1,-1,-1,-1, 1, 0, 0],
        [1,-1,-1,-1,-1,-1,-1, 1, 0, 0, 0],
        [1,-1,-1,-1,-1,-1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
        dtype = np.intc)

    game = init_board_from_array(reference)
    game._turns = MAX_TURNS
    assert np.array_equal(game.win_state(), np.array([False, True, False], dtype = np.intc)) # Player 2 wins
    
    # Winning terminal state for black (player1)
    game = init_board_from_array(-reference)
    game._turns = MAX_TURNS
    assert np.array_equal(game.win_state(), np.array([True, False, False], dtype = np.intc)) # Player 1 wins

# Test 6: Immutable move check
def test_immutable_move():
    game = Game()
    
    clone_game = game.clone()
    assert game.__eq__(clone_game)
    game.play_action(7 + 6*11 + 5)

    assert not game.__eq__(clone_game)
    assert np.array_equal(clone_game._board.hexes, game._board.hexes) == False  # Board should have changed
    assert np.array_equal(clone_game._board.hexes_available, game._board.hexes_available) == False 
    assert np.array_equal(clone_game._board.digit_chosen, game._board.digit_chosen) == False 
    assert np.array_equal(clone_game._board.tiles_left_to_place, game._board.tiles_left_to_place) == False 
    assert np.array_equal(clone_game._board.hexes_available, game._board.hexes_available) == False 

# Test 7: Random Rollout
def test_rollout():
    game = Game()
    while np.array_equal(game.win_state(), np.array([False, False, False])):
        valid_actions = game.valid_moves()
        true_indices = np.where(valid_actions)[0]
        action = np.random.choice(true_indices).item()
        game.play_action(action)

# Test 8: Observation
def check_observation():
    game = Game()
    obs = game.observation()
    assert obs.shape == game.observation_size()
    
# Test 9: Pickleability
def is_pickleable():
    game = Game()
    assert pickle.pickles(game)
    assert pickle.dumps(game)
    assert pickle.pickles(game._board)
    assert pickle.dumps(game._board)

if __name__ == "__main__":
    pytest.main()