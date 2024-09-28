"""
import pytest
import numpy as np
from alphazero.envs.generic.generic import Game
import dill as pickle


# Test 1: Simple dynamics check
def test_simple_moves():
    pass

# Test 2: Overlapping Tiles and catching an error
def test_overlap_tiles():
    pass

# Test 3: Valid moves
def test_get_valid_moves():
    pass

# Test 4: Symmetries of the board
def test_symmetries():
    pass

# Test 5: Game end detection
def test_game_ended():
    pass

# Test 6: Immutable move check
def test_immutable_move():
    pass

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

"""