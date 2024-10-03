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
    game._board.hexes = np.array(board_array, dtype=np.intc)
    return game

# Test 1: Simple dynamics check
def test_simple_moves():
    game = Game()
    expected = np.zeros((DEFAULT_WIDTH,DEFAULT_WIDTH))
    
    assert np.array_equal(game._board.hexes , expected)


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
    
    assert np.array_equal(game._board.hexes , expected)

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
    expected_valid_moves = np.array([False for _ in range(7)] + [game._board.hexes_to_labels[x][y] == 2 for y in range(11) for x in range(11)], dtype=np.intc)
    assert np.array_equal(valid_moves, expected_valid_moves)

# Test 4: Symmetries of the board
def test_symmetries():
    reference_1 = np.array([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype = np.intc)
    
    reference_2 = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        dtype = np.intc)
    
    reference_3 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
        dtype = np.intc)
    
    reference_4 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
        dtype = np.intc)
        
    game = init_board_from_array(reference_1)
    pi = np.zeros((7+DEFAULT_WIDTH**2), dtype = np.intc) 
    data = game.symmetries(pi)

    print(np.reshape(data[2][0]._board.hexes, (DEFAULT_WIDTH, DEFAULT_WIDTH)))

    print(np.reshape(data[3][0]._board.hexes, (DEFAULT_WIDTH, DEFAULT_WIDTH)))

    assert np.array_equal(data[0][0]._board.hexes, reference_1)
    assert np.array_equal(data[1][0]._board.hexes, reference_2)
    assert np.array_equal(data[2][0]._board.hexes, reference_3)
    assert np.array_equal(data[3][0]._board.hexes, reference_4)

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
        [1,-1,-1,-1,-1, 1,-1,-1,-1,-1, 1],
        [1,-1,-1,-1,-1,-1,-1,-1,-1, 1, 0],
        [1,-1,-1,-1,-1,-1,-1,-1, 1, 0, 0],
        [1,-1,-1,-1,-1,-1,-1, 1, 0, 0, 0],
        [1,-1,-1,-1,-1,-1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]],
        dtype = np.intc)

    game = init_board_from_array(reference)
    game._turns = MAX_TURNS
    areas_black, areas_white, areas_empty = game._board.compute_areas()
    assert areas_empty == [0]
    assert areas_white == [60, 0]
    assert areas_black == [30, 1, 0]
    assert np.array_equal(game.win_state(), np.array([False, True, False], dtype = np.intc)) # Player 2 wins


# Test 6: Immutable move check
def test_immutable_move():
    game = Game()
    
    clone_game = game.clone()
    assert game.__eq__(clone_game)
    assert np.array_equal(clone_game._board.hexes, game._board.hexes)   # Board should not have changed
    assert np.array_equal(clone_game._board.hexes_available, game._board.hexes_available) 
    assert np.array_equal(clone_game._board.digit_chosen, game._board.digit_chosen) 
    assert np.array_equal(clone_game._board.rest, game._board.rest) 
    assert np.array_equal(clone_game._board.hexes_available, game._board.hexes_available)

    game.play_action(7 + 6*11 + 5)

    assert not game.__eq__(clone_game)
    assert np.array_equal(clone_game._board.hexes, game._board.hexes) == False  # Board should have changed
    assert np.array_equal(clone_game._board.hexes_available, game._board.hexes_available) == False 
    assert np.array_equal(clone_game._board.digit_chosen, game._board.digit_chosen) == False 
    assert np.array_equal(clone_game._board.rest, game._board.rest) == False 
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

# Test 10: Heuristic Agent
def test_heuristic_agent():
    game = Game()
    from alphazero.envs.strands.players import MCTSPlayerWithHeuristics, OneStepLookAheadPlayerWithHeuristics
    from alphazero.envs.strands.train import args
    import alphazero.Coach as c
    args =  c.get_args(args)
    args['_num_players'] = 2
    args['macro_act'] = True

    # Test MCTSPlayerWithHeuristics
    agent1 = MCTSPlayerWithHeuristics(Game,args)
    M = agent1.play(game)
    for m in M:
        game.play_action(m)
    
    # Test MCTSPlayerWithHeuristics
    agent2 = OneStepLookAheadPlayerWithHeuristics(Game,args)
    M = agent2.play(game)
    for m in M:
        game.play_action(m)

if __name__ == "__main__":
    pytest.main()