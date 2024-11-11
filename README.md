# MegaZero General
This is an adaptation of AlphaZero based on the following repositories:

* The original repo: https://github.com/suragnair/alpha-zero-general

The purpose of the code is to play multi-action games with the AlphaZero framework.

### How to use:

Use **`envs/generic`** to learn how to make program your own multi-action game

To train it, run **`python3 -m alphazero.envs.<env_name>.train --algorithm <alg>`** 
Games implemented are: Connect4, Connect4d, and Strands

### Description of some hyperparameters

**`mode, strategy`**: These are the mode and strategy of the MCTSPlayer used for self play. The mode is either "mcts" or "emcts", and the strategy is either "vanilla" or "bridge_burning".

**`emcts_bb_phases`**: Number of phases for the EMCTS bridge-burning strategy.

**`emcts_horizon`:** The size of the horizon for the EMCTS.

**`workers`:** Number of processes used for self play, Arena comparison, and training the model. Should generally be set to the number of processors - 1.

**`process_batch_size`:** The size of the batches used for batching MCTS during self play. Equivalent to the number of games that should be played at the same time in each worker. For exmaple, a batch size of 128 with 4 workers would create 128\*4 = 512 total games to be played simultaneously.

**`minTrainHistoryWindow`, `maxTrainHistoryWindow`, `trainHistoryIncrementIters`:** The number of past iterations to load self play training data from. Starts at min and increments once every `trainHistoryIncrementIters` iterations until it reaches max.

**`max_moves`:** Number of moves in the game before the game ends in a draw (should be implemented manually for now in getGameEnded of your Game class, automatic draw is planned). Used for the calculation of `default_temp_scaling` function.

**`num_stacked_observations`:** The number of past observations from the game to stack as a single observation. Should also be done manually for now, but take a look at how it was implemented in `alphazero/envs/tafl/tafl.pyx`.

**`numWarmupIters`:** The number of warm up iterations to perform. Warm up is self play but with random policy and value to speed up initial generations of self play data. Should only be 1-3, otherwise the neural net is only getting random moves in games as training data. This can be done in the beginning because the model's actions are random anyways, so it's for performance.

**`skipSelfPlayIters`:** The number of self play data generation iterations to skip. This assumes that training data already exists for those iterations can be used for training. For example, useful when training is interrupted because data doesn't have to be generated from scratch because it's saved on disk.

**`symmetricSamples`:** Add symmetric samples to training data from self play based on the user-implemented method `symmetries` in their Game class. Assumes that this is implemented. For example, in Viking Chess, the board can be rotated 90 degrees and mirror flipped any number of times while still being a valid game state, therefore this can be used for more training data.

**`numMCTSSims`:** Number of Monte Carlo Tree Search simulations to execute for each move in self play. A higher number is much slower, but also produces better value and policy estimates.

**`max_gating_iters`:** If a model doesn't beat its own past iteration this many times, then gating is temporarily reset and the model is allowed to move on to the next iteration. Use `None` to disable this feature.

**`min_next_model_winrate`:** The minimum win rate required for the new iteration against the last model in order to move on. If it doesn't beat this number, the previous model is used again (model gating).

**`cpuct`:** A constant for balancing exploration vs exploitation in the MCTS algorithm. A higher number promotes more exploration of new actions whereas a lower one promotes exploitation of previously known good actions. A normal range is between 1-4, depending on the environment; a game with less possible moves on each turn would need a lower CPUCT.

**`fpu_reduction`:** "First Play Urgency" reduction decreases the initialization Q value of an unvisited node by this factor, must be in the range `[-1, 1]`. The closer this value is to 1, it discourages MCTS to explore unvisited nodes further, which (hopefully) allows it to explore paths that are more familiar. If this is set to 0, no reduction is done and unvisited nodes inherit their parent's Q value. Closer to a value of -1 (not recommended to go below 0), unvisited nodes become more prefered which can lead to more exploration.


