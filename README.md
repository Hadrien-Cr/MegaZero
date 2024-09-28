# AlphaZero General
This is an implementation of AlphaZero based on the following repositories:

* The original repo: https://github.com/suragnair/alpha-zero-general
* A fork of the original repo: https://github.com/bhansconnect/fast-alphazero-general

This project is still work-in-progress, so expect frequent fixes, updates, and much more detailed documentation soon.

You may join the [Discord server](https://discord.gg/MVaHwGZpRC) if you wish to join the community and discuss this project, ask questions, or contribute to the framework's development.

### Current differences from the above repos
1. **Cython:** The most computationally intensive components are written in Cython to be compiled for a runtime speedup of [up to 30x](https://towardsdatascience.com/use-cython-to-get-more-than-30x-speedup-on-your-python-code-f6cb337919b6) compared to pure python.
2. **GUI:** Includes a graphical user interface for easier training and arena comparisons. It also allows for games to be played visually (agent-agent, agent-human, human-human) instead of through a command line interface (work-in-progress). Custom environments must implement their own GUI naturally.
3. **Node-based MCTS:** Uses a better implementation of MCTS that uses nodes instead of dictionary lookups. This allows for a huge increase in performance and much less RAM usage than what the previous implementation used, about 30-50% speed increase and 95% less RAM usage from experimental data. The base code for this was provided by [bhandsconnect](https://github.com/bhansconnect).
4. **Model Gating:** After each iteration, the model is compared to the previous iteration. The model that performs better continues forward based on an adjustable minimum winrate parameter.
5. **Batched MCTS:** [bhandsconnect's repo](https://github.com/bhansconnect/fast-alphazero-general) already includes this for self play, but it has been expanded upon to be included in Arena for faster comparison of models.
6. **N-Player Support:** Any number of players are supported! This allows for training on a greater variety of games such as many types of card games or something like Catan.
7. **Warmup Iterations:** A few self play iterations in the beginning of training can optionnally be done using random policy and value to speed up initial generation of training data instead of using a model that is initally random anyways. This makes these iterations purely CPU-bound.
8. **Root Dirichlet Noise & Root Temperature, Discount:** Allows for better exploration and MCTS doesn't get stuck in local minima as often. Discount allows AlphaZero to "understand" the concept of time and chooses actions which lead to a win more quickly/efficiently as opposed to choosing a win that would occur later on in the game.
9. **More Adjustable Parameters:** This implementation allows for the modification of numerous hyperparameters, allowing for substantial control over the training process. More on hyperparameters below where the usage of some are discussed.

### Structure of the code 
To run a training use the command 
`python3 -m alphazero.envs.<env-name>.train`
To run the test for the environments:
`python3 -c 'import pytest; pytest.main()'`
You can change the training arguments in the file `alphazero/envs/<env-name>/train.py`

**`Game.py`:** the class definition for the adversarial game. It only has argument `num_players`and `d`. `d` denotes the *depth of turn* ie the number of actions that a player has to perform to complete his turn (`d = 1`  for non multi-action game)

**`envs`:** the folder that contains the **`Game`** different `game_cls` implementations. See connect4 for a simple example. In general, use a `_board` class to implement the game logic. See `connect4` for a simple exmaple, and edit the `generic` to implement your own game.

**`GenericPlayers.py`:** the script that define the player classes 
- RandomPlayer, 
- NNPlayer (Acts with the given policy network), 
- MCTSPlayer (Acts with MCTS guide by NN policy and value network) 
- RawMCTSPlayer (Acts with MCTS guide by default uniform policy an value), NNPlayer

**`NNetWrapper.py (as nn)`:** the script that defines the neural nets (policy and value net included, usable with `predict` method)

**`SelfPlayAgents.py`:** the script that performs the self-play. It creates multiple instances of MCTSPlayer with the current network. 
Is is used for the mode training `_is_arena = False` or for the mode arena `_is_arena = False`. Mode Arena means that the self play MCTS logic is used to make a perform a match between players.
The process of move selection is distributed among the workers

**`Arena.py`:** the script that defines the class `Arena(players, game_cls)` manages the games played in the arena and record the statistics.
By default, after each training iteration, an Arena is created against `MCTSPlayer(net iter n) VS RawMCTS` (baseline test) `MCTSPlayer(net iter n) vs MCTSPlayer(net iter n-1)` (net selection test).

**`Coach.py`:** the script that defines the class `Coach((self, game_cls, nnet, args)`  Calling `coach.train` starts the training process.
It performs `n_iters`times the following:
```
self.generateSelfPlayAgents() # creates as many selfplayAgents as "workers" and let them play and collect games
self.processSelfPlayBatches(self.model_iter)
self.saveIterationSamples(self.model_iter)
self.processGameResults(self.model_iter)
self.killSelfPlayAgents()
self.train() # fit the network to the collected experience
self.compareToBaseline() # MCTSPlayer(net iter n) VS RawMCTS
self.compareToPast() # Arena MCTSPlayer(net iter n) vs MCTSPlayer(net iter n-1)
```
**`data`:** the folder where the collected experience is written
**`checkpoint`:** the folder where the model checkpoint are written. By default, the newest checkpoint is loaded, delete the folders if needed.

### Description of some hyperparameters
**`workers`:** Number of processes used for self play, Arena comparison, and training the model. Should generally be set to the number of processors - 1.

**`process_batch_size`:** The size of the batches used for batching MCTS during self play. Equivalent to the number of games that should be played at the same time in each worker. For exmaple, a batch size of 128 with 4 workers would create 128\*4 = 512 total games to be played simultaneously.

**`minTrainHistoryWindow`, `maxTrainHistoryWindow`, `trainHistoryIncrementIters`:** The number of past iterations to load self play training data from. Starts at min and increments once every `trainHistoryIncrementIters` iterations until it reaches max.

**`max_moves`:** Number of moves in the game before the game ends in a draw (should be implemented manually for now in getGameEnded of your Game class, automatic draw is planned). Used for the calculation of `default_temp_scaling` function.

**`num_stacked_observations`:** The number of past observations from the game to stack as a single observation. Should also be done manually for now, but take a look at how it was implemented in `alphazero/envs/tafl/tafl.pyx`.

**`numWarmupIters`:** The number of warm up iterations to perform. Warm up is self play but with random policy and value to speed up initial generations of self play data. Should only be 1-3, otherwise the neural net is only getting random moves in games as training data. This can be done in the beginning because the model's actions are random anyways, so it's for performance.

**`skipSelfPlayIters`:** The number of self play data generation iterations to skip. This assumes that training data already exists for those iterations can be used for training. For example, useful when training is interrupted because data doesn't have to be generated from scratch because it's saved on disk.

**`symmetricSamples`:** Add symmetric samples to training data from self play based on the user-implemented method `symmetries` in their Game class. Assumes that this is implemented. For example, in Viking Chess, the board can be rotated 90 degrees and mirror flipped any number of times while still being a valid game state, therefore this can be used for more training data.

**`numMCTSSims`:** Number of Monte Carlo Tree Search simulations to execute for each move in self play. A higher number is much slower, but also produces better value and policy estimates.

**`probFastSim`:** The probability of a fast MCTS simulation to occur in self play, in which case `numFastSims` simulations are done instead of `numMCTSSims`. However, fast simulations are not saved to training history.

**`max_gating_iters`:** If a model doesn't beat its own past iteration this many times, then gating is temporarily reset and the model is allowed to move on to the next iteration. Use `None` to disable this feature.

**`min_next_model_winrate`:** The minimum win rate required for the new iteration against the last model in order to move on. If it doesn't beat this number, the previous model is used again (model gating).

**`cpuct`:** A constant for balancing exploration vs exploitation in the MCTS algorithm. A higher number promotes more exploration of new actions whereas a lower one promotes exploitation of previously known good actions. A normal range is between 1-4, depending on the environment; a game with less possible moves on each turn would need a lower CPUCT.

**`fpu_reduction`:** "First Play Urgency" reduction decreases the initialization Q value of an unvisited node by this factor, must be in the range `[-1, 1]`. The closer this value is to 1, it discourages MCTS to explore unvisited nodes further, which (hopefully) allows it to explore paths that are more familiar. If this is set to 0, no reduction is done and unvisited nodes inherit their parent's Q value. Closer to a value of -1 (not recommended to go below 0), unvisited nodes become more prefered which can lead to more exploration.

**`num_channels`:** The number of channels each ResNet convolution block has.

**`depth`:** The number of stacked ResNet blocks to use in the network.

**`value_head_channels/policy_head_channels`:** The number of channels to use for the 1x1 value and policy convolution heads respectively. The value and policy heads pass data onto their respective dense layers.

**`value_dense_layers/policy_dense_layers`:** These arguments define the sizes and number of layers in the dense network of the value and policy head. This must be a list of integers where each element defines the number of neurons in the layer and the number of elements defines how many layers there should be.


