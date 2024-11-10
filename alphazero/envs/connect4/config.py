CONFIG_MCTS_VANILLA = {
    "run_name": "connect4-mcts-vanilla",
    "self_play_mode": "mcts",
    "self_play_strategy": "vanilla",
    "numMCTSSims": 100,
}

CONFIG_MCTS_BB = {
    "run_name": "connect4-mcts-bb",
    "self_play_mode": "mcts",
    "self_play_strategy": "bridge-burning",
    "numMCTSSims": 100,
}

CONFIG_EMCTS_VANILLA = {
    "run_name": "connect4-emcts-vanilla",
    "self_play_mode": "emcts",
    "self_play_strategy": "vanilla",
    "emcts_horizon": 4,
    "emcts_bb_phases": 10,
    "numMCTSSims": 100,
}

CONFIG_EMCTS_BB = {
    "run_name": "connect4-emcts-bb",
    "self_play_mode": "emcts",
    "self_play_strategy": "bridge-burning",
    "emcts_horizon": 4,
    "emcts_bb_phases": 10,
    "numMCTSSims": 100,
}