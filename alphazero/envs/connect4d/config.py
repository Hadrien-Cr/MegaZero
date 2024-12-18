CONFIG_MCTS_VANILLA = {
    "run_name": "connect4d-mcts-vanilla",
    "self_play_mode": "mcts",
    "self_play_strategy": "vanilla",
    "numMCTSSims": 500,
    "cpuct": 4, 
    "root_temp": 2
}

CONFIG_MCTS_BB = {
    "run_name": "connect4d-mcts-bb",
    "self_play_mode": "mcts",
    "self_play_strategy": "bridge-burning",
    "numMCTSSims": 500,
    "cpuct": 4, 
    "root_temp": 2
}

CONFIG_EMCTS_VANILLA = {
    "run_name": "connect4d-emcts-vanilla",
    "self_play_mode": "emcts",
    "self_play_strategy": "vanilla",
    "emcts_horizon": 5,
    "emcts_bb_phases": 5,
    "numMCTSSims": 500,
    "cpuct": 0.5, 
    "root_temp": 0.2
}

CONFIG_EMCTS_BB = {
    "run_name": "connect4d-emcts-bb",
    "self_play_mode": "emcts",
    "self_play_strategy": "bridge-burning",
    "emcts_horizon": 5,
    "emcts_bb_phases": 5,
    "numMCTSSims": 500,
    "cpuct": 0.5, 
    "root_temp": 0.2
}