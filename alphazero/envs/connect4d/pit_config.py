PIT_CONFIG_MCTS_VANILLA = {
    "run_name": "connect4d-mcts-vanilla",
    "self_play_mode": "mcts",
    "self_play_strategy": "vanilla",
    "numMCTSSims": 2500,
    "cpuct": 4, 
    "root_temp": 2
}

PIT_CONFIG_MCTS_BB = {
    "run_name": "connect4d-mcts-bb",
    "self_play_mode": "mcts",
    "self_play_strategy": "bridge-burning",
    "numMCTSSims": 2500,
    "cpuct": 4, 
    "root_temp": 2
}

PIT_CONFIG_EMCTS_VANILLA = {
    "run_name": "connect4d-emcts-vanilla",
    "self_play_mode": "emcts",
    "self_play_strategy": "vanilla",
    "emcts_horizon": 10,
    "emcts_bb_phases": 8,
    "numMCTSSims": 2500,
    "cpuct": 1, 
    "root_temp": 0
}

PIT_CONFIG_EMCTS_BB = {
    "run_name": "connect4d-emcts-bb",
    "self_play_mode": "emcts",
    "self_play_strategy": "bridge-burning",
    "emcts_horizon": 10,
    "emcts_bb_phases": 8,
    "numMCTSSims": 2500,
    "cpuct": 1, 
    "root_temp": 0
}