CONFIG_MCTS_VANILLA = {
    "run_name": "strands-mcts-vanilla",
    "self_play_mode": "mcts",
    "self_play_strategy": "vanilla",
    "numMCTSSims": 300,
    "cpuct": 4, 
    "root_temp": 2
}

CONFIG_MCTS_BB = {
    "run_name": "strands-mcts-bb",
    "self_play_mode": "mcts",
    "self_play_strategy": "bridge-burning",
    "numMCTSSims": 300,
    "cpuct": 4, 
    "root_temp": 2
}

CONFIG_EMCTS_VANILLA = {
    "run_name": "strands-emcts-vanilla",
    "self_play_mode": "emcts",
    "self_play_strategy": "vanilla",
    "emcts_horizon": 12,
    "emcts_bb_phases": 6,
    "numMCTSSims": 300,
    "cpuct": 0.5, 
    "root_temp": 0.2
}

CONFIG_EMCTS_BB = {
    "run_name": "strands-emcts-bb",
    "self_play_mode": "emcts",
    "self_play_strategy": "bridge-burning",
    "emcts_horizon": 12,
    "emcts_bb_phases": 6,
    "numMCTSSims": 300,
    "cpuct": 0.5, 
    "root_temp": 0.2
}