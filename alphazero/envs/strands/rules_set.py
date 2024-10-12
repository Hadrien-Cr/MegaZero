import numpy as np
rules_strands4 = {
    'DEFAULT_WIDTH': 7,
    'DEFAULT_HEIGHT': 7,
    'MAX_TURNS': 17,
    'NUM_PLAYERS': 2,
    'AVG_ATOMIC_ACTIONS':  37 / 17,
    'DEFAULT_HEXES_TO_LABELS':np.array([[0, 0, 0, 4, 3, 3, 4],
                                        [0, 0, 3, 2, 2, 2, 3],
                                        [0, 3, 2, 2, 2, 2, 3],
                                        [4, 2, 2, 1, 2, 2, 4],
                                        [3, 2, 2, 2, 2, 3, 0],
                                        [3, 2, 2, 2, 3, 0, 0],
                                        [4, 3, 3, 4, 0, 0, 0]],
                                        dtype = np.intc),
    'DEFAULT_HEXES_AVAILABLE':np.array([0, 1, 18, 12, 6, 0, 0], dtype = np.intc)
    }

rules_strands5 = {
    'DEFAULT_WIDTH': 9,
    'DEFAULT_HEIGHT': 9,
    'MAX_TURNS': 23,
    'NUM_PLAYERS': 2,
    'AVG_ATOMIC_ACTIONS':  61/23,
    'DEFAULT_HEXES_TO_LABELS':np.array([[0, 0, 0, 0, 6, 4, 4, 4, 6],
                                        [0, 0, 0, 4, 3, 3, 3, 3, 4],
                                        [0, 0, 4, 3, 2, 2, 2, 3, 4],
                                        [0, 4, 3, 2, 2, 2, 2, 3, 4],
                                        [6, 3, 2, 2, 1, 2, 2, 3, 6],
                                        [4, 3, 2, 2, 2, 2, 3, 4, 0],
                                        [4, 3, 2, 2, 2, 3, 4, 0, 0],
                                        [4, 3, 3, 3, 3, 4, 0, 0, 0],
                                        [6, 4, 4, 4, 6, 0, 0, 0, 0]],
                                        dtype = np.intc),
    'DEFAULT_HEXES_AVAILABLE': np.array([0, 1, 18, 18, 18, 0, 6], dtype = np.intc)
}

rules_strands6 = {
    'DEFAULT_WIDTH': 11,
    'DEFAULT_HEIGHT': 11,
    'MAX_TURNS': 34,
    'NUM_PLAYERS': 2,
    'AVG_ATOMIC_ACTIONS':  91/34,
    'DEFAULT_HEXES_TO_LABELS':np.array([[0, 0, 0, 0, 0, 6, 5, 5, 5, 5, 6],
                                        [0, 0, 0, 0, 5, 3, 3, 3, 3, 3, 5],
                                        [0, 0, 0, 5, 3, 2, 2, 2, 2, 3, 5],
                                        [0, 0, 5, 3, 2, 2, 2, 2, 2, 3, 5],
                                        [0, 5, 3, 2, 2, 2, 2, 2, 2, 3, 5],
                                        [6, 3, 2, 2, 2, 1, 2, 2, 2, 3, 6],
                                        [5, 3, 2, 2, 2, 2, 2, 2, 3, 5, 0],
                                        [5, 3, 2, 2, 2, 2, 2, 3, 5, 0, 0],
                                        [5, 3, 2, 2, 2, 2, 3, 5, 0, 0, 0],
                                        [5, 3, 3, 3, 3, 3, 5, 0, 0, 0, 0],
                                        [6, 5, 5, 5, 5, 6, 0, 0, 0, 0, 0]],
                                        dtype = np.intc),
    'DEFAULT_HEXES_AVAILABLE':np.array([0, 1, 36, 24, 0, 24, 6],dtype = np.intc)
    }
