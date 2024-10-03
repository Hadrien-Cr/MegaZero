# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: auto_pickle=True
# cython: profile=True


import numpy as np

cdef class Board:
    """
    The class that implements the Connect4d Board Logic.
    It is made of multiple Connect4 boards (num_boards) stacked on top of each other.
    Thus it is represented by a numpy array board of shape: [num_boards, height, width].
    """
    
    cdef public int height
    cdef public int width
    cdef public int win_length
    cdef public int num_boards
    cdef public int[:,:,:] pieces

    def __init__(self, int height, int width, int win_length, int num_boards):
        """Set up initial board configuration."""
        self.height = height
        self.width = width
        self.num_boards = num_boards
        self.win_length = win_length

        self.pieces = np.zeros((num_boards, height, width), dtype=np.intc)

    def __getstate__(self):
        return self.height, self.width, self.win_length, np.asarray(self.pieces)

    def __setstate__(self, state):
        self.height, self.width, self.win_length, pieces = state
        self.pieces = np.asarray(pieces)

    def add_stone(self, int column, int player, int sub_board):
        """Create copy of board containing new stone."""
        cdef Py_ssize_t r
        for r in range(self.height):
            if self.pieces[sub_board, (self.height - 1) - r, column] == 0:
                self.pieces[sub_board, (self.height - 1) - r, column] = player
                return
        raise ValueError(f"Can't play column {column} on board {self}")
        return 
    def get_valid_moves(self, int sub_board):
        """Any zero value in the top row is a valid move."""
        cdef Py_ssize_t c
        cdef int[:] valid = np.zeros((self.width), dtype=np.intc)
        for c in range(self.width):
            if self.pieces[sub_board, 0, c] == 0:
                valid[c] = 1
        return valid

    def get_win_state(self, int sub_board):
        """Check if there's a winning state or a draw."""

        cdef int player
        cdef int total
        cdef int good
        cdef Py_ssize_t r, c, x

        for player in [1, -1]:
            # Check row wins
            for r in range(self.height):
                total = 0
                for c in range(self.width):
                    if self.pieces[sub_board, r, c] == player:
                        total += 1
                    else:
                        total = 0
                    if total == self.win_length:
                        return (True, player)
            
            # Check column wins
            for c in range(self.width):
                total = 0
                for r in range(self.height):
                    if self.pieces[sub_board, r, c] == player:
                        total += 1
                    else:
                        total = 0
                    if total == self.win_length:
                        return (True, player)
            
            # Check diagonals (top-left to bottom-right)
            for r in range(self.height - self.win_length + 1):
                for c in range(self.width - self.win_length + 1):
                    good = True
                    for x in range(self.win_length):
                        if self.pieces[sub_board, r + x, c + x] != player:
                            good = False
                            break
                    if good:
                        return (True, player)

            # Check diagonals (top-right to bottom-left)
            for r in range(self.height - self.win_length + 1):
                for c in range(self.win_length - 1, self.width):
                    good = True
                    for x in range(self.win_length):
                        if self.pieces[sub_board, r + x, c - x] != player:
                            good = False
                            break
                    if good:
                        return (True, player)

            # Additionnal rule
            for r in range(self.height - self.win_length + 1):
                for c in range(0, self.width):
                    good = True
                    for x in range(self.win_length):
                        if sub_board+x >= self.num_boards or self.pieces[sub_board + x, r + x, c] != player:
                            good = False
                            break
                    if good:
                        return (True, player)
                        
        # Check for draw
        if sum(self.get_valid_moves(sub_board=sub_board)) == 0:
            return (True, 0)

        # Game is not ended yet
        return (False, 0)

    def __str__(self):
        return str(np.asarray(self.pieces))