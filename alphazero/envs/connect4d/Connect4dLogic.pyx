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
    cdef public int winner
    cdef public bint game_over

    def __init__(self, int height, int width, int win_length, int num_boards):
        """Set up initial board configuration."""
        self.height = height
        self.width = width
        self.num_boards = num_boards
        self.win_length = win_length
        self.winner = 0

        self.pieces = np.zeros((num_boards, height, width), dtype=np.intc)

    def __getstate__(self):
        return self.height, self.width, self.win_length, self.winner, np.asarray(self.pieces)
    
    def __eq__(self, other):
        return (self.winner == other.winner and np.array_equal(self.pieces, other.pieces))

    def __setstate__(self, state):
        self.height, self.width, self.win_length, self.winner, pieces = state
        self.pieces = np.asarray(pieces)

    def add_stone(self, int column, int player, int sub_board):
        """Create copy of board containing new stone."""
        cdef Py_ssize_t r
        assert self.winner == 0, f"Can't play, because the game has already ended with winner {self.winner} on board {self}"
        for r in range(self.height):
            if self.pieces[sub_board, (self.height - 1) - r, column] == 0:
                self.pieces[sub_board, (self.height - 1) - r, column] = player
                ended, player = self.check_if_win(column, (self.height - 1) - r, sub_board, player)
                if ended:
                    self.winner = player
                return
        raise ValueError(f"Can't play column {column} on board {self}")
    
    def get_valid_moves(self, int sub_board):
        """Any zero value in the top row is a valid move."""
        cdef Py_ssize_t c
        cdef int[:] valid = np.zeros((self.width), dtype=np.intc)
        for c in range(self.width):
            if self.pieces[sub_board, 0, c] == 0:
                valid[c] = 1
        return valid

    def get_win_state(self, int sub_board):
        if self.winner == 0:
            if (sum(self.get_valid_moves(sub_board=sub_board)) == 0):
                return(True, 0)
            else:
                return (False, 0)
        else:
            return(True, self.winner)

    def check_if_win(self, int column, int row, int sub_board, int player):
        """
        Check, after an anction, if there's a winning state or a draw.
        Args:
            - column is the last columpn played
            - sub_board is the las board played
        
        """
        cdef int total
        cdef Py_ssize_t r, c, d, s
        cdef Py_ssize_t direction_d, direction_w, direction_h, bound_d, bound_w, bound_h

        ################### 4 STANDARD WINNING CONDITIONS ##########################
        # Check if row wins (constant depth, width(+/-), constant height )
        total = 1
        for (direction_w, bound_w) in [(1, self.width-1), (-1, 0)]:
            for s in range(1, abs(bound_w/direction_w - column) +1):
                c = column + s*direction_w
                if self.pieces[sub_board, row, c] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)
        
        # Check if column wins (constant depth, constant width, height-)
        total = 1
        for (direction_h, bound_h) in [(1, self.height-1)]:
            for s in range(1, abs(bound_h/direction_h - row) +1):
                r = row + s*direction_h 
                if self.pieces[sub_board, r, column] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)

        # Check (bottom-left to top-right) diagonals (constant depth, width++, height++)
        total = 1
        for (direction_w, bound_w, direction_h, bound_h) in [(-1, 0, -1, 0), (1, self.width-1, 1, self.height-1)]:
            for s in range(1, min(abs(bound_w/direction_w - column), abs(bound_h/direction_h - row)) +1):
                r = row + s*direction_h 
                c = column + s*direction_w
                if self.pieces[sub_board, r, column] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)

        # Check (bottom-right to top-left) diagonals (constant depth, width--, height++)
        total = 1
        for (direction_w, bound_w, direction_h, bound_h) in [(1, self.width-1, -1, 0), (-1, 0, 1, self.height-1)]:
            for s in range(1, min(abs(bound_w/direction_w - column), abs(bound_h/direction_h - row)) + 1):
                r = row + s*direction_h 
                c = column + s*direction_w
                if self.pieces[sub_board, r, column] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)


        ################### 6 ADDITIONAL RULES ##########################
        # Additionnal Rule : check diagonals (depth++, constant width, height++)
        total = 1
        for (direction_d, bound_d, direction_h, bound_h) in [(-1, 0, -1, 0), (1, self.num_boards-1, 1, self.height-1)]:
            for s in range(1, min(abs(bound_d/direction_d - sub_board) , abs(bound_h/direction_h - row)) + 1):
                r = row + s*direction_h 
                d = sub_board + s*direction_d
                if self.pieces[d, r, column] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)

        # Additionnal Rule : check diagonals (depth--, constant width, height++)
        total = 1
        for (direction_d, bound_d, direction_h, bound_h) in [(1, self.num_boards-1, -1, 0), (-1, 0, 1, self.height-1)]:
            for s in range(1, min(abs(bound_d/direction_d - sub_board) , abs(bound_h/direction_h - row)) + 1):
                r = row + s*direction_h 
                d = sub_board + s*direction_d
                if self.pieces[d, r, column] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)    

        

        # Additionnal Rule : check diagonals (depth++, width++, height++)
        total = 1
        for (direction_d, bound_d, direction_w, bound_w,  direction_h, bound_h) in [(-1, 0, -1, 0, -1, 0), (1, self.num_boards-1, 1, self.width-1, 1, self.height-1)]:
            for s in range(1, min(abs(bound_d/direction_d - sub_board) ,  min(abs(bound_w/direction_w - column), abs(bound_h/direction_h - row))) + 1):
                c = column + s*direction_w
                r = row + s*direction_h 
                d = sub_board + s*direction_d
                if self.pieces[d, r, c] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)       

        # Additionnal Rule : check diagonals (depth++, width--, height++)
        total = 1
        for (direction_d, bound_d, direction_w, bound_w,  direction_h, bound_h) in [(-1, 0, 1, self.width-1, -1, 0), (1, self.num_boards-1, -1, 0, 1, self.height-1)]:
            for s in range(1, min(abs(bound_d/direction_d - sub_board) ,  min(abs(bound_w/direction_w - column), abs(bound_h/direction_h - row))) + 1):
                c = column + s*direction_w
                r = row + s*direction_h 
                d = sub_board + s*direction_d
                if self.pieces[d, r, c] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)    

        # Additionnal Rule : check diagonals (depth++, width++, height--)
        total = 1
        for (direction_d, bound_d, direction_w, bound_w,  direction_h, bound_h) in [(-1, 0, -1, 0, 1, self.height-1), (1, self.num_boards-1, self.width-1, -1, -1, 0)]:
            for s in range(1, min(abs(bound_d/direction_d - sub_board) ,  min(abs(bound_w/direction_w - column), abs(bound_h/direction_h - row))) + 1):
                c = column + s*direction_w
                r = row + s*direction_h 
                d = sub_board + s*direction_d
                if self.pieces[d, r, c] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)  

        # Additionnal Rule : check diagonals (depth--, width++, height++)
        total = 1
        for (direction_d, bound_d, direction_w, bound_w,  direction_h, bound_h) in [(1, self.num_boards-1, -1, 0, -1, 0), (1, 0,  1, self.width-1, 1, self.height-1)]:
            for s in range(1, min(sub_board - bound_d/direction_d,  min(column - bound_w/direction_w, abs(bound_h/direction_h - row))) + 1):
                c = column + s*direction_w
                r = row + s*direction_h 
                d = sub_board + s*direction_d
                if self.pieces[d, r, c] == player:
                    total += 1
                else:
                    break
                if total == self.win_length:
                    return (True, player)       
                                                                            
        # Game is not ended yet
        return (False, 0)


    def __str__(self):
        return str(np.asarray(self.pieces))