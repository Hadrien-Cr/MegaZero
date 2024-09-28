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
    Strands Board.
    """

    cdef public int width
    cdef public int[:] hexes_to_labels
    cdef int radius 
    
    cdef public int[:] hexes_available
    cdef public int digit_chosen
    cdef public int tiles_left_to_place
    cdef public int[:] hexes
    
    def __init__(self, int width):
        # Immutable
        self.width = width
        self.radius = (width+1) //2
        self.hexes_to_labels = np.array([    [0, 0, 0, 0, 0, 6, 5, 5, 5, 5, 6],
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
                                        dtype = np.intc).flatten()
        self.hexes_available = np.array([0, 1, 36, 24, 0, 24, 6],dtype = np.intc)
        

        # Mutable (define the state)
        self.hexes = np.zeros((self.width**2), dtype=np.intc)
        self.digit_chosen = 2
        self.tiles_left_to_place = 1

    def __getstate__(self):
        return self.digit_chosen, self.tiles_left_to_place, np.asarray(self.hexes_available), np.asarray(self.hexes)
    
    def __setstate__(self, state):
        self.digit_chosen, self.tiles_left_to_place, hexes_available, hexes = state
        self.hexes = np.asarray(hexes)
        self.hexes_available = np.asarray(self.hexes_available)

    def add_tile(self, int hex, int target):

        if self.tiles_left_to_place == 0:
            raise ValueError(f"No more tiles can be placed, you have to choose a digit instead")
        if self.hexes_to_labels[hex] != self.digit_chosen:
            raise ValueError(f"Hex {hex//self.width,hex%self.width} with label {self.hexes_to_labels[hex]} is not valid for digit chosen {self.digit_chosen}")
        if self.hexes[hex] != 0:
            raise ValueError(f"Hex {hex//self.width,hex%self.width} already taken")

        self.hexes[hex] = target
        self.tiles_left_to_place -= 1
        self.hexes_available[self.digit_chosen] -= 1

        if self.tiles_left_to_place == 0:
            self.digit_chosen = 0

    def update_digit_chosen(self, int new_digit):
        if not (1 <= new_digit <= 6): 
            raise ValueError(f"Invalid digit {new_digit}.")
        
        self.digit_chosen = new_digit

        if self.hexes_available[new_digit] <= 0:
            raise ValueError(f"Digit {new_digit} is not available (only {self.hexes_available[new_digit]} valid free hexes)")

        self.tiles_left_to_place = min(self.digit_chosen, self.hexes_available[new_digit])

    cdef list[int] compute_neighbours(self, int x, int y):
        cdef list[tuple[int, int]] neighbours = []
        cdef tuple[int, int] direction
        cdef int x2, y2
        for direction in [(-1, 0), (1, -1), (1, 0), (-1, 1), (0, -1), (0, 1)]:
            x2 = max(0, min(x + direction[0], self.width - 1))
            y2 = max(0, min(y + direction[1], self.width - 1))
            neighbours.append((x2, y2))

        return neighbours

    cdef inline bint is_oob(self, int x, int y):
        return abs(x + y - (self.width - 1)) > self.width // 2

    cdef int bfs(self, int x, int y, int target, int[:,:] visited):
        if visited[x][y]:
            return 0

        visited[x][y] = 1

        cdef tuple[int, int] neighbour
        cdef int sum = 1        
        if self.hexes[x*self.width + y] == target:
            for neighbour in self.compute_neighbours(x, y):
                sum += self.bfs(neighbour[0], neighbour[1], target, visited)
            return sum

        return 0

    def compute_areas(self, int target):
        cdef int[:,:] visited = np.zeros((self.width, self.width), dtype=np.intc)
        cdef int x, y

        for x in range(self.width):
            for y in range(self.width):
                if self.is_oob(x, y):
                    visited[x][y] = 1

        cdef list[tuple[int, int]] visit_order = [(x, y) for y in range(self.width) for x in range(self.width)]
        visit_order.sort(key=lambda p: abs(p[0] + p[1] - (self.width - 1)))

        cdef list areas = []
        cdef int area
        cdef tuple[int, int] pos

        for pos in visit_order:
            if not visited[pos[0]][pos[1]]:
                area = self.bfs(pos[0], pos[1], target, visited)
                areas.append(area)

        areas.sort(reverse=True)
        return areas

    def __str__(self):
        return str(np.asarray(self.tiles))