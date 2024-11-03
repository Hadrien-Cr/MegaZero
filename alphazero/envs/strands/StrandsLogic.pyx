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
    cdef public int[:,:] hexes_to_labels
    cdef int radius 
    
    cdef public int[:] hexes_available
    cdef public int digit_chosen
    cdef public int rest
    cdef public int[:,:] hexes
    
    def __init__(self, int width, 
                int[:] default_hexes_available, 
                int[:,:] default_hexes_to_labels):
        # Immutable
        self.width = width
        self.radius = (width+1) //2
        self.hexes_available = default_hexes_available
        self.hexes_to_labels = default_hexes_to_labels

        # Mutable (define the state)
        self.hexes = np.zeros((self.width, self.width), dtype=np.intc)
        self.digit_chosen = 2
        self.rest = 1

    def __getstate__(self):
        return self.digit_chosen, self.rest, np.asarray(self.hexes_available), np.asarray(self.hexes)
    
    def __eq__(self, other):
        return (self.digit_chosen == other.digit_chosen \
                and self.rest == other.rest \
                and np.array_equal(self.hexes_available, other.hexes_available) \
                and np.array_equal(self.hexes, other.hexes) )
    
    def __setstate__(self, state):
        self.digit_chosen, self.rest, hexes_available, hexes = state
        self.hexes = np.asarray(hexes)
        self.hexes_available = np.asarray(hexes_available)

    def add_tile(self, int x, int y, int target):
        assert self.hexes_to_labels[x][y]>0, f"Hex {x,y} is out of valid bounds {self}"
        assert self.hexes[x,y] == 0, f"Hex {x,y} already taken {self}"
        if self.digit_chosen == 0:
            assert self.hexes_available[self.hexes_to_labels[x][y]]>0, f"No hexes available for digit {self.hexes_to_labels[x][y]} {self}"
            self.update_digit_chosen(self.hexes_to_labels[x][y])
        assert self.hexes_to_labels[x][y] == self.digit_chosen, f"Hex {x,y} {self}"

        self.hexes[x,y] = target
        self.rest -= 1
        self.hexes_available[self.digit_chosen] -= 1

        if self.rest == 0:
            self.digit_chosen = 0

    def update_digit_chosen(self, int new_digit):
        self.digit_chosen = new_digit
        self.rest = min(self.digit_chosen, self.hexes_available[new_digit])

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
        if visited[x,y] or self.hexes[x,y]!= target:
            return 0

        visited[x,y] = 1

        cdef tuple[int, int] neighbour
        cdef int sum = 1        
        for neighbour in self.compute_neighbours(x, y):
            sum += self.bfs(neighbour[0], neighbour[1], target, visited)
        return sum

    def compute_areas(self):
        cdef int[:,:] visited = np.zeros((self.width, self.width), dtype=np.intc)
        cdef int x, y

        for x in range(self.width):
            for y in range(self.width):
                if self.is_oob(x, y):
                    visited[x,y] = 1

        cdef list[tuple[int, int]] visit_order = [(x, y) for y in range(self.width) for x in range(self.width)]
        visit_order.sort(key=lambda p: abs(p[0] + (self.width - 1)//2) +  abs(p[1] - (self.width - 1)//2))

        cdef list areas_black = [0]
        cdef list areas_white = [0]
        cdef list areas_empty = [0]
        cdef int area
        cdef tuple[int, int] pos

        for pos in visit_order:
            if not visited[pos[0]][pos[1]]:
                target = self.hexes[pos[0],pos[1]]
                area = self.bfs(pos[0], pos[1], target, visited)
                if target == 0:
                    areas_empty.append(area)
                elif target == 1:
                    areas_black.append(area)
                elif target == -1:
                    areas_white.append(area)

        areas_black.sort(reverse=True)
        areas_white.sort(reverse=True)
        areas_empty.sort(reverse=True)

        return areas_black, areas_white, areas_empty

    def __str__(self):
        string =f"Digit Selected: {self.digit_chosen if self.digit_chosen >0 else None}, \
                Rest of Hexes for this turn: {self.rest}, \
                Hexes available: {np.asarray(self.hexes_available)} \n"
        c = [" _ ", " B ", " W "]
        for row in range(self.width):
            offset = abs(row-(self.width//2))
            string+=3*offset*" "
            
            for col in range(self.width):
                label = self.hexes_to_labels[row][col]
                value = self.hexes[row][col]
                if label>0: string += c[value]+ "  "
            
            string+="\n"
            string+=3*offset*" "

            for col in range(self.width):
                label = self.hexes_to_labels[row][col]
                if label>0: string += ("("+ str(label) +")" if label > 0 else "   ") + "  "
            string+=2*"\n"
        return string