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
    cdef public int radius
    cdef public int[:] hexes_to_labels
    cdef public list[tuple] labels_to_hexes
    
    cdef public int[:] hexes_available
    cdef public int digit_chosen
    cdef public int tiles_left_to_place
    cdef public int[:,:] hexes
    
    def __init__(self, int width):
        # Immutable
        self.radius = (width + 1) // 2
        self.width = width

        # Call the internal method to define labels
        self._define_labels()

        # Mutable (define the state)
        self.hexes = np.zeros((self.width, self.width), dtype=np.intc)
        self.digit_chosen = 2
        self.tiles_left_to_place = 1

    cdef void _define_labels(self):
        """
        Define the mapping of hexes to labels and the mapping of labels to hexes.
        Sets the values of self.hexes_to_labels, self.labels_to_hexes, and self.hexes_available.
        """
        cdef int x, y, row, col, dist_from_center, label
        cdef list[tuple[int, int]] corners

        self.hexes_to_labels = np.zeros(self.width * self.width, dtype=np.intc)
        self.labels_to_hexes = [[] for _ in range(7)]
        self.hexes_available = np.zeros(7, dtype=np.intc)

        for x in range(self.width):
            for y in range(self.width):
                row = x - (self.radius - 1)  # centered
                col = y - (self.radius - 1)  # centered

                dist_from_center = max(abs(row), abs(col), abs(row + col))

                # Defining the corners of the hexagonal grid
                corners = [(row, col) for row in (-self.radius + 1, self.radius - 1) for col in (-self.radius + 1, self.radius - 1)]
                corners.extend([(-self.radius + 1, 0), (0, -self.radius + 1), (self.radius - 1, 0), (0, self.radius - 1)])

                # Assign labels based on distance from the center
                if dist_from_center == 0:
                    label = 1
                elif dist_from_center >= self.radius:
                    label = 0
                elif (row, col) in corners:
                    label = 6
                elif abs(row) == self.radius - 1 or abs(col) == self.radius - 1 or abs(row + col) == self.radius - 1:
                    label = 5
                elif abs(row) == self.radius - 2 or abs(col) == self.radius - 2 or abs(row + col) == self.radius - 2:
                    label = 3
                elif abs(row) <= self.radius - 3 and abs(col) <= self.radius - 3 and abs(row + col) <= self.radius - 3:
                    label = 2

                # Store the label for the current hex
                self.hexes_to_labels[x * self.width + y] = label
                self.labels_to_hexes[label].append(x * self.width + y)

        # Define the number of available hexes for each label
        self.hexes_available = np.array([0] + [len(self.labels_to_hexes[i]) for i in range(1, 7)], dtype=np.intc)

    def __getstate__(self):
        return self.digit_chosen, self.tiles_left_to_place, self.hexes_available, np.asarray(self.hexes)
    
    def __setstate__(self, state):
        self.digit_chosen, self.tiles_left_to_place, self.hexes_available, hexes = state
        self.hexes = np.asarray(hexes)

    def add_tile(self, int hex, int target):
        cdef int x = hex // self.width
        cdef int y = hex % self.width

        if self.tiles_left_to_place == 0:
            raise ValueError(f"No more tiles can be placed, you have to choose a digit instead")
        if self.hexes_to_labels[hex] != self.digit_chosen:
            raise ValueError(f"Hex {x,y} with label {self.hexes_to_labels[hex]} is not valid for digit chosen {self.digit_chosen}")
        if self.hexes[x, y] != 0:
            raise ValueError(f"Hex {x,y} already taken")

        self.hexes[x, y] = target
        self.tiles_left_to_place -= 1
        self.hexes_available[self.digit_chosen] -= 1

        if self.tiles_left_to_place == 0:
            self.digit_chosen = 0

    def update_digit_chosen(self, int new_digit):
        if not (1 <= new_digit <= 6): 
            raise ValueError(f"Invalid digit {new_digit}.")
        
        self.digit_chosen = new_digit
        cdef int n = min(self.digit_chosen, self.hexes_available[new_digit])

        if n <= 0:
            raise ValueError(f"Digit {new_digit} is not available (only {self.hexes_available[new_digit]} valid free hexes)")

        self.tiles_left_to_place = n

    def get_digits_available(self):
        cdef int[:] valid = np.zeros((7 + self.width**2), dtype=np.intc)
        cdef int digit

        for digit in range(1, 7):
            if self.hexes_available[digit] > 0:
                valid[digit] = 1

        return valid

    def get_hexes_available(self):
        if not (1 <= self.digit_chosen <= 6): 
            raise ValueError(f"Invalid digit {self.digit_chosen}.")
        
        cdef int[:] valid = np.zeros((7 + self.width**2), dtype=np.intc)
        cdef int hex

        for hex in self.labels_to_hexes[self.digit_chosen]:
            if self.hexes[hex // self.width, hex % self.width] == 0:
                valid[7 + hex] = 1

        return valid
    
    def get_skip_action_only(self) -> list[bool]:
        cdef int[:] valid = np.zeros((7 + self.width**2), dtype=np.intc)
        valid[0] = 1
        return valid

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
        if self.hexes[x][y] == target:
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
