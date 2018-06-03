import random

import numpy as np

from rlearn.prototypes.environment import Environment


class Maze(Environment):
    actions = [0, 1, 2, 3]

    def __init__(self, rows, cols):
        self.env = np.zeros([rows, cols], dtype=float, order='C')

    def __init__(self, maze):
        self.env = np.array(maze)

    # def actions(self):
    #     return self.actions

    def change(self, state, action):
        row, col = state

        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1

        rows = len(self.env)
        cols = len(self.env[0])
        if col < 0 or col >= cols or row < 0 or row >= rows:
            return (-1.0, -1.0), -10.0

        return (row, col), self.env[row][col]