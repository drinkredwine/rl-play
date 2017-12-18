import random

from rlearn.prototypes.environment import Environment


class Maze(Environment):
    def __init__(self, rows, cols):
        self.env = np.zeros([rows, cols], dtype=float, order='C')

    def __init__(self, maze):
        self.env = np.array(maze)

    def actions(self):
        return ['up', 'down', 'left', 'right']

    def initial_state(self):
        return random.randint(0, len(self.env) * len(self.env[0]) - 1)

    def show(self):
        return self.env

    def row_col(self, state):
        return int(state / len(self.env[0])), int(state % len(self.env[0]))

    def change(self, state, action):
        new_state = state
        row, col = self.row_col(state)
        rows = len(self.env)
        cols = len(self.env[0])

        if action == 'up':
            row -= 1
            new_state -= cols
        elif action == 'down':
            row += 1
            new_state += cols
        elif action == 'left':
            col -= 1
            new_state -= 1
        elif action == 'right':
            col += 1
            new_state += 1

        if col < 0 or col >= cols or row < 0 or row >= rows:
            return -1, -10.0
        row, col = self.row_col(new_state)
        return new_state, self.env[row][col]
