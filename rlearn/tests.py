import unittest

from run import Maze


class MyTestCase(unittest.TestCase):
    def test_something(self):
        maze = Maze([
            [1, 2, 5, 0],
            [3, 0, 4, 0],
            [1, 0, 8, 6],
        ])

        self.assertEqual(maze.change(1, 'left'), (0, 1))
        self.assertEqual(maze.change(1, 'right'), (2, 5))
        self.assertEqual(maze.change(1, 'x'), (1, 2))
        self.assertEqual(maze.change(1, 'up'), (1, 2))
        self.assertEqual(maze.change(1, 'down'), (5, 0))

        self.assertEqual(maze.change(5, 'left'), (4, 3))
        self.assertEqual(maze.change(5, 'right'), (6, 4))
        self.assertEqual(maze.change(5, 'x'), (5, 0))
        self.assertEqual(maze.change(5, 'up'), (1, 2))
        self.assertEqual(maze.change(5, 'down'), (9, 0))

        self.assertEqual(maze.change(0, 'up'), (0, 1))
        self.assertEqual(maze.change(0, 'left'), (0, 1))

        print(maze.change(11, 'down'), (11, 6))
        print(maze.change(11, 'right'), (11, 6))


if __name__ == '__main__':
    unittest.main()
