from rlearn.maze.maze_env import Maze
from rlearn.maze.maze_agent import MazeAgent
if __name__ == '__main__':

    maze = Maze([
        [1, 0, 5, 1],
        [4, 0, 4, 1],
        [2, 0, 8, 1],
    ])

    agent = MazeAgent()
    agent.learn(maze)
    agent.go(maze,  0)
