from rlearn.maze.maze_env import Maze
from rlearn.maze.maze_agent_dt import MazeAgent
if __name__ == '__main__':

    maze = Maze([
        [1, 0, 5, 1],
        [4, 0, 4, 1],
        [2, 0, 8, 1],
    ])

    agent = MazeAgent()
    clf = agent.learn(maze)
    agent.greedy_run(maze, (0, 0), clf)
    print(clf.feature_importances_)
