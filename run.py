#!/usr/bin/env python3
from pprint import pprint

from rlearn.maze.maze_env import Maze
from rlearn.maze.maze_agent_dt import MazeAgent
if __name__ == '__main__':

    maze = Maze([
        [1, 0, 5, 2],
        [4, 0, 1, 7],
        [3, 0, 8, 6],
    ])

    agent = MazeAgent()
    clf = agent.learn(maze, iterations=2000)

    pprint(agent.memory[-5:])
    for start in [(0, 0), (0, 3), (1, 1), (2, 0), (2, 3)]:
        path, actions, rewards = agent.greedy_run(maze, start, clf)
        print("path", path)
        print("actions", actions)
        print("rewards", rewards)
        print("total reward", sum(rewards))

    print(clf.feature_importances_)
