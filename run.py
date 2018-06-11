#!/usr/bin/env python3
from collections import defaultdict

from rlearn.maze.maze_env import Maze
#import rlearn.maze.maze_agent as agent
import rlearn.maze.maze_agent_dt as agent
#import rlearn.maze.maze_agent_double as agent



if __name__ == '__main__':

    maze = Maze([
        [1, 0, 3, 1],
        [4, 0, 8, 1],
        [3, 0, 4, 1],
    ])

    #model = defaultdict(defaultdict)
    model = agent.learn(maze, iterations=1000)
    from rlearn.ml.trainer import tree_to_json
    print(tree_to_json(model))

    for greedy in [True]:
        for start in [(0, 0), (0, 3), (1, 1), (2, 0), (2, 3)]:
            path, actions, rewards = agent.run(maze, start, model, greedy=greedy)
            print("greedy", greedy)
            print("path", path)
            print("actions", actions)
            print("rewards", rewards)
            print("total reward", sum(rewards))

