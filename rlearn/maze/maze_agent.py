import random
from copy import copy
from pprint import pprint

import math
import numpy as np
from collections import defaultdict

from rlearn.maze.maze_env import Maze
from rlearn.prototypes.agent import Agent

class MazeAgent(Agent):

    def __init__(self):
        pass

    def get_best_action(self, Q, state, actions):
        max_reward = - math.inf
        best_action = random.choice(actions)
        for action in actions:
            if action in Q[state] and Q[state][action] > max_reward:
                max_reward = Q[state][action]
                best_action = action

        return best_action, max_reward

    def learn(maze):
        pprint(maze.show())

        gama = 0.1
        alpha = 0.2
        epsilon = 0.4

        cost_per_step = 0.2
        Q = defaultdict(defaultdict)

        for generation in range(1000):
            state = 0#maze.initial_state()
            energy = 10

            cum_reward = 0.0
            for iteration in range(energy):
                if state not in Q:
                    Q[state] = {action: random.random() for action in maze.actions()}

                # greedy select action
                if random.random() < epsilon:
                    action = random.choice(maze.actions())
                else:
                    action, r = self.get_best_action(Q, state, maze.actions())

                # if action not in Q[state]:
                #     Q[state][action] = random.random()

                # take action
                state_new, reward = maze.change(state, action)
                reward -= cost_per_step

                if state_new not in Q:
                    Q[state_new] = {action: random.random() for action in maze.actions()}

                best_action, best_reward = self.get_best_action(Q, state_new, maze.actions())
                Q[state][action] += alpha * (reward + gama * best_reward - Q[state][action])


                state = copy(state_new)
                if state == -1:
                    break
        return Q


    def go(maze, Q, state=0):
        energy = 20
        path = [state]
        actions = []
        print(state)
        cum_reward = 0.0
        for iter in range(energy):
            action, r = self.get_best_action(Q, state, maze.actions())
            state, reward = maze.change(state, action)
            cum_reward += reward
            if state < 0 or state > len(maze.show()) * len(maze.show()[0]):
                break
            print(state, action, reward)
            path.append(state)
            actions.append(action)
            print(cum_reward)

        print(path)
        print(actions)


    def row_col(state, maze):
        return int(state / len(maze)), int(state % len(maze))
