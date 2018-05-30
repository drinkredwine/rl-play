import math
import random
from collections import defaultdict
from copy import copy
from pprint import pprint

import numpy as np
from sklearn import tree
from tqdm import tqdm

from rlearn.maze.trainer import train_model, score
from rlearn.prototypes.agent import Agent


class MazeAgent(Agent):

    def __init__(self, Q: defaultdict = None):
        self.Q = Q if Q else defaultdict(defaultdict)
        self.memory = [[0, 0, (0, 0), 0]]

    def get_best_action(self, model: tree.DecisionTreeRegressor, state, actions, iteration):
        data = [self._memory(-1, action, state, iteration) for action in actions]

        max_reward = - math.inf
        best_action = random.choice(actions)

        for reward, action in score(data, model):
            if reward > max_reward:
                max_reward = reward
                best_action = action

        return best_action, max_reward

    @staticmethod
    def _memory(reward, action, state, iteration):
        return reward, action, state, iteration

    def learn(self, maze, iterations: int = 1000):
        epsilon = 0.3  # higher means more exploration

        for generation in tqdm(range(iterations)):
            model = train_model(memory=self.memory)
            state = (0, 0)  # maze.initial_state()
            energy = 20

            cum_reward = 0.0
            for iteration in range(energy):
                if random.random() < epsilon:
                    action = random.choice(maze.actions())
                else:
                    action, _ = self.get_best_action(model, state, maze.actions(), iteration)

                state_new, reward = maze.change(state, action)
                cum_reward += reward
                
                self.memory.append((self._memory(cum_reward, action, state, iteration)))

                state = copy(state_new)
                if state == (-1, -1):
                    break

        pprint(self.memory)
        return model

    def greedy_run(self, maze, state, model):

        energy = 20
        path = [state]
        actions = []

        cum_reward = 0.0
        for iteration in range(energy):
            action, r = self.get_best_action(model, state, maze.actions(), iteration)
            state, reward = maze.change(state, action)
            cum_reward += reward

            path.append(state)
            actions.append(action)

            print(state, action, reward)
            print("cumulative reward so far: ", cum_reward)

        print("path:", path)
        print("actions:", actions)

    @staticmethod
    def row_col(state, maze):
        return int(state / len(maze)), int(state % len(maze))
