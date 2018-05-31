import math
import random
from collections import defaultdict
from copy import copy
from pprint import pprint

from tqdm import tqdm

from rlearn.maze.trainer import train_model, score
from rlearn.prototypes.agent import Agent


class MazeAgent(Agent):

    def __init__(self, Q: defaultdict = None):
        self.Q = Q if Q else defaultdict(defaultdict)
        self.memory = [[0, 0, (0, 0), 0]]

    def get_best_action(self, model, state, actions, iteration):
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
        return [reward, action, state, 0]

    def get_random_action(self, model, state, actions, iteration):
        action = random.choice(actions)

        obs = [self._memory(-1, action, state, iteration)]
        reward, _ = list(score(obs, model))[0]

        return action, reward

    def learn(self, maze, iterations: int = 1000):
        epsilon = 0.5  # higher means more exploration
        energy = 15
        gamma = 0.9
        alpha = 0.4

        for generation in tqdm(range(iterations)):

            if generation == 0 or random.random() < 0.5:
                model = train_model(memory=self.memory)

            state = (0, 0)  # maze.initial_state()

            for iteration in range(energy):

                if random.random() < epsilon:
                    action, qsa = self.get_random_action(model, state, maze.actions(), iteration)
                else:
                    action, qsa = self.get_best_action(model, state, maze.actions(), iteration)

                state_new, reward = maze.change(state, action)
                _, best_q = self.get_best_action(model, state_new, maze.actions(), iteration)

                qsa = qsa + alpha * (reward + gamma * best_q - qsa)

                obs = self._memory(qsa[0], action, state, iteration)
                self.memory.append(obs)

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
