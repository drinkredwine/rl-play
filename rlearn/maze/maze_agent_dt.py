import math
import random
from collections import defaultdict
from copy import copy

from tqdm import tqdm

from rlearn.ml.trainer import train_model, score
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

        last_avg_rewards = 1
        pbar = tqdm(range(iterations))
        for generation in pbar:

            if generation == 0 or random.random() < 0.5:
                train_data = self.memory
                if len(self.memory) > 1000:
                    train_data = random.sample(train_data, 1000)

                model = train_model(memory=train_data)

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

            if generation % 100:
                starts = [(0, 0), (0, 3), (1, 1), (2, 0), (2, 3)]
                avg_rewards = 0

                for start in starts:
                    _, _, rewards = self.greedy_run(maze, start, model)
                    avg_rewards += sum(rewards)
                avg_rewards /= len(starts)

                pbar.set_description("avg reward {}".format(avg_rewards))

        return model

    def greedy_run(self, maze, state, model):

        energy = 20
        path = [state]
        actions = []
        rewards = []

        for iteration in range(energy):
            action, r = self.get_best_action(model, state, maze.actions(), iteration)
            state, reward = maze.change(state, action)

            rewards.append(reward)
            path.append(state)
            actions.append(action)

        return path, actions, rewards
