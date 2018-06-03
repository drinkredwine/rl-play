import math
import random
from collections import defaultdict
from copy import copy

from tqdm import tqdm

from rlearn.ml.trainer import train_model, score
from rlearn.prototypes.agent import Agent


def _obs_actor(t_action, state):
    return t_action, state


def _obs_critic(t_qsa, action, state):
    return t_qsa, action, state


class MazeAgent(Agent):

    def __init__(self, Q: defaultdict = None):
        self.Q = Q if Q else defaultdict(defaultdict)

    def get_best_action(self, model, state, actions):
        data = [_obs_critic(-1, action, state) for action in actions]

        max_reward = - math.inf
        best_action = random.choice(actions)

        for action, reward in zip(actions, score(data, model)):

            if reward > max_reward:
                max_reward = reward
                best_action = action

        return best_action, max_reward

    @staticmethod
    def get_random_action(actions: list):
        action = random.choice(actions)
        return action

    def learn(self, maze, iterations: int = 1000):

        memory_actor = [_obs_actor(0, (0, 0))]  # action, state
        memory_critic = [_obs_critic(0, 0, (0, 0))]  # qsa, action, state

        epsilon = 0.5  # higher means more exploration
        energy = 15
        gamma = 0.9
        alpha = 0.4
        for generation in tqdm(range(iterations)):

            # if generation == 0 or random.random() < 0.5:
            #     train_data = self.memory
            #     if len(self.memory) > 1000:
            #         train_data = random.sample(train_data, 1000)
            #     model = train_model(memory=train_data)

            if generation == 0 or random.random() < 0.5:
                model_actor = train_model(memory_actor[-min(len(memory_actor), 1000):], type='classifier')
            if generation == 0 or random.random() < 0.2:
                model_critic = train_model(memory_critic[-min(len(memory_actor), 1000):], type='regressor')

            state = (1, 1)
            for iteration in range(energy):

                if random.random() < epsilon:
                    action = self.get_random_action(maze.actions())
                else:
                    action = list(score([_obs_actor(-1, state)], model_actor))[0]

                # if random.random() < epsilon:
                #     action, qsa = self.get_random_action(model_actor, state, maze.actions())
                # else:
                #     action, qsa = self.get_best_action(model_actor, state, maze.actions())

                state_new, reward = maze.change(state, action)

                correct_action, _ = self.get_best_action(model_critic, state, maze.actions())
                _, best_q = self.get_best_action(model_critic, state_new, maze.actions())

                qsa = list(score([_obs_critic(-1, action, state)], model_critic))[0]
                qsa = qsa + alpha * (reward + gamma * best_q - qsa)

                memory_actor.append(_obs_actor(correct_action, state))
                memory_critic.append(_obs_critic(qsa, state, action))

                state = copy(state_new)
                if state == (-1, -1):
                    break

            self.memory_actor = memory_actor
            self.memory_critic = memory_critic
            self.model_actor = model_actor
            self.model_critic = model_critic

        return model_actor

    def greedy_run(self, maze, state, model):

        energy = 20
        path = [state]
        actions = []
        rewards = []

        for iteration in range(energy):
            #action, _ = self.get_best_action(model, state, maze.actions())
            action = list(score([_obs_actor(-1, state)], model))[0]
            state, reward = maze.change(state, action)

            rewards.append(reward)
            path.append(state)
            actions.append(action)

        return path, actions, rewards
