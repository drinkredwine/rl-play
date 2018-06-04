import math

# import random
import numpy as np
from tqdm import tqdm

from rlearn.ml.trainer import train_model

meta = {}


def obs(state, action):
    """Construct observation from state and action"""
    x = [i for i in state]
    x.append(action)
    return x


def construct_features(obs: list):
    """Construct new features."""
    return obs


def get_best_action(model, state, actions):
    """Get the best action in current state by model"""
    data = [obs(state, action) for action in actions]

    max_reward = - math.inf
    best_action = actions[0]  # np.random.choice(actions)
    for reward, action in zip(model.predict(data), actions):
        if reward > max_reward:
            max_reward = reward
            best_action = action

    return best_action


def get_random_action(actions, p=None):
    """Get random action. Use uniform distribution by default or provided probabilities vector"""
    return np.random.choice(actions, p=p)


def estimate_reward(model, state, action):
    """Returns estimated reward for give action and state) by model"""
    X = [obs(state, action)]
    return model.predict(X)


def get_action_softmax(model, state, actions, t=1):
    """Softmax - the probabilistic pick based on quality of action"""
    data = [obs(state, action) for action in actions]
    scores = model.predict(data)

    p = map(lambda x: math.exp(x / t), scores)
    total = sum(p)
    p = map(lambda x: x / total, p)

    return get_random_action(model, state, actions, p)


def get_action_epsilon(model, state, actions, epsilon=0.5):
    """Epsilon - random action if epsilon < random(0,1) otherwise the best valued action"""
    if np.random.random() < epsilon:
        action = get_random_action(actions)
    else:
        action = get_best_action(model, state, actions)
    return action


def get_action(model, state, actions):
    policy = meta.get('policy', 'epsilon')
    temperature = meta.get('generation', 1)
    progress = meta.get('generation', 1) / meta.get('iterations', 1)

    if policy == 'epsilon':
        epsilon = 0.5 - 0.5 * progress  # decreasing with progress
        action = get_action_epsilon(model, state, actions, epsilon)
    elif policy == 'softmax':
        action = get_action_softmax(model, state, actions, temperature)

    return action


def learn(maze, iterations: int = 1000):
    energy_capacity = 15
    gamma = 0.7
    alpha = 0.4

    initial_state = (0, 0)
    X1 = list()
    y1 = list()
    X1.append(obs(initial_state, 0))
    y1.append(0)
    X2 = list()
    y2 = list()
    X2.append(obs(initial_state, 0))
    y2.append(0)

    meta['iterations'] = iterations
    meta['policy'] = 'epsilon'

    model1 = train_model(X1, y2)
    model2 = train_model(X1, y2)

    for generation in tqdm(range(iterations)):
        meta['generation'] = generation

        if np.random.random() < 0.2:
            if np.random.random() < 0.5:
                model1 = train_model(X1, y1)
            else:
                model2 = train_model(X2, y2)

        energy = energy_capacity
        state = initial_state
        done = False

        while not done:

            action = get_action(model1, state, maze.actions)  # take decision
            state_new, reward = maze.change(state, action)  # take action

            if np.random.random() < 0.5:
                qsa = estimate_reward(model1, state, action)
                best_action = get_best_action(model1, state_new, maze.actions)  # what best can happen in new state?
                best_q = estimate_reward(model2, state_new, best_action) # what best can happen in new state?

                qsa = qsa + alpha * (reward + gamma * best_q - qsa)

                X1.append(obs(state, action))  # remember state and action
                y1.append(qsa)  # next time predict this expected reward

            else:
                qsa = estimate_reward(model2, state, action)

                best_action = get_best_action(model2, state_new, maze.actions)  # what best can happen in new state?

                best_q = estimate_reward(model1, state_new, best_action) # what best can happen in new state?

                qsa = qsa + alpha * (reward + gamma * best_q - qsa)  # update reward estimate for old state

                X2.append(obs(state, action))  # remember state and action
                y2.append(qsa)  # next time predict this expected reward

            state = state_new  # move on

            energy -= 1
            if reward < 0 or energy == 0:  # terminal state
                done = True

    return model1


def run(maze, state, model, greedy=False):
    energy = 20
    path = [state]
    actions = []
    rewards = []

    for iteration in range(energy):
        if greedy:
            action = get_best_action(model, state, maze.actions)
        else:
            action = get_action(model, state, maze.actions)

        state, reward = maze.change(state, action)

        rewards.append(reward)
        path.append(state)
        actions.append(action)

    return path, actions, rewards
