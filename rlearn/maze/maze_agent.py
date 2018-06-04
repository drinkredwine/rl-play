import math
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

meta = {}

def get_best_action(Q, state, actions):
    """Get the best action in current state by Q table lookup"""

    max_reward = - math.inf
    best_action = actions[0]  # np.random.choice(actions)

    for action in actions:
        if action in Q[state] and Q[state][action] > max_reward:
            max_reward = Q[state][action]
            best_action = action

    return best_action, max_reward


def get_random_action(actions, p=None):
    """Get random action. Use uniform distribution by default or provided probabilities vector"""
    return np.random.choice(actions, p=p)


def estimate_reward(Q: dict, state, action):
    """Returns estimated reward for give action and state) by Q table lookup"""
    return Q.get(state, {}).get(action, np.random.random())


def get_action_softmax(Q, state, actions, t=1):
    """Softmax - the probabilistic pick based on quality of action"""

    scores = Q[state]
    p = map(lambda x: math.exp(x / t), scores)
    total = sum(p)
    p = map(lambda x: x / total, p)

    return get_random_action(Q, state, actions, p)


def get_action_epsilon(Q, state, actions, epsilon=0.5):
    """Epsilon - random action if epsilon < random(0,1) otherwise the best valued action"""
    if np.random.random() < epsilon:
        action = get_random_action(actions)
    else:
        action, _ = get_best_action(Q, state, actions)
    return action


def get_action_and_expected_reward(model, state, actions):
    policy = meta.get('policy', 'epsilon')
    temperature = meta.get('generation', 1)
    progress = meta.get('generation', 1) / meta.get('iterations', 1)

    if policy == 'epsilon':
        epsilon = 0.5 - 0.5 * progress  # decreasing with progress
        action = get_action_epsilon(model, state, actions, epsilon)
    elif policy == 'softmax':
        action = get_action_softmax(model, state, actions, temperature)

    qsa = estimate_reward(model, state, action)

    return action, qsa


def learn(maze, iterations: int = 1000):
    gama = 0.8  # higher value means that we prefer long term reward vs. short t34m
    alpha = 0.2
    epsilon = 0.4  # higher means more exploration

    Q = defaultdict(defaultdict)

    meta['iterations'] = iterations
    meta['policy'] = 'epsilon'

    for generation in tqdm(range(iterations)):
        meta['generation'] = generation

        state = (0, 0)  # maze.initial_state()
        energy = 10

        for iteration in range(energy):
            if state not in Q:
                Q[state] = {action: np.random.random() for action in maze.actions}

            if random.random() < epsilon:
                action = np.random.choice(maze.actions)
            else:
                action, _ = get_best_action(Q, state, maze.actions)

            state_new, reward = maze.change(state, action)

            if state_new not in Q:
                Q[state_new] = {action: random.random() for action in maze.actions}

            best_action, best_reward = get_best_action(Q, state_new, maze.actions)
            Q[state][action] += alpha * (reward + gama * best_reward - Q[state][action])

            state = state_new
            if state == -1:
                break
    return Q


def run(maze, state, Q, greedy=False):
    energy = 20
    path = [state]
    actions = []
    rewards = []

    for iteration in range(energy):
        if greedy:
            action, r = get_best_action(Q, state, maze.actions)
        else:
            action, r = get_action_and_expected_reward(Q, state, maze.actions)
        state, reward = maze.change(state, action)

        path.append(state)
        actions.append(action)
        rewards.append(reward)

    return path, actions, rewards
