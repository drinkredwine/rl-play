import math
import random
from collections import defaultdict
from copy import copy

from rlearn.prototypes.agent import Agent


class MazeAgent(Agent):
    def __init__(self, Q: defaultdict = None):
        self.Q = Q if Q else defaultdict(defaultdict)

    def get_best_action(self, Q, state, actions):
        max_reward = - math.inf
        best_action = random.choice(actions)
        for action in actions:
            if action in Q[state] and Q[state][action] > max_reward:
                max_reward = Q[state][action]
                best_action = action

        return best_action, max_reward

    def learn(self, maze, iterations: int = 1000):
        Q = self.Q
        gama = 0.8  # higher value means that we prefer long term reward vs. short t34m
        alpha = 0.2
        epsilon = 0.4  # higher means more exploration

        memory = []

        for generation in range(iterations):
            state = (0, 0)  # maze.initial_state()
            energy = 10

            for iteration in range(energy):
                if state not in Q:
                    Q[state] = {action: random.random() for action in maze.actions()}

                if random.random() < epsilon:
                    action = random.choice(maze.actions())
                else:
                    action, _ = self.get_best_action(Q, state, maze.actions())

                state_new, reward = maze.change(state, action)
                memory.append([state, action, reward])

                if state_new not in Q:
                    Q[state_new] = {action: random.random() for action in maze.actions()}

                best_action, best_reward = self.get_best_action(Q, state_new, maze.actions())
                Q[state][action] += alpha * (reward + gama * best_reward - Q[state][action])

                state = copy(state_new)
                if state == -1:
                    break
        return Q

    def greedy_run(self, maze, state=0):
        Q = self.Q
        energy = 20
        path = [state]
        actions = []

        cum_reward = 0.0
        for iteration in range(energy):
            action, r = self.get_best_action(Q, state, maze.actions())
            state, reward = maze.change(state, action)
            cum_reward += reward

            path.append(state)
            actions.append(action)

            print(state, action, reward)
            print("cumulative reward so far: ", cum_reward)

        print("path:", path)
        print("actions:",actions)

    @staticmethod
    def row_col(state, maze):
        return int(state / len(maze)), int(state % len(maze))
