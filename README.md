# rl-play

We have a maze with rewards.

    maze = Maze([
        [1, 0, 5, 1],
        [4, 0, 4, 1],
        [2, 0, 8, 1],
    ])
    
An agent moves around the maze. How much reward it can collect in 20 moves?
- Starting from random position (row, col)
- Can move in 4 directions (up, down, right, left)
- Step out of bounds will kill the agent (no walls)

## Q learning

Lets solve it using building Qtable that estimates reward for all combinations of states and actions.

## Policy approximation

Lets replace Qtable with estimator - in this case decision tree or random forest that predicts future reward based on data about state and action.     
