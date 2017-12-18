

from qlrl.rl import Qlearning

if __name__ == '__main__':
    states = [1,2,3,4]
    actions = [0,1]
    initial_reward = 1
    learning_rate = 0.1
    discount_factor = 0.05

    ql = Qlearning(states, actions, initial_reward, learning_rate, discount_factor)
    state = 1
    for x in range(100):
        action = ql.get_best_action(state)

        ql.update_model(state, action, reward=1, next_state=state+1)
        if state > 4:
            state = 1

    


