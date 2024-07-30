"""
Module that will define the dataset for the transformer world model
It will be generated on the fly with the help of the jumanji library
"""

import flashbax

def gathering_data(env, nb_games, len_seq, buffer):
    """
    In this function we will simply gather data from the environment
    return an array of shape (nb_token_state, nb_games, len_seq) for states
    and (nb_token_action, nb_games, len_seq - 1) for actions
    and (nb_games, len_seq) for rewards
    """

    for _ in range(nb_games):
        state_save_element = []
        action_save_element = []
        reward_save_element = []

        state, timestep = env.reset()
        for _ in range(len_seq):
            action = env.action_spec.generate_value()
            state, reward, done, _ = env.step(state, action)

            state_save_element.append(state)
            action_save_element.append(action)

            # we should create a custom reward function
            # TODO

            reward_save_element.append(reward)

            if done:
                break

        # here we should append the state, action and reward
        # to the buffer
        # first we create the state
        # TODO
    return buffer
