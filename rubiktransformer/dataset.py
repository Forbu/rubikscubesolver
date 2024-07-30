"""
Module that will define the dataset for the transformer world model
It will be generated on the fly with the help of the jumanji library
"""
import jax
import jax.numpy as jnp
import flashbax  # replay buffer tool


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
            action = env.action_spec.generate_value()  # random action
            state, reward, done, _ = env.step(state, action)

            state_save_element.append(state)
            action_save_element.append(action)

            # we should create a custom reward function
            reward = compute_reward(state)

            reward_save_element.append(reward)

            if done:
                break

        # here we create the dataset
        # first we concatenate the state and the action
        state = jnp.concatenate(state_save_element, axis=0)
        action = jnp.concatenate(action_save_element, axis=0)
        reward = jnp.concatenate(reward_save_element, axis=0)

        # then we add the state and the action to the buffer
        if len(buffer) == 0:
            buffer_list = buffer.init(
                {"state": state, "action": action, "reward": reward}
            )
        else:
            buffer_list = buffer.add(
                buffer_list, {"state": state, "action": action, "reward": reward}
            )

    return buffer


def compute_reward(state):
    """
    Here we compute the reward for a given state
    the state here is a 3x3x6 array with value between 0 and 5
    that define the state of the rubik cube
    We want to check the distance between the state and the goal
    the goal g is of size 3x3x6 with g[:, :, i] = i
    """
    goal_state = jnp.zeros((3, 3, 6))
    for i in range(6):
        goal_state = goal_state.at[:, :, i].set(i)
    return jnp.linalg.norm(state - goal_state)
