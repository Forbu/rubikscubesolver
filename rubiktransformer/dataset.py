"""
Module that will define the dataset for the transformer world model
It will be generated on the fly with the help of the jumanji library
"""
import random

import jax
import jax.numpy as jnp

import flashbax  # replay buffer tool
import jumanji

# GOAL OBSERVATION
GOAL_OBSERVATION = jnp.zeros((6, 3, 3))
for i in range(6):
    GOAL_OBSERVATION = GOAL_OBSERVATION.at[:, :, i].set(i)


def gathering_data(env, nb_games, len_seq, buffer, buffer_list, key):
    """
    In this function we will simply gather data from the environment
    return an array of shape (nb_token_state, nb_games, len_seq) for states
    and (nb_token_action, nb_games, len_seq - 1) for actions
    and (nb_games, len_seq) for rewards
    """
    keys = jax.random.split(key, nb_games)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    for idx_game in range(nb_games):
        state_save_element = []
        action_save_element = []
        reward_save_element = []

        state, _ = jit_reset(keys[idx_game])

        for _ in range(len_seq):
            # action = env.action_spec.generate_value()  # random action TODO change this
            action = compute_random_action(env)
            state, _ = jit_step(state, action)

            state_save_element.append(state.cube)
            action_save_element.append(action)

            # we should create a custom reward function
            reward = compute_reward(state.cube)

            reward_save_element.append(reward)

        # here we create the dataset
        # first we concatenate the state and the action
        state = jnp.stack(state_save_element, axis=0)
        action = jnp.stack(action_save_element, axis=0)

        # transform action to int32 type
        action = action.astype(jnp.int32)

        reward = jnp.array(reward_save_element)

        # then we add the state and the action to the buffer
        buffer_list = buffer.add(
            buffer_list, {"state": state, "action": action, "reward": reward}
        )

    return buffer, buffer_list


def compute_random_action(env):
    action_spec = env.action_spec

    array_action = action_spec.generate_value()

    for i in range(len(array_action)):
        array_action = array_action.at[i].set(random.randint(0, action_spec.maximum[i]))

    return array_action


@jax.jit
def compute_reward(observation):
    """
    Here we compute the reward for a given observation
    the observation here is a 3x3x6 array with value between 0 and 5
    that define the observation of the rubik cube
    We want to check the distance between the observation and the goal
    the goal g is of size 3x3x6 with g[:, :, i] = i
    """
    return -((observation - GOAL_OBSERVATION) ** 2).mean()


def init_env_buffer(max_length=1024, sample_batch_size=32):
    """
    Initializes the environment and buffer for the Rubik's Cube game.

    Returns:
        env (jumanji.Environment): The initialized environment.
        buffer (flashbax.Buffer): The initialized buffer.
    """
    env = jumanji.make("RubiksCube-v0")

    buffer = flashbax.make_item_buffer(
        max_length=max_length, min_length=2, sample_batch_size=sample_batch_size
    )

    return env, buffer
