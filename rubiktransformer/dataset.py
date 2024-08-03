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
    GOAL_OBSERVATION = GOAL_OBSERVATION.at[i, :, :].set(i)


def fast_gathering_data(
    env, vmap_reset, vmap_step, batch_size, rollout_length, buffer, buffer_list, key
):
    key1, key2 = jax.random.split(key)

    keys = jax.random.split(key1, batch_size)
    state, timestep = vmap_reset(keys)

    # Collect a batch of rollouts
    keys = jax.random.split(key2, batch_size)
    rollout = vmap_step(state, keys, rollout_length)

    # we retrieve the information from the state_first (state), state_next,
    #  the action and the reward
    state_first = timestep.observation.cube
    state_next = rollout.observation.cube
    action = rollout.extras["action"]

    # now we compute the reward :
    reward = jnp.zeros((batch_size, rollout_length))

    # for each batch / rollout we compute the mean difference between the
    # observation and the goal
    # we repeat the goal_observation to match the shape of the observation
    goal_observation = jnp.repeat(
        GOAL_OBSERVATION[None, None, :, :, :], batch_size, axis=0
    )
    goal_observation = jnp.repeat(goal_observation, rollout_length, axis=1)
    reward = jnp.where(state_next != goal_observation, -1.0, 1.0)

    reward = reward.mean(axis=[2, 3, 4])

    for idx_batch in range(batch_size):
        buffer_list = buffer.add(
            buffer_list,
            {
                "state_first": state_first[idx_batch],
                "action": action[idx_batch],
                "reward": reward[idx_batch],
                "state_next": state_next[idx_batch],
            },
        )

    return buffer, buffer_list


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


def init_env_buffer(max_length=1024 * 100, sample_batch_size=32):
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
