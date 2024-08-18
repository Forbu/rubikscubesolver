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
    action_pred = rollout.extras["action_pred"]

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
                "action_pred": action_pred[idx_batch], # here we could manage the action prediction
            },
        )

    return buffer, buffer_list


@jax.jit
def compute_reward(observation):
    """
    Here we compute the reward for a given observation
    the observation here is a 6x3x3 array with value between 0 and 5
    that define the observation of the rubik cube
    We want to check the distance between the observation and the goal
    the goal g is of size 6x3x3 with g[i, :, :] = i
    """
    if observation.shape == (6, 3, 3):
        return -((observation - GOAL_OBSERVATION) ** 2).mean()
    elif len(observation.shape) == 4:
        # we repeat the goal_observation to match the shape of the observation
        goal_observation = jnp.repeat(
            GOAL_OBSERVATION[None, :, :, :], observation.shape[0], axis=0
        )
        return -((observation - goal_observation) ** 2).mean(axis=[1, 2, 3])



def gather_data_policy(
    model_policy,
    model_worldmodel,
    env,
    vmap_reset,
    batch_size,
    len_seq,
    key,
):
    keys = jax.random.split(key, batch_size)
    state, timestep = vmap_reset(keys)

    one_hot = jax.nn.one_hot(state.cube, 6)
    state_first_policy = jnp.reshape(one_hot, (batch_size, 1, -1))

    state_pred = jnp.copy(state_first_policy)
    action_list = None

    state_pred_list = []
    uniform0_list = []
    uniform1_list = []

    # Collect a batch of rollouts
    for i in range(len_seq):
        keys = jax.random.split(key, batch_size)
        key_uniform = jax.random.split(keys[0], 2)
        key = keys[1]

        # generate random values
        # random_uniform0, random_uniform1
        # should be of size (batch_size, 6) and (batch_size, 3)
        uniform0 = jax.random.uniform(key_uniform[0], (batch_size, 1, 6))
        uniform1 = jax.random.uniform(key_uniform[1], (batch_size, 1, 3))

        # apply the policy
        action_result = model_policy(state_pred, uniform0, uniform1)


        if action_list is None:
            action_list = action_result
        else:
            action_list = jnp.concatenate((action_list, action_result), axis=1)

        # save data into a list
        state_pred_list.append(state_pred)
        uniform0_list.append(uniform0)
        uniform1_list.append(uniform1)

        # now we can apply the world model to sample next state
        state_logits, reward = model_worldmodel(state_pred, action_list)

        # reshape then argmax
        state_logits = state_logits.reshape(
            (state_logits.shape[0], state_logits.shape[1], 54, 6)
        )

        state_pred = jnp.argmax(state_logits, axis=3)

        # onehot
        state_pred = jax.nn.one_hot(state_pred, 6)

        # shape to flatten
        state_pred = state_pred.reshape((state_pred.shape[0], state_pred.shape[1], -1))

        # take the last state
        state_pred = state_pred[:, -1, :]

        # add a dimension on axis 1
        state_pred = jnp.expand_dims(state_pred, axis=1)

    # here we create the dataset in a proper format
    state_pred_histo = jnp.concatenate(state_pred_list, axis=1)
    uniform0_histo = jnp.concatenate(uniform0_list, axis=1)
    uniform1_histo = jnp.concatenate(uniform1_list, axis=1)

    return state_pred_histo, uniform0_histo, uniform1_histo, action_list


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
