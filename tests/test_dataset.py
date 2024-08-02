"""
Testing the dataset generation
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import rubiktransformer.dataset as dataset


def test_init_env_buffer():
    """
    Here we test the init function of the environment buffer
    """
    env, buffer = dataset.init_env_buffer()

    assert len(buffer) == 4


def test_compute_reward():
    """
    Here we test the compute reward function
    """
    state = jnp.zeros((6, 3, 3))
    for i in range(6):
        state = state.at[:, :, i].set(i - 1)
    reward = dataset.compute_reward(state)
    assert reward == -1.0


def test_gathering_data():
    env, buffer = dataset.init_env_buffer()

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    key = jax.random.PRNGKey(0)

    nb_games = 10
    len_seq = 6

    state_first = jnp.zeros((6, 3, 3))
    state_next = jnp.zeros((len_seq, 6, 3, 3))
    action = jnp.zeros((len_seq, 3))

    # transform state to int8 type
    state_first = state_first.astype(jnp.int8)
    state_next = state_next.astype(jnp.int8)

    # action to int32 type
    action = action.astype(jnp.int32)

    reward = jnp.zeros((len_seq))

    buffer_list = buffer.init(
        {
            "state_first": state_first,
            "action": action,
            "reward": reward,
            "state_next": state_next,
        }
    )

    buffer, buffer_list = dataset.gathering_data(
        env, jit_reset, jit_step, nb_games, len_seq, buffer, buffer_list, key
    )

    # we test the buffer sampling
    rng_key = jax.random.PRNGKey(0)  # Source of randomness.
    batch0 = buffer.sample(buffer_list, rng_key)  # Sample
    batch1 = buffer.sample(buffer_list, rng_key)

    print(batch0.experience["state_first"].shape)
    print(batch1.experience["state_first"].shape)


def test_fast_gathering_data():
    env, buffer = dataset.init_env_buffer()

    def step_fn(state, key):
        action = jax.random.randint(
            key=key,
            minval=env.action_spec.minimum,
            maxval=env.action_spec.maximum,
            shape=(3,),
        )

        new_state, timestep = env.step(state, action)
        timestep.extras["action"] = action

        return new_state, timestep

    def run_n_steps(state, key, n):
        random_keys = jax.random.split(key, n)
        state, rollout = jax.lax.scan(step_fn, state, random_keys)

        return rollout

    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(run_n_steps, in_axes=(0, 0, None))

    rollout_data = dataset.fast_gathering_data(
        env, vmap_reset, vmap_step, 10, 20, jax.random.PRNGKey(0)
    )

    print(rollout_data.observation.cube.shape)

    rollout_data = dataset.fast_gathering_data(
        env, vmap_reset, vmap_step, 100, 20, jax.random.PRNGKey(0)
    )

    print(rollout_data.observation.cube.shape)

    rollout_data = dataset.fast_gathering_data(
        env, vmap_reset, vmap_step, 100, 20, jax.random.PRNGKey(0)
    )

    print(rollout_data.extras["action"].shape)
