"""
Testing the dataset generation
"""

import jax
import jax.numpy as jnp

import rubiktransformer.dataset as dataset


def test_init_env_buffer():
    """
    Here we test the init function of the environment buffer
    """
    env, buffer = dataset.init_env_buffer()

    assert len(buffer) == 4


def init_buffer(buffer, len_seq=20):

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

    return buffer_list


def test_fast_gathering_data():
    env, buffer = dataset.init_env_buffer()

    len_seq = 20

    buffer_list = init_buffer(buffer, len_seq)

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

    dataset.fast_gathering_data(
        env, vmap_reset, vmap_step, 10, 20, buffer, buffer_list, jax.random.PRNGKey(0)
    )

    dataset.fast_gathering_data(
        env, vmap_reset, vmap_step, 100, 20, buffer, buffer_list, jax.random.PRNGKey(0)
    )

    dataset.fast_gathering_data(
        env, vmap_reset, vmap_step, 100, 20, buffer, buffer_list, jax.random.PRNGKey(0)
    )
