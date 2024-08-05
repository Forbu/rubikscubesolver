"""
Setup to test the model
We will use the same script to test the model
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx

import rubiktransformer.models as models


def test_rubiktransformer():

    key = nnx.Rngs(42)
    len_seq = 7

    tb = models.RubikTransformer(
        d_model=512,
        nhead=8,
        num_decoder_layers=2,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        rngs=key,
        dim_input_state=6 * 6 * 3 * 3,
        dim_output_action=6 + 3,
        dim_output_state=6 * 6 * 3 * 3,
    )

    # input is a random float32 tensor of size (batch_size, 1, 6*6*3*3)

    state_input = jnp.ones((5, 1, 6 * 6 * 3 * 3))
    actions_input = jnp.ones((5, len_seq, 6 + 3))

    states_pred, reward = tb(state_input, actions_input)

    print(states_pred.shape)
    print(reward.shape)

    assert states_pred.shape == (5, len_seq + 1, 6 * 6 * 3 * 3)
    assert reward.shape == (5, len_seq + 1, 1)


def test_policy_model():

    key = nnx.Rngs(42)

    policy = models.PolicyModel(rngs=key, temp=5.)

    # input is a random float32 tensor of size (batch_size, 1, 6*6*3*3)

    state_input = jnp.ones((5, 6 * 6 * 3 * 3))
    uniform_input0 = jnp.ones((5, 6))
    uniform_input1 = jnp.ones((5, 3))

    action = policy(state_input, uniform_input0, uniform_input1)

    print(action.shape)

    assert action.shape == (5, 6 + 3)

    # testing with sequence

    state_input = jnp.ones((5, 7, 6 * 6 * 3 * 3))

    # uniform random between 0 and 1
    key = jax.random.PRNGKey(42)
    uniform_input0 = jax.random.uniform(key, shape=(5, 7, 6))
    uniform_input1 = jax.random.uniform(key, shape=(5, 7, 3))

    action = policy(state_input, uniform_input0, uniform_input1)

    print(action.shape)
    print(action)

    # compute the gradient of the output sum
    def loss_fn(x):
        return jnp.mean(policy(x, uniform_input0, uniform_input1))

    grad = jax.grad(loss_fn)(state_input)

    print(grad)

    assert action.shape == (5, 7, 6 + 3)