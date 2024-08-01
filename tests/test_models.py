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
        dim_output_action=6 * 1 * 3,
        dim_output_state=6 * 6 * 3 * 3,
    )

    # input is a random float32 tensor of size (batch_size, 1, 6*6*3*3)

    state_input = jnp.ones((5, 1, 6 * 6 * 3 * 3))
    actions_input = jnp.ones((5, len_seq, 6 * 1 * 3))

    states_pred, reward = tb(state_input, actions_input)

    assert states_pred.shape == (5, len_seq + 1, 6 * 6 * 3 * 3)
    assert reward.shape == (5, len_seq + 1, 1)
