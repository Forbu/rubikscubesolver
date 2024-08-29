"""
Test module for the decision diffuser
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from rubiktransformer.model_diffusion_dt import RubikDTTransformer, InverseRLModel


def test_decision_diffuser():
    rng = nnx.Rngs(42)

    decision_diffuser = RubikDTTransformer(
        d_model=512,
        nhead=8,
        num_decoder_layers=8,
        dim_feedforward=1024,
        dropout=0.00,
        layer_norm_eps=1e-5,
        causal=False,
        dim_input_state=6 * 6 * 3 * 3,
        dim_context_input=2,  # reward and time step
        dim_output_state=6 * 6 * 3 * 3,
        max_len_seq=50,
        rngs=rng
    )

    len_seq = 5

    # mock input
    state_past = jnp.ones((1, len_seq, 54, 6))
    state_future = jnp.ones((1, len_seq, 54, 6))
    context = jnp.ones((1, 2))

    state_pred_past, state_pred_future = decision_diffuser(state_past, state_future, context)

    assert state_pred_future.shape == (1, len_seq, 54, 6)

def test_inverserlmodel():
    rng = nnx.Rngs(42)

    inverse_rl_model = InverseRLModel(
        dim_input_state = 6 * 6 * 3 * 3,
        dim_output_action = 6 + 3,
        dim_middle = 1024,
        nb_layers = 3,
        rngs=rng,
    )

    state_past = jnp.ones((1, 6 * 6 * 3 * 3))
    state_future = jnp.ones((1, 6 * 6 * 3 * 3))

    action = inverse_rl_model(state_past, state_future)

    assert action.shape == (1, 6 + 3)
