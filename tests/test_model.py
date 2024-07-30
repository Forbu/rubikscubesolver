"""
Setup to test the model
We will use the same script to test the model
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import rubiktransformer.model as model


def test_feedforward():
    d_model = 512
    dim_feedforward = 2048

    key = nnx.Rngs(42)

    ff = model.FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, rngs=key)

    x = jnp.ones((5, d_model))

    y = ff(x)

    assert y.shape == (5, d_model)


def test_transformerblock():
    d_model = 512
    nhead = 8
    len_seq = 6

    key = nnx.Rngs(42)

    tb = model.TransformerBlock(
        d_model=d_model, nhead=nhead, dim_feedforward=2048, rngs=key
    )

    x = jnp.ones((5, len_seq, d_model))
    y = tb(x)

    assert y.shape == (5, len_seq, d_model)


def test_transformer():
    d_model = 512
    nhead = 8
    nb_embedding = 64
    out_features = 124

    key = nnx.Rngs(42)

    tb = model.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_decoder_layers=2,
        rngs=key,
        nb_embedding=nb_embedding,
        out_features=out_features,
    )

    # input is random int tensor between 0 and 64
    x = jax.random.randint(jax.random.PRNGKey(0), (5, 6), 0, nb_embedding)

    y = tb(x)

    assert y.shape == (5, 6, out_features)
