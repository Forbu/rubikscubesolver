"""
Model architecture backbone

It will be specific to the rubiks cube transformer
"""

import jax
import jax.numpy as jnp
from jax import nn, random

from flax import nnx

from rubiktransformer.model_transformer import TransformerBlock

EPS = 1e-11


class Sequential(nnx.Module):
    """
    Sequential module
    """
    def __init__(self, dim_input, dim_output, dim_middle, nb_layer, rngs):
        super().__init__()

        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_middle = dim_middle

        assert nb_layer >= 2, "nb_layer should be at least 2"

        self.layers = nnx.List(
            [
                nnx.Linear(in_features=dim_middle, out_features=dim_middle, rngs=rngs)
                for _ in range(nb_layer - 2)
            ]
        )

        self.input_mapping = nnx.Linear(in_features=dim_input, out_features=dim_middle, rngs=rngs)
        self.output_mapping = nnx.Linear(in_features=dim_middle, out_features=dim_output, rngs=rngs)

    def __call__(self, x):
        x = self.input_mapping(x)
        x = nn.gelu(x)
        for layer in self.layers:
            x = layer(x)
            x = nn.gelu(x)
        x = self.output_mapping(x)
        return x

class RubikDTTransformer(nnx.Module):
    """
    Rubik's cube Transformer

    Specialized design for the rubik's cube transformer world model
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.00,
        layer_norm_eps=1e-5,
        causal=True,
        dim_input_state=6 * 6 * 3 * 3,
        dim_context_input=2, # reward and time step
        dim_output_state=6 * 6 * 3 * 3,
        max_len_seq=50,
        rngs=None,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.dim_input_state = dim_input_state
        self.dim_output_state = dim_output_state

        self.dim_context_input = dim_context_input

        # we setup a stack of transformer blocks
        self.transformer_blocks = nnx.List(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    layer_norm_eps=layer_norm_eps,
                    rngs=rngs,
                    causal=causal,
                )
                for _ in range(num_decoder_layers)
            ],
        )

        # sequential module
        self.mapping_context = Sequential(
            dim_input=dim_context_input,
            dim_output=d_model,
            dim_middle=d_model,
            nb_layer=3,
            rngs=rngs,
        )

        # two mapping functions to transform the state and the action
        self.state_mapping = nnx.Linear(
            out_features=d_model, in_features=dim_input_state, rngs=rngs
        )

        self.layernorm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

        # output layer for state value
        self.linear = nnx.Linear(
            in_features=d_model, out_features=dim_output_state, rngs=rngs
        )

        # output layer for reward value
        self.linear_reward = nnx.Linear(in_features=d_model, out_features=1, rngs=rngs)

        # position embedding
        self.position_embedding = nnx.Embed(
            num_embeddings=max_len_seq, features=d_model, rngs=rngs
        )

    def __call__(self, state_past, state_future, context):
        """
        Dimensions :
        Inputs:
            state_past is (batch_size, seq_len_past, dim_input_state / 6, 6)
            state_future is (batch_size, seq_len_future, dim_input_state / 6, 6)
            context is (batch_size, dim_context_input)

        Outputs:
            state_prediction is (batch_size, seq_len, dim_output_state / 6, 6)

        """
        seq_len_past = state_past.shape[1]
        seq_len_future = state_future.shape[1]

        # concat the two states
        transformer_input = jnp.concatenate([state_past, state_future], axis=1)

        # apply mapping functions
        transformer_input = self.state_mapping(transformer_input)

        # add position embedding
        position_embedding = self.position_embedding(
            jnp.arange(transformer_input.shape[1], dtype=jnp.int32)
        )

        # repeat to handle batch size (seq_len, d_model) => (batch_size, seq_len, d_model)
        position_embedding = position_embedding[None, :, :]

        position_embedding = jnp.repeat(
            position_embedding, transformer_input.shape[0], axis=0
        )

        # adding position embedding
        transformer_input = transformer_input + position_embedding

        # sequential module
        context_value = self.mapping_context(context) # shape (batch, d_model)
        context_value = context_value[None, :, :]

        # repeat to handle batch size (seq_len, d_model) => (batch_size, seq_len, d_model)
        context_value = jnp.repeat(context_value, transformer_input.shape[1], axis=1)

        # adding context value
        transformer_input = transformer_input + context_value

        for i in range(self.num_decoder_layers):
            transformer_input = self.transformer_blocks[i](transformer_input)

        transformer_out = self.layernorm(transformer_input)
        state_prediction = self.linear(transformer_out)

        return state_prediction[:, seq_len_past:, :], state_prediction[:, :seq_len_past, :]
