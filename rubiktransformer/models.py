"""
Model architecture backbone

It will be specific to the rubiks cube transformer
"""
import jax
import jax.numpy as jnp

from flax import nnx

from rubiktransformer.model_transformer import FeedForward, TransformerBlock


class RubikTransformer(nnx.Module):
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
        dropout: float = 0.05,
        layer_norm_eps=1e-5,
        rngs=None,
        causal=True,
        dim_input_state=6 * 6 * 3 * 3,
        dim_output_action=6 + 3,
        dim_output_state=6 * 6 * 3 * 3,
        max_len_seq=30,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.dim_input_state = dim_input_state
        self.dim_output_action = dim_output_action
        self.dim_output_state = dim_output_state

        # we setup a stack of transformer blocks
        self.transformer = nnx.List(
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

        # two mapping functions to transform the state and the action
        self.state_mapping = nnx.Linear(
            out_features=d_model, in_features=dim_input_state, rngs=rngs
        )

        self.action_mapping = nnx.Linear(
            out_features=d_model, in_features=dim_output_action, rngs=rngs
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

    def __call__(self, state, actions=None):
        """
        Dimensions :
        Inputs:
            state is (batch_size, 1, dim_input_state)
            actions is (batch_size, seq_len, dim_output_action)

        Outputs:
            state is (batch_size, seq_len, dim_output_state / 6, 6)

        """

        state = self.state_mapping(state)

        if actions is not None:
            actions = self.action_mapping(actions)

            # concat the state and the action
            # to get dim (batch_size, seq_len + 1, d_model)
            transformer_input = jnp.concatenate([state, actions], axis=1)
        else:
            transformer_input = state

        # add position embedding
        position_embedding = self.position_embedding(
            jnp.arange(transformer_input.shape[1], dtype=jnp.int32)
        )

        # repeat to handle batch size (seq_len, d_model) => (batch_size, seq_len, d_model)
        position_embedding = position_embedding[None, :, :]

        position_embedding = jnp.repeat(
            position_embedding, transformer_input.shape[0], axis=0
        )

        transformer_input = transformer_input + position_embedding

        for i in range(self.num_decoder_layers):
            transformer_input = self.transformer[i](transformer_input)

        transformer_out = self.layernorm(transformer_input)
        state_prediction = self.linear(transformer_out)

        reward = self.linear_reward(transformer_out)

        return state_prediction, reward


class PolicyModel(nnx.Module):
    """
    Policy action
    Mapping from state to action

    """

    def __init__(
        self,
        nb_layers=3,
        d_model=512,
        input_dim_state=6 * 6 * 3 * 3,
        output_dim_action=6 + 3,
        rngs=None,
    ):
        super().__init__()
        self.input_dim_state = input_dim_state
        self.output_dim_action = output_dim_action

        self.layers = nnx.List(
            [
                nnx.Linear(in_features=d_model, out_features=d_model, rngs=rngs)
                for _ in range(nb_layers - 1)
            ],
        )

        self.linear = nnx.Linear(
            in_features=d_model, out_features=output_dim_action, rngs=rngs
        )

        self.linear_in = nnx.Linear(
            in_features=input_dim_state, out_features=d_model, rngs=rngs
        )

    def __call__(self, state):
        state = self.linear_in(state)
        state = nnx.gelu(state)

        for layer in self.layers:
            state = layer(state)
            state = nnx.gelu(state)

        state = self.linear(state)
        return state
