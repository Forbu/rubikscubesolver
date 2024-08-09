"""
Model architecture backbone
It will be two things : 

Essentially, it will be the backbone of the transformer world model

We will try to code the transformer in jax (flax)

Homemade version of the transformer

"""

from flax import nnx
import jax
import jax.numpy as jnp


class FeedForward(nnx.Module):
    """
    Feed forward layer
    """

    def __init__(
        self,
        d_model: int = 512,
        dim_feedforward: int = 2048,
        rngs=None,
    ):
        super().__init__()
        self.linear1 = nnx.Linear(
            in_features=d_model, out_features=dim_feedforward, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=dim_feedforward, out_features=d_model, rngs=rngs
        )

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.gelu(x)
        x = self.linear2(x)

        return x


class TransformerBlock(nnx.Module):
    """
    Transformer block

    1. Layer Norm
    2. Multi-Head Attention
    3. Layer Norm
    4. Feed Forward

    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        rngs=None,
        causal=False,
    ):
        super().__init__()

        self.causal = causal

        # init layernorm
        self.layernorm1 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

        # init multi-head attention
        self.multihead = nnx.MultiHeadAttention(
            num_heads=nhead,
            in_features=d_model,
            qkv_features=d_model,
            decode=False,
            rngs=rngs,
        )

        # init layernorm
        self.layernorm2 = nnx.LayerNorm(num_features=d_model, rngs=rngs)

        # init feed forward
        self.feedforward = FeedForward(
            d_model=d_model, dim_feedforward=dim_feedforward, rngs=rngs
        )

        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        self.layer_norm_eps = layer_norm_eps

    def __call__(self, x):
        x_forward = self.layernorm1(x)

        if self.causal:
            mask = nnx.make_causal_mask(x_forward[:, :, 0])

            # mask is (batch_size, 1, len_seq, len_seq)
            # make it (batch_size, nb_head, len_seq, len_seq)
            mask = jnp.repeat(mask, self.multihead.num_heads, axis=1)

            x_forward = self.multihead(x_forward, mask=mask)
        else:
            x_forward = self.multihead(x_forward)

        x_forward = self.dropout(x_forward)
        x_forward = x + x_forward
        x_forward_second = self.layernorm2(x_forward)
        x_forward_second = self.feedforward(x_forward_second)
        x_forward_second = self.dropout(x_forward_second)
        x_forward_second = x_forward + x_forward_second

        return x_forward_second


class Transformer(nnx.Module):
    """
    Transformer model
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.,
        # decoder only
        layer_norm_eps: float = 1e-5,
        nb_embedding: int = 64,
        out_features: int = 64,
        rngs=None,
        causal=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.causal = causal

        self.nb_embedding = nb_embedding

        # init embedding layer
        self.embedding = nnx.Embed(
            num_embeddings=nb_embedding, features=d_model, rngs=rngs
        )

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

        # now the last layer norm and linear layer
        self.layernorm = nnx.LayerNorm(num_features=d_model, rngs=rngs)
        self.linear = nnx.Linear(
            in_features=d_model, out_features=out_features, rngs=rngs
        )

    def __call__(self, x):
        x = self.embedding(x)

        for i in range(self.num_decoder_layers):
            x = self.transformer[i](x)

        x = self.layernorm(x)
        x = self.linear(x)

        return x
