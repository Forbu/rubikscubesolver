"""
All the function to train properly the transformer world model
with the policy model

TODO in the trainer 

1. Train the transformer world model
2. Train the policy model

"""

import jax
import jax.numpy as jnp

import optax
from flax import nnx

from rubiktransformer.models import RubikTransformer, PolicyModel


def init_models_optimizers(rngs, lr_1=1e-3, lr_2=1e-3):
    """
    Initializes the models and optimizers for the policy model and the transformer world model.

    Args:
        rngs (nnx.Rngs): The random key for initializing the models.
        lr_1 (float, optional): The learning rate for the transformer world model. Defaults to 1e-3.
        lr_2 (float, optional): The learning rate for the policy model. Defaults to 1e-3.

    Returns:
        tuple: A tuple containing the policy model, the transformer world model, the optimizer for the transformer world model, and the optimizer for the policy model.
    """
    # init model for the transformer world model
    # init model for policy model
    # init optimizers
    policy = PolicyModel(rngs=rngs)
    transformer = RubikTransformer(rngs=rngs)

    # init optimizer
    optimizer_worldmodel = nnx.Optimizer(transformer, optax.adam(lr_1))
    optimizer_policy = nnx.Optimizer(policy, optax.adam(lr_2))

    return policy, transformer, optimizer_worldmodel, optimizer_policy


def init_learning():
    # gather data from the environment
    # init models and optimizers
    pass


def train():
    # init learning

    # init models and optimizers

    # train

    # evaluate

    pass


def learning_loop():
    pass


def learning_step():
    pass
