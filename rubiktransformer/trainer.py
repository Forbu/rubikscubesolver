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
import rubiktransformer.dataset as dataset


# BATCH GENERATIVE CONFIG


def init_models_optimizers(config):
    """
    Initializes the models and optimizers for the policy model and the transformer world model.

    Args:
        rngs (nnx.Rngs): The random key for initializing the models.
        lr_1 (float, optional): The learning rate for the transformer world model. Defaults to 1e-3.
        lr_2 (float, optional): The learning rate for the policy model. Defaults to 1e-3.

    Returns:
        tuple: A tuple containing the policy model, the transformer world model,
        the optimizer for the transformer world model, and the optimizer for the policy model.
    """
    # init models
    policy = PolicyModel(rngs=config.rngs)
    transformer = RubikTransformer(rngs=config.rngs)

    # init optimizer
    optimizer_worldmodel = nnx.Optimizer(transformer, optax.adam(config.lr_1))
    optimizer_policy = nnx.Optimizer(policy, optax.adam(config.lr_2))

    # metrics
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
    )

    return (policy, transformer), (optimizer_worldmodel, optimizer_policy), metrics


def init_learning(config):
    # gather data from the environment
    # init models and optimizers
    env, buffer = dataset.init_env_buffer()

    nb_games = config.nb_games
    len_seq = config.len_seq

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

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # init models and optimizers
    models, optimizers, metrics = init_models_optimizers(config)

    policy, transformer = models
    optimizer_worldmodel, optimizer_policy = optimizers

    return (
        env,
        buffer,
        jit_reset,
        jit_step,
        policy,
        transformer,
        optimizer_worldmodel,
        optimizer_policy,
        buffer_list,
        metrics,
    )


def train(config):

    print("Init learning")
    (
        env,
        buffer,
        jit_reset,
        jit_step,
        policy,
        transformer,
        optimizer_worldmodel,
        optimizer_policy,
        buffer_list,
        metrics,
    ) = init_learning(config)

    print("Init display")
    nnx.display(transformer)
    nnx.display(policy)

    print("Start learning")
    for _ in range(config.nb_step):
        learning_loop(
            policy,
            transformer,
            optimizer_worldmodel,
            optimizer_policy,
            metrics,
            env,
            jit_reset,
            jit_step,
            buffer,
            buffer_list,
            key=config.rngs,
            config=config,
        )


def loss_fn_transformer(model: RubikTransformer, batch):
    state_logits, reward_value = model(batch["state_first"], batch["action"])
    loss_crossentropy = optax.softmax_cross_entropy_with_integer_labels(
        logits=state_logits, labels=batch["state_next"]
    ).mean()

    loss_reward = jnp.square(reward_value - batch["reward"]).mean()

    loss = loss_crossentropy + loss_reward

    return loss, state_logits


@nnx.jit
def train_step_transformer(
    model: RubikTransformer, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn_transformer, has_aux=True)
    (loss, _), grads = grad_fn(model, batch)
    metrics.update(loss=loss)
    optimizer.update(grads)


def learning_loop(
    policy,
    transformer,
    optimizer_worldmodel,
    optimizer_policy,
    metrics,
    env,
    jit_reset,
    jit_step,
    buffer,
    buffer_list,
    key,
    config,
):
    """
    Executes the learning loop for the given parameters.

    Args:
        policy (object): The policy object.
        transformer (object): The transformer object.
        optimizer_worldmodel (object): The optimizer for the world model.
        optimizer_policy (object): The optimizer for the policy.
        metrics (object): The metrics object.
        env (object): The environment object.
        buffer (object): The buffer object.
        buffer_list (object): The buffer list object.
        key (object): The key object.
        config (object): The configuration object.

    Returns:
        None
    """
    buffer, buffer_list = dataset.gathering_data(
        env,
        jit_reset,
        jit_step,
        config.nb_games,
        config.len_seq,
        buffer,
        buffer_list,
        config.jax_key,
    )

    # transformer model calibration
    for _ in range(10):
        # we sample from buffer
        sample = buffer.sample(buffer_list, config.jax_key)

        print(sample.experience.keys())

        # we update the policy
        train_step_transformer(transformer, optimizer_policy, metrics, sample.experience)

    # model evluation
