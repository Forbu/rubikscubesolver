"""
In this file, we define the online training process for the RubikTransformer model.

"""

import os
from tqdm import tqdm
import pickle

import wandb  # for logging
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import optax

from rubiktransformer.model_diffusion_dt import RubikDTTransformer, InverseRLModel
import rubiktransformer.dataset as dataset
from rubiktransformer.trainer import reshape_sample

from rubiktransformer.online_training_utils import (
    run_n_steps,
    reshape_diffusion_setup
)


def init_config_wandb():
    @dataclass
    class Config:
        """Configuration class"""

        jax_key: jnp.ndarray = jax.random.PRNGKey(49)
        rngs = nnx.Rngs(48)
        batch_size: int = 128
        lr_1: float = 4e-4
        lr_2: float = 4e-4
        nb_games: int = 128 * 100
        len_seq: int = 32
        nb_step: int = 1000000
        log_every_step: int = 10
        log_eval_every_step: int = 10
        log_policy_reward_every_step: int = 10
        add_data_every_step: int = 500

        save_model_every_step: int = 2000

    config = Config()

    # init wandb config
    user = "forbu14"
    project = "RubikTransformer"
    display_name = "experiment_" + time.strftime("%Y%m%d-%H%M%S")

    wandb.init(entity=user, project=project, name=display_name)

    wandb.config.update(config)

    return config


def init_model_optimizer(config):
    transformer = RubikDTTransformer(rngs=config.rngs, causal=True)

    inverse_rl_model = InverseRLModel(
        dim_input_state=6 * 6 * 3 * 3,
        dim_output_action=6 + 3,
        dim_middle=1024,
        nb_layers=3,
        rngs=config.rngs,
    )

    scheduler = optax.linear_schedule(
        init_value=0.0, end_value=1.0, transition_steps=4000
    )

    # init optimizer
    optimizer_dd = optax.chain(
        optax.clip_by_global_norm(1.0),
        # optax.lion(config.lr_1 / 10.0),
        optax.adamw(config.lr_1),
        optax.scale_by_schedule(scheduler),
    )

    optimizer_rl_inverse = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(config.lr_2),
    )

    optimizer_diffuser = nnx.Optimizer(transformer, optimizer_dd)
    optimizer_inverse = nnx.Optimizer(inverse_rl_model, optimizer_rl_inverse)

    # metrics
    metrics_train = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        loss_cross_entropy=nnx.metrics.Average("loss_cross_entropy"),
    )

    metrics_eval = nnx.MultiMetric(
        loss_eval=nnx.metrics.Average("loss_eval"),
        loss_cross_entropy_eval=nnx.metrics.Average("loss_cross_entropy_eval"),
    )

    # metric for inverse model
    metrics_inverse = nnx.MultiMetric(
        loss_inverse=nnx.metrics.Average("loss_inverse"),
    )

    return (
        optimizer_diffuser,
        optimizer_inverse,
        metrics_train,
        metrics_eval,
        metrics_inverse,
    )


def init_buffer(config):
    # gather data from the environment
    # init models and optimizers
    env, buffer = dataset.init_env_buffer(sample_batch_size=config.batch_size)
    env, buffer_eval = dataset.init_env_buffer(sample_batch_size=config.batch_size)

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

    reward = jnp.zeros((1))

    jit_step = jax.jit(env.step)

    buffer_list = buffer.init(
        {
            "action": action,
            "reward": reward,
            "state_histo": state_next,
        }
    )

    buffer_list_eval = buffer_eval.init(
        {
            "action": action,
            "reward": reward,
            "state_histo": state_next,
        }
    )

    return env, buffer, buffer_eval, buffer_list, buffer_list_eval, jit_step


def loss_fn_transformer_rf(model: RubikDTTransformer, batch):
    # rectified flow setup
    state_past, state_future = model(
        batch["state_past"], batch["state_future_noise"], batch["context"]
    )

    loss_crossentropy = optax.softmax_cross_entropy(
        logits=state_future, labels=batch["state_future"]
    ).mean(axis=[1, 2])

    weight = jnp.clip(1.0 / (1.0 - batch["time_step"][:, 0, 0, 0]), min=0.005, max=1.5)

    loss_cross_entropy_weight = loss_crossentropy * weight

    return loss_cross_entropy_weight.mean(), (loss_crossentropy.mean())


@nnx.jit
def train_step_transformer_rf(
    model: RubikDTTransformer,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch,
):
    """Train for a single step."""

    grad_fn = nnx.value_and_grad(loss_fn_transformer_rf, has_aux=True)
    (loss, (loss_crossentropy)), grads = grad_fn(model, batch)
    metrics.update(loss=loss, loss_cross_entropy=loss_crossentropy)
    optimizer.update(grads)


################# INVERSE RL ####################
def loss_fn_inverse_rl(model: InverseRLModel, batch):
    # rectified flow setup
    action = model(batch["state_histo_inverse_t"], batch["state_histo_inverse_td1"])

    loss_crossentropy_0 = optax.softmax_cross_entropy(
        logits=action[:, :6], labels=batch["action_inverse"][:, :6]
    ).mean()

    loss_cross_entropy_1 = optax.softmax_cross_entropy(
        logits=action[:, 6:], labels=batch["action_inverse"][:, 6:]
    ).mean()

    return loss_crossentropy_0 + loss_cross_entropy_1


@nnx.jit
def train_step_inverse_rl(
    model: InverseRLModel,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch,
):
    """Train for a single step."""

    grad_fn = nnx.value_and_grad(loss_fn_inverse_rl)
    loss, grads = grad_fn(model, batch)
    metrics.update(loss_inverse=loss)
    optimizer.update(grads)


def training_loop(
    config,
    env,
    buffer,
    buffer_eval,
    buffer_list,
    buffer_list_eval,
    vmap_reset,
    vmap_step,
    transformer,
    optimizer_diffuser,
    metrics_train,
    metrics_eval,
    inverse_rl_model,
    optimizer_inverse,
    metrics_inverse,
):
    # transformer model calibration
    for idx_step in tqdm(range(config.nb_step)):
        # training for world model
        key, subkey = jax.random.split(config.jax_key)
        config.jax_key = key

        if idx_step % config.add_data_every_step == 0:
            buffer, buffer_list = dataset.fast_gathering_data_diffusion(
                env,
                vmap_reset,
                vmap_step,
                int(config.nb_games // 10),
                config.len_seq,
                buffer,
                buffer_list,
                config.jax_key,
            )

        sample = buffer.sample(buffer_list, subkey)
        sample = reshape_diffusion_setup(sample, subkey)

        # we update the policy
        train_step_transformer_rf(
            transformer, optimizer_diffuser, metrics_train, sample
        )

        # train the inverse model
        train_step_inverse_rl(
            inverse_rl_model, optimizer_inverse, metrics_inverse, sample
        )

        if idx_step % config.log_every_step == 0:
            metrics_train_result = metrics_train.compute()
            print(metrics_train_result)

            wandb.log(metrics_train_result, step=idx_step)
            metrics_train.reset()

            metrics_inverse_result = metrics_inverse.compute()
            print(metrics_inverse_result)

            wandb.log(metrics_inverse_result, step=idx_step)
            metrics_inverse.reset()

        if idx_step % config.log_eval_every_step == 0:
            key, subkey = jax.random.split(config.jax_key)
            config.jax_key = key

            buffer_eval, buffer_list_eval = dataset.fast_gathering_data_diffusion(
                env,
                vmap_reset,
                vmap_step,
                int(config.batch_size),
                config.len_seq,
                buffer_eval,
                buffer_list_eval,
                subkey,
            )

            sample = buffer_eval.sample(buffer_list_eval, subkey)
            sample = reshape_diffusion_setup(sample, subkey)

            loss, (loss_crossentropy) = loss_fn_transformer_rf(transformer, sample)

            metrics_eval.update(
                loss_eval=loss,
                loss_cross_entropy_eval=loss_crossentropy,
            )
            wandb.log(metrics_eval.compute(), step=idx_step)

            metrics_eval.reset()

        if idx_step % config.save_model_every_step == 0:
            state_weight = nnx.state(transformer)

            with open("state_ddt_model_improved.pickle", "wb") as handle:
                pickle.dump(state_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)

            # save inverse model
            state_weight = nnx.state(inverse_rl_model)

            with open("state_inverse_rl_model_improved.pickle", "wb") as handle:
                pickle.dump(state_weight, handle, protocol=pickle.HIGHEST_PROTOCOL)


def trainer_full():
    """
    Full training

    1. Init config
    2. Init model and optimizer
    3. Init buffer
    4. Train

    """

    ##### INIT #####
    config = init_config_wandb()

    (
        optimizer_diffuser,
        optimizer_inverse,
        metrics_train,
        metrics_eval,
        metrics_inverse,
    ) = init_model_optimizer(config)

    env, buffer, buffer_eval, buffer_list, buffer_list_eval, jit_step = init_buffer(
        config
    )

    vmap_reset = jax.vmap(jax.jit(env.reset))
    vmap_step = jax.vmap(run_n_steps, in_axes=(0, 0, None))

    ##### TRAINING #####
    key, subkey = jax.random.split(config.jax_key)
    config.jax_key = key

    buffer, buffer_list = dataset.fast_gathering_data_diffusion(
        env,
        vmap_reset,
        vmap_step,
        config.nb_games * 10,  # old is int(config.nb_games * 10.0),
        config.len_seq,
        buffer,
        buffer_list,
        subkey,
    )
