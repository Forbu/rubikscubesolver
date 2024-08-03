"""
All the function to train properly the transformer world model
with the policy model

TODO in the trainer

1. Train the transformer world model
2. Train the policy model

"""

from tqdm import tqdm

import wandb  # for logging

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
    transformer = RubikTransformer(rngs=config.rngs, causal=True)

    # init optimizer
    optimizer_worldmodel = nnx.Optimizer(transformer, optax.adamw(config.lr_1))
    optimizer_policy = nnx.Optimizer(policy, optax.adamw(config.lr_2))

    # metrics
    metrics_train = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        loss_reward=nnx.metrics.Average("loss_reward"),
        loss_cross_entropy=nnx.metrics.Average("loss_cross_entropy"),
    )

    metrics_eval = nnx.MultiMetric(
        loss=nnx.metrics.Average("loss"),
        loss_reward=nnx.metrics.Average("loss_reward"),
        loss_cross_entropy=nnx.metrics.Average("loss_cross_entropy"),
        loss_sequence_1=nnx.metrics.Average("loss_sequence_1"),
        loss_sequence_5=nnx.metrics.Average("loss_sequence_5"),
        loss_sequence_10=nnx.metrics.Average("loss_sequence_10"),
    )

    return (policy, transformer), (optimizer_worldmodel, optimizer_policy), (metrics_train, metrics_eval)


def init_learning(config):
    # gather data from the environment
    # init models and optimizers
    env, buffer = dataset.init_env_buffer(sample_batch_size=config.batch_size)

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

    def step_fn(state, key):
        action = jax.random.randint(
            key=key,
            minval=env.action_spec.minimum,
            maxval=env.action_spec.maximum,
            shape=(3,),
        )

        new_state, timestep = env.step(state, action)
        timestep.extras["action"] = action

        return new_state, timestep

    def run_n_steps(state, key, n):
        random_keys = jax.random.split(key, n)
        state, rollout = jax.lax.scan(step_fn, state, random_keys)

        return rollout

    vmap_reset = jax.vmap(env.reset)
    vmap_step = jax.vmap(run_n_steps, in_axes=(0, 0, None))

    # init models and optimizers
    models, optimizers, metrics = init_models_optimizers(config)

    policy, transformer = models
    optimizer_worldmodel, optimizer_policy = optimizers
    metrics_train, metrics_eval = metrics

    return (
        env,
        buffer,
        buffer_list,
        vmap_reset,
        vmap_step,
        policy,
        transformer,
        optimizer_worldmodel,
        optimizer_policy,
        metrics_train,
        metrics_eval,
    )


def train(config):
    print("Init learning")
    (
        env,
        buffer,
        buffer_list,
        vmap_reset,
        vmap_step,
        policy,
        transformer,
        optimizer_worldmodel,
        optimizer_policy,
        metrics_train,
        metrics_eval,
    ) = init_learning(config)

    print("Init display")
    nnx.display(transformer)
    nnx.display(policy)

    print("Start learning")
    learning_loop(
        policy,
        transformer,
        optimizer_worldmodel,
        optimizer_policy,
        metrics_train,
        env,
        vmap_reset,
        vmap_step,
        buffer,
        buffer_list,
        key=config.rngs,
        config=config,
    )


def loss_fn_transformer(model: RubikTransformer, batch):
    state_logits, reward_value = model(batch["state_first"], batch["action"])

    # reshape state_logits
    # from (batch_size, sequence_length, 324) => (batch_size, sequence_length -1, 54, 6)
    state_logits = state_logits[:, 1:, :]
    state_logits = state_logits.reshape(
        (state_logits.shape[0], state_logits.shape[1], 54, 6)
    )

    reward_value = reward_value[:, 1:]

    loss_crossentropy = optax.softmax_cross_entropy_with_integer_labels(
        logits=state_logits, labels=batch["state_next"]
    ).mean()

    loss_reward = jnp.square(reward_value - batch["reward"]).mean()

    loss = loss_crossentropy + loss_reward

    return loss, (loss_crossentropy, loss_reward)


@nnx.jit
def train_step_transformer(
    model: RubikTransformer, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn_transformer, has_aux=True)
    (loss, (loss_crossentropy, loss_reward)), grads = grad_fn(model, batch)
    metrics.update(
        loss=loss, loss_reward=loss_reward, loss_cross_entropy=loss_crossentropy
    )
    optimizer.update(grads)


def learning_loop(
    policy,
    transformer,
    optimizer_worldmodel,
    optimizer_policy,
    metrics,
    env,
    vmap_reset,
    vmap_step,
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

    key, subkey = jax.random.split(config.jax_key)
    config.jax_key = key

    buffer, buffer_list = dataset.fast_gathering_data(
        env,
        vmap_reset,
        vmap_step,
        config.nb_games * 10,
        config.len_seq,
        buffer,
        buffer_list,
        subkey,
    )

    # transformer model calibration
    for idx_step in tqdm(range(config.nb_step)):
        # if idx_step % config.add_data_every_step == 0:
        #     buffer, buffer_list = dataset.fast_gathering_data(
        #         env,
        #         vmap_reset,
        #         vmap_step,
        #         config.nb_games,
        #         config.len_seq,
        #         buffer,
        #         buffer_list,
        #         config.jax_key,
        #     )

        # training for world model
        train_step(
            buffer,
            buffer_list,
            transformer,
            optimizer_worldmodel,
            metrics,
            config,
            idx_step,
        )

        # training for policy
        # TODO


def train_step(
    buffer, buffer_list, transformer, optimizer_worldmodel, metrics, config, idx_step
):
    key, subkey = jax.random.split(config.jax_key)
    config.jax_key = key

    sample = buffer.sample(buffer_list, subkey)
    sample = reshape_sample(sample)

    # we update the policy
    train_step_transformer(
        transformer, optimizer_worldmodel, metrics, sample.experience
    )

    if idx_step % config.log_every_step == 0:
        metrics_result = metrics.compute()
        print(metrics_result)

        wandb.log(metrics_result, step=idx_step)
        metrics.reset()


def reshape_sample(sample):
    """
    Simple reshape function to reshape the sample

    Args:
        sample (object): The sample object.

    Returns:
        sample (object): The sample object.
    """
    # action have to go from (batch_size, seq_len, 3) to (batch_size, seq_len, 6+3)
    # using one hot encoding because there is 6 possibles values in col [batch_size, seq_len, 0] and
    # 3 possibles values in [batch_size, seq_len, 2]
    # (and only one value in [batch_size, seq_len, 1])
    one_hot_0 = jax.nn.one_hot(sample.experience["action"][:, :, 0], 6)
    one_hot_1 = jax.nn.one_hot(sample.experience["action"][:, :, 1], 3)

    sample.experience["action"] = jnp.concatenate([one_hot_0, one_hot_1], axis=2)

    # reward have to go from (batch_size, seq_len) to (batch_size, seq_len, 1)
    sample.experience["reward"] = sample.experience["reward"].reshape(
        sample.experience["reward"].shape[0], -1, 1
    )

    # state_first have to go from (batch_size, seq_len, 6, 3, 3)
    # to (batch_size, seq_len, 6*6*3*3) using one hot encoding on the 6 classes
    one_hot = jax.nn.one_hot(sample.experience["state_first"], 6)
    sample.experience["state_first"] = jnp.reshape(
        one_hot, (sample.experience["state_first"].shape[0], 1, -1)
    )

    # state_next have to go from (batch_size, seq_len, 6, 3, 3)
    # to (batch_size, seq_len, 6*3*3)
    sample.experience["state_next"] = jnp.reshape(
        sample.experience["state_next"],
        (
            sample.experience["state_next"].shape[0],
            sample.experience["state_next"].shape[1],
            -1,
        ),
    )

    return sample
