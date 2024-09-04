import jax
import jax.numpy as jnp
import flax.nnx as nnx

import wandb

from rubiktransformer.dataset import GOAL_OBSERVATION


def step_fn(state, key, env):
    """
    Simple step function
    We choose a random action
    """
    action = jax.random.randint(
        key=key,
        minval=env.action_spec.minimum,
        maxval=env.action_spec.maximum,
        shape=(3,),
    )

    new_state, timestep = env.step(state, action)
    timestep.extras["action"] = action

    return new_state, timestep


def run_n_steps(state, key, n, env):
    random_keys = jax.random.split(key, n)
    state, rollout = jax.lax.scan(lambda s, k: step_fn(s, k, env), state, random_keys)
    return rollout


def generate_past_state_with_random_policy(key, vmap_reset, step_jit_env, config, env):
    """
    Generate past state with random policy

    Args:
        config: configuration object

    Returns:
        state_past: (batch_size, len_seq//4, 6, 3, 3)
    """
    key1, key2 = jax.random.split(key)

    keys = jax.random.split(key1, config.batch_size)
    state, timestep = vmap_reset(keys)

    past_state = []

    actions_all = jax.random.randint(
        key=key2,
        minval=env.action_spec.minimum,
        maxval=env.action_spec.maximum,
        shape=(config.batch_size, config.len_seq // 4, 3),
    )

    for i in range(config.len_seq // 4):
        action = actions_all[:, i, :]
        state, timestep = step_jit_env(state, action)
        past_state.append(state.cube)

    state_past = jnp.stack(past_state, axis=1)

    return state_past, state, actions_all


def apply_decision_diffuser_policy(
    key, state_past, decision_diffuser, inverse_rl_model, config
):
    """
    1. Make an estimation of the targeted reward
    2. Generate future state with those targeted rewards
    3. Choose policy from that
    """
    sample_eval = {
        "state_past": jax.nn.one_hot(state_past, 6),
    }

    state_past = jnp.copy(
        state_past.reshape((state_past.shape[0], state_past.shape[1], -1))
    )
    state_past = jax.nn.one_hot(state_past, num_classes=6)

    state_future = sampling_model(
        key, decision_diffuser, sample_eval, nb_step=100, config=config
    )

    state_to_act = jnp.concatenate([state_past, state_future], axis=1)
    state_to_act_futur_t = state_to_act[:, (config.len_seq // 4 - 1) : (-1), :, :]
    state_to_act_futur_td1 = state_to_act[:, (config.len_seq // 4) :, :, :]

    state_to_act_futur_t = state_to_act_futur_t.reshape(
        (state_to_act_futur_t.shape[0], state_to_act_futur_t.shape[1], -1)
    )

    state_to_act_futur_td1 = state_to_act_futur_td1.reshape(
        (state_to_act_futur_td1.shape[0], state_to_act_futur_td1.shape[1], -1)
    )

    actions = inverse_rl_model(state_to_act_futur_t, state_to_act_futur_td1)

    return actions


def gather_data_with_policy(
    state,
    state_past,
    actions_past,
    actions_futur,
    buffer,
    buffer_list,
    config,
    step_jit_env,
):
    """
    For loop with those policy and state
    log performance compare to target
    """
    state_futur_list = []

    for i in range(config.len_seq - config.len_seq // 4):
        actions_step = actions_futur[:, i, :]
        actions_0 = jnp.argmax(actions_step[:, :6], axis=1)
        actions_1 = jnp.argmax(actions_step[:, 6:], axis=1)

        actions_full = jnp.stack(
            [actions_0, jnp.zeros(config.batch_size), actions_1], axis=1
        )
        actions_full = actions_full.astype(jnp.int32)

        state, timestep = step_jit_env(state, actions_full)
        state_futur_list.append(state.cube)

    actions_0_all_futur = jnp.argmax(actions_futur[:, :, :6], axis=-1)
    actions_1_all_futur = jnp.argmax(actions_futur[:, :, 6:], axis=-1)

    action_all_futur = jnp.stack(
        [
            actions_0_all_futur,
            jnp.zeros((config.batch_size, actions_0_all_futur.shape[1])),
            actions_1_all_futur,
        ],
        axis=-1,
    )

    action_all = jnp.concatenate([actions_past, action_all_futur], axis=1)
    action_all = action_all.astype(jnp.int32)

    state_futur = jnp.stack(state_futur_list, axis=1)
    state_all = jnp.concatenate([state_past, state_futur], axis=1)

    goal_observation = jnp.repeat(
        GOAL_OBSERVATION[None, None, :, :, :], config.batch_size, axis=0
    )
    goal_observation = jnp.repeat(goal_observation, config.len_seq, axis=1)
    reward = jnp.where(state_all != goal_observation, -1.0, 1.0)

    reward = reward.mean(axis=[2, 3, 4])
    reward = reward[:, -1] - reward[:, config.len_seq // 4]

    for idx_batch in range(config.batch_size):
        buffer_list = buffer.add(
            buffer_list,
            {
                "action": action_all[idx_batch],
                "reward": reward[idx_batch],
                "state_histo": state_all[idx_batch],
            },
        )

    return buffer, buffer_list


def improve_training_loop(
    nb_iter,
    config,
    vmap_reset,
    step_jit_env,
    transformer,
    inverse_rl_model,
    buffer,
    buffer_list,
    train_step_transformer_rf,
    optimizer_diffuser,
    metrics_train,
    env,
):
    """
    Relaunch the training loop with those new data incorporated into the buffer

    Full stuff here
    Online transformer setup

    1. We generate env setup
    2. First random action in the different env
    3. Use decision_diffuser to choose the action to do from here
    4. Observe / apply policy to retrieve data
    5. Add the data into the buffer
    6. Train model on those data

    Remember to log the performance data to compare with other run / algorithms
    """

    for idx_step in range(nb_iter):
        key, subkey = jax.random.split(config.jax_key)
        config.jax_key = key

        state_past, state, actions_past = generate_past_state_with_random_policy(
            key, vmap_reset, step_jit_env, config, env
        )

        actions_futur = apply_decision_diffuser_policy(
            config.jax_key, state_past, transformer, inverse_rl_model, config
        )

        buffer, buffer_list = gather_data_with_policy(
            state,
            state_past,
            actions_past,
            actions_futur,
            buffer,
            buffer_list,
            config,
            step_jit_env,
        )

        sample = buffer.sample(buffer_list, subkey)
        sample = reshape_diffusion_setup(sample, subkey)

        train_step_transformer_rf(
            transformer, optimizer_diffuser, metrics_train, sample
        )

        if idx_step % config.log_every_step == 0:
            metrics_train_result = metrics_train.compute()
            print(metrics_train_result)

            wandb.log(metrics_train_result, step=idx_step)
            metrics_train.reset()


def sampling_model(key, model, sample_eval, nb_step=100, config=None):
    """
    Function used to sampling a state from a list
    """
    seq_len_future = config.len_seq - config.len_seq // 4
    noise_future = jax.random.dirichlet(
        key, jnp.ones(6) * 5.0, (config.batch_size, seq_len_future, 54)
    )
    sample_eval["reward"] = jnp.linspace(start=-0.5, stop=0.5, num=config.batch_size)[
        :, None
    ]

    for t_step in range(nb_step):
        t_step_array = jnp.ones((config.batch_size, 1, 1, 1)) * float(t_step / nb_step)
        sample_eval["context"] = jnp.concatenate(
            [sample_eval["reward"], t_step_array[:, :, 0, 0]], axis=1
        )

        estimation_logits_past, estimation_logits_future = model(
            sample_eval["state_past"], noise_future, sample_eval["context"]
        )

        estimation_proba_future = jax.nn.softmax(estimation_logits_future, axis=-1)

        noise_future = noise_future + float(1.0 / nb_step) * 1.0 / (
            1.0 - t_step_array + 0.0001
        ) * (estimation_proba_future - noise_future)

    return noise_future

def reshape_diffusion_setup(sample, key=jax.random.PRNGKey(0)):
    sample.experience["state_histo"] = sample.experience["state_histo"].reshape(
        (
            sample.experience["state_histo"].shape[0],
            sample.experience["state_histo"].shape[1],
            54,
        )
    )

    # one hot encoding for state_histo
    sample.experience["state_histo"] = jax.nn.one_hot(
        sample.experience["state_histo"],
        num_classes=6,
        axis=-1,
    )

    # batch creation
    batch = sample.experience
    len_seq = batch["state_histo"].shape[1]

    time_step = jax.random.uniform(
        key, (batch["state_histo"].shape[0], 1, 1, 1)
    )  # random value between 0 and 1

    batch["time_step"] = time_step

    # now contact the value to have the context for the rectified flow setup
    batch["context"] = jnp.concatenate([batch["reward"], time_step[:, :, 0, 0]], axis=1)

    batch["state_past"] = batch["state_histo"][:, : len_seq // 4, :, :]
    batch["state_future"] = batch["state_histo"][:, len_seq // 4 :, :, :]

    # now we generate the random noise for the rectified flow setup
    simplex_noise = jax.random.dirichlet(
        key, jnp.ones(6), batch["state_future"].shape[:-1]
    )

    batch["state_future_noise"] = (1 - time_step) * simplex_noise + time_step * batch[
        "state_future"
    ]

    batch["action_inverse"] = sample.experience["action"][:, 1:, :]

    # flatten the action_inverse to only have batch data
    batch["action_inverse"] = jnp.reshape(
        batch["action_inverse"],
        (batch["action_inverse"].shape[0] * batch["action_inverse"].shape[1], -1),
    )

    # now we can one hot encode the action_inverse

    action_inverse_0 = jax.nn.one_hot(
        batch["action_inverse"][:, 0], num_classes=6, axis=-1
    )
    action_inverse_1 = jax.nn.one_hot(
        batch["action_inverse"][:, 2], num_classes=3, axis=-1
    )

    batch["action_inverse"] = jnp.concatenate(
        [action_inverse_0, action_inverse_1], axis=1
    )

    state_histo_inverse_t = sample.experience["state_histo"][:, :-1, :, :]
    state_histo_inverse_td1 = sample.experience["state_histo"][:, 1:, :, :]

    batch["state_histo_inverse_t"] = state_histo_inverse_t
    batch["state_histo_inverse_td1"] = state_histo_inverse_td1

    # we flatten the two state_histo_inverse
    batch["state_histo_inverse_t"] = jnp.reshape(
        batch["state_histo_inverse_t"],
        (
            batch["state_histo_inverse_t"].shape[0]
            * batch["state_histo_inverse_t"].shape[1],
            -1,
        ),
    )
    batch["state_histo_inverse_td1"] = jnp.reshape(
        batch["state_histo_inverse_td1"],
        (
            batch["state_histo_inverse_td1"].shape[0]
            * batch["state_histo_inverse_td1"].shape[1],
            -1,
        ),
    )

    return batch
