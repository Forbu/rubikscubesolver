"""
Main script
"""
import time
import wandb

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from dataclasses import dataclass

from rubiktransformer.trainer import train


def main():
    """Main function"""

    @dataclass
    class Config:
        """Configuration class"""
        jax_key: jnp.ndarray = jax.random.PRNGKey(45)
        rngs = nnx.Rngs(44)
        batch_size: int = 128
        lr_1: float = 1e-3
        lr_2: float = 1e-3
        nb_games: int = 128 * 400
        len_seq: int = 5
        nb_step: int = 1000000
        log_every_step: int = 10
        add_data_every_step: int = 1000

    config = Config()

    # init wandb config
    user = "forbu14"
    project = "RubikTransformer"
    display_name = "experiment_" + time.strftime("%Y%m%d-%H%M%S")

    wandb.init(entity=user, project=project, name=display_name)

    train(config=config)


if __name__ == "__main__":
    main()
