"""
Main script
"""

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
        jax_key: jnp.ndarray = jax.random.PRNGKey(43)
        rngs = nnx.Rngs(42)
        lr_1: float = 1e-3
        lr_2: float = 1e-3
        nb_games: int = 100
        len_seq: int = 20
        nb_step: int = 100

    config = Config()

    train(config=config)


if __name__ == "__main__":
    main()
