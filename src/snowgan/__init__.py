"""
snowGAN package
Exposes the main functions for generating and training diffusion models.
"""

# Import functions/classes from internal modules
from snowgan.models.discriminator import Discriminator, load_discriminator
from snowgan.models.generator import Generator, load_generator

from snowgan.trainer import Trainer as trainer
from snowgan.generate import generate
from snowgan.config import load_gen_config, configure_gen, load_disc_config, configure_disc
from snowgan.config import build as configuration

# Define a clean public API
__all__ = [
    "Generator",
    "load_generator",
    "Discriminator",
    "load_discriminator",
    "trainer",
    "generate",
    "configuration",
    "load_gen_config",
    "configure_gen",
    "load_disc_config",
    "configure_disc"
]
