"""
snowGAN package
Exposes the main functions for generating and training diffusion models.
"""

# Import functions/classes from internal modules
from snowgan.models.discriminator import Discriminator, load_discriminator
from snowgan.models.generator import Generator, load_generator

from snowgan.trainer import Trainer as trainer
from snowgan.generate import generate
from snowgan.config import configure_discriminator, configure_generator
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
    "configure_generator",
    "configure_discriminator"
]
