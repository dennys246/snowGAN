"""
snowGAN package
Exposes the main functions for generating and training diffusion models.
"""

# Import functions/classes from internal modules
from snowgan.models.discriminator import Discriminator, load_discriminator
from snowgan.models.generator import Generator, load_generator

from snowgan.trainer import Trainer as trainer
from snowgan.generate import generate
from snowgan.config import configure_gen, configure_disc
from snowgan.config import build

# Define a clean public API
__all__ = [
    "Generator",
    "load_generator",
    "Discriminator",
    "load_discriminator",
    "trainer",
    "generate",
    "build",
    "configure_gen",
    "configure_disc"
]
