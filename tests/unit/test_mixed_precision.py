"""Unit tests for the mixed-precision contract.

UPGRADES #15: under a ``mixed_float16`` global policy, the generator's
``toRGB_curr`` (final tanh) and the discriminator's ``Dense(1)`` output
heads must compute in float32 to avoid saturation and unbounded score
overflow. Both optimizers must additionally wrap in
``LossScaleOptimizer`` so fp16 gradients don't silently underflow.

The fixture sets ``mixed_float16`` for the duration of a test and
restores the prior policy in teardown so other tests are not perturbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import keras
import pytest


@pytest.fixture
def mixed_float16_policy():
    prior = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        yield
    finally:
        keras.mixed_precision.set_global_policy(prior)


def _disc_config() -> SimpleNamespace:
    return SimpleNamespace(
        resolution=(64, 64),
        channels=3,
        depth=2,
        filter_counts=[8, 16],
        kernel_size=[3, 3],
        kernel_stride=[2, 2],
        padding="same",
        negative_slope=0.2,
        learning_rate=1e-4,
        beta_1=0.5,
        beta_2=0.9,
        spectral_norm=False,
    )


def _gen_config() -> SimpleNamespace:
    return SimpleNamespace(
        resolution=(64, 64),
        channels=3,
        depth=2,
        latent_dim=16,
        filter_counts=[16, 8],
        kernel_size=[3, 3],
        kernel_stride=[2, 2],
        padding="same",
        negative_slope=0.2,
        batch_norm=False,
        final_activation="tanh",
        learning_rate=1e-4,
        beta_1=0.5,
        beta_2=0.9,
    )


def test_discriminator_output_dtype_is_float32_under_fp16(mixed_float16_policy):
    from snowgan.models.discriminator import Discriminator

    discriminator = Discriminator(_disc_config())
    final_dense = discriminator.model.layers[-1]
    # The last layer is the Dense(1) (or its SpectralNorm wrapper); the
    # raw output dtype must be float32 even though the global policy is
    # mixed_float16.
    assert final_dense.dtype == "float32"
    assert discriminator.model.output.dtype == "float32"


def test_generator_output_dtype_is_float32_under_fp16(mixed_float16_policy):
    from snowgan.models.generator import Generator

    generator = Generator(_gen_config())
    to_rgb = generator.model.get_layer("toRGB_curr")
    assert to_rgb.dtype == "float32"
    assert generator.model.output.dtype == "float32"


def test_discriminator_optimizer_is_loss_scaled_under_fp16(mixed_float16_policy):
    from snowgan.models.discriminator import Discriminator

    discriminator = Discriminator(_disc_config())
    assert isinstance(
        discriminator.optimizer, keras.mixed_precision.LossScaleOptimizer
    )


def test_generator_optimizer_is_loss_scaled_under_fp16(mixed_float16_policy):
    from snowgan.models.generator import Generator

    generator = Generator(_gen_config())
    assert isinstance(
        generator.optimizer, keras.mixed_precision.LossScaleOptimizer
    )


def test_optimizers_are_plain_adam_under_default_policy():
    # Sanity: the LossScaleOptimizer wrap must be conditional on
    # mixed_float16. Default-policy runs should not pay the loss-scaling
    # overhead.
    from snowgan.models.discriminator import Discriminator
    from snowgan.models.generator import Generator

    assert keras.mixed_precision.global_policy().name != "mixed_float16"

    discriminator = Discriminator(_disc_config())
    generator = Generator(_gen_config())
    assert not isinstance(
        discriminator.optimizer, keras.mixed_precision.LossScaleOptimizer
    )
    assert not isinstance(
        generator.optimizer, keras.mixed_precision.LossScaleOptimizer
    )
