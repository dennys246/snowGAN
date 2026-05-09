"""Unit tests for snowgan.models.discriminator.Discriminator.

Tests build a tiny Discriminator (64×64 input, two small conv filters) so
TF model construction stays sub-second after the framework import.
"""
from __future__ import annotations

from types import SimpleNamespace


def _minimal_config() -> SimpleNamespace:
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


def test_discriminator_flatten_layer_named_features():
    """AvAI Phase 3 hard-commits to model.get_layer("features") as the
    transfer-learning tap point — see docs/UPGRADES.md §3 cross-repo note.
    Renaming this layer breaks AvAI's prepare_backbone_for_transfer.
    """
    import keras
    from snowgan.models.discriminator import Discriminator

    discriminator = Discriminator(_minimal_config())
    layer = discriminator.model.get_layer("features")

    assert layer.name == "features"
    assert isinstance(layer, keras.layers.Flatten)
