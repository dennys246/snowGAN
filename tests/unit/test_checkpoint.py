"""Unit tests for the weights-only checkpoint contract.

Tests build a tiny Discriminator (64×64, two small conv filters) so save
and load round-trip in well under a second per test. The sidecar
contract is implicit in these tests: each test reconstructs the
architecture from the same in-memory config used at save time, mirroring
how AvAI's load_backbone will reconstruct from the persisted
*_config.json sidecar.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from snowgan.checkpoint import resolve_weights_path, to_weights_path


def _minimal_disc_config() -> SimpleNamespace:
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


# --------------------------- resolver behavior ---------------------------


def test_to_weights_path_canonicalizes_extensions():
    assert to_weights_path("a/b/disc.keras") == "a/b/disc.weights.h5"
    assert to_weights_path("a/b/disc.weights.h5") == "a/b/disc.weights.h5"
    assert to_weights_path("a/b/disc") == "a/b/disc.weights.h5"


def test_resolver_returns_none_when_nothing_exists(tmp_path):
    declared = str(tmp_path / "discriminator.weights.h5")
    assert resolve_weights_path(declared) is None


def test_resolver_prefers_new_format_when_both_exist(tmp_path):
    new_path = tmp_path / "discriminator.weights.h5"
    legacy_path = tmp_path / "discriminator.keras"
    new_path.write_bytes(b"new")
    legacy_path.write_bytes(b"legacy")

    # Even when the declared path uses the legacy extension, the resolver
    # picks up the new format if it exists alongside.
    resolved = resolve_weights_path(str(legacy_path))
    assert resolved == str(new_path)


def test_resolver_falls_back_to_legacy_keras(tmp_path):
    legacy_path = tmp_path / "discriminator.keras"
    legacy_path.write_bytes(b"legacy")

    declared = str(tmp_path / "discriminator.weights.h5")
    resolved = resolve_weights_path(declared)
    assert resolved == str(legacy_path)


# --------------------------- weights round-trip --------------------------


def test_weights_round_trip_preserves_outputs(tmp_path):
    """Save weights from one Discriminator, reconstruct architecture from
    the same config, load weights, assert forward outputs match exactly.
    """
    import tensorflow as tf
    from snowgan.models.discriminator import Discriminator

    cfg = _minimal_disc_config()
    src = Discriminator(cfg)

    # Use a fixed input so any weight drift surfaces in the comparison.
    x = tf.constant(
        np.random.RandomState(0).randn(2, cfg.depth, *cfg.resolution, cfg.channels),
        dtype=tf.float32,
    )
    expected = src.model(x, training=False).numpy()

    weights_path = str(tmp_path / "discriminator.weights.h5")
    src.model.save_weights(weights_path)

    fresh = Discriminator(cfg)
    fresh.model.load_weights(weights_path)
    actual = fresh.model(x, training=False).numpy()

    np.testing.assert_array_equal(actual, expected)


def test_load_weights_fails_loud_on_shape_mismatch(tmp_path):
    """A wider config than the one used at save time must NOT silently
    re-init — load_weights raises so the misconfiguration surfaces.
    """
    from snowgan.models.discriminator import Discriminator

    saved_cfg = _minimal_disc_config()
    src = Discriminator(saved_cfg)

    weights_path = str(tmp_path / "discriminator.weights.h5")
    src.model.save_weights(weights_path)

    # Mismatched filter counts: the conv stack ends up with different
    # channel widths, so weight tensors won't align.
    mismatched_cfg = _minimal_disc_config()
    mismatched_cfg.filter_counts = [16, 32]  # was [8, 16]
    fresh = Discriminator(mismatched_cfg)

    with pytest.raises(Exception):
        fresh.model.load_weights(weights_path)
