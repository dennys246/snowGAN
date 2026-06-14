"""Focused tests for the spectral-norm checkpoint consistency guard.

A mixed/stale save_dir (config says spectral_norm: true, but the saved disc is
plain-conv) otherwise crashes deep in Keras with "expected 2 variables,
received 0". These pin the early, clear-message guard and the underlying
weights-inspection helper.
"""

import keras
import pytest
import tensorflow as tf

from snowgan.checkpoint import weights_use_spectral_norm
from snowgan.trainer import assert_spectral_norm_consistency


def _save_disc(path, use_sn):
    inp = keras.Input(shape=(1, 8, 8, 3))
    conv = keras.layers.Conv3D(4, (1, 3, 3), strides=(1, 2, 2), padding="same")
    x = keras.layers.SpectralNormalization(conv)(inp) if use_sn else conv(inp)
    x = keras.layers.Flatten()(x)
    dense = keras.layers.Dense(1, dtype="float32")
    out = keras.layers.SpectralNormalization(dense)(x) if use_sn else dense(x)
    model = keras.Model(inp, out)
    model(tf.zeros([2, 1, 8, 8, 3]), training=True)  # build vector_u
    model.save_weights(path)
    return path


def test_detects_spectral_norm(tmp_path):
    p = str(tmp_path / "sn.weights.h5")
    _save_disc(p, use_sn=True)
    assert weights_use_spectral_norm(p) is True


def test_detects_plain_conv(tmp_path):
    p = str(tmp_path / "plain.weights.h5")
    _save_disc(p, use_sn=False)
    assert weights_use_spectral_norm(p) is False


def test_none_when_uninspectable(tmp_path):
    assert weights_use_spectral_norm(None) is None
    assert weights_use_spectral_norm(str(tmp_path / "missing.weights.h5")) is None
    assert weights_use_spectral_norm("legacy.keras") is None  # bundle, not inspectable


def test_guard_passes_on_match(tmp_path):
    p = str(tmp_path / "sn.weights.h5")
    _save_disc(p, use_sn=True)
    assert_spectral_norm_consistency(True, p)  # no raise


def test_guard_raises_on_mismatch(tmp_path):
    # The exact failure from the live run: config says SN, file is plain conv.
    p = str(tmp_path / "plain.weights.h5")
    _save_disc(p, use_sn=False)
    with pytest.raises(ValueError, match="spectral normalization"):
        assert_spectral_norm_consistency(True, p)


def test_guard_noop_when_uninspectable():
    # Undeterminable -> don't block (best-effort guard, not a hard gate).
    assert_spectral_norm_consistency(True, None)
    assert_spectral_norm_consistency(True, "missing.weights.h5")
