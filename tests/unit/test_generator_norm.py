"""Focused tests for generator normalization (PixelNorm) and added depth.

The v0.1 generator had NO normalization (batch_norm=false, nothing replacing
it) and one conv per resolution. Unbounded activation scale drove the final
tanh into saturation (the monochrome-blue collapse) and the high-resolution
stages were too thin to synthesize texture. v0.2 adds GP-safe PixelNorm and a
second conv per resolution. These tests pin the layer math and that gen_norm
selects/derives correctly.
"""

import types

import numpy as np
import tensorflow as tf

from snowgan.models.generator import Generator, PixelNorm


def _tiny_config(**over):
    cfg = types.SimpleNamespace(
        latent_dim=16,
        filter_counts=[8],
        kernel_size=[3, 3],
        kernel_stride=[2, 2],
        padding="same",
        batch_norm=False,
        gen_norm="pixel",
        negative_slope=0.25,
        channels=3,
        final_activation="tanh",
        depth=1,
        learning_rate=1e-4,
        beta_1=0.5,
        beta_2=0.9,
    )
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def test_pixelnorm_unit_rms_per_pixel():
    tf.random.set_seed(0)
    x = tf.random.normal([2, 1, 4, 4, 16]) * 7.0 + 3.0  # arbitrary scale/offset
    y = PixelNorm()(x).numpy()
    # Each pixel's channel vector should have ~unit mean-square.
    ms = np.mean(np.square(y), axis=-1)
    assert np.allclose(ms, 1.0, atol=1e-4), ms.max()


def test_pixelnorm_layer_present_when_selected():
    tf.random.set_seed(0)
    gen = Generator(_tiny_config(gen_norm="pixel"))
    assert any(isinstance(layer, PixelNorm) for layer in gen.model.layers)


def test_gen_norm_derives_from_batch_norm_when_unset():
    # Legacy config: gen_norm absent, batch_norm False -> no norm layers.
    tf.random.set_seed(0)
    cfg = _tiny_config(batch_norm=False)
    delattr(cfg, "gen_norm")
    gen = Generator(cfg)
    names = [type(layer).__name__ for layer in gen.model.layers]
    assert not any(n in ("PixelNorm", "BatchNormalization") for n in names), names


def test_two_convs_per_resolution():
    # One filter block -> two stride-1 Conv3D feature convs + the 1x1 toRGB.
    tf.random.set_seed(0)
    gen = Generator(_tiny_config())
    conv3d = [l for l in gen.model.layers if type(l).__name__ == "Conv3D"]
    assert len(conv3d) == 3, [l.name for l in conv3d]


def test_output_still_correct_shape_and_range():
    tf.random.set_seed(0)
    gen = Generator(_tiny_config())
    out = gen(tf.random.normal([2, 16]), training=False).numpy()
    assert out.shape == (2, 1, 64, 64, 3), out.shape
    assert out.min() >= -1.0 and out.max() <= 1.0
