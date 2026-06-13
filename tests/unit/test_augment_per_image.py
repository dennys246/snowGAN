"""Focused tests for per-image differentiable augmentation.

The v0.1 augment drew a single scalar decision/magnitude per *batch* — one
coin flip and one delta applied identically to every image — so augmentation
entropy was far below what DiffAugment/ADA assume. These tests pin that each
image in a batch is now augmented independently, while shape and the [-1, 1]
range are preserved.
"""

import numpy as np
import tensorflow as tf

from snowgan import augment as A


def _const_batch(n=8, rank=4):
    shape = (n, 8, 8, 3) if rank == 4 else (n, 2, 8, 8, 3)
    return tf.zeros(shape, dtype=tf.float32)


def test_brightness_is_per_image():
    tf.random.set_seed(0)
    out = A._random_brightness(_const_batch(), p=1.0, max_delta=0.2).numpy()
    per_image = out.reshape(out.shape[0], -1).mean(axis=1)
    # Distinct deltas per image -> the per-image means are not all equal.
    assert np.unique(np.round(per_image, 6)).size > 1


def test_cutout_fires_per_image():
    tf.random.set_seed(1)
    out = A._random_cutout(_const_batch() + 1.0, p=0.5, ratio=0.25).numpy()
    # Some images keep all ones (didn't fire), others have zeroed pixels.
    zero_fraction = (out == 0.0).reshape(out.shape[0], -1).mean(axis=1)
    assert zero_fraction.min() == 0.0  # at least one image untouched
    assert zero_fraction.max() > 0.0   # at least one image cut


def test_augment_preserves_shape_and_range_5d():
    tf.random.set_seed(2)
    x = tf.random.uniform((4, 2, 8, 8, 3), minval=-1.0, maxval=1.0)
    out = A.augment(x, p=0.8).numpy()
    assert out.shape == (4, 2, 8, 8, 3)
    assert out.min() >= -1.0 and out.max() <= 1.0


def test_flip_per_image_decision():
    tf.random.set_seed(3)
    # Asymmetric pattern so a width-flip is detectable; identical across batch.
    base = tf.reshape(tf.range(8, dtype=tf.float32) / 8.0, (1, 1, 8, 1))
    x = tf.tile(base, (16, 8, 1, 3))
    out = A._random_flip(x, p=0.5).numpy()
    flipped = np.isclose(out, np.flip(x.numpy(), axis=2))
    all_flipped = flipped.reshape(out.shape[0], -1).all(axis=1)
    # With p=0.5 over 16 images, expect a mix of flipped and not-flipped.
    assert all_flipped.any() and not all_flipped.all()
