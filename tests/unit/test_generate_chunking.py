"""Focused test: preview generation forwards in sub-batches.

Regression target: generate() forwarded all `count` images in a single
generator call, so preview VRAM scaled with n_samples. At 1024² a 10-wide
forward materialized multi-GB UpSampling intermediates and OOM'd on top of the
resident training graph. Chunking caps peak VRAM at `gen_batch` images while
still writing `count` files.
"""

import os

import tensorflow as tf

from snowgan.generate import generate


class _StubGen:
    """Records the batch size of every forward call."""

    def __init__(self):
        self.call_sizes = []

    def __call__(self, seed, training=False):
        n = int(seed.shape[0])
        self.call_sizes.append(n)
        # (n, depth=1, H, W, C) so each image is rank-4 with depth 1.
        return tf.zeros([n, 1, 4, 4, 3], dtype=tf.float32)


def test_generate_chunks_into_subbatches(tmp_path):
    gen = _StubGen()
    out = str(tmp_path) + "/"
    generate(gen, count=5, seed_size=8, save_dir=out, filename_prefix="x", gen_batch=2)

    # 5 images at gen_batch=2 -> forwards of 2, 2, 1.
    assert gen.call_sizes == [2, 2, 1]
    pngs = [f for f in os.listdir(tmp_path) if f.endswith(".png")]
    assert len(pngs) == 5


def test_generate_default_batch_is_one(tmp_path):
    gen = _StubGen()
    out = str(tmp_path) + "/"
    generate(gen, count=3, seed_size=8, save_dir=out, filename_prefix="y")
    # Default gen_batch=1 -> one image per forward.
    assert gen.call_sizes == [1, 1, 1]
    assert len([f for f in os.listdir(tmp_path) if f.endswith(".png")]) == 3
