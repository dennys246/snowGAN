"""Regression test for the CPU-RAM leak in the preview path.

generate() used to end with ``return np.stack(synthetic_images)`` — a full
copy of the entire preview batch into CPU RAM (e.g. 10x1024x1024x3 ~ 125 MB)
that no caller used (main.py discards it with ``_ =`` and the trainer preview
ignores it). On the per-batch preview cadence that dead allocation ratcheted
process RSS upward via glibc arena fragmentation until long runs were
OOM-killed. The contract now: generate() writes files and returns nothing.

This test fails under the old behavior (return value is an ndarray) and passes
under the fix (return value is None), while still proving the images are
written so the removal didn't break the preview's actual job.
"""
from __future__ import annotations

import os

import pytest

# generate.py imports TensorFlow at module scope, so this test only runs in an
# env where TF is installed (the WSL training env / full-suite gate), not on the
# CPU-only Windows dev box.
pytest.importorskip("tensorflow")

import tensorflow as tf  # noqa: E402

from snowgan.generate import generate  # noqa: E402


def test_generate_returns_none_and_still_writes_files(tmp_path):
    def fake_generator(seed, training=False):
        # Mirror the real rank-5 (B, depth, H, W, C) preview shape at a tiny
        # resolution so the test stays sub-second.
        n = int(seed.shape[0])
        return tf.zeros([n, 1, 8, 8, 3], dtype=tf.float32)

    result = generate(
        fake_generator,
        count=2,
        seed_size=4,
        save_dir=f"{tmp_path}{os.sep}",
        filename_prefix="synthetic",
    )

    # The leak fix: no stacked ndarray handed back to a caller that ignores it.
    assert result is None
    # The preview still does its real job — one PNG per generated sample.
    assert len(list(tmp_path.glob("*.png"))) == 2
