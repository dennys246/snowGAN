"""Unit tests for the dataset/model depth contract.

UPGRADES #2 + #40 documented two failure modes of the previous code:
silent weight loss when ``_ensure_depth_alignment`` rebuilt models
mid-training, and per-batch rebuilds under a mixed-depth manifest. The
fix centers on a single source of truth (``DataManager.PAIR_DEPTH``)
and a hard assertion in ``_ensure_depth_alignment`` that surfaces any
upstream contract violation instead of papering over it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from snowgan.data.dataset import DataManager
from snowgan.modality import Modality
from snowgan.trainer import Trainer


def test_pair_depth_matches_modality_count():
    # Modality enum and PAIR_DEPTH must agree — they are two faces of the
    # same contract. A drift between them would silently let merge_images
    # produce a tensor that doesn't fit the (PROFILE, CORE) ordering test
    # in test_modality.py.
    assert DataManager.PAIR_DEPTH == len(list(Modality))
    assert DataManager.PAIR_DEPTH == 2


def test_merge_images_output_depth_matches_pair_depth():
    # If merge_images is ever extended to stack more modalities, this
    # test fails and PAIR_DEPTH must be updated alongside.
    dm = DataManager.__new__(DataManager)
    profile = tf.zeros([8, 8, 3])
    core = tf.zeros([4, 4, 3])
    merged = dm.merge_images(core, profile)
    assert merged.shape[0] == DataManager.PAIR_DEPTH


def test_ensure_depth_alignment_raises_on_mismatch():
    trainer = Trainer.__new__(Trainer)
    trainer._built_depth = DataManager.PAIR_DEPTH

    with pytest.raises(RuntimeError, match="does not match"):
        trainer._ensure_depth_alignment(DataManager.PAIR_DEPTH + 1)


def test_ensure_depth_alignment_is_noop_on_match():
    trainer = Trainer.__new__(Trainer)
    trainer._built_depth = DataManager.PAIR_DEPTH

    # Returns None, does not raise.
    assert trainer._ensure_depth_alignment(DataManager.PAIR_DEPTH) is None


def test_ensure_depth_alignment_tolerates_none():
    # depth=None reaches this method when an upstream batch is empty;
    # treat as a no-op rather than fabricating a depth assertion.
    trainer = Trainer.__new__(Trainer)
    trainer._built_depth = DataManager.PAIR_DEPTH
    assert trainer._ensure_depth_alignment(None) is None


def test_batch_merged_does_not_mutate_config_depth(monkeypatch):
    """Regression: depth is a structural property of the dataset, not
    something per-batch code should mutate. The earlier bug had
    ``batch_merged`` writing back ``config.depth = merged.shape[0]``,
    which competed with the trainer's depth assertion.
    """
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = ["datatype", "site", "column", "core"]
    dm.manifest = [
        [0, "A", 1, 1],  # core
        [2, "A", 1, 1],  # magnified profile
    ]

    # Stub the HF dataset access that batch_merged performs. Returns
    # synthetic uint8 image arrays; the rest of preprocess_image is real.
    fake_image = np.zeros((64, 64, 3), dtype=np.uint8)

    class _FakeHFDataset:
        def __getitem__(self, idx):
            row = dm.manifest[idx]
            return {
                "image": fake_image,
                "datatype": row[0],
                "site": row[1],
                "column": row[2],
                "core": row[3],
                "segment": "test",
            }

    dm.dataset = {"train": _FakeHFDataset()}
    dm.seen_profiles = set()
    dm.seen_cores = set()
    dm.config = SimpleNamespace(
        train_ind=0,
        seen_profiles=set(),
        resolution=(64, 64),
        depth=99,  # sentinel — must not be mutated
    )

    batch = dm.batch_merged(batch_size=1)

    assert batch is not None
    assert batch.shape[1] == DataManager.PAIR_DEPTH
    assert dm.config.depth == 99  # unchanged
