"""Unit tests for the dataset/model depth contract.

UPGRADES #2 + #40 documented two failure modes of the original code:
silent weight loss when ``_ensure_depth_alignment`` rebuilt models
mid-training, and per-batch rebuilds under a mixed-depth manifest.
The fix centers on a single source of truth (``DataManager.pair_depth``,
derived from ``config.modality`` via ``pair_depth_for_modality``) and
a hard assertion in ``_ensure_depth_alignment`` that surfaces any
upstream contract violation instead of papering over it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from snowgan.data.dataset import DataManager, pair_depth_for_modality
from snowgan.modality import Modality
from snowgan.trainer import Trainer


# --------------------- pair_depth_for_modality contract ---------------------


def test_pair_depth_for_merged_matches_modality_count():
    # Modality enum and merged-mode pair_depth must agree — drifting either
    # without the other would let merge_images produce a tensor that doesn't
    # match the (PROFILE, CORE) ordering pinned in test_modality.py.
    assert pair_depth_for_modality("merged") == len(list(Modality))
    assert pair_depth_for_modality("merged") == 2


@pytest.mark.parametrize(
    "modality", ["magnified_profile", "core", "profile", "crystal_card"]
)
def test_pair_depth_for_single_modality_is_one(modality):
    assert pair_depth_for_modality(modality) == 1


def test_pair_depth_for_unknown_modality_raises():
    with pytest.raises(ValueError, match="Unknown modality"):
        pair_depth_for_modality("nonsense")


def test_merge_images_output_depth_matches_merged_pair_depth():
    # If merge_images is extended to stack more modalities, this test
    # fails and pair_depth_for_modality("merged") must be updated alongside.
    dm = DataManager.__new__(DataManager)
    profile = tf.zeros([8, 8, 3])
    core = tf.zeros([4, 4, 3])
    merged = dm.merge_images(core, profile)
    assert merged.shape[0] == pair_depth_for_modality("merged")


# --------------------- _ensure_depth_alignment assertion --------------------


@pytest.mark.parametrize("built_depth", [1, 2])
def test_ensure_depth_alignment_raises_on_mismatch(built_depth):
    trainer = Trainer.__new__(Trainer)
    trainer._built_depth = built_depth

    with pytest.raises(RuntimeError, match="does not match"):
        trainer._ensure_depth_alignment(built_depth + 1)


@pytest.mark.parametrize("built_depth", [1, 2])
def test_ensure_depth_alignment_is_noop_on_match(built_depth):
    trainer = Trainer.__new__(Trainer)
    trainer._built_depth = built_depth
    assert trainer._ensure_depth_alignment(built_depth) is None


def test_ensure_depth_alignment_tolerates_none():
    # depth=None reaches this method when an upstream batch is empty;
    # treat as a no-op rather than fabricating a depth assertion.
    trainer = Trainer.__new__(Trainer)
    trainer._built_depth = 1
    assert trainer._ensure_depth_alignment(None) is None


# --------------------- batch path does not mutate config --------------------


def _make_manager_with_fake_hf(manifest, modality):
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = ["datatype", "site", "column", "core"]
    dm.manifest = manifest
    dm.translator = {
        'core': 0, 'profile': 1, 'magnified_profile': 2, 'crystal_card': 3
    }
    dm.pair_depth = pair_depth_for_modality(modality)

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
        modality=modality,
    )
    return dm


def test_batch_merged_does_not_mutate_config_depth():
    """Regression: depth is a structural property of the dataset, not
    something per-batch code should mutate. The earlier bug had
    ``batch_merged`` writing back ``config.depth = merged.shape[0]``,
    competing with the trainer's depth assertion.
    """
    dm = _make_manager_with_fake_hf(
        [
            [0, "A", 1, 1],  # core
            [2, "A", 1, 1],  # magnified profile
        ],
        modality="merged",
    )
    batch = dm.batch_merged(batch_size=1)
    assert batch is not None
    assert batch.shape[1] == pair_depth_for_modality("merged")
    assert dm.config.depth == 99  # unchanged


def test_batch_does_not_mutate_config_depth():
    # Same regression for the single-modality `batch()` path.
    dm = _make_manager_with_fake_hf(
        [
            [2, "A", 1, 1],  # magnified profile
        ],
        modality="magnified_profile",
    )
    batch = dm.batch(batch_size=1, datatype="magnified_profile")
    assert batch is not None
    assert batch.shape[1] == pair_depth_for_modality("magnified_profile")
    assert dm.config.depth == 99  # unchanged
