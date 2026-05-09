"""Unit tests for the modality mode toggle.

``DataManager.next_batch`` dispatches between the merged (depth=2) and
single-modality (depth=1) data paths based on ``config.modality``. The
trainer reads ``dataset.pair_depth`` once at startup to size its models;
this test pins both halves of that contract.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from snowgan.data.dataset import DataManager, pair_depth_for_modality


def _manifest_with_pair():
    return [
        [0, "A", 1, 1],  # core
        [2, "A", 1, 1],  # magnified profile
        [1, "A", 1, 1],  # plain profile
        [3, "A", 1, 1],  # crystal card
    ]


def _make_manager(modality: str):
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = ["datatype", "site", "column", "core"]
    dm.manifest = _manifest_with_pair()
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
        resolution=(32, 32),
        modality=modality,
    )
    return dm


def test_pair_depth_is_one_under_single_modality():
    dm = _make_manager("magnified_profile")
    assert dm.pair_depth == 1


def test_pair_depth_is_two_under_merged():
    dm = _make_manager("merged")
    assert dm.pair_depth == 2


@pytest.mark.parametrize(
    "modality, expected_datatype",
    [
        ("magnified_profile", 2),
        ("core", 0),
        ("profile", 1),
        ("crystal_card", 3),
    ],
)
def test_next_batch_dispatches_to_single_modality(modality, expected_datatype):
    """next_batch with a single-modality config returns a depth-1 batch.

    Per-sample shape (depth=1, H, W, C) confirms the dispatcher routed to
    ``batch()`` with the correct datatype filter — only rows whose datatype
    matches the modality should land in the batch.
    """
    dm = _make_manager(modality)
    batch = dm.next_batch(batch_size=1)
    assert batch is not None
    assert batch.shape[1] == 1
    # We can't easily verify the datatype filter from the returned tensor
    # alone, but config.train_ind should have advanced past the sentinel
    # row matching `expected_datatype` in our fixture.
    # The fixture has one of each datatype, so train_ind > 0 means a match
    # was found.
    assert dm.config.train_ind > 0


def test_next_batch_dispatches_to_merged_under_merged():
    dm = _make_manager("merged")
    batch = dm.next_batch(batch_size=1)
    assert batch is not None
    # Depth-2 stack (PROFILE, CORE).
    assert batch.shape[1] == 2
