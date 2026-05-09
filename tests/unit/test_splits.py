"""Unit tests for DataManager.derive_splits.

The split is taken at the group level ((site, column, core) triples) and
must be deterministic given a seed, idempotent across calls, and
JSON-friendly so AvAI's evaluation consumes the same test_pool the GAN
never saw. Tests bypass DataManager.__init__ (which calls load_dataset)
by constructing instances via __new__ and assigning a synthetic manifest
plus a SimpleNamespace config — same pattern as test_dataset.py.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from snowgan.data.dataset import DataManager


_COLUMNS = ["datatype", "site", "column", "core"]


def _manifest_with_n_groups(n: int) -> list[list]:
    """Build a manifest with N pairable groups, one core + one profile each."""
    rows: list[list] = []
    for i in range(n):
        rows.append([0, f"site{i}", i, 1])  # core
        rows.append([2, f"site{i}", i, 1])  # magnified profile
    return rows


def _make_manager(manifest: list[list], seed: int = 42) -> DataManager:
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = _COLUMNS
    dm.manifest = manifest
    dm.config = SimpleNamespace(
        trained_pool=None,
        validation_pool=None,
        test_pool=None,
        seed=seed,
    )
    return dm


def test_split_pools_are_disjoint_and_cover_all_groups():
    dm = _make_manager(_manifest_with_n_groups(20))
    expected_keys = set(dm.pair_index.keys())

    dm.derive_splits()

    train = {tuple(k) for k in dm.config.trained_pool}
    val = {tuple(k) for k in dm.config.validation_pool}
    test = {tuple(k) for k in dm.config.test_pool}

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == expected_keys


def test_split_sizes_follow_80_10_10():
    dm = _make_manager(_manifest_with_n_groups(20))
    dm.derive_splits()
    assert len(dm.config.trained_pool) == 16
    assert len(dm.config.validation_pool) == 2
    assert len(dm.config.test_pool) == 2


def test_split_is_deterministic_given_same_seed():
    dm_a = _make_manager(_manifest_with_n_groups(30), seed=7)
    dm_b = _make_manager(_manifest_with_n_groups(30), seed=7)

    dm_a.derive_splits()
    dm_b.derive_splits()

    assert dm_a.config.trained_pool == dm_b.config.trained_pool
    assert dm_a.config.validation_pool == dm_b.config.validation_pool
    assert dm_a.config.test_pool == dm_b.config.test_pool


def test_different_seed_yields_different_split():
    dm_a = _make_manager(_manifest_with_n_groups(30), seed=1)
    dm_b = _make_manager(_manifest_with_n_groups(30), seed=2)

    dm_a.derive_splits()
    dm_b.derive_splits()

    # Same set of keys partitioned, but the order/membership across pools
    # differs — at least one of the three pools must diverge.
    assert (
        dm_a.config.trained_pool != dm_b.config.trained_pool
        or dm_a.config.validation_pool != dm_b.config.validation_pool
        or dm_a.config.test_pool != dm_b.config.test_pool
    )


def test_derive_splits_is_idempotent_when_pools_populated():
    dm = _make_manager(_manifest_with_n_groups(20))
    dm.derive_splits()

    sentinel_train = list(dm.config.trained_pool)
    sentinel_val = list(dm.config.validation_pool)
    sentinel_test = list(dm.config.test_pool)

    # Re-run with the seed flipped — pools should not move because all three
    # are already populated.
    dm.config.seed = 999
    dm.derive_splits()

    assert dm.config.trained_pool == sentinel_train
    assert dm.config.validation_pool == sentinel_val
    assert dm.config.test_pool == sentinel_test


def test_pools_are_json_serializable():
    dm = _make_manager(_manifest_with_n_groups(10))
    dm.derive_splits()

    payload = {
        "trained_pool": dm.config.trained_pool,
        "validation_pool": dm.config.validation_pool,
        "test_pool": dm.config.test_pool,
    }
    serialized = json.dumps(payload)
    round_tripped = json.loads(serialized)

    assert round_tripped == payload
    # Each entry is itself a list (group key persisted as list, not tuple).
    for entry in dm.config.trained_pool + dm.config.validation_pool + dm.config.test_pool:
        assert isinstance(entry, list)


def test_empty_manifest_yields_empty_pools():
    dm = _make_manager([])
    dm.derive_splits()
    assert dm.config.trained_pool == []
    assert dm.config.validation_pool == []
    assert dm.config.test_pool == []
