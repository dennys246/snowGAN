"""Unit tests for snowgan.data.dataset.DataManager.

Tests bypass DataManager.__init__ (which calls load_dataset) by constructing
instances via __new__ and assigning a synthetic in-memory manifest. This keeps
the suite hermetic — no HF download, no TF eager work, sub-second runtime.
"""
from __future__ import annotations

import pytest

from snowgan.data.dataset import DataManager


def _make_manager(manifest_columns: list[str], manifest: list[list]) -> DataManager:
    """Construct a DataManager wired only for manifest-level introspection.

    pair_index reads only self.manifest and self.manifest_columns via
    self._get_manifest_entry, so the HF dataset, config, and seen-set
    attributes are intentionally absent here.
    """
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = manifest_columns
    dm.manifest = manifest
    return dm


# Synthetic manifest from the spec: group A has 2 cores × 3 profiles (6 pairs),
# group B has 1 core × 1 profile (1 pair). Indices are interleaved deliberately
# so a correct implementation must match by (site, column, core), not adjacency.
_COLUMNS = ["datatype", "site", "column", "core"]
_MANIFEST = [
    [0, "A", 1, 1],   # idx 0 — core, group A
    [2, "A", 1, 1],   # idx 1 — profile, group A
    [0, "A", 1, 1],   # idx 2 — core, group A
    [2, "B", 2, 1],   # idx 3 — profile, group B
    [2, "A", 1, 1],   # idx 4 — profile, group A
    [0, "B", 2, 1],   # idx 5 — core, group B
    [2, "A", 1, 1],   # idx 6 — profile, group A
]


def test_pair_index_returns_full_cartesian_product():
    dm = _make_manager(_COLUMNS, _MANIFEST)

    pi = dm.pair_index

    assert set(pi.keys()) == {("A", 1, 1), ("B", 2, 1)}

    a_pairs = set(pi[("A", 1, 1)])
    expected_a = {(c, p) for c in (0, 2) for p in (1, 4, 6)}
    assert a_pairs == expected_a
    assert len(pi[("A", 1, 1)]) == 6

    assert pi[("B", 2, 1)] == [(5, 3)]
    assert len(pi[("B", 2, 1)]) == 1


def test_pair_index_skips_groups_missing_a_modality():
    # Group "lonely" has profiles but no core; group "core_only" has the inverse.
    # Neither should appear in pair_index — a pair requires both endpoints.
    columns = ["datatype", "site", "column", "core"]
    manifest = [
        [2, "lonely", 1, 1],
        [2, "lonely", 1, 1],
        [0, "core_only", 1, 1],
        [0, "paired", 1, 1],
        [2, "paired", 1, 1],
    ]
    dm = _make_manager(columns, manifest)

    pi = dm.pair_index

    assert set(pi.keys()) == {("paired", 1, 1)}
    assert pi[("paired", 1, 1)] == [(3, 4)]


def test_pair_index_ignores_unrelated_datatypes():
    # datatype=1 (profile) and datatype=3 (crystal_card) must not enter the index;
    # the pair contract is core (0) × magnified_profile (2) only.
    columns = ["datatype", "site", "column", "core"]
    manifest = [
        [0, "A", 1, 1],
        [1, "A", 1, 1],   # plain profile — ignore
        [2, "A", 1, 1],
        [3, "A", 1, 1],   # crystal_card — ignore
    ]
    dm = _make_manager(columns, manifest)

    pi = dm.pair_index

    assert pi == {("A", 1, 1): [(0, 2)]}


def test_pair_index_does_not_mutate_manager_state():
    dm = _make_manager(_COLUMNS, _MANIFEST)
    # Sentinel attributes that batch / batch_merged would touch. pair_index must
    # not read or write any of them.
    config = type("C", (), {"train_ind": 0, "seen_profiles": set()})()
    dm.config = config
    dm.seen_profiles = set()
    dm.seen_cores = set()

    _ = dm.pair_index

    assert dm.config.train_ind == 0
    assert dm.config.seen_profiles == set()
    assert dm.seen_profiles == set()
    assert dm.seen_cores == set()


def test_pair_index_is_cached():
    dm = _make_manager(_COLUMNS, _MANIFEST)
    first = dm.pair_index
    second = dm.pair_index
    assert first is second


def test_pair_index_handles_empty_manifest():
    dm = _make_manager(_COLUMNS, [])
    assert dm.pair_index == {}
