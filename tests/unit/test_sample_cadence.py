"""Unit tests for per-batch seeded-preview sample emission.

The trainer used to emit synthetic preview PNGs every batch, then
(in commit 10cba23, May 2026) moved emission to epoch-end only. Long
runs that never close an epoch (1024x1024 manifests, mid-epoch crashes)
were silently sample-starved. ``sample_batch_interval`` restores
opt-in per-batch emission; this file pins the predicate that gates it
and the config round-trip that persists it.
"""
from __future__ import annotations

import json
import os

import pytest

from snowgan.config import build, config_template
from snowgan.trainer import Trainer


# ---------------------------------------------------------------------------
# Predicate: Trainer._should_emit_batch_sample
# ---------------------------------------------------------------------------

def test_predicate_off_when_interval_zero():
    # interval=0 is the documented "off" sentinel. Must never emit.
    assert Trainer._should_emit_batch_sample(batch=1, interval=0, n_samples=10) is False
    assert Trainer._should_emit_batch_sample(batch=100, interval=0, n_samples=10) is False


def test_predicate_off_when_n_samples_zero():
    # n_samples=0 also disables — there's nothing to emit.
    assert Trainer._should_emit_batch_sample(batch=100, interval=10, n_samples=0) is False


def test_predicate_off_when_interval_negative():
    # Negative intervals are nonsense; treat as off rather than crashing
    # with a ZeroDivision-ish modulo behaviour or emitting weirdly.
    assert Trainer._should_emit_batch_sample(batch=10, interval=-1, n_samples=10) is False


def test_predicate_fires_on_multiples():
    # Mirrors the existing checkpoint-cadence semantics: batch is
    # 1-indexed in the train loop, so an interval of N fires at
    # batch == N, 2N, 3N, ...
    assert Trainer._should_emit_batch_sample(batch=100, interval=100, n_samples=10) is True
    assert Trainer._should_emit_batch_sample(batch=200, interval=100, n_samples=10) is True


def test_predicate_skips_non_multiples():
    assert Trainer._should_emit_batch_sample(batch=99, interval=100, n_samples=10) is False
    assert Trainer._should_emit_batch_sample(batch=101, interval=100, n_samples=10) is False
    assert Trainer._should_emit_batch_sample(batch=150, interval=100, n_samples=10) is False


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------

def test_template_includes_sample_batch_interval():
    # The template is the source of truth for new configs; any field the
    # trainer reads from config must exist here so brand-new save_dirs
    # don't AttributeError on first run.
    assert "sample_batch_interval" in config_template


def test_config_default_is_off(tmp_path):
    # Default value is 0 (off) so existing runs aren't surprised by a
    # sudden I/O storm when they upgrade.
    cfg_path = tmp_path / "generator_config.json"
    cfg = build(str(cfg_path))
    assert cfg.sample_batch_interval == 0


def test_config_round_trips_through_save_load(tmp_path):
    # Persist a non-default value, re-load via build(), and confirm the
    # field survives the JSON round-trip — protects against silent drops
    # in dump() or configure().
    cfg_path = tmp_path / "generator_config.json"
    cfg = build(str(cfg_path))
    cfg.sample_batch_interval = 250
    cfg.save_config(str(cfg_path))

    with open(cfg_path) as f:
        on_disk = json.load(f)
    assert on_disk["sample_batch_interval"] == 250

    reloaded = build(str(cfg_path))
    assert reloaded.sample_batch_interval == 250


def test_legacy_config_without_field_defaults_to_zero(tmp_path):
    # Configs written before this field landed must keep loading. The
    # setdefault() in build.__init__ provides the back-compat path; if
    # someone deletes that line, this test fails loudly.
    cfg_path = tmp_path / "generator_config.json"
    legacy = {k: v for k, v in config_template.items() if k != "sample_batch_interval"}
    with open(cfg_path, "w") as f:
        json.dump(legacy, f)

    cfg = build(str(cfg_path))
    assert cfg.sample_batch_interval == 0
