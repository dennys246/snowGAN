"""Regression: trainer persists derived splits to *both* configs.

The original PR #9 only persisted splits to the generator config, because
``DataManager.derive_splits`` mutates the config its ``DataManager`` was
constructed with — and ``Trainer`` builds the manager from
``self.gen.config``. ``discriminator_config.json`` was left with null pools,
which blocks AvAI's transfer-learning evaluation (it reads ``test_pool``
from the discriminator-side config).

This test pins ``Trainer._persist_initial_splits`` against the four states
the bug taught us to care about:

  - both configs null on a fresh run (writes both)
  - both configs already populated (idempotent, no write)
  - the torn state on disk today: gen has splits, disc is null (repairs disc)
  - the data must match between the two configs (no silent divergence)
"""
from __future__ import annotations

from types import SimpleNamespace

from snowgan.data.dataset import DataManager
from snowgan.trainer import Trainer


_COLUMNS = ["datatype", "site", "column", "core"]


def _manifest_with_n_groups(n: int) -> list[list]:
    rows: list[list] = []
    for i in range(n):
        rows.append([0, f"site{i}", i, 1])  # core
        rows.append([2, f"site{i}", i, 1])  # magnified profile
    return rows


class _RecordingConfig(SimpleNamespace):
    """SimpleNamespace whose ``save_config()`` records call count.

    Avoids pulling in ``snowgan.config.build`` (which registers an
    atexit hook and touches the filesystem); the trainer code only
    requires the three pool attributes and a ``save_config()``.
    """

    def save_config(self):
        self.save_calls = getattr(self, "save_calls", 0) + 1


def _make_dataset(manifest, gen_cfg) -> DataManager:
    dm = DataManager.__new__(DataManager)
    dm.manifest_columns = _COLUMNS
    dm.manifest = manifest
    dm.config = gen_cfg
    return dm


def _make_trainer_stub(gen_cfg, disc_cfg, dataset) -> Trainer:
    t = Trainer.__new__(Trainer)
    t.gen = SimpleNamespace(config=gen_cfg)
    t.disc = SimpleNamespace(config=disc_cfg)
    t.dataset = dataset
    return t


def _fresh_configs():
    gen_cfg = _RecordingConfig(
        trained_pool=None,
        validation_pool=None,
        test_pool=None,
        seed=42,
    )
    disc_cfg = _RecordingConfig(
        trained_pool=None,
        validation_pool=None,
        test_pool=None,
        seed=42,
    )
    return gen_cfg, disc_cfg


def test_disc_config_receives_splits_alongside_gen():
    """After the first init, disc.config has the same pools as gen.config."""
    gen_cfg, disc_cfg = _fresh_configs()
    dm = _make_dataset(_manifest_with_n_groups(10), gen_cfg)
    trainer = _make_trainer_stub(gen_cfg, disc_cfg, dm)

    trainer._persist_initial_splits()

    assert disc_cfg.trained_pool == gen_cfg.trained_pool
    assert disc_cfg.validation_pool == gen_cfg.validation_pool
    assert disc_cfg.test_pool == gen_cfg.test_pool
    # Manifest has 10 pairable groups -> at least one entry must land in each
    # pool (80/10/10 split rounds to 8/1/1).
    assert disc_cfg.trained_pool
    assert disc_cfg.test_pool


def test_save_called_on_both_when_freshly_derived():
    gen_cfg, disc_cfg = _fresh_configs()
    dm = _make_dataset(_manifest_with_n_groups(10), gen_cfg)
    trainer = _make_trainer_stub(gen_cfg, disc_cfg, dm)

    trainer._persist_initial_splits()

    assert getattr(gen_cfg, "save_calls", 0) == 1
    assert getattr(disc_cfg, "save_calls", 0) == 1


def test_save_skipped_when_both_already_populated():
    """Idempotent: if both pools are present, the immediate save is skipped.

    The periodic ``_sync_fade_progress`` save still persists later state;
    skipping here just avoids an extra disk write on resume.
    """
    populated = [[0, 0, 0]]
    gen_cfg = _RecordingConfig(
        trained_pool=populated,
        validation_pool=populated,
        test_pool=populated,
        seed=42,
    )
    disc_cfg = _RecordingConfig(
        trained_pool=populated,
        validation_pool=populated,
        test_pool=populated,
        seed=42,
    )
    dm = _make_dataset(_manifest_with_n_groups(10), gen_cfg)
    trainer = _make_trainer_stub(gen_cfg, disc_cfg, dm)

    trainer._persist_initial_splits()

    assert getattr(gen_cfg, "save_calls", 0) == 0
    assert getattr(disc_cfg, "save_calls", 0) == 0


def test_save_fires_when_only_gen_was_previously_populated():
    """Repair path: matches the torn state of the two existing on-disk runs
    (gen has splits, disc is null). Mirror + save must run so the next time
    AvAI reads discriminator_config.json the pools are present.
    """
    populated = [[0, 0, 0]]
    gen_cfg = _RecordingConfig(
        trained_pool=populated,
        validation_pool=populated,
        test_pool=populated,
        seed=42,
    )
    disc_cfg = _RecordingConfig(
        trained_pool=None,
        validation_pool=None,
        test_pool=None,
        seed=42,
    )
    dm = _make_dataset(_manifest_with_n_groups(10), gen_cfg)
    trainer = _make_trainer_stub(gen_cfg, disc_cfg, dm)

    trainer._persist_initial_splits()

    # derive_splits is idempotent on gen (already populated); the mirror
    # copies the populated pools onto disc and triggers the save.
    assert disc_cfg.trained_pool == gen_cfg.trained_pool
    assert disc_cfg.validation_pool == gen_cfg.validation_pool
    assert disc_cfg.test_pool == gen_cfg.test_pool
    assert getattr(disc_cfg, "save_calls", 0) == 1
    assert getattr(gen_cfg, "save_calls", 0) == 1


def test_save_failure_is_swallowed_with_warning(capsys):
    """A failing save_config must not crash trainer init — splits are still
    correct in memory, and the next periodic save can retry. (Mirrors the
    pre-fix behavior for gen.config.)
    """
    gen_cfg, disc_cfg = _fresh_configs()

    def _raise(self_):
        raise IOError("simulated disk full")

    # Override save_config on disc_cfg to raise.
    disc_cfg.save_config = lambda: (_ for _ in ()).throw(IOError("simulated disk full"))

    dm = _make_dataset(_manifest_with_n_groups(10), gen_cfg)
    trainer = _make_trainer_stub(gen_cfg, disc_cfg, dm)

    # Should not raise.
    trainer._persist_initial_splits()

    # Splits still in memory on both configs.
    assert disc_cfg.trained_pool == gen_cfg.trained_pool
    captured = capsys.readouterr()
    assert "failed to persist derived data splits" in captured.out
