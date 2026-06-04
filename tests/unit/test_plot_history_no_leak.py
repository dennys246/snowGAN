"""Regression test: plot_history must reuse a single figure, not leak one per call.

Creating a fresh matplotlib figure on every plot_history() call leaked CPU RAM
in the Agg backend even with plt.close(fig) (isolation test: ~1.8 GB over 300
calls). Since plot_history runs every ~10 batches over long runs, that was a
multi-GB CPU-RAM source. The fix reuses one persistent figure and clears its
axes each call. This test fails under the old per-call-figure behavior (the
pyplot figure registry grows with every call) and passes under the fix.
"""
from __future__ import annotations

import os

import pytest

# Trainer imports TensorFlow at module scope, so this runs in the TF env
# (WSL ml-env / full-suite gate), not the CPU-only Windows dev box.
pytest.importorskip("tensorflow")

from matplotlib import pyplot as plt  # noqa: E402

from snowgan.trainer import Trainer  # noqa: E402


def test_plot_history_reuses_single_figure(tmp_path):
    # Build a Trainer shell without the heavy __init__ (mirrors test_dataset.py's
    # __new__ approach); plot_history only needs save_dir, loss, and the two
    # persistent-figure handles.
    t = Trainer.__new__(Trainer)
    t.save_dir = f"{tmp_path}{os.sep}"
    t.loss = {"gen": [1.0, 2.0, 3.0], "disc": [0.5, 0.25, 0.1]}
    t._history_fig = None
    t._history_ax = None

    fignums_before = len(plt.get_fignums())
    for _ in range(25):
        t.plot_history()
    fignums_after = len(plt.get_fignums())

    # At most one new figure should exist regardless of how many calls happen.
    assert fignums_after - fignums_before <= 1
    assert os.path.exists(os.path.join(str(tmp_path), "history.png"))

    plt.close(t._history_fig)
