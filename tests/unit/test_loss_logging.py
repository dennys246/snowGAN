"""Unit tests for the per-train_step mean-loss logging contract.

UPGRADES #33: train_step previously appended only the final iteration's
loss after the inner update loops, hiding two of every three values
under the default `gen_steps=3`. The recorded curve must reflect the
mean across all inner iterations so AvAI's "frozen backbone trained on
this curve" claim is supportable.
"""
from __future__ import annotations

import pytest

from snowgan.trainer import Trainer


def test_mean_loss_returns_arithmetic_mean():
    # The doc-specified example: gen_steps=3, per-step values [1.0, 2.0, 3.0]
    # → recorded value 2.0 (the mean), not 3.0 (the last).
    assert Trainer._mean_loss([1.0, 2.0, 3.0]) == 2.0


def test_mean_loss_handles_single_element():
    assert Trainer._mean_loss([4.5]) == 4.5


def test_mean_loss_handles_empty_list():
    # Guard against training_steps=0: the appended value must always be
    # a finite float so downstream plotting and EMA updates don't break.
    assert Trainer._mean_loss([]) == 0.0


def test_mean_loss_handles_negative_values():
    # WGAN-GP losses can be negative (Wasserstein distance is unbounded);
    # the mean must compute correctly across mixed-sign values.
    assert Trainer._mean_loss([-1.0, 1.0]) == 0.0
    assert Trainer._mean_loss([-3.0, -1.0, 2.0]) == pytest.approx(-2.0 / 3.0)
