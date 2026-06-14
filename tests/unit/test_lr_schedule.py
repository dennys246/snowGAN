"""Focused regression tests for the cosine LR schedule horizon.

Regression target: the cosine decay horizon was hard-coded to 200k steps in
`_update_learning_rates`. With fade_steps=50k that floored BOTH learning rates
at lr_min the moment global_step crossed 250k — pinning a long run at 1e-7 and
freezing learning (the observed post-250k destabilization of the core run).
The horizon is now config-driven (`lr_decay_steps`); these tests pin that
behavior so the freeze cannot silently return.
"""

import math

from snowgan.trainer import cosine_decayed_lr


BASE = 1e-4
LR_MIN = 1e-5


def test_starts_at_base():
    assert cosine_decayed_lr(BASE, LR_MIN, 0, 500_000) == BASE


def test_reaches_min_at_horizon_end():
    assert math.isclose(cosine_decayed_lr(BASE, LR_MIN, 500_000, 500_000), LR_MIN, rel_tol=1e-9)


def test_holds_min_past_horizon():
    assert math.isclose(cosine_decayed_lr(BASE, LR_MIN, 900_000, 500_000), LR_MIN, rel_tol=1e-9)


def test_midpoint_is_halfway():
    # cos(pi/2) = 0 -> factor 0.5 -> exactly halfway between min and base.
    mid = cosine_decayed_lr(BASE, LR_MIN, 250_000, 500_000)
    assert math.isclose(mid, LR_MIN + (BASE - LR_MIN) * 0.5, rel_tol=1e-9)


def test_not_floored_past_old_200k_boundary():
    """The core regression: at the step that USED to floor the LR (post-fade
    250k -> post_fade_step 200k under the old hard-coded horizon), a run whose
    real horizon is longer must still be actively learning, not pinned at min.
    """
    long_horizon = 500_000
    lr = cosine_decayed_lr(BASE, LR_MIN, 200_000, long_horizon)
    # Under the old hard-coded 200k horizon this point was exactly lr_min.
    # With a horizon matched to the real run it is still well above the floor.
    assert lr > LR_MIN * 5
    # And strictly below base (decay is actually happening, not disabled).
    assert lr < BASE
