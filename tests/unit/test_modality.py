"""Unit tests for the Modality depth-axis contract.

The depth axis of stacked snow tensors is ordered (PROFILE, CORE). These
tests pin the integer values and confirm DataManager.merge_images stacks
the inputs in that order.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from snowgan.modality import Modality
from snowgan.data.dataset import DataManager


def test_modality_integer_values_are_stable():
    # AvAI's data layer and any saved backbone weights depend on these
    # integer values. Renaming the enum members is fine; reordering them
    # is a breaking change.
    assert int(Modality.PROFILE) == 0
    assert int(Modality.CORE) == 1


def test_modality_is_reexported_from_snowgan_package():
    # AvAI plans to import via `from snowgan import Modality`. The
    # canonical definition lives at `snowgan.modality`; both must point
    # at the same enum class.
    from snowgan import Modality as Reexported
    assert Reexported is Modality


def test_merge_images_stacks_profile_then_core():
    # Sentinel pixel values let us identify which input ended up at each
    # depth index without depending on the resize implementation.
    profile = tf.fill([8, 8, 3], 1.0)
    core = tf.fill([4, 4, 3], 2.0)  # different spatial size to exercise resize

    dm = DataManager.__new__(DataManager)
    merged = dm.merge_images(core, profile)

    assert merged.shape == (2, 8, 8, 3)
    np.testing.assert_array_equal(
        merged[Modality.PROFILE].numpy(), profile.numpy()
    )
    # Core was resized to profile's spatial dims; fill value is preserved
    # by bilinear resize of a constant tensor, so every pixel is 2.0.
    np.testing.assert_allclose(
        merged[Modality.CORE].numpy(), np.full((8, 8, 3), 2.0)
    )
