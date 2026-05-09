"""Unit tests for snowgan.utils.set_seed.

UPGRADES #12: training runs must be replayable across same-host
restarts. set_seed pins ``random``, ``numpy``, and ``tf.random`` so
weight init, noise sampling, and dataset shuffling all produce
identical sequences when the same seed is applied.
"""
from __future__ import annotations

import os
import random as py_random

import numpy as np
import tensorflow as tf

from snowgan.utils import set_seed


def test_set_seed_makes_python_random_deterministic():
    set_seed(42)
    first = [py_random.random() for _ in range(5)]
    set_seed(42)
    second = [py_random.random() for _ in range(5)]
    assert first == second


def test_set_seed_makes_numpy_random_deterministic():
    set_seed(7)
    first = np.random.rand(5)
    set_seed(7)
    second = np.random.rand(5)
    np.testing.assert_array_equal(first, second)


def test_set_seed_makes_tensorflow_random_deterministic():
    set_seed(123)
    first = tf.random.normal([5]).numpy()
    set_seed(123)
    second = tf.random.normal([5]).numpy()
    np.testing.assert_array_equal(first, second)


def test_different_seeds_yield_different_streams():
    # Sanity: the seed is honored, not ignored. With different seeds,
    # at least one of the three RNGs should diverge.
    set_seed(1)
    first_py = py_random.random()
    set_seed(2)
    second_py = py_random.random()
    assert first_py != second_py


def test_set_seed_writes_env_vars():
    # Best-effort env-var seeding for PYTHONHASHSEED and
    # TF_DETERMINISTIC_OPS. These only fully take effect when set
    # before interpreter / TF startup; UPGRADES #4 tracks that.
    set_seed(99)
    assert os.environ["PYTHONHASHSEED"] == "99"
    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"
