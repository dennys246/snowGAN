"""Focused tests for v0.2 config-schema back-compat and round-trip.

The config is a shared loose-kwarg schema (also used by the magnified_profile
backbone), so new v0.2 fields must load on legacy configs that predate them and
round-trip through dump() intact.
"""

import copy
import json

from snowgan.config import build, config_template


def _write(tmp_path, data):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_legacy_config_without_new_fields_loads(tmp_path):
    # Simulate a pre-v0.2 config: strip the fields added this branch.
    data = copy.deepcopy(config_template)
    for key in ("gen_norm", "lr_decay_steps"):
        data.pop(key, None)
    cfg = build(_write(tmp_path, data))

    # Defaults applied without error.
    assert cfg.lr_decay_steps == 0
    # gen_norm derives from legacy batch_norm (False -> "none").
    assert cfg.gen_norm == "none"


def test_gen_norm_derives_from_batch_norm_true(tmp_path):
    data = copy.deepcopy(config_template)
    data.pop("gen_norm", None)
    data["batch_norm"] = True
    cfg = build(_write(tmp_path, data))
    assert cfg.gen_norm == "batch"


def test_explicit_gen_norm_wins(tmp_path):
    data = copy.deepcopy(config_template)
    data["batch_norm"] = False
    data["gen_norm"] = "pixel"
    cfg = build(_write(tmp_path, data))
    assert cfg.gen_norm == "pixel"


def test_new_fields_round_trip(tmp_path):
    data = copy.deepcopy(config_template)
    data["gen_norm"] = "pixel"
    data["lr_decay_steps"] = 400000
    data["lambda_gp"] = 0.0
    cfg = build(_write(tmp_path, data))
    dumped = cfg.dump()
    assert dumped["gen_norm"] == "pixel"
    assert dumped["lr_decay_steps"] == 400000
    assert dumped["lambda_gp"] == 0.0
