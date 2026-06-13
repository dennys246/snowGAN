"""Focused test: lambda_gp=0 (gradient penalty disabled) survives config load.

For the SN-only v0.2 critic we disable the gradient penalty with lambda_gp=0.
The old coercion `float(lambda_gp) or None` mapped 0.0 -> None, which the disc
configuration then forced back to 10.0 — so GP could never actually be turned
off from config. This pins that 0.0 round-trips intact.
"""

import copy
import json

from snowgan.config import build, config_template


def test_lambda_gp_zero_preserved(tmp_path):
    data = copy.deepcopy(config_template)
    data["architecture"] = "discriminator"
    data["lambda_gp"] = 0.0
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))

    cfg = build(str(cfg_path))
    assert cfg.lambda_gp == 0.0

    # And it serializes back as 0.0, not None/10.0.
    assert cfg.dump()["lambda_gp"] == 0.0


def test_lambda_gp_default_still_ten(tmp_path):
    data = copy.deepcopy(config_template)
    data["lambda_gp"] = 10.0
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))

    cfg = build(str(cfg_path))
    assert cfg.lambda_gp == 10.0
