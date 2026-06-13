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


def test_cli_disc_lambda_gp_zero_disables_gp(tmp_path):
    """`--disc_lambda_gp 0` must reach config (not be dropped as falsy), so a
    fresh run can select the spectral-norm-only critic."""
    import types
    from snowgan.config import configure_disc

    data = copy.deepcopy(config_template)
    data["architecture"] = "discriminator"
    data["lambda_gp"] = 10.0  # template/default GP on
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(data))
    cfg = build(str(cfg_path))

    # Minimal args namespace: only disc_lambda_gp set, everything else None/falsy.
    args = types.SimpleNamespace(
        disc_lambda_gp=0.0, disc_checkpoint=None, disc_kernel=None, disc_stride=None,
        disc_lr=None, disc_beta_1=None, disc_beta_2=None, disc_negative_slope=None,
        disc_steps=None, disc_filters=None, save_dir=None, dataset_dir=None, rebuild=None,
        resolution=None, n_samples=None, batch_size=None, epochs=None, latent_dim=None,
        fade=None, fade_steps=None, cleanup_milestone=None, spectral_norm=None, augment=None,
        lr_decay=None, lr_min=None, lr_decay_steps=None, ema_decay=None, fid_interval=None,
        multiscale_disc=None, grad_clip_norm=None, ada_target=None, adaptive_steps=None,
        gen_norm=None, modality=None, sample_epoch_interval=None, sample_batch_interval=None,
    )
    configure_disc(cfg, args)
    assert cfg.lambda_gp == 0.0

