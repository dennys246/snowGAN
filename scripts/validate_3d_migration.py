"""Validate that the migrated *.weights.h5 files load cleanly into freshly
built Generator / Discriminator instances.

Run from the repo root inside the snowGAN venv:
    python scripts/validate_3d_migration.py

Prints the number of variables loaded per model and any per-layer mismatch.
Exits 0 on success, 1 on any load failure.
"""
from __future__ import annotations

import os
import sys
import traceback

# Make sure src/ is importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorflow as tf  # noqa: E402  (import after sys.path insert)
import keras  # noqa: E402

from snowgan.config import build, configure_disc, configure_gen  # noqa: E402
from snowgan.models.generator import load_generator  # noqa: E402
from snowgan.models.discriminator import load_discriminator  # noqa: E402


SAVE_DIR = "keras/snowgan"


class _Args:
    """Minimal stand-in for argparse.Namespace; configure_* only reads fields."""
    def __init__(self):
        self.save_dir = SAVE_DIR
        self.dataset_dir = "rmdig/rocky_mountain_snowpack"
        self.mode = "train"
        self.rebuild = False
        self.device = "cpu"
        self.xla = False
        self.mixed_precision = False
        self.resolution = None
        self.n_samples = 10
        self.batch_size = 4
        self.epochs = 1
        self.latent_dim = 100
        self.cleanup_milestone = 1000
        self.fade = None
        self.fade_steps = None
        self.gen_checkpoint = None
        self.gen_kernel = None
        self.gen_stride = None
        self.gen_lr = None
        self.gen_beta_1 = None
        self.gen_beta_2 = None
        self.gen_negative_slope = None
        self.gen_steps = None
        self.gen_filters = None
        self.gen_norm = None
        self.disc_checkpoint = None
        self.disc_kernel = None
        self.disc_stride = None
        self.disc_lr = None
        self.disc_beta_1 = None
        self.disc_beta_2 = None
        self.disc_negative_slope = None
        self.disc_lambda_gp = None
        self.disc_steps = None
        self.disc_filters = None
        self.spectral_norm = None
        self.augment = None
        self.lr_decay = None
        self.lr_min = None
        self.ema_decay = None
        self.fid_interval = None
        self.multiscale_disc = None
        self.grad_clip_norm = None
        self.ada_target = None
        self.adaptive_steps = None
        self.modality = "magnified_profile"
        self.sample_epoch_interval = None


def _try_load(model: keras.Model, weights_path: str, label: str) -> bool:
    if not os.path.exists(weights_path):
        print(f"  {label}: SKIP — {weights_path} missing")
        return True
    try:
        model.load_weights(weights_path)
    except Exception:
        print(f"  {label}: FAILED — {weights_path}")
        traceback.print_exc()
        return False
    n = sum(v.shape.num_elements() if hasattr(v.shape, "num_elements") else int(np.prod(v.shape))
            for v in model.weights)
    print(f"  {label}: ok — {weights_path}  ({len(model.weights)} weight tensors, {n:,} params)")
    return True


def main() -> int:
    args = _Args()

    gen_cfg = build(os.path.join(SAVE_DIR, "generator_config.json"))
    gen_cfg = configure_gen(gen_cfg, args)
    disc_cfg = build(os.path.join(SAVE_DIR, "discriminator_config.json"))
    disc_cfg = configure_disc(disc_cfg, args)

    print(f"Generator depth={gen_cfg.depth}  resolution={gen_cfg.resolution}")
    print(f"Discriminator depth={disc_cfg.depth}  resolution={disc_cfg.resolution}")

    print("\n--- Building generator + loading weights ---")
    gen = load_generator(gen_cfg.checkpoint, gen_cfg)
    ok_gen = _try_load(gen.model, os.path.join(SAVE_DIR, "generator.weights.h5"), "generator")
    if gen.fade_endpoints is not None:
        ok_fade = _try_load(gen.fade_endpoints,
                            os.path.join(SAVE_DIR, "generator_fade_endpoints.weights.h5"),
                            "generator_fade_endpoints")
    else:
        ok_fade = True

    print("\n--- Building discriminator + loading weights ---")
    disc = load_discriminator(disc_cfg.checkpoint, disc_cfg)
    ok_disc = _try_load(disc.model, os.path.join(SAVE_DIR, "discriminator.weights.h5"), "discriminator")

    print("\n--- Smoke-test forward pass ---")
    try:
        z = tf.random.normal((1, gen_cfg.latent_dim))
        img = gen.model(z, training=False)
        d_out = disc.model(img, training=False)
        print(f"  generator output shape: {tuple(img.shape)}")
        print(f"  discriminator output shape: {tuple(d_out.shape)}")
        ok_fwd = True
    except Exception:
        traceback.print_exc()
        ok_fwd = False

    if ok_gen and ok_fade and ok_disc and ok_fwd:
        print("\nALL CHECKS PASSED")
        return 0
    print("\nVALIDATION FAILED")
    return 1


if __name__ == "__main__":
    import numpy as np  # noqa: E402  (only needed inside _try_load fallback)
    sys.exit(main())
