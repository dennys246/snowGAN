"""Warm-start a depth=2 paired-modality snowGAN from depth=1 single-modality weights.

The depth-axis architecture's Conv3D / Conv3DTranspose kernels use
``ksize = (1, kH, kW)`` — the kernel itself is depth-agnostic, so the
parameter tensor shape is identical at depth=1 and depth=2. Those weights
transfer directly.

The Generator's *first* Dense (``latent_dim → start_depth·16·16·F``) and the
Discriminator's *final* Dense (``Flatten → 1``) are depth-coupled in their
input/output dims, so their parameter shapes differ between depth=1 and
depth=2 builds. ``model.load_weights(..., skip_mismatch=True)`` skips those
two layers, leaving them at the depth=2 model's random initialization.

For AvAI's transfer-learning consumer this is harmless: AvAI only reads
``model.get_layer("features").output`` (the Flatten layer, *before* the final
Dense). The Conv3D backbone is what matters, and it transfers intact. The
Generator's random Dense and the Discriminator's random Dense will adapt
during the subsequent depth=2 merged-modality training.

Caveat — the inherited Conv3D filters were trained on single-modality
inputs (magnified profile alone, depth=1). When fed paired (profile, core)
batches at depth=2, the filter responses on the core slice are *plausible
but never validated*. The follow-on training cycle is what actually adapts
the features to paired-modality discrimination.

Usage:
    python scripts/migrate_depth1_to_depth2_weights.py \\
        --source keras/snowgan/magnified_profiles \\
        --target keras/snowgan-avai-run-1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def _ensure_paths(source: str, target: str) -> None:
    src_disc_cfg = os.path.join(source, "discriminator_config.json")
    src_disc_w = os.path.join(source, "discriminator.weights.h5")
    src_gen_cfg = os.path.join(source, "generator_config.json")
    src_gen_w = os.path.join(source, "generator.weights.h5")
    for p in (src_disc_cfg, src_disc_w, src_gen_cfg, src_gen_w):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required source artifact missing: {p}")
    os.makedirs(target, exist_ok=True)


def _build_target_config(src_cfg_path: str, tgt_cfg_path: str, target_dir: str, architecture: str) -> dict:
    """Derive a depth=2 merged-modality config from the depth=1 source config.

    Copies all training hyperparameters (filter_counts, resolution, learning
    rates, advanced training flags) so the migrated run starts at the same
    operating point as the source. Reroots save_dir / checkpoint to the
    target dir. Clears resume state (train_ind, seen_profiles, current_epoch,
    fade_step) so the run starts fresh in its own save_dir.
    """
    with open(src_cfg_path) as f:
        cfg = json.load(f)

    weights_filename = "generator.weights.h5" if architecture == "generator" else "discriminator.weights.h5"
    save_dir = target_dir.replace("\\", "/").rstrip("/") + "/"

    cfg["save_dir"] = save_dir
    cfg["checkpoint"] = save_dir + weights_filename
    cfg["depth"] = 2
    cfg["modality"] = "merged"
    cfg["architecture"] = architecture

    # Clear resume state — this run starts fresh in its own dir.
    cfg["train_ind"] = 0
    cfg["seen_profiles"] = []
    cfg["current_epoch"] = 0
    cfg["fade_step"] = 0
    cfg["trained_data"] = []
    # Splits will be re-derived (and now persisted to both configs) on Trainer init.
    cfg["trained_pool"] = None
    cfg["validation_pool"] = None
    cfg["test_pool"] = None

    with open(tgt_cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    return cfg


def _migrate_weights(model, src_weights_path: str, tgt_weights_path: str) -> None:
    """Load source weights with shape-mismatch skipping, save to target.

    Layers whose parameter shapes match between depth=1 and depth=2 (every
    Conv3D / Conv3DTranspose, plus the SpectralNormalization wrappers' u
    vectors) are transferred. Layers whose shapes differ (Generator's first
    Dense, Discriminator's final Dense) keep the target model's random init.
    """
    model.load_weights(src_weights_path, skip_mismatch=True)
    model.save_weights(tgt_weights_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--source", required=True,
                        help="Source dir containing depth=1 *_config.json + *.weights.h5")
    parser.add_argument("--target", required=True,
                        help="Target dir for depth=2 artifacts (created if missing)")
    args = parser.parse_args(argv)

    _ensure_paths(args.source, args.target)

    # Import here to keep --help fast and to ensure TF env vars (if any) are
    # set by the caller's shell before TF loads.
    from snowgan.config import build
    from snowgan.models.discriminator import Discriminator
    from snowgan.models.generator import Generator

    print(f"[migrate] source: {args.source}")
    print(f"[migrate] target: {args.target}")

    # --- Generator -----------------------------------------------------
    src_gen_cfg_path = os.path.join(args.source, "generator_config.json")
    tgt_gen_cfg_path = os.path.join(args.target, "generator_config.json")
    tgt_gen_cfg_dict = _build_target_config(
        src_gen_cfg_path, tgt_gen_cfg_path, args.target, architecture="generator"
    )
    print(f"[migrate] wrote {tgt_gen_cfg_path} (depth=2, modality=merged)")

    tgt_gen_cfg = build(tgt_gen_cfg_path)
    tgt_gen = Generator(tgt_gen_cfg)
    tgt_gen.model.build((None, tgt_gen_cfg.latent_dim))
    if tgt_gen.fade_endpoints is not None:
        tgt_gen.fade_endpoints.build((None, tgt_gen_cfg.latent_dim))

    src_gen_w = os.path.join(args.source, "generator.weights.h5")
    tgt_gen_w = os.path.join(args.target, "generator.weights.h5")
    _migrate_weights(tgt_gen.model, src_gen_w, tgt_gen_w)
    print(f"[migrate] generator weights migrated -> {tgt_gen_w}")

    # --- Discriminator -------------------------------------------------
    src_disc_cfg_path = os.path.join(args.source, "discriminator_config.json")
    tgt_disc_cfg_path = os.path.join(args.target, "discriminator_config.json")
    tgt_disc_cfg_dict = _build_target_config(
        src_disc_cfg_path, tgt_disc_cfg_path, args.target, architecture="discriminator"
    )
    print(f"[migrate] wrote {tgt_disc_cfg_path} (depth=2, modality=merged)")

    tgt_disc_cfg = build(tgt_disc_cfg_path)
    tgt_disc = Discriminator(tgt_disc_cfg)
    tgt_disc.model.build(
        (None, tgt_disc_cfg.depth, tgt_disc_cfg.resolution[0],
         tgt_disc_cfg.resolution[1], tgt_disc_cfg.channels)
    )

    src_disc_w = os.path.join(args.source, "discriminator.weights.h5")
    tgt_disc_w = os.path.join(args.target, "discriminator.weights.h5")
    _migrate_weights(tgt_disc.model, src_disc_w, tgt_disc_w)
    print(f"[migrate] discriminator weights migrated -> {tgt_disc_w}")

    # --- Optional sidecars (fade endpoints, EMA, lowres disc) -----------
    # Fade endpoints and lowres disc are depth-coupled in the same way as
    # their parent models; load with skip_mismatch where they exist.
    src_fade = os.path.join(args.source, "generator_fade_endpoints.weights.h5")
    if os.path.exists(src_fade) and tgt_gen.fade_endpoints is not None:
        tgt_gen.fade_endpoints.load_weights(src_fade, skip_mismatch=True)
        tgt_fade = os.path.join(args.target, "generator_fade_endpoints.weights.h5")
        tgt_gen.fade_endpoints.save_weights(tgt_fade)
        print(f"[migrate] generator fade endpoints -> {tgt_fade}")

    # EMA shadow weights mirror the generator's main weights. If present,
    # apply the same skip-mismatch load via a temporary clone of the
    # target generator (the clone is identical in shape to the live one).
    src_ema = os.path.join(args.source, "generator_ema.weights.h5")
    if os.path.exists(src_ema):
        import keras
        tmp_clone = keras.models.clone_model(tgt_gen.model)
        tmp_clone.build((None, tgt_gen_cfg.latent_dim))
        tmp_clone.load_weights(src_ema, skip_mismatch=True)
        tgt_ema = os.path.join(args.target, "generator_ema.weights.h5")
        tmp_clone.save_weights(tgt_ema)
        del tmp_clone
        print(f"[migrate] generator EMA -> {tgt_ema}")

    # Lowres disc: only meaningful if multiscale_disc is configured. The
    # lowres path is 2-D (B, 256, 256, 3) regardless of depth — its weights
    # transfer directly without depth-axis concerns. The trainer rebuilds
    # the lowres disc on init; copying the file is sufficient.
    src_lowres = os.path.join(args.source, "discriminator_lowres.weights.h5")
    if os.path.exists(src_lowres):
        tgt_lowres = os.path.join(args.target, "discriminator_lowres.weights.h5")
        shutil.copy2(src_lowres, tgt_lowres)
        print(f"[migrate] lowres disc copied (depth-agnostic) -> {tgt_lowres}")

    print("[migrate] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
