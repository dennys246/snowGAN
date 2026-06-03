"""Bundle a snowGAN training run into a HuggingFace Hub model release.

Uploads the weights + sidecar configs + an auto-generated MANIFEST.md to a
HuggingFace model repo at a tagged revision. Downstream consumers (AvAI,
etc.) then pin exactly which backbone they're using via
``snowgan.weights.fetch("RMDig/snowGAN-core", "v0.1.0")``.

Usage:
    python scripts/release_weights.py \\
        --save-dir keras/snowgan/magnified_profiles \\
        --repo RMDig/snowGAN-magnified-profile \\
        --tag v0.1.0 \\
        --notes "First public release; trained to fade_step 428k."

Requires (only when actually uploading):
    pip install huggingface_hub
    huggingface-cli login   # or set HF_TOKEN env var

Use ``--dry-run`` to inspect the manifest and the upload list without
contacting HF Hub. Use ``--create-repo`` to bootstrap a new model repo on
first release.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


REQUIRED_ARTIFACTS = (
    "discriminator.weights.h5",
    "discriminator_config.json",
    "generator.weights.h5",
    "generator_config.json",
)
OPTIONAL_ARTIFACTS = (
    "generator_ema.weights.h5",
    "generator_fade_endpoints.weights.h5",
    "discriminator_lowres.weights.h5",
)
MANIFEST_FILENAME = "MANIFEST.md"


def build_manifest(
    save_dir: Path,
    tag: str,
    notes: str,
    *,
    snowgan_version: Optional[str] = None,
    git_sha: Optional[str] = None,
) -> str:
    """Generate the MANIFEST.md body for a release.

    Pulled out as a standalone function so tests can exercise the
    formatting without touching the filesystem or HF Hub. ``snowgan_version``
    and ``git_sha`` are injectable so tests can pin them; in production
    they're auto-discovered.
    """
    disc_cfg = json.loads((save_dir / "discriminator_config.json").read_text())
    gen_cfg = json.loads((save_dir / "generator_config.json").read_text())

    if snowgan_version is None:
        try:
            import snowgan  # noqa: WPS433 — runtime introspection
            snowgan_version = getattr(snowgan, "__version__", "unknown")
        except Exception:
            snowgan_version = "unknown"

    if git_sha is None:
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(Path(__file__).resolve().parent.parent),
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            git_sha = "unknown"

    artifacts = sorted(
        p.name for p in save_dir.iterdir()
        if p.is_file() and (p.suffix in (".h5", ".json") or p.name == MANIFEST_FILENAME)
    )
    artifact_lines = "\n".join(f"- `{a}`" for a in artifacts)

    splits = (
        f"- trained_pool: {len(disc_cfg.get('trained_pool') or [])} groups\n"
        f"- validation_pool: {len(disc_cfg.get('validation_pool') or [])} groups\n"
        f"- test_pool: {len(disc_cfg.get('test_pool') or [])} groups"
    )

    return f"""# snowGAN release {tag}

Snapshot of a snowGAN training run, packaged for downstream consumers
(AvAI etc.). Consume with `snowgan.weights.fetch("{{repo}}", "{tag}")`.

## Provenance
- snowGAN package version: `{snowgan_version}`
- snowGAN git SHA at release: `{git_sha}`
- Training dataset: `{gen_cfg.get('dataset')}`
- Source save_dir: `{save_dir.as_posix()}`

## Model architecture
- depth: {gen_cfg.get('depth')}
- resolution: {gen_cfg.get('resolution')}
- modality: {gen_cfg.get('modality')}
- channels: {gen_cfg.get('channels')}
- latent_dim: {gen_cfg.get('latent_dim')}
- filter_counts (gen): {gen_cfg.get('filter_counts')}
- filter_counts (disc): {disc_cfg.get('filter_counts')}
- kernel_size / kernel_stride: {gen_cfg.get('kernel_size')} / {gen_cfg.get('kernel_stride')}

## Training state at release
- fade_step: {gen_cfg.get('fade_step')} (gen) / {disc_cfg.get('fade_step')} (disc)
- fade_steps target: {gen_cfg.get('fade_steps')}
- current_epoch: {gen_cfg.get('current_epoch')}
- training_steps: gen={gen_cfg.get('training_steps')}, disc={disc_cfg.get('training_steps')}
- lambda_gp: {disc_cfg.get('lambda_gp')}

## Advanced training options
- spectral_norm: {disc_cfg.get('spectral_norm')}
- augment: {disc_cfg.get('augment')}
- multiscale_disc: {disc_cfg.get('multiscale_disc')}
- ema_decay: {gen_cfg.get('ema_decay')}
- lr_decay: {gen_cfg.get('lr_decay')} (lr_min={gen_cfg.get('lr_min')})
- ada_target: {gen_cfg.get('ada_target')}
- adaptive_steps: {gen_cfg.get('adaptive_steps')}
- grad_clip_norm: {gen_cfg.get('grad_clip_norm')}
- fid_interval: {gen_cfg.get('fid_interval')}

## Persisted dataset splits
{splits}

## Artifacts
{artifact_lines}

## Notes
{notes or '(none)'}
"""


def _validate_required(save_dir: Path) -> list[str]:
    return [a for a in REQUIRED_ARTIFACTS if not (save_dir / a).exists()]


def _list_release_files(save_dir: Path) -> list[str]:
    files = list(REQUIRED_ARTIFACTS) + [MANIFEST_FILENAME]
    files.extend(a for a in OPTIONAL_ARTIFACTS if (save_dir / a).exists())
    return files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--save-dir", required=True,
                        help="Local training save_dir containing the artifacts.")
    parser.add_argument("--repo", required=True,
                        help="HF Hub model repo id (e.g. RMDig/snowGAN-core).")
    parser.add_argument("--tag", required=True,
                        help="Release tag (e.g. v0.1.0). Becomes a git tag on the HF repo.")
    parser.add_argument("--notes", default="",
                        help="Free-text notes appended to MANIFEST.md.")
    parser.add_argument("--create-repo", action="store_true",
                        help="Create the model repo on HF Hub if it doesn't exist.")
    parser.add_argument("--private", action="store_true",
                        help="If creating the repo, make it private.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the manifest and list files; do not upload.")
    args = parser.parse_args(argv)

    save_dir = Path(args.save_dir).resolve()
    if not save_dir.is_dir():
        print(f"error: {save_dir} does not exist or is not a directory", file=sys.stderr)
        return 1

    missing = _validate_required(save_dir)
    if missing:
        print(f"error: missing required artifacts in {save_dir}:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 1

    manifest_body = build_manifest(save_dir, args.tag, args.notes).replace(
        "{repo}", args.repo
    )
    manifest_path = save_dir / MANIFEST_FILENAME
    manifest_path.write_text(manifest_body)
    print(f"[release] wrote {manifest_path}")

    files_to_upload = _list_release_files(save_dir)
    total_mb = sum((save_dir / f).stat().st_size for f in files_to_upload) / 1024 / 1024
    print(f"[release] {args.repo}@{args.tag} -> {len(files_to_upload)} files ({total_mb:.1f} MB):")
    for f in files_to_upload:
        size_mb = (save_dir / f).stat().st_size / 1024 / 1024
        print(f"  - {f}  ({size_mb:.1f} MB)")

    if args.dry_run:
        print("[release] dry-run: not uploading.")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print(
            "error: huggingface_hub not installed. "
            "Install with `pip install snowgan[hub]` or `pip install huggingface_hub`.",
            file=sys.stderr,
        )
        return 1

    api = HfApi()

    if args.create_repo:
        api.create_repo(
            repo_id=args.repo, repo_type="model",
            exist_ok=True, private=args.private,
        )
        print(f"[release] ensured repo {args.repo} exists (private={args.private})")

    # Upload as a single commit on the main branch, then create the tag.
    # allow_patterns is the explicit whitelist — anything else in save_dir
    # (batch_* snapshots, loss txts, history.png) is intentionally NOT shipped.
    api.upload_folder(
        folder_path=str(save_dir),
        repo_id=args.repo,
        repo_type="model",
        allow_patterns=list(files_to_upload),
        commit_message=f"Release {args.tag}",
    )
    api.create_tag(
        repo_id=args.repo, tag=args.tag, repo_type="model",
        tag_message=f"snowGAN release {args.tag}",
    )
    print(f"[release] tagged {args.repo}@{args.tag} on HF Hub")
    print(f"[release] consumers can now: snowgan.weights.fetch('{args.repo}', '{args.tag}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
