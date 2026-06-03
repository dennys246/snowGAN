"""Fetch and cache snowGAN weights from HuggingFace Hub model repos.

Trained snowGAN backbones are published as tagged releases on HuggingFace
Hub (e.g. ``RMDig/snowGAN-core``, ``RMDig/snowGAN-magnified-profile``).
Each release contains the weights-only artifacts plus their sidecar
configs and a ``MANIFEST.md`` describing training provenance.

This module is the consumer-side counterpart to
``scripts/release_weights.py``: downstream projects (AvAI, etc.) call
``fetch(repo, version)`` and get back a local filesystem path containing
the artifacts. HuggingFace's local cache de-dupes across versions, so a
fresh checkout of a previously-cached release returns instantly.

Optional dependency: requires ``huggingface_hub`` (pip install
``snowgan[hub]`` or ``pip install huggingface_hub``). The package itself
does not pull in HF Hub by default — it's only needed by callers that
actually want to fetch weights.

Usage:
    from snowgan.weights import fetch
    path = fetch("RMDig/snowGAN-core", "v0.1.0")
    # path / "discriminator.weights.h5"   <- ready to load
    # path / "discriminator_config.json"  <- sidecar
    # path / "MANIFEST.md"                <- provenance
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


_HF_IMPORT_ERROR = (
    "snowgan.weights.fetch requires the 'huggingface_hub' package. "
    "Install it with `pip install snowgan[hub]` or `pip install huggingface_hub`."
)


def fetch(repo: str, version: str, *, cache_dir: Optional[str] = None) -> Path:
    """Download a tagged snowGAN weights release from HF Hub; return local dir.

    Args:
        repo: HuggingFace Hub model repo id (e.g. ``"RMDig/snowGAN-core"``).
        version: Release tag (e.g. ``"v0.1.0"``) or any other git revision
            accepted by ``huggingface_hub.snapshot_download`` (branch, commit
            SHA, etc.). Always pin to a *tag* for reproducibility.
        cache_dir: Optional override for the local cache root. Defaults to
            HuggingFace's standard cache (``~/.cache/huggingface/hub`` on
            Linux/macOS, ``%LOCALAPPDATA%\\huggingface\\hub`` on Windows).

    Returns:
        Local directory containing the release's artifacts. Subsequent
        calls with the same arguments return the same path instantly via
        the HF cache.

    Raises:
        ImportError: if ``huggingface_hub`` is not installed.
        Exception: HF Hub errors (repo not found, revision not found,
            network failure) propagate from ``snapshot_download``.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(_HF_IMPORT_ERROR) from e

    return Path(snapshot_download(
        repo_id=repo,
        revision=version,
        cache_dir=cache_dir,
        repo_type="model",
    ))
