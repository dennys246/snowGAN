"""Checkpoint path resolution for the weights-only save/load contract.

snowGAN persists trained weights as ``*.weights.h5`` files, paired with a
sidecar JSON describing the architecture (the existing per-model
``*_config.json``). Reload reconstructs the model from the sidecar via
``Discriminator(config)`` / ``Generator(config)`` and loads weights into
the fresh architecture; ``model.load_weights`` fails loud on shape
mismatch.

Legacy artifacts saved as ``*.keras`` (architecture + weights bundled)
remain loadable: ``model.load_weights`` accepts the bundled format too,
extracting only the weights from the zip. Until users explicitly delete
their old ``*.keras`` files, this resolver keeps mid-training resumes
working — the next save writes the new format alongside, and from then
on the resolver picks the new path.

See ``docs/UPGRADES.md`` #17 for the rationale.
"""
from __future__ import annotations

import os
from typing import Optional


_PREFERRED_EXT = ".weights.h5"
_LEGACY_EXT = ".keras"


def resolve_weights_path(declared_path: Optional[str]) -> Optional[str]:
    """Return the on-disk checkpoint path nearest to ``declared_path``.

    Preference order: the new ``*.weights.h5`` format first, then the legacy
    ``*.keras`` bundle. ``declared_path`` is typically ``config.checkpoint``,
    which may use either extension depending on when the config was first
    written. Returns ``None`` if nothing is found.
    """
    if not declared_path:
        return None

    candidates: list[str] = []
    if declared_path.endswith(_PREFERRED_EXT):
        candidates.append(declared_path)
        candidates.append(declared_path[: -len(_PREFERRED_EXT)] + _LEGACY_EXT)
    elif declared_path.endswith(_LEGACY_EXT):
        candidates.append(declared_path[: -len(_LEGACY_EXT)] + _PREFERRED_EXT)
        candidates.append(declared_path)
    else:
        candidates.append(declared_path + _PREFERRED_EXT)
        candidates.append(declared_path + _LEGACY_EXT)
        candidates.append(declared_path)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def weights_use_spectral_norm(path: Optional[str]) -> Optional[bool]:
    """Best-effort: does this ``.weights.h5`` contain SpectralNormalization state?

    Returns ``True``/``False`` when determinable, ``None`` when the format can't
    be inspected (legacy ``.keras`` bundle, missing file, or no ``h5py``). Used
    to fail fast with a clear message when a checkpoint's spectral-norm
    structure disagrees with the model being built, instead of surfacing the
    opaque Keras "expected N variables, received 0" traceback.

    Detection keys on the wrapper's persisted ``vector_u`` weight and the
    ``spectral_normalization`` layer-group name, both of which exist iff the
    saved model wrapped its convs in ``keras.layers.SpectralNormalization``.
    """
    if not path or not path.endswith(_PREFERRED_EXT) or not os.path.exists(path):
        return None
    try:
        import h5py
    except ImportError:
        return None
    try:
        found = False
        with h5py.File(path, "r") as handle:
            names: list[str] = []
            handle.visit(names.append)
            found = any("vector_u" in n or "spectral_normalization" in n for n in names)
        return found
    except OSError:
        return None


def to_weights_path(declared_path: str) -> str:
    """Return the canonical ``*.weights.h5`` path derived from ``declared_path``.

    Used by save paths so the on-disk artifact always uses the new format
    regardless of what the legacy ``config.checkpoint`` value still says.
    """
    if declared_path.endswith(_PREFERRED_EXT):
        return declared_path
    if declared_path.endswith(_LEGACY_EXT):
        return declared_path[: -len(_LEGACY_EXT)] + _PREFERRED_EXT
    return declared_path + _PREFERRED_EXT
