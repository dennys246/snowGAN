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
