"""Depth-axis modality contract for stacked snow images.

snowGAN operates on rank-5 tensors ``(B, depth=2, H, W, C)``. The depth
axis encodes two modalities of the same snow sample in a fixed order:

- index 0: ``PROFILE`` — high-magnification crystal/grain detail
- index 1: ``CORE`` — column-level core photograph

The enum centralizes this contract so dataset stacking, generator output
splitting, and downstream consumers (AvAI transfer learning) all agree on
the axis order without relying on bare integers or string literals. See
``docs/UPGRADES.md`` #25 for the rationale and the cross-repo contract
note.

Changing the enum integer values is a breaking change for any artifact
trained under the previous order — including AvAI's data layer.
"""
from __future__ import annotations

from enum import IntEnum


class Modality(IntEnum):
    PROFILE = 0
    CORE = 1
