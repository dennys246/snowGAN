"""Diagnostic: list datasets inside *.weights.h5 / *.keras to compare layer
naming between saved checkpoints and a freshly-built generator."""
from __future__ import annotations

import os
import sys
import zipfile

import h5py


def list_h5(path: str) -> list[tuple[str, tuple]]:
    out: list[tuple[str, tuple]] = []
    with h5py.File(path, "r") as f:
        def walk(name, obj):
            if isinstance(obj, h5py.Dataset):
                out.append((name, tuple(obj.shape)))
        f.visititems(walk)
    return out


def list_keras_bundle(path: str) -> list[tuple[str, tuple]]:
    """A .keras file is a zip; pull model.weights.h5 out and walk it."""
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        print(f"  zip entries: {names}")
        weights_member = next((n for n in names if n.endswith(".h5")), None)
        if not weights_member:
            return []
        tmp = path + ".inspect.h5"
        with zf.open(weights_member) as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        try:
            return list_h5(tmp)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass


def summarise(label: str, entries: list[tuple[str, tuple]]) -> None:
    print(f"\n=== {label} ({len(entries)} datasets) ===")
    for name, shape in entries[:60]:
        print(f"  {name}  shape={shape}")
    if len(entries) > 60:
        print(f"  ... {len(entries) - 60} more")


def layer_prefixes(entries: list[tuple[str, tuple]]) -> dict[str, int]:
    """Group by the first one or two path components to spot naming offsets."""
    prefixes: dict[str, int] = {}
    for name, _ in entries:
        head = name.split("/")[0]
        prefixes[head] = prefixes.get(head, 0) + 1
    return prefixes


def main() -> int:
    base = "keras/snowgan"
    targets = [
        ("generator.weights.h5", "h5"),
        ("generator_ema.weights.h5", "h5"),
        ("generator_fade_endpoints.weights.h5", "h5"),
        ("generator.keras.bak", "keras"),
        ("discriminator.keras", "keras"),
    ]
    for fname, kind in targets:
        path = os.path.join(base, fname)
        if not os.path.exists(path):
            print(f"\n=== {path} (missing) ===")
            continue
        try:
            entries = list_h5(path) if kind == "h5" else list_keras_bundle(path)
        except Exception as e:
            print(f"\n=== {path} (error: {e}) ===")
            continue
        summarise(path, entries)
        print(f"  top-level layer prefixes: {layer_prefixes(entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
