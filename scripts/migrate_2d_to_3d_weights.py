"""Migrate 2-D Conv weights to depth-axis 3-D by inflating kernels.

The pre-modality codebase used Conv2DTranspose / Conv2D layers and saved
4-D kernels of shape ``(kH, kW, ...)``. Commit 7f47989 introduced the
depth-axis modality contract by switching to Conv3DTranspose / Conv3D
with ``ksize = (1, kH, kW)``. The depth-1 spatial conv is mathematically
identical to the prior 2-D conv, so the saved kernels can be inflated to
``(1, kH, kW, ...)`` without changing outputs (depth=1 modes only).

Pure h5py — no TF/Keras import required. Layer-name remapping matches
what ``Generator(config)`` / ``Discriminator(config)`` produce when
freshly built against the current src/snowgan/models/*.py.

Usage:
    python scripts/migrate_2d_to_3d_weights.py --save-dir keras/snowgan/
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from typing import Mapping, Optional

import h5py
import numpy as np


def _extract_keras_bundle(bundle_path: str) -> str:
    """Pull model.weights.h5 out of a .keras zip into a temp file."""
    out = tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False).name
    with zipfile.ZipFile(bundle_path, "r") as zf:
        member = next((n for n in zf.namelist() if n.endswith(".weights.h5")), None)
        if member is None:
            raise RuntimeError(f"No .weights.h5 inside {bundle_path}")
        with zf.open(member) as src, open(out, "wb") as dst:
            dst.write(src.read())
    return out


def _open_source(path: str) -> tuple[str, Optional[str]]:
    if path.endswith(".weights.h5"):
        return path, None
    extracted = _extract_keras_bundle(path)
    return extracted, extracted


def _inflate_4d_to_5d(arr: np.ndarray) -> np.ndarray:
    """Insert depth=1 axis at position 0."""
    if arr.ndim != 4:
        raise ValueError(f"Expected 4-D kernel, got {arr.shape}")
    return arr[np.newaxis, ...]


# Keras 3 stores weights under each layer's graph-position path (e.g.
# layers/conv3d_transpose_5/) and records the user-given name= argument
# only as the vars subgroup's `name` attr. So we keep slot-numbered paths
# and override the name attribute for explicitly-named layers.

# Generator backbone (5 hidden Conv2DTranspose + 1 explicitly-named toRGB).
GEN_RENAME: Mapping[str, str] = {
    "conv2d_transpose":   "conv3d_transpose",
    "conv2d_transpose_1": "conv3d_transpose_1",
    "conv2d_transpose_2": "conv3d_transpose_2",
    "conv2d_transpose_3": "conv3d_transpose_3",
    "conv2d_transpose_4": "conv3d_transpose_4",
    "conv2d_transpose_5": "conv3d_transpose_5",
}
GEN_NAME_ATTR: Mapping[str, str] = {
    "conv3d_transpose_5": "toRGB_curr",
}

# Fade-endpoints adds a 1x1 Conv2D (toRGB_prev) over the prev feature map.
# In the new code that head is a Conv3D, so its slot path becomes "conv3d".
FADE_RENAME: Mapping[str, str] = {
    **GEN_RENAME,
    "conv2d": "conv3d",
}
FADE_NAME_ATTR: Mapping[str, str] = {
    "conv3d_transpose_5": "toRGB_curr",
    "conv3d":             "toRGB_prev",
}


def _copy_layer_with_rename(
    fin: h5py.File,
    fout: h5py.File,
    layers_in: h5py.Group,
    layers_out: h5py.Group,
    old_name: str,
    new_name: str,
    inflate_kernels: bool,
    name_attr_override: Optional[str] = None,
) -> None:
    """Copy one layer group, renaming it and (optionally) inflating its kernel.

    The ``name`` attribute on the ``vars`` subgroup defaults to ``new_name``
    (the slot path), but ``name_attr_override`` lets us record the user-given
    Keras layer name (e.g. ``toRGB_curr``) without changing the path.
    """
    src_group = layers_in[old_name]
    dst_group = layers_out.create_group(new_name)
    for sub_name, sub_obj in src_group.items():
        if isinstance(sub_obj, h5py.Group) and sub_name == "vars":
            new_vars = dst_group.create_group("vars")
            new_vars.attrs["name"] = name_attr_override or new_name
            for var_idx, ds in sub_obj.items():
                arr = ds[()]
                if inflate_kernels and var_idx == "0" and arr.ndim == 4:
                    arr = _inflate_4d_to_5d(arr)
                new_vars.create_dataset(var_idx, data=arr)
        else:
            fin.copy(sub_obj, dst_group, name=sub_name)


def _copy_spectral_norm_layer(
    fin: h5py.File,
    layer_in: h5py.Group,
    layer_out: h5py.Group,
    inflate: bool = True,
) -> None:
    """SpectralNormalization layout: <sn>/layer/vars/{0,1} (inner Conv) +
    <sn>/vars/{0,1} (wrapper kernel snapshot + u-vector). When ``inflate``,
    4-D kernels are lifted to 5-D; otherwise pass through unchanged. 2-D
    Dense kernels and 1-D biases / u-vectors always pass through.
    """
    for sub_name, sub_obj in layer_in.items():
        if isinstance(sub_obj, h5py.Group) and sub_name in ("vars", "layer"):
            new_sub = layer_out.create_group(sub_name)
            if sub_name == "vars":
                if "name" in sub_obj.attrs:
                    new_sub.attrs["name"] = sub_obj.attrs["name"]
                for var_idx, ds in sub_obj.items():
                    arr = ds[()]
                    if inflate and var_idx == "0" and arr.ndim == 4:
                        arr = _inflate_4d_to_5d(arr)
                    new_sub.create_dataset(var_idx, data=arr)
            else:  # "layer" subgroup wrapping the inner Conv
                inner_vars = sub_obj["vars"]
                new_inner_vars = new_sub.create_group("vars")
                if "name" in inner_vars.attrs:
                    new_inner_vars.attrs["name"] = inner_vars.attrs["name"]
                for var_idx, ds in inner_vars.items():
                    arr = ds[()]
                    if inflate and var_idx == "0" and arr.ndim == 4:
                        arr = _inflate_4d_to_5d(arr)
                    new_inner_vars.create_dataset(var_idx, data=arr)
                for inner_sub, inner_obj in sub_obj.items():
                    if inner_sub == "vars":
                        continue
                    fin.copy(inner_obj, new_sub, name=inner_sub)
        else:
            fin.copy(sub_obj, layer_out, name=sub_name)


def migrate_generator_like(
    src_h5_path: str,
    out_path: str,
    rename: Mapping[str, str],
    name_attrs: Mapping[str, str],
    label: str,
) -> None:
    """Migrate a generator-family file (no SpectralNormalization)."""
    print(f"[{label}] {src_h5_path} -> {out_path}")
    with h5py.File(src_h5_path, "r") as fin, h5py.File(out_path, "w") as fout:
        if "vars" in fin:
            fin.copy("vars", fout)
        layers_in = fin["layers"]
        layers_out = fout.create_group("layers")
        for old_name in layers_in:
            new_name = rename.get(old_name, old_name)
            _copy_layer_with_rename(fin, fout, layers_in, layers_out,
                                    old_name, new_name, inflate_kernels=True,
                                    name_attr_override=name_attrs.get(new_name))
    print(f"[{label}]   ok ({os.path.getsize(out_path):,} bytes)")


def migrate_discriminator(src_h5_path: str, out_path: str, label: str = "disc",
                          inflate: bool = True) -> None:
    """Migrate a discriminator file (SpectralNormalization wrappers).

    The main discriminator uses Conv3D (depth-axis), so 4-D kernels are
    inflated to 5-D. The multi-scale 256x256 ``disc_lowres`` head still uses
    plain Conv2D, so its 4-D kernels must pass through unchanged — pass
    ``inflate=False`` for that case.
    """
    print(f"[{label}] {src_h5_path} -> {out_path}  (inflate={inflate})")
    with h5py.File(src_h5_path, "r") as fin, h5py.File(out_path, "w") as fout:
        if "vars" in fin:
            fin.copy("vars", fout)
        layers_in = fin["layers"]
        layers_out = fout.create_group("layers")
        for layer_name, group in layers_in.items():
            new_layer = layers_out.create_group(layer_name)
            if "layer" in group:
                _copy_spectral_norm_layer(fin, group, new_layer, inflate=inflate)
            else:
                for sub_name, sub_obj in group.items():
                    if isinstance(sub_obj, h5py.Group) and sub_name == "vars":
                        new_vars = new_layer.create_group("vars")
                        if "name" in sub_obj.attrs:
                            new_vars.attrs["name"] = sub_obj.attrs["name"]
                        for var_idx, ds in sub_obj.items():
                            arr = ds[()]
                            if inflate and var_idx == "0" and arr.ndim == 4:
                                arr = _inflate_4d_to_5d(arr)
                            new_vars.create_dataset(var_idx, data=arr)
                    else:
                        fin.copy(sub_obj, new_layer, name=sub_name)
    print(f"[{label}]   ok ({os.path.getsize(out_path):,} bytes)")


def _migrate_one(label: str, src: str, dst: str, fn) -> None:
    if not os.path.exists(src):
        print(f"[{label}] skip — {src} not found")
        return
    src_h5, tmp = _open_source(src)
    try:
        fn(src_h5, dst, label)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default="keras/snowgan")
    args = ap.parse_args()

    d = args.save_dir
    backup = os.path.join(d, "_pre_3d_migration")
    if not os.path.isdir(backup):
        print(f"ERROR: expected backup at {backup} (run the bash backup step first).")
        return 1

    gen_bundle = os.path.join(backup, "generator.keras.bak")
    gen_src = gen_bundle if os.path.exists(gen_bundle) else os.path.join(backup, "generator.weights.h5")
    _migrate_one("gen", gen_src, os.path.join(d, "generator.weights.h5"),
                 lambda s, o, lab: migrate_generator_like(s, o, GEN_RENAME, GEN_NAME_ATTR, lab))

    _migrate_one("gen-ema", os.path.join(backup, "generator_ema.weights.h5"),
                 os.path.join(d, "generator_ema.weights.h5"),
                 lambda s, o, lab: migrate_generator_like(s, o, GEN_RENAME, GEN_NAME_ATTR, lab))

    _migrate_one("gen-fade", os.path.join(backup, "generator_fade_endpoints.weights.h5"),
                 os.path.join(d, "generator_fade_endpoints.weights.h5"),
                 lambda s, o, lab: migrate_generator_like(s, o, FADE_RENAME, FADE_NAME_ATTR, lab))

    _migrate_one("disc", os.path.join(backup, "discriminator.keras"),
                 os.path.join(d, "discriminator.weights.h5"),
                 migrate_discriminator)

    # disc_lowres is the multi-scale 256x256 head; it stays 2-D Conv2D, so
    # its 4-D kernels must NOT be inflated. See trainer._build_lowres_disc.
    _migrate_one("disc-lowres", os.path.join(backup, "discriminator_lowres.keras"),
                 os.path.join(d, "discriminator_lowres.weights.h5"),
                 lambda s, o, lab: migrate_discriminator(s, o, lab, inflate=False))

    print("\nMigration complete. Inspect with: python scripts/inspect_weights.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
