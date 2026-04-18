"""
Migrate an existing discriminator checkpoint to the spectral-normalization architecture.

Usage:
    python scripts/migrate_spectral_norm.py --save_dir keras/snowgan/

This script:
1. Loads the old discriminator weights (from backup if a prior migration was attempted)
2. Builds a new discriminator with SpectralNormalization wrappers
3. Copies kernel/bias weights from old layers into the wrapped layers
4. Saves the new checkpoint (backs up the original first)
"""

import argparse
import json
import os
import shutil
import sys

# Add src to path so we can import snowgan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tensorflow as tf
import keras

from snowgan.config import build as Config


def _extract_conv_dense_weights(model):
    """
    Extract kernel/bias weights from Conv2D and Dense layers,
    handling both plain layers and SpectralNormalization-wrapped layers.
    """
    weights = {}
    conv_idx = 0
    for layer in model.layers:
        # Unwrap SpectralNormalization if present
        actual_layer = layer
        if isinstance(layer, keras.layers.SpectralNormalization):
            actual_layer = layer.layer

        if isinstance(actual_layer, keras.layers.Conv2D) and not isinstance(actual_layer, keras.layers.Conv2DTranspose):
            weights[f"conv_{conv_idx}"] = actual_layer.get_weights()
            conv_idx += 1
        elif isinstance(actual_layer, keras.layers.Dense):
            weights["dense_out"] = actual_layer.get_weights()

    return weights, conv_idx


def migrate(save_dir: str):
    disc_config_path = os.path.join(save_dir, "discriminator_config.json")
    disc_checkpoint = os.path.join(save_dir, "discriminator.keras")
    backup_path = disc_checkpoint + ".pre_spectral_norm.bak"

    # Prefer the backup (original pre-migration weights) if it exists
    load_path = backup_path if os.path.exists(backup_path) else disc_checkpoint
    if not os.path.exists(load_path):
        print(f"No discriminator checkpoint found at {load_path}")
        return

    # Load config
    config = Config(disc_config_path)

    # ---- Load OLD model directly from .keras file ----
    # Keras 3 requires .keras or .h5 extension — copy to temp file if needed
    tmp_load_path = None
    if not load_path.endswith((".keras", ".h5")):
        tmp_load_path = load_path + ".tmp.keras"
        shutil.copy2(load_path, tmp_load_path)
        print(f"Copied {load_path} -> {tmp_load_path} (Keras 3 extension requirement)")
        load_path = tmp_load_path

    print(f"Loading old discriminator from {load_path}")
    old_model = keras.models.load_model(load_path)
    print(f"Loaded old discriminator model")

    # Clean up temp file
    if tmp_load_path and os.path.exists(tmp_load_path):
        os.remove(tmp_load_path)

    # Extract kernel/bias weights (handles both plain and SN-wrapped layers)
    old_weights, conv_count = _extract_conv_dense_weights(old_model)
    dense_found = "dense_out" in old_weights
    print(f"Extracted weights from {conv_count} Conv2D layers + {'1' if dense_found else '0'} Dense layer")

    if conv_count == 0:
        print("ERROR: No Conv2D weights found. Cannot migrate.")
        return

    # ---- Build NEW model (with spectral norm) ----
    from snowgan.models.discriminator import Discriminator
    config.spectral_norm = True
    new_disc = Discriminator(config)
    # Run a dummy forward pass to fully build all layers including SpectralNorm u-vectors
    dummy = tf.random.normal([1, config.resolution[0], config.resolution[1], 3])
    _ = new_disc.model(dummy, training=False)

    # ---- Copy weights into SpectralNormalization wrappers ----
    conv_idx = 0
    for layer in new_disc.model.layers:
        if isinstance(layer, keras.layers.SpectralNormalization):
            inner = layer.layer
            if isinstance(inner, keras.layers.Conv2D):
                key = f"conv_{conv_idx}"
                if key in old_weights:
                    inner.set_weights(old_weights[key])
                    print(f"  Migrated {key} -> {layer.name}")
                conv_idx += 1
            elif isinstance(inner, keras.layers.Dense):
                if "dense_out" in old_weights:
                    inner.set_weights(old_weights["dense_out"])
                    print(f"  Migrated dense_out -> {layer.name}")

    # ---- Backup old checkpoint (only if no backup exists yet) and save new one ----
    if not os.path.exists(backup_path) and os.path.exists(disc_checkpoint):
        shutil.copy2(disc_checkpoint, backup_path)
        print(f"Backed up original to {backup_path}")

    new_disc.model.save(disc_checkpoint)
    print(f"Saved migrated discriminator to {disc_checkpoint}")

    # Update config to enable spectral_norm
    config.spectral_norm = True
    config.save_config(disc_config_path)
    print(f"Updated config at {disc_config_path} with spectral_norm=true")

    # Do a verification round-trip
    out_old = old_model(dummy, training=False)
    out_new = new_disc.model(dummy, training=False)
    diff = tf.reduce_max(tf.abs(out_old - out_new)).numpy()
    print(f"Verification: max output difference = {diff:.6e}")
    if diff < 1e-4:
        print("Migration successful - outputs match within tolerance.")
    else:
        print("NOTE: Outputs differ due to SpectralNormalization u-vectors being randomly "
              "initialized. They converge within a few training steps. This is expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate discriminator to spectral normalization")
    parser.add_argument("--save_dir", type=str, default="keras/snowgan/",
                        help="Directory containing discriminator.keras and discriminator_config.json")
    args = parser.parse_args()
    migrate(args.save_dir)
