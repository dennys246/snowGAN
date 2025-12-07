import os
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from datasets import load_dataset
from matplotlib import pyplot as plt

_DATATYPE_TRANSLATOR = {
    "core": 0,
    "profile": 1,
    "magnified_profile": 2,
    "crystal_card": 3,
}


def _normalize_resolution(resolution) -> tuple:
    if isinstance(resolution, (list, tuple)):
        return int(resolution[0]), int(resolution[1])
    if isinstance(resolution, set):
        res_list = list(resolution)
        if len(res_list) >= 2:
            return int(res_list[0]), int(res_list[1])
    try:
        size = int(resolution)
        return size, size
    except Exception:
        return 1024, 1024


def build_inference_model(discriminator, avalanche_classes: int = 21, wind_classes: int = 4) -> tf.keras.Model:
    """
    Rebuild the discriminator head to predict avalanche counts and wind loading.
    Returns a compiled model ready for evaluation.
    """
    base = discriminator.model
    feature_output = base.layers[-2].output

    avalanche_head = tf.keras.layers.Dense(avalanche_classes, activation="softmax", name="avalanches_spotted")(feature_output)
    wind_head = tf.keras.layers.Dense(wind_classes, activation="softmax", name="wind_loading")(feature_output)

    inference_model = tf.keras.Model(base.input, [avalanche_head, wind_head], name="DiscriminatorInference")

    lr = getattr(discriminator.config, "learning_rate", 1e-4) or 1e-4
    inference_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            "avalanches_spotted": tf.keras.losses.SparseCategoricalCrossentropy(),
            "wind_loading": tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        metrics={
            "avalanches_spotted": ["accuracy"],
            "wind_loading": ["accuracy"],
        },
        jit_compile=False,
    )
    return inference_model


def _matches_datatype(sample: Dict) -> bool:
    target_code = _DATATYPE_TRANSLATOR["magnified_profile"]
    return sample.get("datatype") == target_code


def _preprocess_sample(sample: Dict, resolution) -> Optional[tuple]:
    image = sample.get("image")
    if image is None:
        return None

    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))
    target_res = _normalize_resolution(resolution)
    image = tf.image.resize(image, target_res)
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    avalanche_label = sample.get("avalanches_spotted")
    wind_label = sample.get("wind_loading")
    if avalanche_label is None or wind_label is None:
        return None

    labels = {
        "avalanches_spotted": tf.clip_by_value(tf.cast(avalanche_label, tf.int32), 0, 20),
        "wind_loading": tf.clip_by_value(tf.cast(wind_label, tf.int32), 0, 3),
    }
    return image, labels


def _build_dataset(dataset_name: str, resolution, sample_count: int, batch_size: int):
    print(f"Loading dataset '{dataset_name}' for inference (first {sample_count} samples, magnified_profile only)...")
    split = load_dataset(dataset_name, split=f"train[:{sample_count}]")

    def generator():
        processed = 0
        for row in split:
            if processed >= sample_count:
                break
            if not _matches_datatype(row):
                continue

            sample = _preprocess_sample(row, resolution)
            if sample is None:
                continue

            processed += 1
            if processed % 100 == 0:
                print(f"Prepared {processed} samples for inference...")
            yield sample

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            {
                "avalanches_spotted": tf.TensorSpec(shape=(), dtype=tf.int32),
                "wind_loading": tf.TensorSpec(shape=(), dtype=tf.int32),
            },
        ),
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _plot_metrics(history: Dict[str, list], save_dir: str) -> Optional[str]:
    if not history["loss"]:
        return None

    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["loss"], label="total")
    axes[0].plot(history["avalanches_spotted_loss"], label="avalanche head")
    axes[0].plot(history["wind_loading_loss"], label="wind head")
    axes[0].set_title("Inference loss")
    axes[0].set_xlabel("Batches")
    axes[0].legend()

    axes[1].plot(history["avalanches_spotted_accuracy"], label="avalanche acc")
    axes[1].plot(history["wind_loading_accuracy"], label="wind acc")
    axes[1].set_title("Inference accuracy")
    axes[1].set_xlabel("Batches")
    axes[1].legend()

    fig.tight_layout()
    plot_path = os.path.join(save_dir, "inference_metrics.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def run_inference(
    discriminator,
    dataset_name: str,
    resolution,
    batch_size: int = 8,
    sample_count: int = 1000,
    save_dir: str = "keras/snowgan/",
) -> Dict:
    """
    Swap the discriminator head for avalanche/wind predictions and evaluate on labeled samples.
    """
    inference_model = build_inference_model(discriminator)
    print("Building inference dataset...")
    eval_dataset = _build_dataset(dataset_name, resolution, sample_count, batch_size)
    print("Inference dataset ready; starting evaluation.")

    metrics_history = {
        "loss": [],
        "avalanches_spotted_loss": [],
        "wind_loading_loss": [],
        "avalanches_spotted_accuracy": [],
        "wind_loading_accuracy": [],
    }

    total_seen = 0
    for batch_images, batch_labels in eval_dataset:
        results = inference_model.test_on_batch(batch_images, batch_labels, return_dict=True)
        for key in metrics_history:
            if key in results:
                metrics_history[key].append(float(results[key]))
        total_seen += int(batch_images.shape[0])
        if total_seen >= sample_count:
            break

    plot_path = _plot_metrics(metrics_history, save_dir)
    return {
        "total_seen": total_seen,
        "batches": len(metrics_history["loss"]),
        "plot_path": plot_path,
        "history": metrics_history,
    }
