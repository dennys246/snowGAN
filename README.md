---
language:
- en
license: apache-2.0
tags:
- gan
- image-generation
pipeline_tag: image-generation
library_name: snowgan
---

# The Abominable SnowGAN

## Model Description
The snowGAN is a generative adversarial network built to take in magnified pictures of snowpack and train a generator and discriminator to generate and discriminate pictures of the snow respectively. The end goal is to pre-train an AI that could potentially be rebuilt to assess other things like avalanche risk or wind loading.

This is an example of the data fed into the snowGAN...
![IMG_3357](https://github.com/user-attachments/assets/23c833e4-5664-4ccf-aeb3-3defd1af1478)

This is an example of a picture generated from the snowGAN after training on ~1500 images over 50 epochs...
![IMG_3451](https://github.com/user-attachments/assets/466bdbd6-0186-488e-8f8a-fd426b7bf2d2)

## Intended uses & limitations
- Intended for research and artistic generation of snow scenes.
- Not suitable for safety-critical applications.
- Model outputs may not generalize well to very different domains.

## Training data
- Dataset: Custom winter landscape dataset (~3,000 images)
- Source: Collected from open-source winter photography repositories

## Training procedure
- Architecture: Wasserstein GAN with gradient penalty
- Optimizers: Adam with learning rate 1e-4
- Number of epochs: 100

## Model Details
- **Framework**: TensorFlow
- **Generator Input**: 100D latent vector
- **Geneator Output**: 1024x1024 RGB magnified snowpack image
- **Discriminator Input**: 1024x1024 RGB magnified snowpack image
- **Discriminator Output**: 1 classification of real or fake

## Ethical considerations
- Model may produce unrealistic snow patterns — avoid misuse

## How to Use

```bash
snowgan --mode train --save_dir D:/Models/snowGAN/keras/ --gen_steps 4 --disc_lr 0.0001

snowgan --mode generate --n_samples 10
```

## Documentation

- [CLAUDE.md](CLAUDE.md) — systems-engineering operating manual (session workflow,
  two-lens review, no-bandaids rule). Read this first if you're contributing.
- [docs/architecture.md](docs/architecture.md) — system map: runtime flow, data pipeline,
  model shapes, loss math, checkpointing.
- [docs/UPGRADES.md](docs/UPGRADES.md) — tiered roadmap of known bugs and production-
  readiness gaps with a suggested sequencing.
- [docs/TRANSFER_LEARNING_PLAN.md](docs/TRANSFER_LEARNING_PLAN.md) — design for
  transfer-learning this model's backbone into downstream metric prediction.
- [docs/AVAI_BOOTSTRAP_PLAN.md](docs/AVAI_BOOTSTRAP_PLAN.md) — executable plan for the
  AvAI downstream project to consume this repo as a pretrained backbone.

## TensorFlow + GPU (Blackwell) Setup

SnowGAN works with stable TensorFlow and tf-nightly. For NVIDIA Blackwell GPUs, use a recent TensorFlow that supports CUDA 12.x and cuDNN 9, or tf-nightly. The CLI selects GPU by default and enables memory growth; use `--device cpu` to force CPU.

- Linux (recommended):
  - Install latest NVIDIA driver (R555+).
  - Create/activate a virtual environment.
  - Install tf-nightly and, if needed, CUDA runtime libs from PyPI:
    ```bash
    pip install --upgrade pip
    pip install tf-nightly
    # If needed for GPU runtime on Linux:
    pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12
    ```
  - Verify GPU visibility:
    ```bash
    python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```

- Windows: GPU wheels availability for nightly varies. If tf-nightly GPU is not available, use the latest stable `tensorflow` with GPU support, or run under WSL2 and follow Linux steps.

- Mixed precision: enable with `--mixed_precision True` to use `mixed_float16` (recommended on recent NVIDIA GPUs).

