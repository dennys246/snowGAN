# Getting Started with snowGAN

## Installation

### Requirements

- Python >= 3.9
- TensorFlow >= 2.15 (or `tf-nightly` for Python >= 3.11)
- CUDA-capable GPU recommended (training at 1024x1024 requires ~13GB+ VRAM)

### Install from Source

```bash
git clone https://github.com/dennys246/snowGAN.git
cd snowGAN
pip install -e .
```

This installs the `snowgan` CLI command and makes the package importable.

### Dependencies

Installed automatically via pip:
- `numpy>=1.24`
- `matplotlib>=3.7`
- `pillow>=9.5`
- `datasets>=2.16` (HuggingFace)
- `tensorflow>=2.15`
- `opencv-python>=4.8` (for video generation)

---

## Quick Start

### Train

```bash
snowgan --mode train
```

This loads (or creates) configs from `keras/snowgan/`, downloads the Rocky Mountain Snowpack dataset from HuggingFace, and begins WGAN-GP training at 1024x1024 resolution.

Training saves automatically:
- Checkpoints every 1000 batches
- Synthetic samples after every batch (to `synthetic_images/`)
- Loss history plot (`history.png`)
- Auto-saves on exit (Ctrl+C is safe)

### Generate Synthetic Images

```bash
snowgan --mode generate --n_samples 20
```

Loads the trained generator and produces 20 synthetic snow crystal images.

### Run Inference

```bash
snowgan --mode infer --infer_samples 500
```

Evaluates the discriminator's learned features on avalanche count and wind loading classification tasks.

---

## CLI Reference

### Required

| Flag | Description |
|------|-------------|
| `--mode {train,generate,infer}` | Execution mode |

### Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--save_dir` | `keras/snowgan/` | Checkpoint and output directory |
| `--batch_size` | `8` | Training batch size |
| `--epochs` | `10` | Number of training epochs |
| `--n_samples` | `10` | Synthetic images per batch |
| `--latent_dim` | `100` | Latent vector dimension |
| `--device {cpu,gpu}` | `gpu` | Compute device |
| `--xla` | `False` | Enable XLA compilation |
| `--mixed_precision` | `False` | Enable mixed_float16 |

### Generator Options

| Flag | Description |
|------|-------------|
| `--gen_checkpoint` | Path to generator weights |
| `--gen_lr` | Learning rate (default: 1e-4) |
| `--gen_steps` | Training steps per batch (default: 3) |
| `--gen_filters` | Filter counts, space-separated (e.g., "1024 512 256 128 64") |
| `--gen_kernel` | Kernel size (e.g., "3 3") |
| `--gen_stride` | Stride (e.g., "2 2") |
| `--gen_norm` | Enable batch normalization |
| `--gen_beta_1` | Adam beta_1 |
| `--gen_beta_2` | Adam beta_2 |
| `--gen_negative_slope` | LeakyReLU slope |

### Discriminator Options

| Flag | Description |
|------|-------------|
| `--disc_checkpoint` | Path to discriminator weights |
| `--disc_lr` | Learning rate (default: 1e-4) |
| `--disc_steps` | Training steps per batch (default: 2) |
| `--disc_filters` | Filter counts, space-separated (e.g., "64 128 256 512 1024") |
| `--disc_kernel` | Kernel size |
| `--disc_stride` | Stride |
| `--disc_lambda_gp` | Gradient penalty weight (default: 10.0) |
| `--disc_beta_1` | Adam beta_1 |
| `--disc_beta_2` | Adam beta_2 |
| `--disc_negative_slope` | LeakyReLU slope |

### Progressive Training

| Flag | Description |
|------|-------------|
| `--fade` | Enable progressive fade-in |
| `--fade_steps` | Steps for alpha ramp (default: 50000) |
| `--cleanup_milestone` | Checkpoint save/cleanup frequency (default: 1000) |

### Post-Progressive Improvements

| Flag | Description |
|------|-------------|
| `--spectral_norm` | Enable spectral normalization on discriminator |
| `--augment` | Enable differentiable augmentation |
| `--lr_decay cosine` | Enable cosine annealing LR decay |
| `--lr_min` | Minimum LR for decay (default: 1e-7) |
| `--ema_decay` | Generator EMA decay (e.g., 0.999; 0 to disable) |
| `--fid_interval` | Steps between FID evaluations (0 to disable) |

---

## Configuration Persistence

All settings are stored in JSON config files:
- `{save_dir}/generator_config.json`
- `{save_dir}/discriminator_config.json`

CLI flags override config values only when explicitly provided. Omitted flags preserve the stored values. This means you can configure once and simply run:

```bash
snowgan --mode train
```

on subsequent runs — all hyperparameters are remembered.

---

## Python API

snowGAN can also be used as a Python library:

```python
import snowgan

# Load a trained generator
gen_config = snowgan.build("keras/snowgan/generator_config.json")
generator = snowgan.load_generator(gen_config.checkpoint, gen_config)

# Generate images
snowgan.generate(generator, count=5, seed_size=100, save_dir="outputs/")

# Training
disc_config = snowgan.build("keras/snowgan/discriminator_config.json")
discriminator = snowgan.load_discriminator(disc_config.checkpoint, disc_config)
trainer = snowgan.trainer(generator, discriminator)
trainer.train(batch_size=8, epochs=1)
```

### Public API (`snowgan.__init__`)

| Export | Description |
|--------|-------------|
| `Generator` | Generator model class |
| `load_generator(checkpoint, config)` | Load generator from checkpoint |
| `Discriminator` | Discriminator model class |
| `load_discriminator(checkpoint, config)` | Load discriminator from checkpoint |
| `trainer` | Trainer class (aliased from `Trainer`) |
| `generate(generator, count, seed_size, save_dir)` | Generate synthetic images |
| `build(config_path)` | Load/create configuration |
| `configure_gen(config, args)` | Apply generator CLI args to config |
| `configure_disc(config, args)` | Apply discriminator CLI args to config |

---

## Project Structure

```
snowGAN/
├── pyproject.toml                  # Package metadata & entry point
├── requirements.txt                # ML library dependencies
├── src/snowgan/
│   ├── __init__.py                 # Public API exports
│   ├── __main__.py                 # python -m snowgan entry point
│   ├── main.py                     # CLI routing & initialization
│   ├── config.py                   # Configuration persistence
│   ├── trainer.py                  # WGAN-GP training loop
│   ├── losses.py                   # Wasserstein loss & gradient penalty
│   ├── generate.py                 # Image generation & video creation
│   ├── augment.py                  # Differentiable augmentation
│   ├── inference.py                # Avalanche/wind classification
│   ├── log.py                      # Loss history I/O
│   ├── utils.py                    # CLI args, device config, fade utils
│   ├── models/
│   │   ├── generator.py            # Generator architecture
│   │   └── discriminator.py        # Discriminator architecture
│   └── data/
│       └── dataset.py              # HuggingFace dataset loading
├── scripts/
│   └── migrate_spectral_norm.py    # Weight migration utility
├── keras/snowgan/                  # Default checkpoint directory
│   ├── generator.keras
│   ├── discriminator.keras
│   ├── generator_config.json
│   ├── discriminator_config.json
│   └── synthetic_images/
└── docs/                           # Documentation
```

---

## Dataset

snowGAN trains on the [Rocky Mountain Snowpack](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack) dataset hosted on HuggingFace. The dataset is downloaded automatically on first run.

The dataset contains four image types:
| Type | Code | Description |
|------|------|-------------|
| Core | 0 | Snow core cross-sections |
| Profile | 1 | Snow layer profiles |
| Magnified Profile | 2 | Close-up crystal images **(used for training)** |
| Crystal Card | 3 | Crystal card photographs |

Only `magnified_profile` images are used for GAN training. Each image includes metadata labels for `avalanches_spotted` (0-20) and `wind_loading` (0-3), which are used in inference mode.

---

## Tips

- **Resume training**: Just run `snowgan --mode train` again. Configs, weights, loss history, and training position all persist automatically.
- **GPU memory**: At 1024x1024, batch_size=8 requires ~13GB VRAM. Reduce `--batch_size` if you hit OOM errors.
- **Monitor progress**: Check `keras/snowgan/history.png` for loss curves and `synthetic_images/` for visual quality.
- **Safe interruption**: Ctrl+C triggers `atexit` handlers that save the current model state.
- **Custom save directory**: Use `--save_dir my_experiment/` to run independent experiments with separate configs and checkpoints.
