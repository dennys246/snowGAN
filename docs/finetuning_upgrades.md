# Post-Progressive Training Upgrades

This document details the six training improvements added to snowGAN's post-progressive (fine-tuning) phase. These upgrades activate after the progressive fade-in completes (i.e., when `fade_step >= fade_steps`) and are designed to stabilize long-horizon WGAN-GP training at full resolution.

All six features are controlled via config JSON fields and CLI flags, and persist across training restarts.

---

## 1. Cosine Annealing Learning Rate Decay

**Config fields:** `lr_decay`, `lr_min`
**CLI flags:** `--lr_decay cosine`, `--lr_min 1e-7`

Gradually reduces both generator and discriminator learning rates using a cosine schedule after the fade-in phase completes. This prevents the optimizer from overshooting as the model converges.

**Schedule:**
```
post_fade_step = global_step - fade_steps
progress = min(post_fade_step / 200000, 1.0)
cosine_factor = 0.5 * (1 + cos(pi * progress))
lr = lr_min + (lr_base - lr_min) * cosine_factor
```

The decay horizon is 200,000 post-fade steps. Both generator and discriminator LRs decay independently from their respective base rates down to `lr_min`.

**Implementation:** `trainer.py` - `_update_learning_rates()`, called after every training step.

---

## 2. Spectral Normalization (Discriminator)

**Config field:** `spectral_norm`
**CLI flag:** `--spectral_norm`

Wraps all Conv2D and the final Dense layer in the discriminator with `keras.layers.SpectralNormalization`. This enforces a Lipschitz constraint on the discriminator, which is complementary to WGAN-GP's gradient penalty and helps prevent mode collapse.

**Architecture change:**
```python
# Without spectral norm:
x = Conv2D(filters, ...)(x)

# With spectral norm:
x = SpectralNormalization(Conv2D(filters, ...))(x)
```

All 5 convolutional layers and the output Dense(1) layer are wrapped.

**Migration:** Existing discriminator checkpoints (without spectral norm) can be migrated to the new architecture using:
```bash
python scripts/migrate_spectral_norm.py --save_dir keras/snowgan/
```

This script:
1. Loads the original discriminator weights (prefers `.pre_spectral_norm.bak` backup)
2. Builds a new discriminator with SpectralNormalization wrappers
3. Copies kernel/bias weights from old layers into the wrapped layers
4. Saves the new checkpoint and backs up the original
5. Verifies output consistency

**Implementation:** `models/discriminator.py` - `_build_model()`, `scripts/migrate_spectral_norm.py`

---

## 3. Differentiable Augmentation

**Config field:** `augment`
**CLI flag:** `--augment`

Applies random augmentations to both real and fake images before they reach the discriminator. This prevents the discriminator from overfitting on the limited training set, which is critical for training on small datasets like Rocky Mountain Snowpack.

**Augmentations (each applied with p=0.5):**
1. **Random horizontal flip** - `tf.image.random_flip_left_right`
2. **Random brightness** - delta in [-0.2, 0.2], clamped to [-1, 1]
3. **Random saturation** - factor in [0.8, 1.2]
4. **Random cutout** - zeros out a random 25% rectangle

All operations are differentiable, so gradients flow through augmentation back to the generator during generator training.

**Memory optimization:** During discriminator training, augmentation is applied *outside* the gradient tape and generator output uses `tf.stop_gradient`. This avoids storing intermediate augmentation tensors at 1024x1024 resolution in GPU memory, which previously caused OOM errors.

```python
# Discriminator training (memory-efficient):
synthetic_images = tf.stop_gradient(generator(noise))
disc_real = augment(images)      # outside tape
disc_fake = augment(synthetic)   # outside tape
with tf.GradientTape() as tape:
    ...

# Generator training (gradients flow through augmentation):
with tf.GradientTape() as tape:
    synthetic = generator(noise)
    disc_fake = augment(synthetic)  # inside tape - differentiable
    ...
```

**Implementation:** `augment.py`, `trainer.py` - `train_step()`

---

## 4. Generator EMA (Exponential Moving Average)

**Config field:** `ema_decay`
**CLI flag:** `--ema_decay 0.999`

Maintains shadow weights for the generator using an exponential moving average. The EMA weights produce smoother, higher-quality outputs and are used for sample generation and FID evaluation.

**Update rule (every training step):**
```
ema_weight = decay * ema_weight + (1 - decay) * current_weight
```

With `decay=0.999`, the shadow weights lag behind the training weights by roughly 1000 steps, filtering out training noise.

**Usage:**
- Sample generation during training uses EMA weights (swapped in temporarily)
- FID evaluation uses EMA weights
- Training itself uses the real (non-EMA) weights
- EMA weights are saved to `generator_ema.weights.h5` and restored on resume

**Implementation:** `trainer.py` - `_init_ema()`, `_update_ema()`, `_apply_ema_to_generator()`, `_restore_generator_weights()`

---

## 5. FID-Based Checkpointing

**Config field:** `fid_interval`
**CLI flag:** `--fid_interval 5000`

Periodically computes the Frechet Inception Distance between generated and real images, saving a "best model" checkpoint whenever FID improves.

**FID computation:**
1. Generate 256 samples using EMA weights
2. Load 256 real samples from the dataset
3. Extract 2048-dim features from InceptionV3 (pooling='avg')
4. Compute FID: `||mu_r - mu_f||^2 + Tr(C_r + C_f - 2*sqrt(C_r @ C_f))`

**Checkpointing:**
- Best FID model is saved to `{save_dir}/best_fid/`
- FID scores are logged to stdout at each evaluation

**Implementation:** `trainer.py` - `_compute_fid()`, evaluated in `train()` loop.

---

## 6. Fade Endpoint Weight Persistence

**Bug fix** (not a new feature, but critical for progressive training)

Previously, the `toRGB_prev` layer weights (used for blending during progressive fade-in) were not saved or loaded. This meant pausing and resuming during the fade phase would reset the previous-resolution path to random weights, degrading output quality.

**Fix:**
- `save_model()` now saves fade endpoint weights to `generator_fade_endpoints.weights.h5`
- `Trainer.__init__()` loads them on resume if they exist

**Implementation:** `trainer.py` - `save_model()` and `__init__()`

---

## Current Configuration

The active model (`keras/snowgan/`) is configured with all improvements enabled:

| Setting | Generator | Discriminator |
|---------|-----------|---------------|
| `spectral_norm` | `true` | `true` |
| `augment` | `true` | `true` |
| `lr_decay` | `"cosine"` | `"cosine"` |
| `lr_min` | `1e-7` | `1e-7` |
| `ema_decay` | `0.999` | `0.999` |
| `fid_interval` | `5000` | `5000` |
| `fade_step` | `288734` | `288734` |
| `fade_steps` | `50000` | `50000` |

The model is well past the fade phase (`fade_step >> fade_steps`), so all post-progressive improvements are fully active.

---

## Enabling These Features

For a fresh training run with all improvements:
```bash
snowgan --mode train \
  --spectral_norm \
  --augment \
  --lr_decay cosine \
  --lr_min 1e-7 \
  --ema_decay 0.999 \
  --fid_interval 5000
```

Once set, these values persist in the config JSON files and do not need to be re-specified on resume:
```bash
snowgan --mode train
```

---

## Files Modified/Created

| File | Change |
|------|--------|
| `src/snowgan/config.py` | Added 6 config fields, wired through `configure()`, `dump()`, `configure_generic()` |
| `src/snowgan/utils.py` | Added 6 CLI arguments |
| `src/snowgan/models/discriminator.py` | Conditional `SpectralNormalization` wrapping |
| `src/snowgan/augment.py` | **New file** - differentiable augmentation module |
| `src/snowgan/trainer.py` | EMA, LR decay, FID, augmentation integration, fade endpoint persistence |
| `scripts/migrate_spectral_norm.py` | **New file** - weight migration for spectral norm |
