# Model Refactoring Plan

Changes deferred to next fresh training run to avoid disrupting the current model mid-training.

## R1 Regularization (replace WGAN-GP)

**What:** Replace gradient penalty with R1 regularization: `R1 = gamma/2 * ||grad(D(real))||^2`

**Why:**
- GP requires an extra forward+backward pass on interpolated images — R1 only needs gradients on reals
- Pairs better with spectral normalization (SN enforces per-layer Lipschitz, R1 penalizes global gradient magnitude — complementary rather than redundant)
- Used by StyleGAN2/3, proven at 1024x1024+
- Removes the need for lambda_gp tuning relative to SN

**Implementation:**
1. Add `r1_gamma` config field (default 10.0)
2. In `losses.py`, add:
   ```python
   def compute_r1_penalty(discriminator, real_images):
       with tf.GradientTape() as tape:
           tape.watch(real_images)
           real_pred = discriminator(real_images, training=True)
           real_pred = tf.reduce_sum(real_pred)  # scalar output for clean per-pixel gradients
       grads = tape.gradient(real_pred, real_images)
       r1 = tf.reduce_mean(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
       return r1
   ```
3. Use **lazy regularization** (compute R1 every 16 steps, scale gamma accordingly) per StyleGAN2 — this cuts the extra backward pass cost by ~16x
4. Replace GP call in `train_step` disc loop
5. Update disc loss: `wasserstein + gamma/2 * r1`
6. **Migration from GP:** Set `lambda_gp=0` and `r1_gamma=10.0` in config. Keep `lambda_gp` in config with default 0 so old configs still load. Remove `compute_gradient_penalty` from `losses.py` once R1 is validated.

**Risk:** Cannot apply mid-training — the discriminator has learned with GP regularization dynamics. Switching to R1 changes what the loss surface looks like, causing transient instability.

**When:** Next time the model is trained from scratch or a new resolution phase begins.

---

## Perceptual / LPIPS Loss

**What:** Add a VGG-based perceptual loss term to the generator objective, encouraging structurally realistic outputs at the feature level rather than just pixel level.

**Why:**
- WGAN generator loss (`-E[D(G(z))]`) only provides scalar feedback from the discriminator
- Perceptual loss adds mid-level feature matching (textures, edges, structure) that helps with fine detail in snow crystal imagery
- ~500MB VRAM overhead for frozen VGG16 — manageable on 16GB cards

**Implementation:**
1. Add `perceptual_weight` config field (default 0.0, e.g. 0.1 to enable)
2. Load frozen VGG16 feature extractor (layers `block3_conv3`, `block4_conv3`)
3. **VGG preprocessing** — VGG16 expects 224x224 ImageNet-normalized inputs:
   ```python
   # Generated images are in [-1, 1], VGG expects [0, 255] with ImageNet mean subtraction
   def preprocess_for_vgg(images):
       images = (images + 1.0) * 127.5  # [-1, 1] -> [0, 255]
       images = tf.image.resize(images, [224, 224])
       return tf.keras.applications.vgg16.preprocess_input(images)
   ```
4. In gen training loop, compute:
   ```python
   real_features = vgg(preprocess_for_vgg(real_images))
   fake_features = vgg(preprocess_for_vgg(fake_images))
   perceptual_loss = sum(MSE(rf, ff) for rf, ff in zip(real_features, fake_features))
   gen_loss = wasserstein_loss + perceptual_weight * perceptual_loss
   ```
5. Real images need to be passed into the gen training loop (currently only noise is passed)

**Risk:** Introduces a new loss term that changes the generator's optimization landscape. Must be started from the beginning or introduced very gradually with a small weight that ramps up.

**When:** Next fresh training run. Could also be tested on a branch with a reduced-epoch trial run.

---

## tf.function Compilation

**What:** Wrap the inner training loops with `@tf.function` to compile the computation graph, eliminating Python overhead.

**Why:**
- At 1024x1024, GPU compute dominates so the speedup is modest (~10-15%)
- At lower resolutions or with smaller batch sizes, the speedup is more significant (~30-50%)
- Required for XLA compilation support

**Implementation:**
1. Convert ALL augmentation functions in `augment.py` from Python `if` to `tf.cond`:
   - `_random_flip` — horizontal flip
   - `_random_brightness` — brightness jitter
   - `_random_contrast` — contrast jitter
   - `_random_saturation` — saturation jitter
   - `_random_cutout` — cutout augmentation
   - Any other augmentation functions using `tf.random.uniform` in a Python conditional
   ```python
   def _random_flip(images, p):
       return tf.cond(
           tf.random.uniform([]) < p,
           lambda: tf.image.random_flip_left_right(images),
           lambda: images
       )
   ```
2. Extract disc and gen inner loops into separate methods
3. Decorate with `@tf.function` (handles retracing when tensor shapes change)
4. Move Python conditionals (fade check, augment flag) outside the traced function or convert to `tf.cond`
5. **Retrace at resolution transitions:** `tf.function` will retrace when input shapes change during progressive growing. Use `input_signature` or accept the one-time retrace cost per resolution phase.

**Risk:** Low risk but requires careful testing — tf.function tracing can surface subtle bugs with Python state capture. The augmentation conversion is the main work.

**When:** Can be done anytime. Lower priority since 1024x1024 training is GPU-bound.

---

## Progressive Fade Schedule (Updated)

Increase fade duration at higher resolutions for more stable transitions:

| Resolution | Fade Steps | Rationale |
|------------|-----------|-----------|
| Up to 256x256 | 50,000 | Lower complexity, fast convergence |
| 512x512 | 75,000 | Moderate spatial complexity |
| 1024x1024 | 100,000–150,000 | High-res loss landscape is complex, needs time to stabilize |

These numbers assume a dataset of ~1,000–5,000 images. For significantly smaller datasets, reduce proportionally to avoid overfitting during fade. For larger datasets, the above should be sufficient.

**Stabilization phase:** After each fade completes (alpha reaches 1.0), train for at least 25–50% of the fade duration at full alpha before starting the next resolution phase. This lets the model consolidate before the next disruption.

The cosine LR decay horizon should scale proportionally (e.g., 300k–400k for 1024x1024 training).

Update `fade_steps` in config to 100000+ for the current resolution phase.

---

## Background Grid: Keep As-Is (Decision)

**Decision:** Do NOT remove the crystal card grid background from training data.

**Rationale:**
- The grid-snow boundary carries real structural information about crystal thickness, opacity, and morphology
- Grid uniformity vs. snow irregularity creates a natural contrast that forces the discriminator to learn fine-grained snow features rather than relying on low-frequency cues
- Clear/thin snow refracts the white grid in optically meaningful ways — these refraction patterns encode crystal type and orientation. The generator must learn these to produce realistic output.
- Background subtraction would introduce segmentation artifacts at snow-grid boundaries, injecting noise that the generator would learn as real features
- The grid is a small fixed pattern that the generator memorizes early — minimal wasted capacity at 1024x1024

---

## Implementation Order & Dependencies

The deferred items have ordering dependencies — implement in this order:

1. **tf.function compilation** — No dependencies, can be done anytime. Do this first since it speeds up all subsequent training.
2. **R1 regularization** — Must be done at the start of a fresh training run. Replaces GP so the loss landscape changes fundamentally.
3. **Perceptual/LPIPS loss** — Add after R1 is stable (at least 10k steps). Introduces a second loss term; easier to debug when R1 is already working.

Do NOT apply R1 and perceptual loss simultaneously from step 0 — if training destabilizes, you won't know which caused it.

---

## Recently Implemented Features

These features were added in the current development cycle and are available via CLI flags:

- **Adaptive disc/gen step ratio** (`--adaptive_steps`) — dynamically adjusts discriminator training steps based on loss EMA
- **Multi-scale discriminator** (`--multiscale_disc`) — adds a 256x256 discriminator head for low-frequency feedback
- **Gradient clipping** (`--grad_clip_norm 1.0`) — global gradient norm clipping on both gen and disc
- **Adaptive Data Augmentation (ADA)** (`--ada_target 0.6`) — adjusts augment probability based on disc overfitting signal
- **Fixed tracking seed** — consistent latent vectors for visual progress comparison across training
- **SN + GP auto-balancing** — lambda_gp auto-reduced from 10 to 1 when spectral normalization is active
- **VRAM optimizations:**
  - EMA shadow weights stored on CPU instead of GPU (~50-100MB freed)
  - InceptionV3 loaded on-demand for FID and freed after (~100MB freed between intervals)
  - Disc/lowres disc trained with separate gradient tapes to reduce peak activation memory
  - Explicit cleanup of augmented image tensors between disc and gen training
- **Mixed precision** (`--mixed_precision`) — fixed `argparse` flag and dtype-aware augmentation/GP for float16 compatibility. Halves activation VRAM; safe to enable mid-training.

---

## Evaluation Criteria

How to know if a change is working:

- **FID score** — primary metric. Track per-resolution. A successful R1 migration should reach comparable FID within 50k steps.
- **Fixed-seed visual tracking** — compare generated images from the same latent vectors across checkpoints. Look for mode collapse (all outputs identical), color drift (gradual tint), or loss of detail.
- **Disc/gen loss ratio** — disc loss should not be orders of magnitude larger than gen loss (indicates over-constrained disc, as seen with the GP² bug).
- **ADA augment probability** — if it saturates at 0 or 1, the target is miscalibrated.

---

## Rollback Strategy

If a deferred change destabilizes training:

1. **Checkpoints:** Save a checkpoint immediately before applying any deferred change. Name it clearly (e.g., `pre_r1_migration_checkpoint`).
2. **Config snapshots:** Save the full config alongside the checkpoint so you can restore exact settings.
3. **Revert window:** Give each change at least 10k steps before judging. Transient instability in the first 2-5k steps is expected when changing regularization.
4. **Partial rollback:** If perceptual loss hurts quality, set `perceptual_weight=0` without reverting R1. Each change is independent once applied.

---

## Additional Future Considerations

- **Progressive growing to 2048x2048** — The existing fade infrastructure supports this. Would require adding another conv block to both gen and disc, plus more VRAM (mixed precision becomes essential).
- **Style-based generator (StyleGAN)** — Replace the current direct generator with a mapping network + style injection architecture. Major rewrite but produces highest-quality results.
- **Latent space interpolation tools** — Add utilities to interpolate between latent vectors for exploring the learned manifold of snow crystal morphologies.
