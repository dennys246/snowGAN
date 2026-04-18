# snowGAN Architecture

A narrative map of the codebase as it exists on `feature/modality_blending`. Paired with
[UPGRADES.md](UPGRADES.md) (what to change) and [TRANSFER_LEARNING_PLAN.md](TRANSFER_LEARNING_PLAN.md)
(how to extend into AvAI).

## 1. Purpose

snowGAN trains a WGAN-GP on the [`rmdig/rocky_mountain_snowpack`](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack)
Hugging Face dataset. The current branch extends it so that generator output and discriminator
input are a *stack* of views — a matched `core` image and `magnified_profile` image — along a new
`depth` axis (modality blending). Intent: produce a backbone that has learned joint
core/profile structure, then transfer it downstream (AvAI) to predict avalanche risk, wind
loading, etc.

## 2. Top-level layout

```
src/snowgan/
├── main.py                 # CLI entry (`snowgan` console script)
├── config.py               # build(): God-config with atexit save
├── trainer.py              # Trainer: loop, fade, checkpoint, cleanup
├── models/
│   ├── generator.py        # Conv3DTranspose stack + fade_endpoints sub-model
│   └── discriminator.py    # Conv3D stack → Flatten → Dense(1)
├── data/
│   └── dataset.py          # DataManager: HF manifest, batch() / batch_merged()
├── generate.py             # generate(): sample + save; make_movie()
├── inference.py            # run_inference(): swap heads for avalanche/wind labels
├── losses.py               # emd_loss, generator_loss, compute_gradient_penalty
├── utils.py                # CLI parse, device config, fade helpers
└── log.py                  # plain-text loss/history persistence
```

Artifacts live under `keras/snowgan/`: `generator.keras`, `discriminator.keras`, their JSON
configs, `synthetic_images/batch_*_synthetic_*_{core,profile}.png`, `history.png`,
`generator_loss.txt`, `discriminator_loss.txt`, `trained.txt`.

## 3. Runtime flow (train mode)

```
snowgan --mode train --save_dir keras/snowgan/
  → main.parse_args()
  → configure_device(args)                       # env vars set *after* TF import (bug)
  → build(gen_config_path) + configure_gen       # merge CLI → saved JSON
  → build(disc_config_path) + configure_disc
  → load_generator(checkpoint, gen_config)       # Generator(config); keras.Model.build
  → load_discriminator(checkpoint, disc_config)
  → Trainer(generator, discriminator)
      • load weights via keras.models.load_model + set_weights (survives shape-compat)
      • DataManager(config) loads HF manifest to pandas, drops image/audio, keeps rows as list
      • atexit.register(save_model)              # not crash-safe
  → Trainer.train(batch_size, epochs)
      for epoch in [current_epoch, current_epoch+epochs):
        dataset.reset_seen_profiles()
        while batch_merged() returns data:
          _ensure_depth_alignment(batch.depth)   # silently rebuilds models & loses weights
          train_step(x)
          plot_history()
          generate(...)                          # always runs → I/O heavy
          if batch % cleanup_milestone == 0:
            save_model(batch_subdir)
            _cleanup_saved_batches(100)
          batch += 1
```

## 4. Data pipeline (`src/snowgan/data/dataset.py`)

- `load_dataset(dataset_name)` loads the full HF split into memory.
- `to_pandas().drop(['image','audio'], errors='ignore').values.tolist()` ⇒ in-memory manifest
  for O(1) metadata lookup per index.
- Image bytes are still fetched row-by-row from the underlying `self.dataset['train'][idx]`.
- Two batching modes:
  - `batch(batch_size, datatype=...)`: single-view, filters `sample_datatype == translator[datatype]`, expands to depth=1.
  - `batch_merged(batch_size)`: walks the manifest linearly, and for each `datatype==0` (core)
    it forward-scans until it finds a `datatype==2` (magnified_profile) with matching
    `site`/`column`/`core`. That gives a `(depth=2, H, W, C)` stack: `[profile, core_resized]`.
  - Profile consumption is tracked in `self.seen_profiles` (a set mirrored back to
    `config.seen_profiles`) so they are not reused within an epoch.
- `preprocess_image`: `tf.image.resize → cast float32 → /127.5 - 1` (in [-1, 1], matches `tanh`).

### Data bugs worth naming

- `train_ind` is advanced only when a match is found — but for the *profile* scan it is not
  bounded when profiles are exhausted, resulting in the double-increment path in `batch_merged`
  at lines 181–185 skipping cores rather than reusing them.
- `if profile_image == None:` compares a tensor to `None`; works by accident because the var
  is only ever reassigned from `None` → tensor, but it will raise if the branch ever evolves.
- `batch()` re-computes `self.dataset['train'][self.config.train_ind]` *after* reading the
  manifest row — two round trips per hit.

## 5. Model architecture

### Generator (`models/generator.py`)

```
Input: (latent_dim,)                        # default 256 in config, 100 in CLI default
Dense(depth * 16 * 16 * filters[0], no bias)
Reshape((depth, 16, 16, filters[0]))        # starts 3D
For f in filter_counts (default [1024,512,256,128,64]):
  Conv3DTranspose(f, (1,kH,kW), stride=(1,2,2), padding='same')
  [optional BatchNorm]
  LeakyReLU(negative_slope)
Conv3DTranspose(channels, (1,kH,kW), stride=(1,2,2), padding='same', activation='tanh') → curr_img
```

A parallel `fade_endpoints` model outputs `(prev_up, curr_img)` where `prev_up` is the second
last feature map projected to RGB with a 1×1×1 Conv3D and spatially resized to match
`curr_img` via a Lambda layer (`tf.image.resize`, nearest). The Trainer blends them by
`(1-α)·prev_up + α·curr_img` during the fade window (`fade_steps` → default 50k).

Spatial math: six Conv3DTranspose stride-2 blocks starting from 16×16 give
`16 → 32 → 64 → 128 → 256 → 512 → 1024`. That matches the configured resolution. If
`filter_counts` length changes or a block is skipped, the `(1024, 1024)` requirement breaks
silently.

### Discriminator (`models/discriminator.py`)

```
Input: (depth, H, W, C) = (2, 1024, 1024, 3)
For f in filter_counts (default [64,128,256,512,1024]):
  Conv3D(f, (1,kH,kW), stride=(1,2,2), padding='same')
  LeakyReLU(negative_slope)
Flatten
Dense(1)                                    # no activation (WGAN critic)
```

## 6. Loss & optimization (`losses.py`, `trainer.train_step`)

- Critic loss: `E[D(fake)] - E[D(real)] + λ · GP`.
- GP: 1-Lipschitz penalty via interpolated samples; `alpha` is rank-5 `(B,1,1,1,1)` so
  interpolation broadcasts over depth. Gradient norms are computed across axes `[1,2,3,4]`.
- Generator loss: `-E[D(fake)]`.
- Optimizers: Adam β₁=0.5, β₂=0.9 (separate instances per model).
- Training schedule: `disc.training_steps` critic updates per `gen.training_steps` generator
  updates, per batch.

### Double-λ bug (confirmed)

`compute_gradient_penalty(...)` already multiplies `mean((‖∇‖-1)²)` by `lambda_gp` before
returning. Then `Discriminator.get_loss(real, fake, gp, lambda_gp)` adds `lambda_gp * gp` again,
giving an effective penalty of `λ² · mean((‖∇‖-1)²)`. With the configured `λ=10`, the
discriminator is training with a 100× stronger penalty than intended.

### Other numerical concerns worth naming here

- The interpolation `alpha` is shape `(B,1,1,1,1)` so the same value broadcasts across
  both depth slices (core + profile). See [UPGRADES #37](UPGRADES.md).
- Training records only the last inner-step's loss (not an average), so plotted curves
  understate the number of updates per batch. See [UPGRADES #33](UPGRADES.md).
- The trainer builds its own optimizer instances but never uses them; all gradient
  applies go through `self.gen.optimizer` / `self.disc.optimizer` instead. See
  [UPGRADES #32](UPGRADES.md).

## 7. Fade scheduling

- `global_step` is max of persisted `fade_step` from gen/disc configs and increments at end of
  each `train_step`. `compute_fade_alpha(step, total)` produces `clip(step/total, 0, 1)`.
- `_use_fade()` only returns True while `fade_complete == False` and `fade_endpoints` exists.
  Once `global_step >= fade_steps`, fade is latched off permanently for the rest of the run.
- `_sync_fade_progress(persist=True)` writes both configs to disk every step — 2 JSON writes
  per step is IO-bound on long runs. Those writes are non-atomic (`open(..., 'w') +
  json.dump`), so a crash mid-write corrupts the config file and kills resume. See
  [UPGRADES #34](UPGRADES.md).

## 8. Checkpointing & persistence

- Weights: `self.gen.model.save(.../generator.keras)` + discriminator (full-model dump).
- Configs: JSON via `build.save_config()`, registered at `atexit`; not signal-safe.
- Loss histories: plain `.txt` newline-delimited floats, rewritten from scratch each save.
- `synthetic_images/` grows unbounded; `_cleanup_saved_batches(100)` removes non-milestone
  checkpoint directories and trims all but the last 7 synthetic PNGs per non-milestone batch.
  Runs only when `batch % cleanup_milestone == 0`, so a run that stops mid-milestone leaves
  the scratch dir fat.
- `generate(...)` is called *every* train_step (not just at milestones), writing
  `n_samples * depth` PNGs per step. See [UPGRADES #38](UPGRADES.md).
- No lockfile on `save_dir`; two processes sharing a directory silently corrupt each
  other's writes. See [UPGRADES #39](UPGRADES.md).
- Batch counter is recovered by globbing `synthetic_images/batch_*.png` — the same
  files `_cleanup_saved_batches` deletes at milestones, producing silent overwrite of
  older `batch_N/` snapshots. See [UPGRADES #36](UPGRADES.md).
- `reset_seen_profiles()` is called at epoch top and immediately persisted by the next
  per-step config sync, so a crash mid-epoch resumes with a cleared set and reuses
  profiles. See [UPGRADES #35](UPGRADES.md).

These items all share one root cause — persistence has no transaction boundary. See
the cross-lens section in [UPGRADES.md](UPGRADES.md) for the unified-`Checkpointer` fix.

## 9. Inference head (`inference.py`)

Builds a new Keras model from the loaded discriminator: takes `base.layers[-2].output` (the
Flatten output) and attaches two softmax heads — `avalanches_spotted` (21 classes) and
`wind_loading` (4 classes). Uses sparse categorical crossentropy. Walks the first `N` rows of
the HF train split, filters to `datatype==magnified_profile`, yields `(image, labels)` via
`tf.data.Dataset.from_generator`.

### Inference bugs worth naming

- **Input-rank mismatch**: the discriminator's input shape is `(depth=2, H, W, C)`; the
  inference dataset yields rank-3 tensors `(H, W, C)`. Feeding the model raises a shape
  error at the first batch.
- **Eval-only**: `test_on_batch` evaluates an untrained head with random weights. The function
  is framed as "inference" but never calls `.fit()` or tunes the heads, so reported accuracies
  are baseline noise.
- **No backbone freezing**: even if `.fit()` were added, there is no `trainable=False` on the
  conv blocks; the whole GAN critic would be fine-tuned, defeating transfer learning.

## 10. CLI surface (`utils.parse_args`)

~30 flags spread across generic/generator/discriminator. Rough groupings:

- Runtime: `--mode {train,generate,infer}`, `--device`, `--xla`, `--mixed_precision`,
  `--batch_size`, `--epochs`, `--save_dir`, `--dataset_dir`, `--rebuild`.
- Sampling: `--n_samples`, `--latent_dim` (typed `float`, used as int),
  `--resolution` (typed `set` — argparse can't build a set from a single arg, so this is
  effectively dead).
- Fade: `--fade`, `--fade_steps`, `--cleanup_milestone`.
- Per-model overrides: `--gen_*`, `--disc_*` for checkpoint, kernel, stride, lr, betas,
  leaky-slope, steps, filter counts, norm.
- Inference: `--infer_samples`.

Several `type=bool` flags: argparse casts any non-empty string to `True`, so
`--fade False` becomes `True`. Should be `action='store_true'` or `BooleanOptionalAction`.

## 11. AvAI (neighbour project, `~/Scripts/AvAI/`)

AvAI is the downstream "what do we do with a trained snowpack backbone" project.

- `src/snowGAN/snowGAN.py` — a predecessor of this repo; obsolete copy kept for reference.
  Contains a `sys.path.append` **before** `import sys` (NameError on load). Not wired to the
  current snowgan package.
- `src/snowMaker/` — preprocessing pipeline: `colorSegmenter` extracts core/profile/label
  panels from raw field photos (red/green/blue color keying + perspective warp), `Labeler`
  is an MNIST-trained CNN for reading the hand-written numeric label panel.
- `src/coreDiffusor/coreDiff.py` — separate diffusion model (`Diffusor`, U-Net w/ sinusoidal
  time embedding) for generating synthetic snow cores.
- `models/` — `crystaldig/label_model.h5`, `corediff/*.keras`, `snowgan/` legacy artifacts.

AvAI's `pyproject.toml` is minimal (`tensorflow` only) and has no CLI or installed entry
point. It is effectively a research scratchpad right now.

## 12. Dependency & environment notes

- snowGAN depends on `tensorflow>=2.15` (or `tf-nightly` on py ≥ 3.11), plus `datasets`,
  `numpy`, `matplotlib`, `pillow`, `click` (unused), `opencv-python-headless` (generate.py,
  make_movie).
- `.gitignore` excludes `keras/` and `pytorch/` — trained artifacts are not tracked, good.
- `__pycache__/` is ignored at the top level but `src/snowgan/**/__pycache__/` is committed
  (inspect `src/snowgan.egg-info/` and the `*.pyc` files visible on disk). Worth cleaning.
- No `Pipfile.lock` / `poetry.lock` / `uv.lock`; no reproducible install.
- Submodule `external/snowMaker` is declared in `.gitmodules` but the directory is absent in
  the tree.

## 13. Reproducibility posture

- No global seed set anywhere (tf, numpy, python random).
- Config persists `train_ind`, `current_epoch`, `seen_profiles`, `fade_step` — good for
  resume; but `batch` counter is recovered by globbing `synthetic_images/batch_*.png` and
  taking the max — fragile to user-side cleanup.
- Loss history is appended in-memory then rewritten in full on every save; a crash mid-save
  truncates to zero bytes with no atomic rename.

## 14. Observability posture

- All output is `print(...)`. No `logging` module, no level control, no structured logs.
- Metrics are losses only. No FID/KID/IS for GAN quality, no gradient norms, no per-layer
  activation stats, no W&B/TensorBoard integration.

## 15. Test / CI posture

- Zero tests (`tests/`, `test_*.py`, `__tests__/` all absent).
- No CI config (`.github/workflows/` absent).
- No type checking (`mypy`, `pyright` absent).
- No linter config (`ruff`, `black`, `flake8` absent beyond a mention in the gitignore).
