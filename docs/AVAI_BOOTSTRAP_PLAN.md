# AvAI Bootstrap Plan — extract infra from snowGAN

> **Transfer instructions:** this file is written for a Claude session running inside
> `~/Scripts/AvAI/`. Copy it to `AvAI/docs/BOOTSTRAP_FROM_SNOWGAN.md` (or move it to the
> AvAI repo root as `CLAUDE_BOOTSTRAP.md` if you want it to be the first thing Claude
> reads). It is self-contained — the receiving session does not need access to the snowGAN
> conversation that produced it.

---

## 0. Who you are, what you're doing

You are a Claude session assisting in the `avai` Python package at
`~/Scripts/AvAI/`. AvAI's purpose is **downstream transfer learning from snowGAN** — take
a pretrained WGAN-GP discriminator (learned on paired snowpack core + magnified profile
images) and reuse its convolutional backbone to predict physical metrics carried by the
Hugging Face dataset [`rmdig/rocky_mountain_snowpack`](https://huggingface.co/datasets/rmdig/rocky_mountain_snowpack)
— starting with `avalanches_spotted` and `wind_loading`, extensible to any labeled column.

Your job is to **bootstrap AvAI as a clean downstream consumer of snowGAN**, not to
re-implement GAN training or copy snowGAN source. Depend on snowGAN as a library (git-URL
dependency is fine for now); do not vendor its source into AvAI.

## 1. Required reading before touching code

All authoritative design lives in the snowGAN repo at `~/Scripts/snowGAN/`. Read in this
order:

1. `~/Scripts/snowGAN/CLAUDE.md` — the systems-engineering operating manual. Apply all
   of its rules to AvAI work too: four-phase session workflow, two-lens review before
   full-suite runs, hard no-bandaids rule. These rules travel with the project family,
   not with a single repo.
2. `~/Scripts/snowGAN/docs/ARCHITECTURE.md` — the system map you're depending on. Pay
   close attention to §4 (data pipeline), §5 (model architecture — **the input is
   rank-5** `(B, depth=2, H, W, C)`), §9 (the existing broken `inference.py`).
3. `~/Scripts/snowGAN/docs/UPGRADES.md` — known issues in the upstream. §2 below lists
   which of these block AvAI work vs. which AvAI can work around.
4. `~/Scripts/snowGAN/docs/TRANSFER_LEARNING_PLAN.md` — the architectural shape of the
   transfer-learning pipeline AvAI is meant to implement. This doc (the one you're
   reading) is the executable version of that plan.

## 2. When to start — prerequisites in snowGAN

Two categories: hard prerequisites (block AvAI meaningfully) and soft prerequisites (AvAI
can stub around them).

### 2.1 Hard prerequisites — must be resolved in snowGAN before the final wiring

These touch the **correctness of the backbone AvAI will load** or **the contract AvAI
consumes**. Until they land, the pretrained `keras/snowgan/*.keras` artifacts and their
input-shape guarantees cannot be trusted.

| snowGAN item | Why it blocks AvAI |
| --- | --- |
| 🔴 #1 double-λ gradient penalty | Current artifacts were trained with λ=100 effective. Transfer features are suspect; a retrain on the fixed code is required. |
| 🔴 #2 depth-rebuild weight discard + 🟠 #40 single-depth assertion | Without these the backbone's shape contract is unstable — AvAI cannot safely assume rank-5 `(B,2,H,W,C)`. |
| 🔴 #32 stale trainer optimizers | Any reported LR / β tuning on the backbone was silently ignored; retrain after fix. |
| 🔴 #33 last-step-only loss logging | Reported training curves misrepresent convergence; needed to trust "this backbone finished training." |
| 🟠 #15 mixed precision | If the backbone was trained with `--mixed_precision True`, LossScaleOptimizer and output-dtype bugs taint the weights; retrain. |
| 🟠 #17 weights-only checkpoint format | `.keras` with the Lambda in the fade path is reload-fragile across Keras versions. AvAI needs `.weights.h5` + sidecar to load reliably. |
| 🟠 #11 train/val/test split persisted in config | AvAI must consume the same `test_pool` indices the backbone never saw. Without a frozen split, any eval number is contaminated. |
| 🟡 #25 modality enum (`Modality.PROFILE=0`, `Modality.CORE=1`) | AvAI inherits this convention. A named enum prevents a whole class of axis-order bugs on both sides. |

**Also required but not a UPGRADES.md item:** the snowGAN `Flatten` layer must be given a
stable name (e.g., `name="features"`) so AvAI can tap features by name, not by
`layers[-2]` position. This is noted in UPGRADES #3.

Once those are merged into snowGAN `main` and a **fresh training run** has produced new
weights, AvAI can wire to the real backbone. Until then, AvAI develops against stubs
(§5.1).

### 2.2 Soft prerequisites — nice, but workarounds exist

These improve quality-of-life but do not block AvAI:

- 🟠 #8 atomic checkpointing, #10 tf.data pipeline, #13 structured logging — affect
  future snowGAN runs, not the artifact AvAI consumes.
- 🟠 #18 HF Hub push — convenient for `from_pretrained()` calls; until it exists, load
  from the local `keras/snowgan/` path.
- 🟢 #27 README example, #30 pre-commit — independent of AvAI.

### 2.3 Recommended sequencing

Read `~/Scripts/snowGAN/docs/UPGRADES.md` §"Suggested sequencing". The snowGAN-side plan
puts the hard prerequisites in Weeks 1–3. **AvAI should start Phase 1–3 of this plan in
parallel with snowGAN Week 1**, wait to begin Phase 4 until snowGAN Week 2 lands
(splits + inference rewrite deleted from snowGAN), and do the final wiring (Phase 5–6)
only after a fresh training run on snowGAN `main` produces a clean backbone.

## 3. What to extract vs. build fresh

### 3.1 Consume from `snowgan` (do not copy)

Add `snowgan` to AvAI's `pyproject.toml` dependencies (git URL for now, PyPI eventually).
Consume these named surfaces:

- `snowgan.models.discriminator.load_discriminator(checkpoint, config)` — load the
  pretrained critic.
- `snowgan.data.dataset.DataManager` — the HF manifest loader. *Do not* re-implement
  pairing; call `DataManager.batch_merged(...)` (or the pair-index dict the
  post-UPGRADES pipeline exposes).
- `snowgan.config.build` — the config schema. AvAI's own config should compose with
  snowGAN's (not inherit, not duplicate).
- `snowgan.utils.configure_device` — device/XLA/mixed-precision bootstrap. Called the
  same way from `avai`'s entry point, **before** any `import tensorflow`.

### 3.2 Build fresh in AvAI

Do not port snowGAN's GAN training loop, fade scheduler, sample reporter, or checkpoint
cleanup. AvAI is a classifier/regressor pipeline; it has different lifecycle semantics.

Build in AvAI:

- `src/avai/data.py` — `tf.data` pipeline that filters to labeled rows, builds
  `(image, {head_name: label})` batches, honors the snowGAN-persisted train/val/test
  split (by index).
- `src/avai/backbone.py` — thin wrapper around the loaded `Discriminator` that exposes a
  frozen feature extractor keyed to the named Flatten layer, with `.trainable` toggles
  per conv block for partial unfreezing.
- `src/avai/heads.py` — factory of metric-specific heads keyed by HF column name. Each
  entry defines `(output_layer, loss, metrics, class_weights_from_histogram)`.
- `src/avai/train.py` — Keras `fit()` loop, multi-task loss, TensorBoard callback,
  frozen-backbone default. Writes `avai/{run_name}/` with weights, head configs, split
  manifest, metric history.
- `src/avai/evaluate.py` — per-head evaluation: accuracy, macro-F1, confusion matrix,
  calibration, baseline comparisons (majority-class, random, from-scratch CNN of the
  same head shape).
- `src/avai/cli.py` — single entry point `avai train --head avalanches_spotted`,
  `avai eval --head ...`, `avai push-hf`.

### 3.3 Hard don'ts

- **Do not edit `~/Scripts/AvAI/src/snowGAN/snowGAN.py`.** It is a legacy predecessor copy
  of snowGAN, kept for reference only. (It also has a broken `sys.path.append` before
  `import sys` at line 7 — NameError on load. Leave it alone.)
- **Do not extend `~/Scripts/AvAI/src/snowMaker/` or `~/Scripts/AvAI/src/coreDiffusor/`**
  as part of this plan. snowMaker is the upstream field-photo → HF-row preprocessing
  pipeline; coreDiffusor is a separate diffusion-model experiment. New transfer-learning
  code lives in a new `src/avai/` package.
- **Do not re-implement paired-sample loading.** The input shape is `(B, 2, H, W, C)`
  with `depth=[profile, core]`. Use the upstream `DataManager` (or the post-UPGRADES
  pair-index dict). Copying the pairing logic will drift out of sync.
- **Do not vendor snowGAN source into `avai/`.** Depend on the package.
- **Do not restart training from scratch.** The whole point is to reuse the trained
  convs. If you find yourself training conv blocks without `trainable=False`, stop and
  re-read the plan.

## 4. Shape contract AvAI inherits

From snowGAN `feature/modality_blending`:

```
Input tensor:  (B, depth=2, H=1024, W=1024, C=3)
Depth axis:    [Modality.PROFILE = 0, Modality.CORE = 1]
Value range:   [-1, 1]   # tanh-scaled, matches generator output
```

AvAI's data loader must produce this shape. Two realistic options per
`~/Scripts/snowGAN/docs/TRANSFER_LEARNING_PLAN.md` §4:

1. **Paired** — default. Drop samples that don't have a matched core/profile pair.
2. **Tile-to-depth-2** — fallback for profile-only inference. Repeat the profile along
   the depth axis. Gate behind a `--single-view` flag and log it in every run output.

Pick (1) unless label coverage forces you to (2).

## 5. Execution phases

Apply snowGAN's `CLAUDE.md` four-phase workflow to each phase below: plan → focused
tests → two-lens review → full suite before push.

### 5.1 Phase 1 — project skeleton (start: snowGAN Week 1, in parallel)

- Rewrite `~/Scripts/AvAI/pyproject.toml`:
  - `name = "avai"`, `requires-python = ">=3.10"`.
  - Dependencies: `snowgan @ git+https://github.com/.../snowGAN.git@main`,
    `tensorflow`, `datasets`, `pydantic`, `numpy`, `matplotlib`.
  - Scripts entry: `avai = "avai.cli:main"`.
- Create `src/avai/{__init__.py, cli.py, config.py, backbone.py, heads.py, data.py,
  train.py, evaluate.py}` as empty stubs with module docstrings describing the intent.
- Create `tests/` with `tests/conftest.py` that builds a tiny fake
  `Discriminator`-shaped model (5 Conv3D blocks → Flatten named `features` → Dense(1))
  against which AvAI code can be unit-tested without loading real weights. This lets
  Phase 2–4 progress while snowGAN prerequisites land.
- Add `.github/workflows/ci.yml` running `pytest`, `ruff check`, `mypy src/avai`.
- Add `AvAI/CLAUDE.md` that points at `~/Scripts/snowGAN/CLAUDE.md` as the canonical
  operating manual ("rules travel with the project family").

**Gate:** CI green on an empty test suite. Stop here if CI is red.

### 5.2 Phase 2 — data layer (start: parallel with snowGAN Week 1)

- `src/avai/data.py::load_rmdig_dataset(split_name)` — returns a `tf.data.Dataset`
  yielding `(image, {head_name: label})`. Under the hood: reuse
  `snowgan.data.dataset.DataManager` for the manifest + pair index.
- Implement both paired and tile-to-depth-2 modes. Log which mode is active on every run.
- Write the split manifest to `avai/{run_name}/splits.json` as a list of
  `(site, column, core)` triples per `{train, val, test}`. **Consume the snowGAN-side
  persisted split once UPGRADES #11 lands** — until then, derive your own split with a
  fixed seed and document that AvAI-derived splits must be replaced with snowGAN-derived
  splits when #11 ships.
- Tests: small synthetic manifest, assert correct pair count per split, assert no
  leakage across splits, assert stable ordering across repeated loads at the same seed.

**Gate:** full data loader round-trips a fake dataset into the fake-backbone stub.

### 5.3 Phase 3 — backbone wrapper & head factory (start: parallel)

- `src/avai/backbone.py::load_backbone(checkpoint_path, freeze=True)`:
  - Uses `snowgan.models.discriminator.load_discriminator` internally.
  - Sets `layer.trainable = False` on every layer when `freeze=True`.
  - Returns a model whose output is the `features` layer (by name, not by index).
- `src/avai/heads.py::HEAD_REGISTRY: dict[str, HeadSpec]`:
  - Keys are HF column names.
  - Values declare `output_layer_factory`, `loss`, `metrics`,
    `class_weights_from_histogram(labels) -> dict`.
  - Register `avalanches_spotted` and `wind_loading` initially; the factory pattern makes
    adding a third a 10-line change.
- Tests: backbone freeze toggles actually freeze, head factory produces a model that
  compiles and `fit`s on a tiny dummy batch, class weights are computed correctly from
  a known histogram.

**Gate:** a `build_transfer_model(fake_backbone, heads=["avalanches_spotted"])` produces
a model that `fit`s for one step without errors.

### 5.4 Phase 4 — training loop (start: after snowGAN #17 lands + Week 2 done)

By this point snowGAN has shipped #17 (weights-only checkpoint), #11 (split persistence),
#25 (modality enum), and #3 (the broken `inference.py` has been removed from snowGAN, not
patched). A fresh snowGAN training run has produced a real backbone.

- `src/avai/train.py::train(head_names, backbone_ckpt, ...)`:
  - Loads the real backbone.
  - Builds the multi-task head.
  - Configures optimizer, class weights, callbacks (TensorBoard, EarlyStopping,
    ModelCheckpoint — atomic writes, see CLAUDE.md §4).
  - Runs `model.fit(train_ds, validation_data=val_ds, ...)`.
  - Writes artifacts to `avai/{run_name}/` with a run manifest (git SHA, seed, snowGAN
    backbone SHA, split manifest, config snapshot).
- Honor snowGAN seeding conventions (`tf.random.set_seed`, `np.random.seed`,
  `random.seed`, `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS=1`). Seed is a CLI arg and is
  logged.
- Integration test: real backbone loaded, tiny split, 1 epoch, validate metric file
  is written atomically.

**Gate:** one clean run end-to-end on `avalanches_spotted` at 64×64 downscale with a
real backbone. Reviewed under both lenses (architecture: head contract stable, backbone
not accidentally trained; execution: split honored, atomic writes, no cross-leak).

### 5.5 Phase 5 — evaluation & baselines (start: after Phase 4 gate)

- `src/avai/evaluate.py::evaluate(run_dir)`:
  - Loads the held-out `test_pool` split only (never retrains anything).
  - Reports per-head metrics with 95% CIs computed from 3 seeds minimum.
  - Produces a confusion matrix PNG and a calibration plot per head.
  - Computes baselines on the same split: majority-class, random, from-scratch CNN at
    the same head size trained on the same label budget.
- Tests: evaluation is deterministic at a fixed seed; baselines are computed against
  the same split; any attempt to evaluate on `val` or `train` fails loudly.

**Gate:** the test_pool number is believable (not a wild outlier vs. val) and the
from-scratch baseline is meaningfully lower than the transfer-learned number. If
transfer ≈ scratch, stop and investigate before adding more heads.

### 5.6 Phase 6 — HF push + model card (start: after Phase 5 gate)

- `src/avai/hub.py::push(run_dir)` uploads weights + head configs + split manifest +
  model card to `rmdig/avai-{head_name}`.
- Model card lists: dataset commit, snowGAN commit, seeds, compute, per-head metrics
  with CIs, known failure modes, intended use, not-intended use (matches snowGAN's
  "not safety-critical" stance).
- Tests: push is dry-runnable; model card parses as valid HF front-matter.

**Gate:** a freshly-pulled machine can `pip install avai` and
`avai eval --head avalanches_spotted` reproduces the reported test metric within CI.

## 6. Guardrails — apply to every phase

From `~/Scripts/snowGAN/CLAUDE.md`. Re-stated here so they're non-optional in AvAI:

- **No bandaids.** If a test fails, find the root cause upstream. Casting, try/except,
  tolerance bumping, and skipping cases are all forbidden as default fixes. If a bandaid
  is genuinely the right call right now, mark it `# BANDAID(<owner>, <date>, <issue>):`
  and track the debt.
- **Four phases per session.** Plan → focused tests → two-lens review → full suite.
  Never skip the two-lens review to get to the full suite faster.
- **Two lenses, in order.** Architecture first (what contract moved, who depends on the
  old one), then execution (walk through CLI → data → model → train → checkpoint).
- **Atomic writes for all persistence.** `tmp + os.replace()`. Applies to split
  manifests, run manifests, metrics, weights.
- **No `atexit` for correctness-critical state.**
- **Environment vars before TF import.** Bootstrap module runs before anything touches
  `tensorflow`.
- **Names over indices.** `model.get_layer("features").output`, never `layers[-2]`.
- **Stop and ask** before: deleting or overwriting any `avai/{run_name}/` directory,
  modifying the HF dataset, pushing to HF Hub, changing `AvAI/src/snowGAN/` or
  `AvAI/src/snowMaker/`.

## 7. Anti-goals — things this plan is explicitly NOT

- Not a GAN implementation. AvAI does not train GANs.
- Not a replacement for snowGAN. snowGAN stays the pretraining library; AvAI is the
  downstream transfer library. One-way dependency.
- Not a rewrite of `snowMaker` or `coreDiffusor`. Those live separately.
- Not a port of `~/Scripts/snowGAN/src/snowgan/inference.py`. That file is scheduled for
  deletion in snowGAN (UPGRADES #3). The logic it *intended* to perform lives here, in
  AvAI, and is built fresh per this plan.

## 8. Success criteria

- AvAI installs cleanly as `pip install -e .` against a snowGAN git-URL dependency.
- `avai train --head avalanches_spotted --seed 0` produces deterministic results across
  reruns.
- `avai eval --head avalanches_spotted` on the persisted `test_pool` beats the
  from-scratch baseline with 95% confidence.
- A reviewer can reproduce a run from the on-disk artifacts alone (run manifest, split
  manifest, config, weights, seed). No tribal knowledge, no manual steps.
- The model card on HF Hub lets a stranger understand what the model is, how to use it,
  what it's not for, and how it was evaluated.
