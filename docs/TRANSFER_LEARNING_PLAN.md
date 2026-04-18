# Transfer Learning Plan: snowGAN → AvAI metric prediction

Goal: take the trained snowGAN discriminator (learned features of paired core + magnified
profile images of snowpack) and reuse its convolutional backbone to predict physical metrics
carried in the Hugging Face dataset (`rmdig/rocky_mountain_snowpack`) — starting with
`avalanches_spotted` and `wind_loading`, extensible to any labeled column.

This plan lives alongside the current broken [inference.py](../src/snowgan/inference.py) and
the in-progress AvAI project at `~/Scripts/AvAI/`.

**Scope of this doc vs. the execution plan.** This document is the *architectural design*
— what to build and why. The step-by-step *execution plan* (phases, prerequisites,
guardrails, start conditions) lives in [AVAI_BOOTSTRAP_PLAN.md](AVAI_BOOTSTRAP_PLAN.md),
which is written for a Claude session running inside the AvAI repo. If you are about to
start coding the transfer pipeline, read that plan too; it specifies which snowGAN
upgrades must land before each phase can begin.

## 1. What the dataset gives us

From the code references we already have (`dataset.py`, `inference.py`), the HF dataset
carries these columns per row:

| Column | Type | Used by |
| --- | --- | --- |
| `image` | PIL | training input |
| `datatype` | int (0=core, 1=profile, 2=magnified_profile, 3=crystal_card) | pipeline routing |
| `site`, `column`, `core`, `segment` | str/int | pairing cores ↔ profiles |
| `avalanches_spotted` | int 0..20 | regression / classification target |
| `wind_loading` | int 0..3 | ordinal / classification target |

Before coding anything, run a one-off notebook that prints the dataset's full column schema
and null-rate per column. The value of this exercise compounds: every additional labeled
column is a candidate head.

## 2. Architecture

```
              ┌──────────────────────────────────────┐
              │         snowGAN Discriminator        │
              │   Conv3D × 5  (frozen in TL mode)    │
              └──────────────┬───────────────────────┘
                             │ features (Flatten output)
                             ▼
                    ┌──────────────────┐
                    │ shared projection│  Dense(512, ReLU) + Dropout
                    └────────┬─────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   avalanches_head      wind_loading_head   (next metric…)
   Dense(21, softmax)   Dense(4, softmax)   Dense(K, softmax|linear)
```

Three modes:

- `backbone_frozen`: train only heads + projection (fast baseline).
- `backbone_partial`: unfreeze last N conv blocks with 10× lower LR (refinement).
- `backbone_full`: everything trainable, tiny LR (only if you have enough labels).

## 3. Where this code lives

A clean separation keeps snowGAN focused on pretraining:

- `snowgan` (this repo) publishes the pretrained discriminator backbone as a
  first-class object.
- `avai` (the `~/Scripts/AvAI/` repo) depends on `snowgan` and owns the transfer-learning
  pipeline.

Concretely, in this repo add:

- `src/snowgan/models/backbone.py` — `DiscriminatorBackbone` that wraps the loaded critic and
  exposes a named `features` output with `.trainable` toggles per block.
- `src/snowgan/hub.py` — `push_to_hf(...)` and `load_from_hf(...)` helpers.

In the AvAI repo, create:

- `src/avai/heads.py` — factory for heads keyed by HF column name
  (`avalanches_spotted`, `wind_loading`, future ones).
- `src/avai/data.py` — `tf.data` loader that streams the HF dataset, filters to
  `datatype=magnified_profile` (or merged pairs to match the backbone's expected
  `depth=2` input), and yields `(image, {head_name: label})`.
- `src/avai/train.py` — Keras `fit()` training loop with frozen-backbone default, W&B/TB
  logging, class-weight handling for imbalanced `avalanches_spotted`.
- `src/avai/evaluate.py` — per-head metrics: accuracy, macro-F1, confusion matrix, calibration.

## 4. Reconciling the input-shape problem

The snowGAN critic's input is rank-5 `(depth=2, H, W, C)` — matched pairs. Three realistic
options for AvAI:

1. **Feed the same paired structure.** Build the AvAI loader identically to
   `dataset.batch_merged`: pair each `magnified_profile` with its `core`, stack along depth.
   Pros: zero change to the backbone; maximally reuses what the GAN learned. Cons: you can
   only score samples that have both views — if a profile has no matching core, it is
   excluded from evaluation.

2. **Tile a single view to `depth=2`.** Repeat the profile along the depth axis for
   profile-only inference. Pros: works for any sample. Cons: the network is being fed
   out-of-distribution input; reported metrics may mislead. Acceptable as a baseline, not
   for shipping.

3. **Train a second, rank-4 critic from scratch** with the same filter schedule, then
   port weights block-by-block (depth-slice projection). Pros: clean inference surface.
   Cons: weight migration is fiddly and you still lose the cross-modality inductive bias
   you spent compute learning.

Start with (1). If label coverage turns out to be sparse, fall back to (2) gated behind a
clear `--single-view` flag that logs the fact in every run.

## 5. Minimal correct replacement for `inference.py`

The existing function is a misnomer — it builds heads but does not train them. A clean
rewrite, which should live in AvAI rather than in snowGAN:

```python
# src/avai/train.py (sketch — not yet implemented)

from snowgan.models.discriminator import load_discriminator

def build_transfer_model(discriminator_checkpoint, heads, freeze_backbone=True):
    disc = load_discriminator(discriminator_checkpoint)
    backbone = disc.model
    if freeze_backbone:
        for layer in backbone.layers:
            layer.trainable = False

    features = backbone.get_layer("features").output  # rename Flatten in snowgan
    shared = tf.keras.layers.Dense(512, activation="relu")(features)
    shared = tf.keras.layers.Dropout(0.25)(shared)

    outputs = {name: head_factory(shared) for name, head_factory in heads.items()}
    model = tf.keras.Model(backbone.input, outputs, name="AvAI")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={name: h.loss for name, h in heads.items()},
        metrics={name: h.metrics for name, h in heads.items()},
    )
    return model
```

Train with `model.fit(train_ds, validation_data=val_ds, epochs=N, callbacks=[...])` against
a **real** train/val split derived from the pairing index, with `class_weight` set from the
label histogram.

## 6. Concrete upgrade dependencies from snowGAN

Block the AvAI work on these snowgan-side fixes (they are all in
[UPGRADES.md](UPGRADES.md)):

- #3 — rewrite `inference.py` or move it into AvAI (we choose: move, then delete from
  snowGAN). Keeps snowGAN single-purpose.
- #11 — split HF data into `trained_pool / validation_pool / test_pool` with a fixed seed,
  persisted in config. AvAI must consume the same test_pool indices to evaluate on held-out
  data the GAN has never seen.
- #17 — weights-only checkpoint format. Loading full `.keras` files with Lambda layers into
  a second project (AvAI) is a known Keras pain point.
- #18 — HF Hub push of the pretrained discriminator so AvAI can `from_pretrained(...)` it
  rather than point at an absolute local path.
- #25 — encode modality order as an explicit enum. AvAI inherits the convention.

## 7. Metric plan per head

- `avalanches_spotted` (0..20, long-tailed) — classification head with label smoothing;
  evaluate macro-F1 and RMSE-on-class-index (treat as ordinal). Baseline: majority-class
  classifier. Consider a Poisson-regression head instead of 21-way softmax if counts are
  truly unbounded.
- `wind_loading` (0..3) — ordinal classification; use CORAL / CORN loss or simple softmax
  with macro-F1. Confusion matrix is the primary human-read metric.
- Future columns — document each in `src/avai/heads.py` with its range, loss, metrics, and
  a short "what does a wrong answer cost" note so we can pick thresholds.

## 8. Evaluation protocol (non-negotiable)

- Splits are frozen at index level, derived from the pair index and persisted as
  `splits.json` (list of `(site, column, core)` triples per split). Any future rerun loads
  the same splits.
- Baselines report on val and test: majority class, random, and a CNN-from-scratch with the
  same head layout trained on the same label budget.
- Every reported number is a median across ≥ 3 seeds; variance is shown.
- The HF model card for the fine-tuned AvAI model lists: dataset commit hash, snowGAN commit
  hash, seed(s), compute used, per-head metrics with 95% CIs, known failure modes.

## 9. Execution sequence

1. Land snowGAN Tier 🔴 fixes (#1, #2, #4) — one week, because a mis-trained critic taints
   every downstream transfer.
2. Land snowGAN #11 (splits), #17 (weights-only), and name the Flatten layer (`name="features"`).
3. Bootstrap `avai` as a proper package: `pyproject.toml` with `snowgan` as a dependency
   (git URL is fine for now), `src/avai/{data,heads,train,evaluate}.py`, a `tests/` dir.
4. Wire `avai.train` end-to-end with `avalanches_spotted` only. Target: a single working
   frozen-backbone run with TensorBoard output and a sensible val metric on a toy slice
   (100 pairs) before scaling.
5. Add `wind_loading` as the second head. Validate multi-task loss balancing (either
   uncertainty weighting or simple weight sweep).
6. Add the evaluation protocol (splits, baselines, model card). Push the first AvAI model
   to HF Hub.
7. Generalize `heads.py` so a new metric is "register a column name + loss + metrics" —
   opens the door to any labeled column the dataset grows to carry.

## 10. Open questions to answer before coding

- Label coverage per column: how many rows have a non-null `avalanches_spotted` /
  `wind_loading`? If ≪ 500, transfer learning is the right call; if ≫ 5k, a bespoke CNN
  might outperform.
- Profile↔core pairing rate: how many profiles have a matching core? Drives the choice
  between §4(1) and §4(2).
- Test-time expectations: is AvAI's output consumed by a notebook, a CLI, or a service? That
  decides whether we also need a FastAPI/TF Serving layer.
- Deployment target: laptop CPU, on-prem GPU, or cloud? Decides whether quantization / ONNX
  export matters.
