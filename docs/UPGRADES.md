# snowGAN Upgrade Roadmap

Prioritized by blast radius. Tiers labeled 🔴 (bugs that silently corrupt training or break
features), 🟠 (production blockers — correctness, reproducibility, observability), 🟡 (code
health / velocity), 🟢 (nice-to-have). Paired with [ARCHITECTURE.md](ARCHITECTURE.md).

## Tier 🔴 — correctness bugs to fix before the next training run

1. **Double-applied gradient penalty.**
   [losses.py:55](../src/snowgan/losses.py#L55) returns `mean((‖∇‖−1)²) · λ`. Then
   [models/discriminator.py:63](../src/snowgan/models/discriminator.py#L63) adds
   `λ · gp` a second time. Effective penalty = `λ² · mean((‖∇‖−1)²)` (100× at λ=10).
   Fix: either return the raw `mean((‖∇‖−1)²)` from `compute_gradient_penalty` and multiply
   once in the loss, or stop multiplying in the loss. Add a unit test that asserts the
   penalty magnitude for a known input.

2. **`_ensure_depth_alignment` silently discards trained weights.**
   [trainer.py:287-305](../src/snowgan/trainer.py#L287-L305) rebuilds generator/discriminator
   via `type(self.gen)(self.gen.config)` when batch depth changes, with no weight migration.
   A run that sees a depth-1 batch after a depth-2 batch (or vice versa) restarts from random
   init. Fix: refuse to mix depths within a run (assert), or reinitialize only the layers
   whose shape depends on depth and port weights for the rest.

3. **`inference.run_inference` is broken end-to-end.**
   - Feeds rank-3 `(H,W,C)` tensors into a model whose input is rank-5 `(depth,H,W,C)` →
     shape error on first batch. Stack to `(1,H,W,C)` at minimum (or tile to `depth=2`).
   - Never trains the heads it builds (`test_on_batch` only) — "avalanches spotted accuracy"
     is random-init noise. Replace with `model.fit()` against a train/val split, with the
     backbone frozen (`layer.trainable = False` on all conv blocks).
   - `base.layers[-2].output` is brittle; name the Flatten layer (`name="features"`) and
     reference it by name so architecture tweaks don't silently move the tap.

4. **`configure_device` sets env vars after TF is already imported.**
   [main.py:1-8](../src/snowgan/main.py#L1-L8) triggers `import tensorflow` transitively
   before `configure_device` runs, so `CUDA_VISIBLE_DEVICES=-1` and `TF_XLA_FLAGS` are no-ops.
   Fix: create a `snowgan/bootstrap.py` that parses `--device`/`--xla` from `sys.argv`, sets
   env, and only then lets the rest of the package import TF. Invoke it first in the
   console-script entry.

5. **`--resolution` flag is dead.**
   [utils.py:97](../src/snowgan/utils.py#L97) declares `type=set` — argparse cannot build a
   set from a single string. The flag accepts input but silently discards it. Replace with
   `nargs=2, type=int` or `type=parse_resolution` where the parser splits `"1024x1024"`.

6. **Boolean CLI flags accept any truthy string.**
   `--fade`, `--xla`, `--mixed_precision`, `--rebuild`, `--gen_norm` all use `type=bool`.
   `--fade False` evaluates to `True`. Swap to `argparse.BooleanOptionalAction` (py ≥ 3.9)
   for `--fade / --no-fade` semantics.

7. **`latent_dim` is typed `float` but used as int.**
   [utils.py:101](../src/snowgan/utils.py#L101) → `type=float`. `config.latent_dim = int(...)`
   downstream works but `args.latent_dim` is a float. Risk: `tf.random.normal([B, 100.0])`
   is not accepted in strict TF builds. Type as `int`.

32. **Stale trainer-owned optimizers are never used.**
    [trainer.py:75-81](../src/snowgan/trainer.py#L75-L81) constructs
    `self.gen_optimizer` / `self.disc_optimizer` from the configs, but `train_step` at
    [trainer.py:233](../src/snowgan/trainer.py#L233) and
    [trainer.py:259](../src/snowgan/trainer.py#L259) applies gradients via
    `self.disc.optimizer` / `self.gen.optimizer` — the optimizers attached to the Keras
    models at build time. Any learning-rate / β changes Denny thinks he's injecting through
    the trainer are silently ignored. Root cause: two optimizer instances per model, one of
    them dead. Fix: delete the trainer-level optimizer fields entirely and delegate to the
    model's own optimizer (single source of truth).

33. **Logged loss is last-step-only, not averaged over `gen_steps` / `disc_steps`.**
    [trainer.py:262-263](../src/snowgan/trainer.py#L262-L263) appends `gen_loss` and
    `disc_loss` *after* the inner update loops, capturing only the final iteration. With
    `gen_steps=3`, two of every three generator updates are invisible in the logged
    history; the plotted curve misrepresents training. Root cause: loss recorded outside
    the loop, overwriting on each iteration. Fix: accumulate inside the loops and append
    the mean; additionally guard against `training_steps=0` so `gen_loss` / `disc_loss`
    are always defined.

34. **Config JSON is rewritten non-atomically every training step.**
    [config.py:139](../src/snowgan/config.py#L139) opens the config file with `open(..., 'w')`
    + `json.dump` — no temp-file + `os.replace()`. `Trainer._sync_fade_progress(persist=True)`
    at [trainer.py:268](../src/snowgan/trainer.py#L268) calls `save_config()` every single
    `train_step`, so a crash or SIGKILL mid-write leaves the config truncated. On resume,
    JSON parse fails and the whole run state is lost. Root cause: non-atomic writes in a
    high-frequency path. Fix: atomic writes (`tmp + os.replace()`), and demote the sync
    frequency from per-step to per-milestone. Pair with #8.

35. **`reset_seen_profiles` persists the cleared set before the epoch makes progress.**
    [trainer.py:132](../src/snowgan/trainer.py#L132) clears `seen_profiles` at epoch top;
    the very next `_sync_fade_progress(persist=True)` writes the empty set to disk. A
    SIGKILL anywhere later in the epoch means resume loses that epoch's profile bookkeeping
    and will re-pair profiles it already used. Root cause: mutation is eagerly persisted
    outside a transaction boundary. Fix: treat in-epoch `seen_profiles` as append-only;
    persist the reset only once the next epoch has committed at least one pair (or on
    graceful epoch-end only).

36. **Batch counter is recovered by globbing files that `_cleanup_saved_batches` deletes.**
    [trainer.py:137-141](../src/snowgan/trainer.py#L137-L141) derives the next `batch`
    number by scanning `synthetic_images/batch_*.png` for the max. `_cleanup_saved_batches`
    (called on milestone saves) trims those very files. After a milestone, resume reads a
    stale max → restarts at a lower batch number → overwrites existing
    `batch_N/generator.keras` snapshots. Silent data loss of trained checkpoints. Root
    cause: counter lives in an external filesystem artifact that another subsystem is
    authorized to delete. Fix: persist `global_batch` in config alongside `fade_step`,
    drive the loop from that, delete the glob path.

## Tier 🟠 — production readiness (do before calling this a product)

8. **Replace `atexit` with explicit, atomic, signal-safe checkpointing.**
   `atexit` does not fire on SIGKILL, OOM, or kernel death. Move saves to a
   `ModelCheckpoint`-style callback on a step interval. Write via `tmp + os.replace()` so a
   half-written file never replaces a good checkpoint. Persist config JSON the same way.

9. **Split the god-config.**
   [config.py](../src/snowgan/config.py) is a 350-line dict + mutable class with `atexit`
   save that mirrors itself across generator and discriminator. Replace with a
   `pydantic.BaseModel` (or `@dataclass(frozen=False)`) hierarchy:
   `RuntimeConfig`, `DataConfig`, `ModelConfig(generator=..., discriminator=...)`,
   `TrainingConfig(fade=..., checkpoint=...)`. Validate types, ranges, and invariants
   (`filter_counts` monotonicity, `resolution` compatible with `convolution_depth`).

10. **Introduce `tf.data` pipeline with prefetch, parallel map, and caching.**
    Current pipeline is synchronous, one-sample-at-a-time, recomputes resize every epoch.
    Goal: `tf.data.Dataset.from_generator(manifest_iter, ...).map(decode, num_parallel_calls=AUTOTUNE).cache().shuffle().batch().prefetch(AUTOTUNE)`.
    Precompute core→profile pairing index once at startup
    (`dict[(site, column, core), list[profile_idx]]`) to kill the linear rescan in
    `batch_merged`.

11. **Honor `trained_pool / validation_pool / test_pool`.**
    The config has slots; nothing populates them. Split the pair index 80/10/10 with a fixed
    seed, persist the split, and gate training / validation metrics accordingly. This is a
    precondition for any transfer-learning evaluation.

12. **Seed everything, log the seeds.**
    `random.seed`, `np.random.seed`, `tf.random.set_seed`, `os.environ["PYTHONHASHSEED"]`
    and `os.environ["TF_DETERMINISTIC_OPS"]="1"`. Persist the seed in the config snapshot.
    Without this, training runs are not replayable even across same-host restarts.

13. **Structured logging + TensorBoard / W&B.**
    Replace `print(...)` with the stdlib `logging` module (JSON handler for production, human
    handler for dev). Emit scalar metrics (disc_loss, gen_loss, grad_norm, fade_alpha,
    steps/sec, samples/sec) to TensorBoard. Wire an optional W&B or MLflow sink for
    cross-run comparison.

14. **GAN quality metrics, not just loss.**
    Add FID (Fréchet Inception Distance), KID, and per-modality equivalents (e.g., FID on
    core slice, FID on profile slice separately). Compute on a held-out validation pool
    every M batches. This is the only way to know the model is improving — WGAN-GP loss is
    not monotonic.

15. **Mixed precision: audit and fix.** Two independent breakages under the same flag:
    - **(a) Output dtype.** `--mixed_precision True` enables `mixed_float16` globally, but
      the generator's final `tanh` Conv3DTranspose computes in float16 and can
      saturate/overflow. Force the output head to float32: `dtype='float32'` on the
      `toRGB_curr` layer. Same for the critic's final `Dense(1)`.
    - **(b) No loss scaling.** [utils.py:78-79](../src/snowgan/utils.py#L78-L79) sets
      `set_global_policy("mixed_float16")` but never wraps the Adam optimizers in
      `tf.keras.mixed_precision.LossScaleOptimizer`. fp16 gradients silently underflow to
      zero below ~1e-7, especially early in fade-in at small α. Wrap both optimizers at
      construction when the policy is fp16.
    Both (a) and (b) must ship together; fixing only one leaves mixed precision broken.

16. **Loss history storage.**
    Append-only JSON Lines (`loss.jsonl`: `{"step", "gen_loss", "disc_loss", "epoch", "batch"}`)
    instead of rewriting two `.txt` files on every save. Atomic append; easy to resume; easy
    to plot externally.

17. **Checkpoint format: weights-only + sidecar.**
    `keras.Model.save(*.keras)` serializes architecture + weights + custom layers. With the
    `Lambda` in the generator's fade path, reload has already been known to break across
    Keras versions. Prefer `model.save_weights(*.weights.h5)` plus a sidecar JSON describing
    the architecture (the config you already have). Construct-then-load on reload; fail loud
    if shapes don't match.

18. **Hugging Face Hub integration.**
    You already host the dataset there; mirror the code. Add an `hf push-model` command that
    uploads `generator.keras`, `discriminator.keras`, the generator/discriminator config
    JSONs, and a model card (training curves, sample grids, dataset commit, seed, git SHA).
    That makes the pretrained backbone discoverable for AvAI.

19. **Dockerfile + CUDA pin.**
    CUDA/cuDNN version drift will be the #1 cause of failed re-runs. Ship a `Dockerfile`
    (or `uv` lockfile + devcontainer) that pins TF, CUDA runtime, and driver expectations.
    Today, onboarding a new host is a bespoke README incantation.

37. **Gradient-penalty α broadcasts uniformly across depth.**
    [losses.py:46-47](../src/snowgan/losses.py#L46-L47) reshapes `alpha` to `(B,1,1,1,1)`,
    so the same interpolation weight is applied to both the core slice and the profile
    slice of every paired sample. WGAN-GP theory assumes independent interpolation between
    real and fake; coupled α couples the two modality gradients and can bias the critic
    toward correlated features across depth. Root cause: rank-4 GP formula was lifted to
    rank-5 without rethinking the modality semantics. Fix: sample
    `alpha = tf.random.uniform([B, depth, 1, 1, 1])` so each modality gets its own
    interpolation. Alternative: document explicitly that coupled α is intentional and
    justify it; otherwise default to independent.

38. **`generate()` runs every batch during training.**
    [trainer.py:173-174](../src/snowgan/trainer.py#L173-L174) calls `generate(...)`
    unconditionally in the inner loop, writing `n_samples * depth` PNGs per `train_step`
    plus a generator forward pass. This reads as debug tracing left in the hot loop —
    hundreds of filesystem writes per second at the configured `n_samples=10`, `depth=2`.
    Root cause: no cadence control; the sample step shares the training step. Fix: add a
    `--sample_interval` (default = `cleanup_milestone`) and gate sample generation behind
    it. Also move the filesystem write off the training thread (async queue / dedicated
    reporter).

39. **No lock on `save_dir` → concurrent runs silently corrupt each other.**
    Two `snowgan --mode train --save_dir ./models` processes will race on
    `config.save_config()`, on `keras.Model.save()` (not atomic — writes several temp files
    internally), and on `synthetic_images/` counter recovery. No lockfile, no PID file.
    Root cause: `save_dir` is shared mutable state with no concurrency discipline. Fix:
    acquire `filelock.FileLock("{save_dir}/.snowgan.lock")` in `Trainer.__init__`;
    release on graceful shutdown; include the current PID and host in the lock payload
    so a stuck lock can be diagnosed.

40. **Mixed-depth datasets trigger a model rebuild every batch.**
    `_ensure_depth_alignment` (already cataloged under #2 for weight loss) has a second
    failure mode: if the incoming manifest contains both depth-1 and depth-2 samples, the
    trainer rebuilds models on every batch that flips depth. Each rebuild leaks TF graph
    state and loses training momentum even if weights were preserved. Root cause: no
    validation that the dataset has a consistent depth. Fix: `DataManager.__init__`
    precomputes the depth of every pair and asserts homogeneity; refuse mixed-depth
    datasets with a clear error. This is the companion fix to #2 — #2 addresses
    *what* happens on rebuild, #40 prevents rebuilds from happening in steady-state.

## Tier 🟡 — code health & velocity

20. **Tests.**
    - `tests/unit/test_losses.py`: `compute_gradient_penalty` on a frozen conv, assert the
      double-λ fix.
    - `tests/unit/test_config.py`: round-trip JSON, CLI overrides, type validation.
    - `tests/unit/test_dataset.py`: pair index built from a tiny synthetic manifest; asserts
      pairing correctness and no profile reuse within an epoch.
    - `tests/integration/test_train_step.py`: 1 step at 64×64 with `depth=2` completes and
      updates weights (compare `sum(weights)` before/after).
    - GitHub Actions CI: `pytest`, `ruff check`, `mypy src/snowgan`.

21. **Type hints + `mypy --strict` on `src/snowgan/`.**
    Most functions are untyped. `config.build.configure` takes 30+ positional args — a
    `TypedDict` or dataclass erases a class of bugs.

22. **Extract `Trainer` into focused collaborators.**
    Current trainer owns loading, saving, fade scheduling, sample generation, cleanup, plot.
    Aim for:
    - `Trainer` — loop + train_step.
    - `Checkpointer` — save/load, atomic writes, rotation.
    - `FadeScheduler` — alpha, step counter, completion.
    - `SampleReporter` — synthetic grid generation + TensorBoard image writer.
    - `ArtifactCleaner` — disk hygiene.
    Each ~100 lines, each unit-testable.

23. **Fix `load_discriminator` / `load_generator` path mangling.**
    [models/discriminator.py:65-71](../src/snowgan/models/discriminator.py#L65-L71) does
    `split.pop()` and then `"/".join(split) + "/"` — order-dependent, mutates the list
    mid-use, drops the filename. Use `pathlib.Path` and `parent / name`.

24. ~~**Stop committing `__pycache__` and `*.egg-info`.**~~ **Resolved 2026-04-18:**
    `src/**/__pycache__/` and `src/**/*.egg-info/` patterns added to `.gitignore`. No
    tracked files needed removal (check ran clean at resolution time).

25. **Document the modality-blending contract.**
    The current shape convention `(B, depth, H, W, C)` with `depth=2` = `[profile, core]`
    is encoded only by a line in `generate.py` (`view_names = ["profile", "core"]`). Make
    it an explicit named tuple / enum (`Modality.PROFILE = 0`, `Modality.CORE = 1`) and use
    it in dataset, generator, reporter.

26. **Replace hand-rolled batch-counter recovery with a persistent step counter.**
    [trainer.py:137-141](../src/snowgan/trainer.py#L137-L141) globs `synthetic_images/batch_*.png`
    to resume the `batch` number. Persist it in config (`global_batch`), just like
    `fade_step`. See #36 for the silent-data-loss angle when `_cleanup_saved_batches`
    trims the files this glob depends on — both fold into the same fix.

27. ~~**Fix the README CLI example.**~~ **Resolved 2026-04-18:** space inserted between
    `keras/` and `--gen_steps` in the README. A `Documentation` section was also added
    to the README pointing at `CLAUDE.md` and all files under `docs/`.

41. **LeakyReLU `negative_slope=0.25` is non-standard.**
    [config.py:85](../src/snowgan/config.py#L85) (and the layer constructors in
    [generator.py:46](../src/snowgan/models/generator.py#L46) /
    [discriminator.py:39](../src/snowgan/models/discriminator.py#L39)) default to 0.25; the
    near-universal GAN default is 0.2. Larger leak softens the critic's feature gating and
    may weaken the Lipschitz signal. Root cause: default value chosen without documented
    justification. Fix: either (a) revert to 0.2 and treat 0.25 as a tuning knob per-run,
    or (b) document the empirical reason 0.25 won and pin it.

42. **Fade path blends RGB, not feature-map resolution.**
    [generator.py:49-74](../src/snowgan/models/generator.py#L49-L74) implements "fade" as a
    1×1×1 Conv3D on the second-to-last feature map, resized to match the current RGB
    output, then linearly blended with the final `toRGB_curr` output. That is *not* the
    ProgGAN / StyleGAN pattern (grow spatial resolution, then blend old + new block
    outputs) — it's a color-only blend at a fixed spatial resolution. Training doesn't get
    the smoothing benefit of true progressive growth. Root cause: naming vs. implementation
    drift. Fix: either rename to avoid the "progressive fade" connotation (e.g.,
    `output_blend`) and document the actual semantics, or refactor to genuine progressive
    growth where spatial resolution ramps up. Do not ship under the current name.

43. **Matplotlib plot uses implicit global state each batch.**
    [trainer.py:348-360](../src/snowgan/trainer.py#L348-L360) calls `plt.plot` / `plt.title`
    / `plt.savefig` / `plt.close()` without ever creating an explicit `Figure`. `plt.close()`
    with no argument closes the *current* figure, but if the implicit figure was never
    reset, state (legends, titles, axes) accumulates across saves and can leak. Root cause:
    pyplot state-machine used as if it were stateless. Fix: `fig, ax = plt.subplots();
    ax.plot(...); fig.savefig(...); plt.close(fig)` — explicit figure lifecycle per save.

44. **Manifest duplicated in memory alongside the HF dataset.**
    [data/dataset.py:17-20](../src/snowgan/data/dataset.py#L17-L20) materializes the HF
    split to pandas, drops `image`/`audio`, then stores the remaining rows as a list. The
    full HF `DatasetDict` also stays live via `self.dataset`. For a few-thousand-row
    snowpack split it's fine; for a production-scale dataset it's two copies of the
    metadata. Root cause: eager materialization used as a lookup layer instead of an
    index. Fix: at init, precompute the pair index
    `dict[(site, column, core), list[profile_idx]]`, then drop `self.manifest`
    entirely — iterate via the index and `self.dataset[i]`.

## Tier 🟢 — nice-to-have

28. **Submodule cleanup.** `.gitmodules` references `external/snowMaker` but the directory is
    missing. Either restore or drop.

29. **Remove dead imports / dead code.**
    `generate.make_movie` imports `cv2`, which is only needed for that path. `click` is in
    deps but unused. `emd_loss` in `losses.py` duplicates `Discriminator.get_loss`.

30. **Pre-commit hooks.** `ruff`, `ruff-format`, `end-of-file-fixer`, `check-added-large-files`
    (helpful given `.keras` weights lurk).

31. **Model card + dataset card cross-links.** Describe licensing, intended-use, limitations,
    evaluation results, bias considerations.

---

## Cross-lens observation — unify persistence as a single subsystem

A striking pattern across items #8, #16, #17, #26, #34, #35, #36, #39, and part of #40: all
of them are persistence bugs with the same shape — **some piece of state is written in the
middle of an operation, without atomicity, and without a transaction boundary that defines
when the state is "committed."** Each finding is individually fixable, but every
independent fix risks drifting out of sync with the others (a new atomic write that doesn't
know about the lockfile; a lockfile that doesn't cover the JSONL loss log; a persisted
`global_batch` that updates before or after `fade_step` depending on the code path).

Proposed root-cause fix: a single `Checkpointer` module owns all persistence:

- Atomic writes (tmp + `os.replace()`) for configs, weights, histories, and sample indexes.
- A file-lock (`{save_dir}/.snowgan.lock`) acquired at Trainer init.
- A single "commit point" at the end of each `train_step` (or every N steps for
  throughput). Before the commit, state changes are staged in memory; on commit, all
  persisted artifacts — config JSON, loss JSONL, `global_batch`, `seen_profiles`,
  `fade_step` — land together. A crash between commits rewinds cleanly to the previous
  committed state.
- Resume reads the last committed snapshot only; no globbing the filesystem.

Shipping this retires #8, #16, #17, #26, #34, #35, #36, #39 in one coherent change and
eliminates a whole class of future bugs. Until it exists, the individual fixes are
bandaids (in the sense of CLAUDE.md §2 — the symptom goes away, the design defect persists).

## Suggested sequencing

Week 1 (unblock training quality):
- 🔴 #1 double-λ, #2 depth rebuild guard, #4 device env ordering, #32 stale optimizers,
  #33 last-step loss, #40 single-depth assertion.
- 🟠 #12 seeding, #15 mixed precision (both parts), #37 per-modality α.

Week 2 (unblock transfer learning):
- 🔴 #3 inference rewrite (frozen backbone + `fit`), #35 seen-profile persist order.
- 🟠 #10 tf.data, #11 train/val/test split.

Week 3 (persistence subsystem — the cross-lens fix):
- 🔴 #34 atomic config writes, #36 persistent batch counter.
- 🟠 #8 atexit → Checkpointer, #17 weights-only, #39 save-dir lockfile.
- 🟠 #38 sample cadence (hot-loop cleanup while inside Trainer).
- 🟡 #20 tests + CI (regression tests for every 🔴 shipped so far).

Week 4 (production shell):
- 🟠 #9 pydantic config, #13 structured logging, #14 FID, #18 HF Hub push, #19 Docker pin.

Week 5 (hardening):
- 🟡 #21 mypy, #22 Trainer split (will be easier after Checkpointer exists),
  #42 rename-or-rebuild fade, #43 matplotlib lifecycle, #44 drop manifest duplication.
