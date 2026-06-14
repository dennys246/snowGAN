# CPU-RAM Leak Investigation (June 2026)

Training was OOM-killed in **CPU RAM** (not VRAM) after ~a day of running. This
documents the diagnosis, the fixes that landed, what remains open, and the
methodology — so a future debugging session starts with full context instead of
re-deriving it. Paired with [UPGRADES.md](UPGRADES.md).

## Symptom

- Process RSS climbs steadily over a long run until the Linux OOM-killer
  `SIGKILL`s it. GPU/VRAM is fine — it is **host RAM**.
- The kill often lands mid-checkpoint-write, which truncates
  `generator.weights.h5` (see **Recovery** below).

## Diagnostic methodology (what actually worked)

1. **RSS vs `tracemalloc` ⇒ native vs Python.** Log process RSS
   (`/proc/self/status` → `VmRSS`) alongside `tracemalloc.get_traced_memory()`.
   If RSS climbs while `tracemalloc` stays flat, the leak is **native** (C/C++:
   PIL/libpng/zlib, TF, cuDNN) — and `gc.collect()` / `tracemalloc` top-N will
   neither find nor fix it. This single distinction unblocked the whole hunt.
2. **The bench could not reproduce it.** Every component in isolation
   (`DataManager.batch`, `generate`, the EMA swap + per-step update, a real
   `train_step` loop, even the full loop body) was **flat** over dozens–hundreds
   of iterations. The leak only manifested in the long-running *production*
   process. Lesson: **instrument the real run**; don't trust short bench loops.
3. **Per-substep RSS deltas in the real loop** localized it: wrap each suspect
   (`train_step` / `plot_history` / preview `generate`, and *inside* `generate`:
   forward vs `.numpy()` vs PIL-save) with an RSS read before/after and print it.
4. **Confirm the install is editable.** `python -c "import snowgan;
   print(snowgan.__file__)"` must point into `src/`, not site-packages —
   otherwise source edits never take effect. A frozen `pip install .` snapshot
   silently ran stale code and wasted early effort; now installed with
   `pip install -e . --no-deps`.
5. A reusable probe lives on branch **`diag/memtrace`** (env-gated
   `SNOWGAN_MEMTRACE=N`): a `tracemalloc` snapshot-diff that names the growing
   `file:line`, plus RSS and per-substep deltas. Re-run it to re-localize.

## Confirmed leaks and fixes

| Source | Mechanism | Fix | PR |
|--------|-----------|-----|----|
| `generate()` returned `np.stack(...)` | ~125 MB CPU copy of the whole preview batch, discarded by every caller | drop the dead return | #21 ✅ |
| `plot_history()` made a new figure every 10 batches | the Agg backend leaks per-figure even with `plt.close()` (isolation: 61→1867 MiB over 300 calls; flat when reused) | reuse one persistent figure + `ax.cla()` | #22 ✅ |
| **preview PIL save (dominant)** | `Image.fromarray` + `save` retains ~15 MiB of **native** (libImaging/libpng/zlib) heap per preview — freed by Python but glibc keeps the arenas | `del` the per-image/array buffers + `malloc_trim(0)` after the save loop | #23 |

Net effect: the every-N-batch preview sawtooth is gone — the per-preview RSS
delta went from a steady **+9.4 MiB to net-negative** (now reclaiming).

## Still open

- **cuDNN-on-Blackwell native leak, ~+1 MiB/batch in `train_step`.** The GPU is
  an **RTX 5080 (sm_120 / "CC 12.0a")**, but the installed CUDA/`ptxas`
  (12.0.140) predates Blackwell, so TF JIT-compiles kernels from PTX and **cuDNN
  falls back to driver compilation every step** (`None of the algorithms
  provided by cuDNN frontend heuristics worked; trying fallback algorithms`),
  holding native workspace that `malloc_trim` cannot reclaim (it is *live*).
  `TF_CUDNN_USE_FRONTEND=0` / `TF_CUDNN_USE_AUTOTUNE=0` did **not** help. This is
  **environmental, not a snowGAN bug**; it predated the OOM problem and never
  killed jobs on its own, so it was deferred.
  - **Durable fix:** a CUDA 12.8+ / Blackwell-capable TF + cuDNN build so kernels
    stop falling back.
  - **Stopgap:** periodic checkpoint + restart (RSS resets); fewer
    `disc_steps`/`gen_steps`.
- **The leak still recurs as of this writing.** Next-session starting points:
  (1) re-run the `diag/memtrace` probe on the *current* code; (2) confirm whether
  the residual is the cuDNN +1/batch or a **new** source; (3) verify #23 is
  actually deployed (`snowgan.__file__` in `src/`, and the running branch carries
  the fix); (4) watch RSS-vs-`tracemalloc` to re-confirm native vs Python.

## Checkpoint corruption + recovery

`save_model` calls `model.save_weights(path)` **directly** (not atomically), so an
OOM-kill or any crash mid-write **truncates the live `*.weights.h5`** (observed:
`generator.weights.h5` shrank to 96 bytes; `h5py` then fails with
`truncated file: eof = 96 ... stored_eof = 2048`).

**Recovery:** restore the full, consistent set from the newest intact
`keras/snowgan/core/batch_<N>/` snapshot — generator **and** discriminator **and**
EMA **and** fade-endpoints **and** both configs together (so the adversarial pair
is from one training moment; do not mix a restored generator with a
different-vintage discriminator). Back up the corrupt top-level files first.
Training resumes from the highest `batch_*` directory.

**Recommended fix (🔴):** make `save_model` write to `tmp + os.replace()` per the
atomic-write rule in CLAUDE.md, so a crash can never corrupt the last-good
checkpoint again.
