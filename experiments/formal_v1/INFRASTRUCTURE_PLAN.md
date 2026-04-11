# Formal v1 — Infrastructure Plan for Full-Scale Training

**Audience:** AI researcher running and reporting a 15+ model ASR comparison study.
**Goal:** Build a unified framework that (a) trains any model reproducibly,
(b) captures every metric and artifact we'll want later, and (c) turns raw runs
into analysis-ready reports with zero manual glue.

Current state: we have a working codebase for all backbones, a clean Mamba
comparison (PyTorch eager vs CUDA mamba-ssm) and a `torch.compile` path that is
ready for the move to a bigger GPU. What's missing is the *research harness*
around the training loop — metric tracking, checkpoint management, plotting,
and the experiment registry that will make the thesis chapter writable in a
day instead of a week.

---

## 1. Move to a bigger GPU — what changes

**Context.** The 32 GB RTX 5090 cannot run PyTorch Mamba with `torch.compile`
because `torch.compile` is incompatible with gradient checkpointing, and
without checkpointing the 6-layer activation graph for a `batch_max_seconds=300`
batch exceeds 32 GB. On A100 40/80 GB or H100 we expect `torch.compile` to
match or beat the CUDA `mamba-ssm` kernels (microbenchmark shows 26 ms vs 35 ms
fwd+bwd).

**Deliverable now — `scripts/run_mamba_compiled.py`** (already created):
- Forces `torch.compile(encoder, mode=...)` with `default` / `reduce-overhead` /
  `max-autotune` selectable at the CLI.
- Uses a persistent `TORCHINDUCTOR_CACHE_DIR` under the run's output dir, so
  reruns reuse the compiled graph.
- Adds **bf16 autocast** by default. bf16 halves activation memory (making it
  useful even on A100 40 GB) and is numerically safer than fp16 for CTC losses.
  CTC loss is computed in fp32.
- Accepts `--batch-max-seconds` override for falling back on smaller GPUs.
- Gradient checkpointing is already auto-disabled under compile via
  `torch.compiler.is_compiling()` in `MambaEncoder`.

**Expected first-run compile cost:** ~100 s for `default`, ~5–10 min for
`max-autotune`. Cached afterwards.

**Why a separate script instead of just flags on `run_experiment.py`:**
`run_experiment.py` is the generic entry point; `run_mamba_compiled.py` is
a performance-hardened path that also enables AMP, custom CTC-fp32 handling,
and a persistent Inductor cache. Keeping them separate lets the generic
runner stay simple and readable for non-Mamba backbones.

---

## 2. `pyproject.toml` / dependencies — what to add

Current `pyproject.toml` has the minimum: torch, torchaudio, datasets, jiwer,
matplotlib, pyyaml, einops, soundfile.

**Additions needed for the research harness:**

```toml
dependencies = [
    # Existing
    "torch>=2.2.0",
    "torchaudio>=2.2.0",
    "datasets>=2.18.0",
    "jiwer>=3.0.0",
    "matplotlib>=3.8.0",
    "pyyaml>=6.0",
    "einops>=0.7.0",
    "soundfile>=0.12.0",

    # Research harness (no TensorBoard — everything is flat files in git)
    "numpy>=1.26",              # was implicit via torch, pin explicitly
    "pandas>=2.2",              # results DataFrame, CSV export for thesis tables
    "seaborn>=0.13",            # convergence + chunked-eval plots
    "tqdm>=4.66",               # replace ad-hoc logging every 50 batches
    "rich>=13.7",               # pretty run summaries on the console
    "tabulate>=0.9",            # markdown tables for RESULTS.md autogen
    "psutil>=5.9",              # RAM / CPU monitoring in metrics
]

[project.optional-dependencies]
cuda-mamba = ["mamba-ssm>=2.0.0"]  # keep CUDA kernels as an optional backend
```

**Rationale (AI researcher lens):**
- `pandas` + `tabulate` are the difference between copy-pasting 15 numbers into
  a markdown table by hand and running `python -m src.reporting.make_tables`.
- `seaborn` gives us paper-quality convergence plots with minimal code.
- `rich` for the summary table at the end of each run is low-cost polish that
  saves several seconds per glance.
- **No TensorBoard.** We store everything as flat files committed to git:
  `history.csv` for per-epoch data, `metrics.jsonl` for per-step data, and
  pre-rendered PNG plots under `plots/` in each run dir. Rationale: git is
  the single source of truth, plots render in the GitHub web UI, and a
  reviewer reading the repo never has to run a local process. For
  mid-training debugging we have `tail -f history.csv` and `rich`-rendered
  console summaries — enough to catch divergence without a web UI.
- Keep `mamba-ssm` as an **optional extra**, not a core dep: core install must
  succeed on any CUDA box, optional extras cover the fast backend.

**Lock file:** commit `uv.lock` (currently untracked). Without it we cannot
reproduce exact dependency versions three months from now when writing up.

---

## 3. Metrics — what to track, why, and how often

The draft-phase runs logged only `train_loss`, `dev_cer`, `dev_wer`, and
`epoch_time_sec`. That's enough for a rough comparison but far too little for
the analysis we need to write. Here's a layered scheme.

### 3.1 Per-step metrics (appended to `metrics.jsonl`, one JSON line per step logged)

Logged every N=50 steps. `.jsonl` is chosen because it's append-only
(crash-safe), grep-able, and `pandas.read_json(lines=True)` loads it in one
call. File size for an 80-epoch run at 50-step granularity is ~2 MB — fits
comfortably in git.

| Metric | Why |
|---|---|
| `step`, `epoch` | index |
| `train_loss` | catch divergence instantly, not once per epoch |
| `lr` | verify schedule is doing what we think |
| `grad_norm` (post-clip) | detect exploding grads, especially in Delta/LUCID |
| `grad_norm_raw` (pre-clip) | see how often clipping fires — if every step, the clip value is too tight |
| `tokens_per_sec` | throughput; needed to compare backbones fairly |
| `batch_duration_sec` | verify the DurationBatchSampler is packing efficiently |
| `gpu_mem_gb` | for the OOM / memory section of the thesis |

**Design note:** these are *not* saved in `history[]`. `history[]` has one
entry per epoch so that `history.csv` stays readable and thesis tables can
be built from it directly. Fine-grained data lives in `metrics.jsonl` and is
only loaded on demand by the plotting code.

### 3.2 Per-epoch metrics (saved in `history[]` + TensorBoard)

| Metric | Why |
|---|---|
| `epoch` | index |
| `train_loss` (mean over epoch) | the smoothed signal for the thesis curve |
| `train_loss_std` | quantify within-epoch noise |
| `dev_loss` | CTC loss on dev — correlates with CER but sometimes diverges late |
| `dev_cer`, `dev_wer` | primary metrics |
| `epoch_time_sec` | for throughput tables |
| `learning_rate_end` | schedule sanity check |
| `grad_norm_mean` | regime summary (did grads settle?) |
| `peak_mem_gb` | for the memory-vs-compute plot |

### 3.3 Final metrics (saved once, in `results.json`)

| Metric | Why |
|---|---|
| `test_cer`, `test_wer` | the headline numbers |
| `best_dev_cer`, `best_dev_epoch` | which checkpoint we're reporting |
| `total_train_time_sec` | wall clock for one full run |
| `params {total, encoder, frontend, ctc_head}` | parity sanity check |
| `chunked.{2s,5s,10s}.{reset,carry}.{cer,wer}` | streaming robustness |
| `git_sha`, `torch_version`, `cuda_version`, `gpu_name` | reproducibility |
| `config_snapshot` | the full resolved config at run time |
| `cli_args` | exactly how it was launched |

### 3.4 What we intentionally don't log
- Per-batch reference/hypothesis pairs — huge, low value, easy to regenerate
  from the checkpoint.
- Per-parameter statistics (mean/std per tensor) — `grad_norm` is sufficient;
  per-param adds megabytes of metrics and helps ~never.
- Attention visualizations — expensive, only generate on demand for the paper.

**Researcher rationale:** every metric above answers a specific question we
know we'll need to answer. Anything else we add "just in case" bloats the
artifact and signals that we don't know what we're measuring.

---

## 4. Checkpointing strategy

Current code saves only `best_model.pt` (by dev CER). That's enough to reproduce
the reported number, but insufficient for the kind of post-hoc analysis a
thesis needs.

### 4.1 What to save

| File | When | In git? | Why |
|---|---|---|---|
| `best_model.pt` | whenever `dev_cer` improves | **no** (gitignored, ~36 MB) | the number we report |
| `last_model.pt` | every epoch (overwrite) | no | resume-on-interrupt |
| `checkpoint_ep{N}.pt` | at epochs {1, 5, 10, 20, 40, final} | no | **snapshots for convergence analysis** — we can re-run chunked eval at intermediate epochs without re-training |
| `optimizer_last.pt` | every epoch (overwrite) | no | resume-on-interrupt |
| `results.json` | end of run | **yes** | analysis artifact (~10 KB) |
| `history.csv` | updated every epoch | **yes** | tail-able, pandas-readable, <5 KB |
| `metrics.jsonl` | appended every 50 steps | **yes** | step-granularity curves, ~2 MB per 80-epoch run |
| `plots/*.png` | rendered at end of each epoch, overwritten | **yes** | convergence curves, grad-norm, LR schedule — viewable directly on GitHub |
| `config.yaml` | start of run (snapshot the resolved config) | **yes** | reproducibility |
| `cli_args.txt` | start of run | **yes** | exactly how it was launched |
| `git_sha.txt` | start of run | **yes** | code version at run time |
| `train.log` | streaming | **yes** (gzipped on success) | debugging |

**Binary exclusion rule:** anything `.pt`, `.bin`, `.safetensors` is in
`.gitignore`. The trained weights are regenerable from `git_sha + config`
at the cost of compute. If we ever need weights versioned for a specific
release, we publish them as a GitHub Release asset, not in the repo itself.

### 4.2 Checkpoint file format

A checkpoint is a single `.pt` containing:
```python
{
    "epoch": int,
    "model": state_dict,
    "optimizer": state_dict,
    "scheduler_state": dict,
    "rng_state": {"python", "numpy", "torch", "cuda"},
    "best_cer": float,
    "patience_counter": int,
    "config": dict,  # snapshot of the full resolved config
    "git_sha": str,
}
```

This makes training **resumable after a crash or preemption** — not optional
for the 80-epoch runs. The draft experiments had to be restarted from scratch
on every CUDA driver hiccup; we're not doing that again.

### 4.3 Disk budget

With 15 planned backbones × 6 saved checkpoints × 36 MB ≈ 3.2 GB total.
Easily manageable. The `last_model.pt` for each run gets overwritten so no
additional cost.

---

## 5. Chunked evaluation & the carry-state story

This is the single most important section of the thesis's practical-relevance
claim, so it deserves a careful rework — not just "is it worth 20 minutes of
compute".

### 5.1 What each mode measures

- **Full-utterance eval.** The whole utterance goes through the encoder in
  one forward pass. "Offline ASR" — the standard benchmark setting.
- **Chunked, reset mode.** Split the utterance into 2 s / 5 s / 10 s chunks,
  run each chunk independently, concatenate the CTC outputs. **Streaming
  without memory.** Bidirectional models (LION, Mamba bidir, Transformer with
  full attention) are strictly disadvantaged — they lose their long-range view.
- **Chunked, carry-state mode.** Same chunks, but recurrent encoders propagate
  their internal state across chunks. **Streaming with memory.** The number
  that matters for a real-time system.

### 5.2 Is our current implementation correct? — two bugs to fix

I audited the code. Carry-state *runs* but has two correctness issues we
need to fix before reporting numbers.

**Bug 1 — Mamba does not carry the `conv_state`.**

`MambaBlock.forward()` applies a causal depthwise `Conv1d` with kernel=4
before the SSM. The conv's boundary is handled by PyTorch's implicit zero
padding at the start of each chunk. In the carry-state code path, we carry
`ssm_state` (the S6 hidden state) but we do *not* carry the last `d_conv - 1`
frames of the conv input. That means the first three frames of every new
chunk see a zero history for the conv instead of the real previous-chunk
frames, producing a small but systematic discontinuity at chunk boundaries.

*Fix:* extend the returned state to `{"ssm": ssm_state, "conv": last_conv_tail}`
where `last_conv_tail = x_inner[:, -(d_conv-1):]`. On the next call, prepend
this to the chunk's input before the conv, then slice off the prepended
frames from the conv output. Already prototyped in the existing `step()`
method, just needs to be lifted into the chunked forward path.

**Bug 2 — `MambaEncoder.init_state()` returns `[None] * n_layers`**,
not actual zero-valued state tuples. Works today because `MambaBlock.forward()`
treats `state=None` as "no carry", but it means the *first* chunk and all
subsequent chunks take different code paths. Harmless numerically; ugly
as an API. *Fix:* return a list of `{"ssm": zeros, "conv": zeros}` dicts
and let the forward pass handle them uniformly.

**RWKV-6** looks correct to me: `init_state` returns the right `(B, H, K, K)`
zero tensors, and `_forward_recurrent` accepts and returns them. Good.

### 5.3 Keep / drop decision, per backbone

| Backbone | Full | Reset | Carry | Reasoning |
|---|---|---|---|---|
| Transformer | keep | keep | **n/a** | Bidirectional attention, no notion of carry state in our setup. |
| RWKV-6 (recurrent) | keep | keep | **keep** | Designed to be streaming. Carry is the headline use case. |
| Mamba (PyTorch) | keep | keep | **keep (after fix)** | Same as RWKV-6. Fix conv_state carry first. |
| Mamba (CUDA) | keep | keep | **n/a** | CUDA wrapper doesn't expose carry. Report only full + reset. |
| LION (bidirectional) | keep | keep | **n/a** | Bidirectional by construction. Reset looks bad — that's the finding. |
| LION + mechanisms | keep | keep | **n/a** | Same as LION. |
| Mamba bidir | keep | keep | **n/a** | Bidirectional — same story as LION. |

### 5.4 The memory-footprint demonstration — is it worth it?

**Short answer: yes, and it's the strongest practical argument in the thesis.**
But we need to frame it carefully and not overclaim.

**The honest story (numbers below assume our config: d_model=256, H=4, d_k=64,
6 layers, bf16).**

| Backbone | State type | Size per sample | Scales with audio length? |
|---|---|---|---|
| RWKV-6 (recurrent) | per-layer WKV state `(H, K, K)` = `(4, 64, 64)` bf16 | ~32 KB / layer, **~192 KB total** | **no** — constant |
| Mamba (PyTorch) | per-layer `{ssm: (d_inner, d_state), conv: (d_inner, d_conv-1)}` = `(512, 16) + (512, 3)` bf16 | ~19 KB / layer, **~114 KB total** | **no** — constant |
| Bidirectional Transformer, chunked+overlap | re-computed per chunk, **no state carried** | activation-only, `O(T_chunk²)` for attention | chunk size, not total audio |
| Causal Transformer with KV cache *(hypothetical)* | per-layer `(T, H, d_k × 2)` | **24 KB × T frames**, e.g. 6 MB at 10 s, 36 MB at 60 s, 1.4 GB at 40 min | **yes — linearly** |

The dramatic "KV cache explodes" framing applies to *causal* Transformers
(decoder-only LLMs, autoregressive speech models). Our Transformer baseline
is *bidirectional*, so strictly speaking it carries no state — it just
re-runs attention on the next chunk. If we want the comparison to land hard
we need to either:

1. **Reframe as "streaming state vs audio duration"** and plot theoretical
   numbers for a causal-Transformer baseline alongside the actual measured
   Mamba / RWKV-6 state sizes. Clearly label the causal Transformer as
   *hypothetical* ("what you'd need if you made it streaming").
2. **Actually implement a causal-mask Transformer variant** and measure it.
   ~50 lines of code: swap `nn.MultiheadAttention` for a KV-cache-aware
   causal attention. We don't need to train it — just measure its state
   size and peak VRAM during streaming inference on a long-audio sample.
3. **Do both.** Theoretical curve + one measured point at, say, 30 seconds
   audio, to validate the theoretical model.

**My recommendation: option 3.** The causal Transformer variant is small,
reusable (we can also use it as an ablation: "does causal attention hurt
offline CER?"), and the measured point gives the theoretical curve its
credibility. The whole demonstration is maybe half a day of work and one
dedicated figure in the thesis. For the practical-relevance claim, it's
worth it.

### 5.5 What we will *measure* and *plot*

A new evaluation script `scripts/measure_streaming_memory.py` runs this
once per backbone:

1. Take a single 30-second LibriSpeech utterance (pad/crop).
2. For each chunk size in `[0.5, 1, 2, 5, 10, 30]` s:
   - Stream the utterance through the encoder with `carry_state=True`
     (or equivalent for bidirectional models: re-run on a sliding window).
   - At each chunk boundary, measure `sum(t.numel() * t.element_size() for t in state)`.
   - Measure peak GPU memory during the forward pass with `torch.cuda.max_memory_allocated()`.
3. Write one row per (backbone, chunk_size) to `outputs/_streaming_memory.csv`.
4. Produce `outputs/_plots/streaming_memory_vs_duration.png`: state size on
   the y-axis (log scale), audio duration on the x-axis, one line per
   backbone. Mamba/RWKV-6 flat. Causal Transformer linear. Bidirectional
   Transformer and LION: dashed flat lines annotated "no cross-chunk
   state (recomputation)".

This figure is the thesis's single strongest argument for recurrent
encoders in streaming ASR and it fits in one plot.

### 5.6 Optimizations for the regular chunked-eval loop (orthogonal)

Independent of the memory story, the current `evaluate_chunked` is slow
(20 min / run). Make it faster but don't drop it:

1. **Batch the chunked inference for reset mode.** Today it processes one
   utterance at a time. We can pad a mini-batch of chunks from different
   utterances and run them all in one forward pass — ~20× faster.
2. **Cap reset-mode eval at 500 stratified-by-length utterances.** Dev-clean
   has 2642 utterances, which is overkill for a stable CER; 500 keeps the
   standard error below the inter-model gap.
3. **Only run carry-mode on the shortlist.** Top-5 configurations get full
   chunked+carry. Everyone else gets full + reset only.

### 5.7 Deliverables

1. Fix Mamba conv_state carry + `init_state()`. Add a numerical test:
   concatenated chunks must match full-utterance output to within float
   tolerance for a randomly-initialized Mamba.
2. Rewrite `evaluate_chunked` with batched reset mode and `max_utterances`.
3. New `scripts/measure_streaming_memory.py` (see §5.5).
4. New `src/models/transformer_causal.py` for the hypothetical-comparison
   measurement point (~50 lines).
5. New plot module `src/reporting/plots/streaming_memory.py`.

---

## 6. The unified research framework — what to build

This is the core of the plan. Right now every run is a self-contained script
output directory. To make the thesis chapter writable we need three layers
on top:

```
Experiment registry  →  Run executor  →  Analysis / reporting
```

### 6.1 Layer 1 — Experiment registry (`configs/experiments.yaml`)

A single YAML file that lists every run we plan to do:

```yaml
experiments:
  - id: exp001_transformer
    backbone: transformer
    seed: 42
    epochs: 80
    tags: [baseline]
    shortlist: false

  - id: exp002_rwkv6
    backbone: rwkv6
    seed: 42
    epochs: 80
    tags: [baseline, recurrent]
    shortlist: true   # top-5 — runs with all evaluations

  - id: exp003_mamba_cuda
    backbone: mamba_cuda
    seed: 42
    epochs: 80
    tags: [baseline, recurrent]

  - id: exp004_lion
    backbone: lion
    seed: 42
    epochs: 80
    tags: [baseline, bidir]
    shortlist: true

  - id: exp005_lion_convshift
    backbone: lion_convshift
    seed: 42
    epochs: 80
    tags: [mechanism, bidir]

  # …
```

Each entry produces one run directory `outputs/{id}/`. Re-running the same id
resumes from `last_model.pt` if present.

**Why a registry and not just shell scripts:**
- Prevents typos: the backbone name is validated against the factory.
- The same file drives training, reporting, and the final thesis table —
  one source of truth.
- Easy to filter (`--tag mechanism`, `--shortlist`) without rewriting
  commands.

### 6.2 Layer 2 — Run executor (`scripts/run_registry.py`)

```
uv run scripts/run_registry.py --all
uv run scripts/run_registry.py --ids exp004_lion,exp005_lion_convshift
uv run scripts/run_registry.py --tag mechanism --gpu 1
uv run scripts/run_registry.py --shortlist --seeds 42,123,777
```

Behavior:
- Resolves each id to a fully-loaded `ExperimentConfig`.
- Writes `config.yaml`, `git_sha`, `cli_args` to the run dir **before** the
  first training step (so even a 30-second crash leaves a trail).
- Runs training. On failure, leaves `results.json` absent so the reporter
  can mark it as "failed" rather than silently missing.
- On success, writes `results.json` and appends one row to
  `outputs/_index.csv` (the master table).
- Emits a compact final summary with `rich`.

### 6.3 Layer 3 — Analysis / reporting (`src/reporting/`)

Four modules, each one a plain Python script invocable as `python -m`:

| Module | Produces | Committed to git? |
|---|---|---|
| `reporting.collect` | Scans `outputs/*/results.json`, builds a master `pandas.DataFrame` and writes `outputs/_index.csv` | yes, `_index.csv` is the single source of truth for tables |
| `reporting.tables` | Reads `_index.csv`, emits markdown tables into `RESULTS.md` between `<!-- AUTOGEN: ... -->` markers. Idempotent. | yes, `RESULTS.md` is regenerated in-place |
| `reporting.plots` | Produces `outputs/_plots/*.png`: convergence curves (all backbones overlaid), CER vs params scatter, chunked-CER vs chunk-size lines, training time vs backbone bar, streaming-memory vs duration | yes, PNGs committed |
| `reporting.diff` | `python -m src.reporting.diff exp004 exp005` — pairwise comparison with per-epoch CER delta and final test gap | no, ad-hoc tool |

**In-run plots** (written under each run's own `plots/` directory, not
`_plots/`): `loss_curve.png`, `cer_curve.png`, `lr_schedule.png`,
`grad_norm.png`. Regenerated at the end of every epoch, overwriting
previous versions. These are committed to git per run so a reviewer
browsing `outputs/exp004_lion/plots/cer_curve.png` on the GitHub web UI
sees the training dynamics immediately.

**Why separate from training:** reporting must be runnable without a GPU,
without the dataset, on just the output artifacts. If a reviewer asks a
question six months from now you should be able to answer it by running
`python -m src.reporting.tables` on your laptop. Since all plots are
pre-rendered PNGs committed to git, even the laptop-regeneration is
optional — the figures are already there.

### 6.4 Layer 4 — The `RESULTS.md` auto-update (optional but valuable)

`RESULTS.md` is currently hand-edited. Long-term: mark sections with HTML
comments like `<!-- AUTOGEN: core_baselines -->` and have
`reporting.tables` rewrite between the markers. The thesis narrative stays
hand-written, but the numbers never drift.

---

## 7. Concrete next steps (ordered, each ≤ 1 day)

**Phase A — Harness scaffolding (before the big GPU is ready)**

1. Update `pyproject.toml` with the new deps (no TensorBoard). Commit `uv.lock`.
2. Add a `MetricLogger` class in `src/training/metrics.py` that writes
   `metrics.jsonl` (per-step) + `history.csv` (per-epoch). No TensorBoard.
   Wire into `train.py`. Backwards-compatible with `run_experiment.py`.
3. Upgrade checkpointing: `last_model.pt`, `optimizer_last.pt`,
   epoch-snapshots `{1,5,10,20,40,final}`, `config.yaml`, `git_sha.txt`,
   `cli_args.txt`. Add `--resume` flag.
4. Add in-run plot generation at epoch end: `plots/loss_curve.png`,
   `plots/cer_curve.png`, `plots/lr_schedule.png`, `plots/grad_norm.png`.
   Committed to git under the run dir.
5. Update `.gitignore`: exclude `*.pt`, `*.bin`, `*.safetensors`. Include
   everything else under `outputs/`.

**Phase B — Carry-state correctness + memory demonstration**

6. Fix Mamba `conv_state` carry-through (see §5.2 Bug 1). Add numerical
   equivalence test: concatenated-chunks output matches full-utterance output.
7. Fix `MambaEncoder.init_state()` to return dicts of zero tensors (§5.2 Bug 2).
8. Add `src/models/transformer_causal.py` — a minimal causal-attention
   Transformer variant with KV-cache support. Not for training, for
   measurement only.
9. Write `scripts/measure_streaming_memory.py` (§5.5): produces
   `outputs/_streaming_memory.csv` and
   `outputs/_plots/streaming_memory_vs_duration.png`.
10. Rewrite `evaluate_chunked` with batched reset mode and `max_utterances`.
    Add equivalence test against the old single-utterance path.

**Phase C — Registry + reporting**

11. Write `configs/experiments.yaml` for the 15-run plan.
12. Write `scripts/run_registry.py` with `--all`, `--ids`, `--tag`,
    `--shortlist`, `--seeds` filters.
13. Write `src/reporting/collect.py` — scans `outputs/*/results.json` and
    produces `outputs/_index.csv`.
14. Write `src/reporting/tables.py` — regenerates markdown tables in
    `RESULTS.md` between `<!-- AUTOGEN: ... -->` markers from `_index.csv`.
15. Write `src/reporting/plots.py` — produces cross-run comparison PNGs
    under `outputs/_plots/`. All committed.
16. Smoke test: 2-epoch run of each backbone on RTX 5090 through the
    registry. Verify `_index.csv`, plots, tables regenerate cleanly.
    Resume-from-crash test: kill a run mid-training, resume, check continuity.

**Phase D — Full training on the big GPU**

17. Move to A100/H100. Verify `run_mamba_compiled.py` gives the target
    speedup. Compare 10-epoch CER against the CUDA kernels on the new GPU.
18. Run `scripts/run_registry.py --all --epochs 80`. Expected wall-clock:
    ~20 h on a single H100 for the baseline set, another ~30 h for mechanisms.
19. Run `scripts/measure_streaming_memory.py` for all backbones.
20. Run the shortlist with 3 seeds for statistical validation.
21. Generate final tables / plots, update `RESULTS.md`. Commit the whole
    `outputs/` tree (except `.pt` files).

**Success criteria**
- One command reproduces every number and plot in the thesis chapter.
- A crash at epoch 40 of 80 costs no more than 30 minutes of lost compute.
- Adding a new mechanism means: one new encoder file, one new entry in the
  registry, one `run_registry.py --ids exp016_newthing`, and the tables
  update themselves.

---

## 8. Out of scope for this plan

- Multi-GPU DDP training. 7 M-param models on 100 h of audio do not benefit
  from data parallelism — the per-step compute is too small to hide the
  gradient-sync latency. If we scale up to LibriSpeech-960, revisit.
- Hyperparameter sweeps. The canonical config is deliberately fixed to keep
  the comparison meaningful; sweeps belong in a follow-up.
- LibriSpeech-other (noisy subset). Can be added as another dataset id once
  the clean comparison is complete.
- Triton / custom CUDA kernels for the PyTorch Mamba scan. Not worth the
  engineering cost now that the `torch.compile` path is the plan for the big
  GPU and CUDA `mamba-ssm` is the fallback on the RTX 5090.
