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

    # Research harness
    "numpy>=1.26",              # was implicit via torch, pin explicitly
    "pandas>=2.2",              # results DataFrame, CSV export for thesis tables
    "seaborn>=0.13",            # convergence + chunked-eval plots
    "tqdm>=4.66",               # replace ad-hoc logging every 50 batches
    "rich>=13.7",               # pretty run summaries on the console
    "tabulate>=0.9",            # markdown tables for RESULTS.md autogen

    # Optional but highly recommended
    "tensorboard>=2.16",        # live training curves during long runs
    "psutil>=5.9",              # RAM / CPU monitoring in metrics
]

[project.optional-dependencies]
cuda-mamba = ["mamba-ssm>=2.0.0"]  # keep CUDA kernels as an optional backend
```

**Rationale (AI researcher lens):**
- `pandas` + `tabulate` are the difference between copy-pasting 15 numbers into
  a markdown table by hand and running `python -m src.reporting.make_tables`.
- `seaborn` gives us paper-quality convergence plots with minimal code.
- `tensorboard` is non-negotiable for 80-epoch runs — you need to watch LR,
  grad-norm, and loss curves in real time to catch divergence early. The raw
  log file is fine for after-the-fact analysis but terrible for live debugging.
- `rich` for the summary table at the end of each run is low-cost polish that
  saves several seconds per glance.
- Keep `mamba-ssm` as an **optional extra**, not a core dep: core install must
  succeed on any CUDA box, optional extras cover the fast backend.

**Lock file:** commit `uv.lock` (currently untracked). Without it we cannot
reproduce exact dependency versions three months from now when writing up.

---

## 3. Metrics — what to track, why, and how often

The draft-phase runs logged only `train_loss`, `dev_cer`, `dev_wer`, and
`epoch_time_sec`. That's enough for a rough comparison but far too little for
the analysis we need to write. Here's a layered scheme.

### 3.1 Per-step metrics (flushed every N=50 steps, stored in TensorBoard only)

| Metric | Why |
|---|---|
| `train/loss` | catch divergence instantly, not once per epoch |
| `train/lr` | verify schedule is doing what we think |
| `train/grad_norm` (post-clip) | detect exploding grads, especially in Delta/LUCID |
| `train/grad_norm_raw` (pre-clip) | see how often clipping fires — if it's on every step, the clip value is too tight |
| `train/tokens_per_sec` | throughput; needed to compare backbones fairly |
| `train/batch_duration_sec` | verify the DurationBatchSampler is packing efficiently |
| `sys/gpu_mem_gb` | for the OOM / memory section of the thesis |

**Design note:** these are not saved in `history[]`. They live in TensorBoard.
Keeping `history[]` to one entry per epoch keeps JSON files small and tables
readable.

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
- Full loss history at step granularity in JSON — TensorBoard is the right tool.
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

| File | When | Why |
|---|---|---|
| `best_model.pt` | whenever `dev_cer` improves | the number we report |
| `last_model.pt` | every epoch (overwrite) | resume-on-interrupt |
| `checkpoint_ep{N}.pt` | at epochs {1, 5, 10, 20, 40, final} | **snapshots for convergence analysis** — we can re-run chunked eval at intermediate epochs without re-training |
| `optimizer_last.pt` | every epoch (overwrite) | resume-on-interrupt |
| `results.json` | end of run | analysis artifact |
| `history.csv` | updated every epoch | tail-able, pandas-readable, doesn't require JSON parsing mid-run |
| `config.yaml` | start of run (snapshot the resolved config) | reproducibility |
| `train.log` | streaming | debugging |
| `tb/` | streaming | live curves |

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

## 5. Chunked evaluation: full, reset, and carry — keep or drop?

This is worth a careful look because chunked eval currently takes ~20 min per
run and dominates the post-training phase.

### 5.1 What each mode measures

- **Full-utterance eval.** The whole utterance goes through the encoder in
  one forward pass. This is the "offline ASR" number — equivalent to the usual
  benchmark setting.
- **Chunked, reset mode.** Split the utterance into fixed-length chunks
  (2 s / 5 s / 10 s), run each chunk independently, concatenate the CTC
  outputs. This measures **streaming-without-memory** — the encoder sees no
  context across chunks. All bidirectional models (LION, bidir Mamba,
  Transformer with full attention) are strictly disadvantaged here because
  they lose their long-range view.
- **Chunked, carry-state mode.** Same chunks, but recurrent encoders propagate
  their internal state across chunks. This measures **streaming-with-memory**
  and is the number that actually matters for a real-time system.

### 5.2 Keep / drop decision, per backbone

| Backbone | Full | Reset | Carry | Reasoning |
|---|---|---|---|---|
| Transformer | keep | keep | **drop** | No carry state — bidirectional attention, full context only. Reset shows graceful degradation. |
| RWKV-6 (recurrent) | keep | keep | **keep** | Designed to be streaming. Carry is the headline use case. |
| Mamba (PyTorch) | keep | keep | **keep** | Same reason as RWKV-6. Our PyTorch impl exposes carry; CUDA mamba-ssm wrapper doesn't. |
| Mamba (CUDA) | keep | keep | n/a | CUDA wrapper doesn't expose carry — report only full + reset. |
| LION (bidirectional) | keep | keep | n/a | Bidirectional, carry is not well-defined. Reset will look bad here, and that's a *finding*, not a bug. |
| LION + mechanisms | keep | keep | n/a | Same as LION. |
| Mamba bidir | keep | keep | n/a | Bidirectional — same story as LION. |

### 5.3 Is chunked eval worth the 20 min per run?

**Yes — but we should make it faster, not drop it.** The streaming numbers are
the reason a thesis on "lightweight offline ASR encoders" can still claim
practical relevance. Dropping them would leave a reviewer asking "is it
actually streamable?".

**Optimizations to implement:**

1. **Batch the chunked inference.** Today `evaluate_chunked` processes one
   utterance at a time. For reset mode we can pad a mini-batch of chunks from
   different utterances and run them all in one forward pass. Carry mode must
   stay sequential per utterance, but reset mode can be ~20× faster.
2. **Cap reset-mode eval at `max_eval_utterances`** (e.g. 500 utterances
   stratified by length). Dev-clean has 2642 utterances, which is overkill
   when you just want a stable CER number — 500 is enough to make the
   standard error smaller than the inter-model gap.
3. **Only run carry-mode on the shortlist.** During the mechanism-ablation
   phase we run many configurations; only the top-5 need full chunked+carry
   evaluation.

### 5.4 Deliverable

Rewrite `evaluate_chunked` to accept a `batch_size` and a `max_utterances`
argument, and change the runner to call:

- `evaluate_chunked(..., mode="reset", batch_size=16, max_utterances=500)`
  for every run.
- `evaluate_chunked(..., mode="carry", batch_size=1, max_utterances=500)`
  only if the backbone supports carry AND the run is marked `shortlist=True`
  in the registry (see §6).

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

| Module | Produces |
|---|---|
| `reporting.collect` | Scans `outputs/*/results.json`, builds a master `pandas.DataFrame` and writes `outputs/_index.csv` + `outputs/_index.parquet` |
| `reporting.tables` | Reads `_index.csv`, emits the exact markdown tables in `RESULTS.md` (Core Baselines, LION+Mechanisms, Chunked, Parameter Counts). Idempotent — re-running overwrites. |
| `reporting.plots` | Produces `outputs/_plots/`: convergence curves (all backbones overlaid), CER vs params scatter, chunked-CER vs chunk-size lines, training time vs backbone bar |
| `reporting.diff` | `python -m src.reporting.diff exp004 exp005` — pairwise comparison with per-epoch CER delta and final test gap |

**Why separate from training:** reporting must be runnable without a GPU,
without the dataset, on just the output artifacts. If a reviewer asks a
question six months from now you should be able to answer it by running
`python -m src.reporting.tables` on your laptop.

### 6.4 Layer 4 — The `RESULTS.md` auto-update (optional but valuable)

`RESULTS.md` is currently hand-edited. Long-term: mark sections with HTML
comments like `<!-- AUTOGEN: core_baselines -->` and have
`reporting.tables` rewrite between the markers. The thesis narrative stays
hand-written, but the numbers never drift.

---

## 7. Concrete next steps (ordered, each ≤ 1 day)

**Phase A — Harness scaffolding (before the big GPU is ready)**

1. Update `pyproject.toml` with the new deps and commit `uv.lock`.
2. Add a `MetricLogger` class in `src/training/metrics.py` (per-step +
   per-epoch + TensorBoard) and wire it into `train.py`. Backwards-compatible
   with `run_experiment.py`.
3. Upgrade checkpointing in `run_experiment.py`: save `last_model.pt`,
   `optimizer_last.pt`, epoch-snapshots, `history.csv`, `config.yaml`,
   `git_sha`. Add `--resume` flag.
4. Rewrite `evaluate_chunked` with batched reset mode and
   `max_utterances`. Add tests against the current single-utterance path to
   prove equivalence.
5. Write `configs/experiments.yaml` for the 15-run plan.
6. Write `scripts/run_registry.py`.
7. Write `src/reporting/collect.py` + `tables.py` + `plots.py`. Make the
   current Mamba comparison the first entry that flows end-to-end.

**Phase B — Dry runs on the RTX 5090**
8. Run 2 epochs of each backbone through the registry to validate the harness.
9. Verify `_index.csv`, plots, and tables regenerate cleanly.
10. Resume-from-crash test: kill a run mid-training, resume, check continuity.

**Phase C — Full training on the big GPU**
11. Move to the A100/H100. Verify `run_mamba_compiled.py` gives the target
    speedup. Compare 10-epoch CER against the CUDA kernels on the new GPU.
12. Run `scripts/run_registry.py --all --epochs 80`. Expected wall-clock:
    ~20 h on a single H100 for the baseline set, another ~30 h for
    mechanisms.
13. Run the shortlist with 3 seeds for statistical validation.
14. Generate final tables/plots, update `RESULTS.md`.

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
