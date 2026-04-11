# Formal v1 — Infrastructure Plan for Full-Scale Training

**Purpose.** Describe the research harness, experiment design, and execution
plan for the ASR encoder comparison study. Written so that any contributor
(or a future self) can pick up from the last completed step, understand the
rationale behind every decision, and continue without reconstructing lost
context.

**Audience.** AI researcher running the 16-run comparison and producing the
thesis chapter.

**Status.** The code for all backbones exists and trains correctly. The
Mamba PyTorch vs CUDA comparison (10 epochs, seed=42) is complete and
accuracy-equivalent. What is missing is (a) the research harness that turns
raw runs into analysis-ready artifacts, (b) the causal-Transformer baseline
that closes the causal comparison group, and (c) two correctness fixes in
Mamba carry-state handling. See §7 for the execution checklist.

---

## 1. Framing: two comparison groups

Prior drafts mixed causal and bidirectional encoders in one results table.
This is unfair and makes the thesis narrative muddy. The formal comparison
is split into two coherent groups, each with its own baseline, its own
evaluation protocol, and its own headline claim.

### Group A — Causal / streaming-capable

All encoders are trained with causal information flow. Each has a
well-defined recurrent state that can be propagated across chunk boundaries,
so the streaming eval is meaningful.

| Backbone | Role | Training state |
|---|---|---|
| `transformer_causal` | causal baseline | **to add** (~50 lines on top of existing Transformer) |
| `rwkv6` | recurrent RWKV-6 | already trained |
| `mamba` (PyTorch) | open-box Mamba | trained 10 ep; needs conv_state fix |
| `mamba_cuda` | fast Mamba | trained 10 ep; no carry-state (CUDA wrapper limitation) |
| `rwkv6_lucid`, `rwkv6_delta`, `rwkv6_lucid_delta` | mechanism ablations | to train |

**Claim for Group A:** RWKV-6 and Mamba match a causal Transformer on
accuracy while using constant (rather than linearly-growing) state memory.

### Group B — Bidirectional / offline-only

All encoders use past + future context. None has a meaningful carry state;
streaming is only tested via chunked reset mode.

| Backbone | Role |
|---|---|
| `transformer` | bidirectional baseline |
| `lion` | bidirectional RWKV-6 (central thesis candidate) |
| `mamba_bidir` | bidirectional Mamba (forward + backward SSM) |
| `lion_convshift`, `lion_lucid`, `lion_lucid_chunked`, `lion_delta`, `lion_headscale`, `lion_convshift_headscale` | LION + mechanism ablations |

**Claim for Group B:** LION with corrected LUCID and causal-only Delta Rule
is the best lightweight offline encoder for LibriSpeech clean.

### Why two groups and not one

1. **Fair evaluation.** A causal model cannot attend to future context; a
   bidirectional model cannot stream. Comparing them head-to-head on one
   axis hides the real tradeoff.
2. **Two independent narratives.** Each group supports a separable thesis
   claim. A reader can read one group without the other.
3. **Clean memory-footprint plot.** The KV-cache-vs-constant-state
   demonstration lives entirely in Group A, where the causal Transformer
   is an honest data point and not a hypothetical one.
4. **Free ablation.** Training both `transformer` and `transformer_causal`
   gives the delta "how much does removing future context hurt a
   Transformer on ASR?" without extra design work.

Carry-state for bidirectional models is out of scope: there is no
well-defined quantity to propagate forward in time for a model that was
trained assuming it sees its future. Our code already marks these
`supports_carry_state = False`.

---

## 2. Training infrastructure

### 2.1 Canonical runner and the big-GPU path

`scripts/run_experiment.py` is the generic entry point. It accepts
`--backbone`, `--seed`, `--epochs`, `--gpu`, `--compile`, and `--resume`.
The `--compile` flag wraps the encoder with `torch.compile(encoder)`; on
the current RTX 5090 this OOMs because `torch.compile` is incompatible with
gradient checkpointing, so it is only used once we move to a larger GPU.

`scripts/run_mamba_compiled.py` is a hardened Mamba-only path with bf16
autocast, a persistent Inductor cache under the run directory, and CTC
loss computed in fp32. Kept separate so the generic runner stays readable.

**Expected speed at full scale (H100, 80 epochs, `batch_max_seconds=300`):**

| Backbone | Eager + checkpoint | `torch.compile` + bf16 (target) |
|---|---:|---:|
| Mamba PyTorch | ~420 s/epoch | ~60 s/epoch |
| Mamba CUDA | ~60 s/epoch | n/a |
| RWKV-6 / LION / Transformer | ~60–120 s/epoch | — |

### 2.2 Checkpointing and resumability

Long runs crash. Every run writes:

| File | Cadence | In git |
|---|---|---|
| `best_model.pt` | whenever dev CER improves | no (binary) |
| `last_model.pt` | every epoch | no |
| `checkpoint_ep{N}.pt` at {1, 5, 10, 20, 40, final} | sparse snapshots | no |
| `optimizer_last.pt` | every epoch | no |
| `history.csv` | every epoch | yes |
| `metrics.jsonl` | every 50 steps | yes |
| `plots/*.png` | end of each epoch, overwritten | yes |
| `results.json` | end of run | yes |
| `config.yaml`, `cli_args.txt`, `git_sha.txt` | start of run | yes |
| `train.log.gz` | end of run | yes |

Each checkpoint contains `{epoch, model, optimizer, scheduler, rng_state,
best_cer, patience_counter, config, git_sha}` so `--resume` restarts from
the exact pre-crash state. Required for 80-epoch runs.

**Binary exclusion:** `*.pt`, `*.bin`, `*.safetensors` are in `.gitignore`.
Weights are regenerable from `git_sha + config`. Anything we want versioned
permanently goes as a GitHub Release asset, not in the repo.

### 2.3 Metrics and artifacts — no TensorBoard

All artifacts are plain files committed to git. No TensorBoard process,
no web UI. Rationale: git is the single source of truth, PNG plots render
in the GitHub web UI, and a reviewer reading the repo never has to run a
local service. For live debugging we have `tail -f history.csv` and
`rich`-rendered console summaries at epoch boundaries.

**Three metric tiers:**

*Per-step* (flushed every 50 steps to `metrics.jsonl`):
`step, epoch, train_loss, lr, grad_norm, grad_norm_raw, tokens_per_sec,
batch_duration_sec, gpu_mem_gb`. Used for curves; loaded on demand by
plotting code.

*Per-epoch* (one row in `history.csv` and one entry in `results.json.history`):
`epoch, train_loss, train_loss_std, dev_loss, dev_cer, dev_wer,
epoch_time_sec, learning_rate_end, grad_norm_mean, peak_mem_gb`. The
thesis curves are built from this.

*Final* (top level of `results.json`): `test_cer, test_wer, best_dev_cer,
best_dev_epoch, total_train_time_sec, params {total, encoder, frontend,
ctc_head}, chunked.{2s,5s,10s}.{reset,carry}.{cer,wer}, git_sha,
torch_version, cuda_version, gpu_name, config_snapshot, cli_args`.

**Not logged:** per-batch reference/hypothesis pairs, per-parameter
statistics, attention visualizations. These are regenerable from a
checkpoint on demand.

---

## 3. Evaluation protocol

### 3.1 What gets evaluated, per group

| Mode | Group A (causal) | Group B (bidirectional) |
|---|---|---|
| Full utterance | ✓ headline | ✓ headline |
| Chunked reset (2 / 5 / 10 s) | ✓ (robustness) | ✓ (context-window sensitivity) |
| Chunked carry-state (2 / 5 / 10 s) | ✓ (streaming; skipped for `mamba_cuda`) | — not meaningful |
| Streaming memory vs duration | ✓ (Group A only) | — |

Carry-state is reported only for backbones whose `supports_carry_state`
flag is True **and** which are in the run registry's shortlist (§4).

### 3.2 Carry-state correctness — two fixes required before reporting

Audit of the current code identified two issues to fix before any
carry-state numbers are trusted:

1. **Mamba does not carry the depthwise conv state.** `MambaBlock` applies
   a causal `Conv1d(kernel=d_conv=4)` before the SSM. The existing carry
   path propagates `ssm_state` but not the last `d_conv - 1` frames of the
   conv input, so each new chunk starts with zero-padded conv history and
   produces a discontinuity at every chunk boundary. Fix: extend the
   returned state to `{"ssm": ssm_state, "conv": x_inner[:, -(d_conv-1):]}`
   and prepend the previous tail on the next call. The existing `step()`
   method already handles this correctly for single-step inference and
   can be lifted.

2. **`MambaEncoder.init_state()` returns `[None] * n_layers`** instead of
   actual zero-state dicts. Harmless today (the block treats `None` as
   "no carry") but means chunk 0 and chunks 1+ take different code paths.
   Fix: return a list of zero-state dicts and make the forward pass
   uniform.

**Equivalence test** (required before merging the fix): for a randomly
initialized Mamba, output of full-utterance forward pass must match
concatenated chunked forward pass to within float tolerance.

RWKV-6 recurrent carry-state path was audited and is correct.

### 3.3 Chunked evaluation cost

Current `evaluate_chunked` processes one utterance at a time and takes
~20 minutes per run. Three optimizations, all orthogonal to correctness:

1. **Batch the reset mode.** Pad chunks from different utterances into a
   mini-batch of fixed chunk length, process in one forward pass. Target
   ~20× speedup. Carry mode must stay sequential per utterance.
2. **Cap reset-mode eval at 500 utterances** (stratified by length).
   Dev-clean has 2642 utterances; 500 makes the standard error smaller
   than the inter-model gap.
3. **Run carry-mode only on `shortlist=true` runs** in the registry.

### 3.4 Streaming-memory demonstration (Group A only)

A separate script `scripts/measure_streaming_memory.py` runs once per
Group A backbone after training:

1. Take one 30-second LibriSpeech utterance.
2. For each chunk length in `[0.5, 1, 2, 5, 10, 30]` s, stream the
   utterance through the encoder with carry-state enabled.
3. Record `sum(t.numel() * t.element_size() for t in state)` and
   `torch.cuda.max_memory_allocated()` after each chunk boundary.
4. Append one row per (backbone, chunk_length) to
   `outputs/_streaming_memory.csv`.

Plot `outputs/_plots/streaming_memory_vs_duration.png`: log-scale y-axis
(bytes), linear x-axis (audio duration). Expected shape:

- `rwkv6`, `mamba` — flat horizontal lines at ~100–200 KB
- `transformer_causal` — linearly growing line (KV cache scales with
  tokens seen so far; ~6 MB at 10 s, ~1.4 GB at 40 min for our config)

This is the single strongest figure for the streaming-practical-relevance
claim in the thesis, and it lives entirely within Group A.

---

## 4. Unified research framework

Four layers, each runnable independently.

### 4.1 Experiment registry — `configs/experiments.yaml`

Single source of truth for the 16-run plan. Each entry fully specifies a
run; the executor validates against the backbone factory to catch typos.

```yaml
experiments:
  # ── Group A: causal / streaming-capable ──────────────────
  - {id: exp01_transformer_causal,  backbone: transformer_causal, seed: 42, epochs: 80, tags: [groupA, baseline],            shortlist: true}
  - {id: exp02_rwkv6,                backbone: rwkv6,              seed: 42, epochs: 80, tags: [groupA, baseline, recurrent], shortlist: true}
  - {id: exp03_mamba_cuda,           backbone: mamba_cuda,         seed: 42, epochs: 80, tags: [groupA, baseline, recurrent], shortlist: true}
  - {id: exp04_mamba_pytorch,        backbone: mamba,              seed: 42, epochs: 80, tags: [groupA, reimplementation]}
  - {id: exp05_rwkv6_lucid,          backbone: rwkv6_lucid,        seed: 42, epochs: 80, tags: [groupA, mechanism]}
  - {id: exp06_rwkv6_delta,          backbone: rwkv6_delta,        seed: 42, epochs: 80, tags: [groupA, mechanism]}
  - {id: exp07_rwkv6_lucid_delta,    backbone: rwkv6_lucid_delta,  seed: 42, epochs: 80, tags: [groupA, mechanism, stacking]}

  # ── Group B: bidirectional / offline ─────────────────────
  - {id: exp08_transformer,          backbone: transformer,                 seed: 42, epochs: 80, tags: [groupB, baseline],   shortlist: true}
  - {id: exp09_lion,                  backbone: lion,                        seed: 42, epochs: 80, tags: [groupB, baseline],   shortlist: true}
  - {id: exp10_mamba_bidir,           backbone: mamba_bidir,                 seed: 42, epochs: 80, tags: [groupB, baseline]}
  - {id: exp11_lion_convshift,        backbone: lion_convshift,              seed: 42, epochs: 80, tags: [groupB, mechanism]}
  - {id: exp12_lion_lucid,            backbone: lion_lucid,                  seed: 42, epochs: 80, tags: [groupB, mechanism]}
  - {id: exp13_lion_lucid_chunked,    backbone: lion_lucid_chunked,          seed: 42, epochs: 80, tags: [groupB, mechanism]}
  - {id: exp14_lion_delta,            backbone: lion_delta,                  seed: 42, epochs: 80, tags: [groupB, mechanism]}
  - {id: exp15_lion_headscale,        backbone: lion_headscale,              seed: 42, epochs: 80, tags: [groupB, mechanism]}
  - {id: exp16_lion_convshift_headscale, backbone: lion_convshift_headscale, seed: 42, epochs: 80, tags: [groupB, mechanism, stacking]}
```

Each entry produces `outputs/{id}/`. Re-running the same id resumes from
`last_model.pt` if present. The `shortlist` flag controls whether
carry-state eval and multi-seed validation run for that backbone.

### 4.2 Run executor — `scripts/run_registry.py`

Invocations:

```
uv run scripts/run_registry.py --all
uv run scripts/run_registry.py --ids exp09_lion,exp11_lion_convshift
uv run scripts/run_registry.py --tag mechanism --gpu 1
uv run scripts/run_registry.py --shortlist --seeds 42,123,777
```

Contract:
- Resolves each id to a fully-loaded `ExperimentConfig`.
- Writes `config.yaml`, `git_sha.txt`, `cli_args.txt` before the first
  training step so even a 30-second crash leaves a trail.
- Runs training; on failure leaves `results.json` absent so the reporter
  marks the run as failed rather than silently missing.
- On success appends one row to `outputs/_index.csv`.
- Emits a compact final summary with `rich`.

### 4.3 Reporting — `src/reporting/`

All modules run offline on the artifacts in `outputs/`; no GPU, no
dataset, no training code imported.

| Module | Output | In git |
|---|---|---|
| `reporting.collect` | `outputs/_index.csv` | yes |
| `reporting.tables` | Markdown tables written into `RESULTS.md` between `<!-- AUTOGEN: ... -->` markers | yes |
| `reporting.plots` | `outputs/_plots/*.png` (convergence overlay, CER-vs-params, chunked CER vs chunk size, streaming memory vs duration, training time bar) | yes |
| `reporting.diff` | `python -m src.reporting.diff exp09 exp11` — pairwise per-epoch comparison | ad-hoc |

In-run plots (per run directory, not per reporting pass): `loss_curve.png`,
`cer_curve.png`, `lr_schedule.png`, `grad_norm.png`. Rendered at the end
of each epoch, overwritten in place, committed. A reviewer browsing
`outputs/exp09_lion/plots/cer_curve.png` on the GitHub web UI sees the
training dynamics immediately.

### 4.4 Dependencies

Additions to `pyproject.toml` (no TensorBoard):

```toml
dependencies = [
    # existing: torch, torchaudio, datasets, jiwer, matplotlib, pyyaml, einops, soundfile
    "numpy>=1.26",
    "pandas>=2.2",      # _index.csv, thesis tables
    "seaborn>=0.13",    # paper-quality plots
    "tqdm>=4.66",       # training progress
    "rich>=13.7",       # console run summaries
    "tabulate>=0.9",    # markdown tables for RESULTS.md autogen
    "psutil>=5.9",      # CPU/RAM in metrics
]

[project.optional-dependencies]
cuda-mamba = ["mamba-ssm>=2.0.0"]
```

`uv.lock` will be committed so exact versions reproduce three months from
now when writing up.

---

## 5. What we deliberately do not do

Scope limits, stated explicitly so the project does not drift:

1. **No DDP / multi-GPU training.** 7 M-param models on 100 h of audio do
   not benefit from data parallelism; per-step compute is too small to
   hide gradient-sync latency. Revisit only if we scale to LibriSpeech-960.
2. **No hyperparameter sweeps.** The canonical config is fixed. Sweeps
   belong in a follow-up study.
3. **No LibriSpeech-other.** The noisy subset can be added as another
   dataset id after the clean comparison is complete.
4. **No Triton / custom CUDA kernels for the PyTorch Mamba scan.** The
   `torch.compile` path on a bigger GPU plus `mamba-ssm` as fallback
   covers the speed story.
5. **No training of bidirectional Transformer with KV cache.** Bidirectional
   attention has no KV cache in our setup; the causal Transformer covers
   that axis.
6. **No evaluation of carry-state on bidirectional models.** Not
   meaningful for models that were trained assuming access to future
   frames.

---

## 6. Deliverables (what to build)

Grouped by area. Each deliverable is a small, reviewable unit.

**Harness.**
- [ ] `pyproject.toml` updates + commit `uv.lock`
- [ ] `src/training/metrics.py` — `MetricLogger` (step + epoch, jsonl + csv)
- [ ] `src/training/checkpoint.py` — save/load with full state, `--resume`
- [ ] In-run plot generation at epoch end (matplotlib, overwritten PNGs)
- [ ] `.gitignore` update: exclude `*.pt`, `*.bin`, `*.safetensors`

**Carry-state correctness.**
- [ ] Fix Mamba `conv_state` carry-through in `MambaBlock.forward()`
- [ ] Fix `MambaEncoder.init_state()` to return real zero-state dicts
- [ ] Equivalence test: chunked forward matches full-utterance forward
- [ ] Rewrite `evaluate_chunked` with batched reset mode + `max_utterances`

**Causal Transformer.**
- [ ] `src/models/transformer_causal.py` — causal-masked attention with
  KV-cache support for inference-time state measurement
- [ ] Register as backbone `transformer_causal` in the encoder factory

**Streaming memory demonstration.**
- [ ] `scripts/measure_streaming_memory.py`
- [ ] `outputs/_streaming_memory.csv` schema
- [ ] `src/reporting/plots/streaming_memory.py`

**Registry and reporting.**
- [ ] `configs/experiments.yaml` — the 16-run plan
- [ ] `scripts/run_registry.py` — executor with filters
- [ ] `src/reporting/collect.py` — scans `outputs/*/results.json` to
  `_index.csv`
- [ ] `src/reporting/tables.py` — autogen markdown into `RESULTS.md`
- [ ] `src/reporting/plots.py` — cross-run comparison plots

---

## 7. Execution phases

Ordered so each phase produces a self-contained, committable result.
Unchecked items are open work; checked items are done. Any contributor can
read this section and know exactly where to resume.

### Phase A — Harness scaffolding (RTX 5090)
- [ ] A1. `pyproject.toml` + `uv.lock`
- [ ] A2. `MetricLogger` (jsonl + csv, no TensorBoard)
- [ ] A3. Checkpointing + `--resume` flag
- [ ] A4. In-run plot generation at epoch end
- [ ] A5. `.gitignore` update for binary weights

### Phase B — Carry-state correctness + causal Transformer
- [ ] B1. Fix Mamba `conv_state` carry
- [ ] B2. Fix `MambaEncoder.init_state()`
- [ ] B3. Equivalence test (full vs chunked forward)
- [ ] B4. Batched `evaluate_chunked` with `max_utterances` cap
- [ ] B5. `transformer_causal.py` + factory registration
- [ ] B6. `scripts/measure_streaming_memory.py`
- [ ] B7. Streaming-memory plot module

### Phase C — Registry + reporting
- [ ] C1. `configs/experiments.yaml` (16 entries)
- [ ] C2. `scripts/run_registry.py`
- [ ] C3. `reporting.collect`
- [ ] C4. `reporting.tables` with AUTOGEN markers in `RESULTS.md`
- [ ] C5. `reporting.plots` (cross-run)
- [ ] C6. Smoke test: 2 epochs of each backbone through the registry on
  the RTX 5090, verify `_index.csv`, tables, plots regenerate cleanly
- [ ] C7. Resume-from-crash test

### Phase D — Full training on the big GPU
- [ ] D1. Verify `run_mamba_compiled.py` speed on the target GPU
- [ ] D2. `run_registry.py --all --epochs 80` — Group A
- [ ] D3. `run_registry.py --all --epochs 80` — Group B
- [ ] D4. `measure_streaming_memory.py` for Group A
- [ ] D5. Multi-seed validation: `--shortlist --seeds 42,123,777`
- [ ] D6. Final `reporting.tables` + `reporting.plots` refresh
- [ ] D7. Commit `outputs/` (excluding `.pt`) and push

---

## 8. Success criteria

- One command reproduces every number and plot in the thesis chapter
  (`python -m src.reporting.tables && python -m src.reporting.plots`).
- A crash at epoch 40 of 80 costs no more than 30 minutes of lost compute.
- Adding a new mechanism means: one new encoder file, one new entry in
  `configs/experiments.yaml`, one `run_registry.py --ids exp17_newthing`
  invocation, and the tables update themselves on the next
  `reporting.tables` run.
- A reviewer browsing the repo on GitHub sees convergence curves for every
  run without cloning or running anything locally.
