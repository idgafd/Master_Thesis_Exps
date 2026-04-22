# CLAUDE.md — Synthetics v1 (Tier-1 Expressivity Benchmarks)

## Project Overview

Sibling experiment to `../formal_v1/`. Reuses the same encoder backbones
(Transformer, RWKV-6, Mamba, Mamba-2, LION + all mechanism variants) on
**synthetic sequence-modeling tasks** that probe architectural expressivity
along axes the LibriSpeech ASR backbone task in `formal_v1` does not exercise:

- **MQAR** — Multi-Query Associative Recall (Arora et al., *Zoology* 2024) —
  separates models that can vs cannot do in-context retrieval.
- **State tracking** *(planned)* — parity, modular addition, Dyck-k —
  separates models that can vs cannot maintain finite-state structure.
- **Selective copy / induction heads** *(planned)*.

Why this tier exists: the PI feedback in `../formal_v1/stage10_feedback.md §2.2`
identified that the LibriSpeech CER ceiling at ~0.115 is **input-representation-
bound, not sequence-mixer-bound**. Mechanism work on the mixer cannot break
that ceiling. To defend the central thesis claim (closing the expressivity gap
between linear-time models and Transformers), we need tasks where the mixer
*is* the bottleneck. Synthetics deliver that cleanly and at low compute cost.

## Current Stage: Stage 1 — MQAR Setup

**Goal:** Reproduce the canonical MQAR benchmark from the Zoology paper at our
matched 7M-parameter / 6-layer / d_model=256 model size, across all formal_v1
backbones and the most informative subset of mechanism variants.

See `PLAN.md` for the experiment matrix and success criteria.

## Architecture

```
TokenEmbedding (vocab → 256-dim)
  → Encoder (swappable: Transformer / RWKV-6 / Mamba / Mamba-2 / LION)
  → LMHead (LayerNorm + Linear → vocab logits)
```

Encoder is **identical to formal_v1** — same `build_encoder()` factory, same
mechanism dispatch, same parameter envelope (~7 M params, 6 layers, d_model=256,
4 heads, head_size=64, FFN dim=896). The only differences from `formal_v1`'s
ASRModel are:

1. **Frontend:** `nn.Embedding(vocab_size, d_model)` instead of `ConvSubsampling`
2. **Head:** LayerNorm + Linear → CE loss instead of CTCHead → CTC loss
3. **Data:** synthetic generators instead of LibriSpeech

## Code Style

(Same as formal_v1.)

- Pure PyTorch, no custom CUDA kernels (Mamba may use optional CUDA backend).
- Type hints on all public functions.
- Docstrings on classes and non-trivial functions.
- No print statements in model code (use logging if needed).
- Model forward() returns `(output, new_state_or_None)`.
- Substring-driven backbone naming (`rwkv6_lucid_delta`, `lion_convshift`, etc.)
  — the dispatcher in `encoder.py` parses substrings to set mechanism flags.
- **Exact reduction at init**: every backbone variant must reduce to vanilla
  output at step 0 (zero-regression contract), exactly as in formal_v1.

## File Layout

```
synthetics_v1/
├── pyproject.toml                    # No torchaudio / librosa / soundfile
├── README.md
├── CLAUDE.md
├── PLAN.md
├── src/
│   ├── config.py                     # SyntheticsConfig (audio fields removed)
│   ├── tasks/
│   │   ├── mqar.py                   # MQAR generator + scorer
│   │   ├── state_tracking.py         # parity / mod-N / Dyck-k (planned)
│   │   └── induction.py              # selective copy / induction heads (planned)
│   ├── data/
│   │   ├── vocab.py                  # TokenVocab (trivial int-id)
│   │   ├── dataset.py                # SyntheticDataset + collate
│   │   └── generator.py              # batched generator orchestrator
│   ├── models/
│   │   ├── synthetic_model.py        # TokenEmbed → Encoder → LMHead
│   │   ├── encoder.py                # SYMLINK ../formal_v1/...
│   │   ├── transformer.py            # SYMLINK
│   │   ├── rwkv6_encoder.py          # SYMLINK
│   │   ├── rwkv6_block.py            # SYMLINK
│   │   ├── rwkv6_time_mix.py         # SYMLINK
│   │   ├── rwkv6_channel_mix.py      # SYMLINK
│   │   ├── mamba_encoder.py          # SYMLINK
│   │   ├── mamba_block.py            # SYMLINK
│   │   ├── mamba2_encoder.py         # SYMLINK
│   │   ├── lion_attention.py         # SYMLINK
│   │   └── mechanisms/               # SYMLINK directory
│   ├── training/
│   │   ├── train.py                  # ADAPT formal_v1: parameterize loss_fn
│   │   ├── evaluate.py               # ADAPT: per-query recall accuracy
│   │   ├── schedulers.py             # SYMLINK
│   │   ├── metrics.py                # SYMLINK
│   │   └── checkpoint.py             # SYMLINK
│   └── utils/
│       ├── misc.py                   # SYMLINK
│       └── plots.py                  # SYMLINK
├── configs/
│   └── default.yaml                  # synthetics base config
├── scripts/
│   ├── setup_symlinks.sh             # mirrors backbones from formal_v1
│   ├── run_experiment.py             # main runner (mirrors formal_v1)
│   ├── debug_run.py                  # smoke test all backbones
│   └── analyze_mqar.py               # post-hoc plots / per-position recall
├── tests/
│   ├── test_mqar_generator.py
│   └── test_synthetic_model.py
└── outputs/                          # gitignored binaries; metrics committed
```

## Symlink Strategy

Backbones live ONLY in `formal_v1`. `scripts/setup_symlinks.sh` creates symlinks
into `synthetics_v1/src/models/` so any patch to a formal_v1 backbone is picked
up automatically here. Files we adapt locally (config, training loop,
components) are NOT symlinked.

Run `bash scripts/setup_symlinks.sh` after `uv sync`.

## How to Run

```bash
cd experiments/synthetics_v1
uv sync
bash scripts/setup_symlinks.sh

# Smoke test (small, 200 steps, all backbones)
uv run scripts/debug_run.py

# Single MQAR run
uv run scripts/run_experiment.py \
    --task mqar \
    --backbone rwkv6 \
    --seq-len 256 \
    --n-kv-pairs 64

# Full sweep (defined in configs/default.yaml)
uv run scripts/run_experiment.py --config configs/default.yaml
```

## Backbone Inventory (inherited from formal_v1)

Same list as `../formal_v1/CLAUDE.md` "Backbone Naming Convention" table.
For Stage 1 we will run a **focused subset**, not the full 14-variant matrix —
see `PLAN.md` cohort spec.

## Common Pitfalls (Synthetics-Specific)

1. **MQAR training loss must mask non-query positions.** Only positions where
   a query token expects a value should contribute to the cross-entropy. Use
   `ignore_index=-100` on non-query positions.

2. **Recall accuracy ≠ token accuracy.** Report per-query accuracy averaged
   over queries in the batch, not per-token accuracy (which is dominated by
   trivially-easy KV-pair memorization positions).

3. **Sequence length stress-tests assume length-extrapolation reporting.**
   Train at one length, evaluate at multiple — DO NOT only report
   train-length accuracy.

4. **LION is bidirectional → trivially solves causal MQAR.** Causal MQAR must
   use causal backbones only (`transformer_causal`, `rwkv6`, `mamba`,
   `mamba2`). Bidirectional MQAR (or non-causal probes) is a separate axis.

5. **Vocab size affects parameter count.** Embedding + LMHead together are
   `2 * V * d_model`. At V=8192, d=256 that's 4.2M params, comparable to the
   encoder itself. Hold V fixed across backbones to keep param parity meaningful;
   tie input/output embeddings if you want strict parity with formal_v1's 7M.

## Research Context

Companion to `../formal_v1/` (LibriSpeech ASR mechanism evaluation). The thesis
core claim — that LION + corrected mechanisms close the expressivity gap to
softmax attention — needs evidence on tasks where softmax attention has a
known, theoretically-motivated advantage. MQAR is exactly that task. State
tracking will be the second leg.
