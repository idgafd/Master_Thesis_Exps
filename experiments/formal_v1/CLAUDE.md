# CLAUDE.md — Formal Experiments v1

## Project Overview

CTC-based ASR encoder backbone comparison on LibriSpeech (English).
Comparing Transformer, RWKV-6, Mamba, and LION (bidirectional RWKV-6)
with mechanism improvements (LUCID, Delta Rule, ConvShift, Headscale).

This is a master thesis project. The draft phase (in `../asr_backbone_comparison/`)
established LION as the best architecture and identified promising mechanisms.
This formal version rewrites everything cleanly with corrected implementations.

## Current Stage: Stage 1 — Foundation Setup

See `PLAN.md` for full task breakdown. We are building infrastructure, not
running experiments yet. The goal is a clean unified codebase that can run
all planned experiments reproducibly.

## Architecture

```
ConvSubsampling (4× downsample, 80-mel → 256-dim)
  → Encoder (swappable: Transformer / RWKV-6 / Mamba / LION)
  → CTCHead (LayerNorm + Linear → vocab logits)
```

All encoders: ~7M params, 6 layers, d_model=256, head_size=64, 4 heads.

## Key Technical Decisions

### Dataset
- LibriSpeech `clean` via `datasets` (HuggingFace): train-clean-100, dev-clean, test-clean
- English characters: a-z, space, apostrophe (+ CTC blank at index 0)
- 16kHz, 80-mel, 25ms window, 10ms hop

### Model Architecture Rules
- **All encoders use pre-norm** (LayerNorm before attention/FFN)
- **FFN dim = `int((d_model * 3.5) // 32 * 32)` = 896** for ALL architectures
  (RWKV-6 ChannelMix and Transformer FFN must match)
- **Layer 0 has extra `ln0`** (input normalization, matches RWKV convention)
- **Parameter counts must be within 5%** across architectures
- **Transformer uses `norm_first=True`** and activation=`"squared_relu"` to
  match RWKV-6's `relu² → value` pattern as closely as possible

### RWKV-6 Implementation
- **Self-contained** — NO external `rwkv-block` dependency
- **Single unified `RWKV6TimeMix`** class with `mode` parameter:
  - `mode="recurrent"`: causal chunked WKV, carry-state capable
  - `mode="lion"`: LION parallel T×T attention, bidirectional
- **Mechanisms are compositional flags**, not separate model files:
  - `conv_shift`, `headscale`, `delta_rule`, `lucid`, `temperature`
- **Token shift**: bidirectional `(x[t-1]+x[t+1])/2` for LION mode,
  causal `x[t-1]` for recurrent mode
- **ConvShift**: replaces fixed token shift with learned DWConv1d(kernel=3)

### LUCID Preconditioner (CORRECTED from draft)
- **Draft (wrong):** `Y = A @ P^{-1} @ V` (decorrelates values, then attend)
- **Correct:** `Y = P^{-1} @ (A @ V)` (decorrelates attention output)
- **Chunked LUCID:** Apply within `chunk_size` windows (default 64 frames)
  rather than full T×T for LION mode
- Preconditioner: `P = I + exp(τ * K_norm @ K_norm^T)`, solve `P @ Y_out = A @ V`
- Temperature `τ` is learnable per head (1 param per head)

### Delta Rule (CORRECTED from draft)
- On LION: **causal-only** correction. NO anticausal delta.
  ```
  A_delta = -tril(A_fwd @ kk_corr_causal)
  A_total = A_fwd + A_bwd + A_delta  # no A_delta_bwd
  ```
- On recurrent RWKV-6: full delta rule state update (same as draft, works)

### Mamba
- **Pure PyTorch implementation** for modifiability (no `mamba-ssm` hard dep)
- Selective scan (S6), 1D depthwise conv, gated output
- Supports carry-state natively
- Can be wrapped bidirectionally (forward + backward passes)

### Training
- **Scheduler:** cosine + warmup (canonical, no WSD option)
- **SpecAugment:** freq=27, time=100, 2+2 masks (LibriSpeech LD policy)
- **Optimizer:** AdamW, lr=3e-4, weight_decay=0.01
- **Epochs:** 80 with early stopping patience=15
- **Dropout:** 0.1 (reduced from 0.15 — more data, cleaner audio)

## Code Style

- Pure PyTorch, no custom CUDA kernels (except optional Mamba backend)
- Type hints on all public functions
- Docstrings on all classes and non-trivial functions
- No print statements in model code (use logging if needed)
- Model forward() returns `(output, new_state_or_None)`

## File Layout

```
src/
├── config.py                    # ExperimentConfig dataclass
├── data/
│   ├── librispeech.py           # HuggingFace dataset loading
│   ├── vocab.py                 # English char vocab
│   ├── dataset.py               # ASRDataset, sampler, collate
│   └── augment.py               # SpecAugment
├── models/
│   ├── components.py            # ConvSubsampling, CTCHead, SinusoidalPE
│   ├── asr_model.py             # Top-level: frontend → encoder → CTC head
│   ├── encoder.py               # build_encoder() factory
│   ├── transformer.py           # Matched Transformer baseline
│   ├── mamba_block.py           # Pure PyTorch Mamba SSM block
│   ├── mamba_encoder.py         # Mamba encoder (uni + bidir wrapper)
│   ├── rwkv6_time_mix.py        # THE unified TimeMix (all modes+mechanisms)
│   ├── rwkv6_channel_mix.py     # ChannelMix
│   ├── rwkv6_block.py           # Layer block (ln + tmix + drop + cmix + drop)
│   ├── rwkv6_encoder.py         # Encoder (wraps N blocks + pos_enc)
│   ├── lion_attention.py        # LION parallel attention kernels
│   └── mechanisms/
│       ├── conv_shift.py        # DWConvShift
│       ├── delta_rule.py        # Delta rule parameters + corrections
│       ├── lucid.py             # LUCID preconditioner (CORRECTED)
│       ├── headscale.py         # Per-head decay bias
│       └── temperature.py       # Per-head attention temperature
├── training/
│   ├── train.py                 # train_one_epoch()
│   ├── evaluate.py              # evaluate(), evaluate_chunked()
│   ├── decode.py                # greedy_ctc_decode(), compute_cer()
│   └── schedulers.py            # WarmupCosineScheduler
└── utils/
    ├── misc.py                  # Seeding, param counting
    └── plots.py                 # Result visualization
```

## How to Run

```bash
# Install
cd experiments/formal_v1
uv sync

# Debug run (5 epochs, all backbones)
uv run scripts/debug_run.py

# Full experiment
uv run scripts/run_experiment.py --config configs/default.yaml

# Single backbone
uv run scripts/run_experiment.py --backbone lion_convshift
```

## Backbone Naming Convention

Config-driven names for the encoder registry:

| Name | Architecture | Mode | Mechanisms |
|------|-------------|------|------------|
| `transformer` | Transformer | — | — |
| `rwkv6` | RWKV-6 | recurrent | — |
| `rwkv6_lucid` | RWKV-6 | recurrent | LUCID |
| `rwkv6_delta` | RWKV-6 | recurrent | Delta Rule |
| `rwkv6_lucid_delta` | RWKV-6 | recurrent | LUCID + Delta |
| `mamba` | Mamba | recurrent | — |
| `mamba_bidir` | Mamba | bidir_serial | — |
| `lion` | RWKV-6 | lion | — |
| `lion_convshift` | RWKV-6 | lion | ConvShift |
| `lion_lucid` | RWKV-6 | lion | LUCID (corrected) |
| `lion_lucid_chunked` | RWKV-6 | lion | LUCID (chunked-64) |
| `lion_delta` | RWKV-6 | lion | Delta Rule (causal-only) |
| `lion_headscale` | RWKV-6 | lion | Headscale |
| `lion_convshift_headscale` | RWKV-6 | lion | ConvShift + Headscale |

## Implementation Priorities

**Write code in this order:**

1. `config.py` + `configs/default.yaml`
2. `data/` pipeline (librispeech → vocab → dataset → augment)
3. `models/components.py` (ConvSubsampling, CTCHead, PE)
4. `models/transformer.py` (fixed baseline)
5. `models/rwkv6_time_mix.py` (recurrent mode only first)
6. `models/rwkv6_channel_mix.py` + `rwkv6_block.py` + `rwkv6_encoder.py`
7. `models/lion_attention.py` (clean kernel)
8. Add LION mode to `rwkv6_time_mix.py`
9. `training/` pipeline
10. `scripts/debug_run.py` — validate everything works

Then mechanisms:
11. `mechanisms/conv_shift.py`
12. `mechanisms/lucid.py` (CORRECTED)
13. `mechanisms/delta_rule.py` (causal-only for LION)
14. Wire mechanisms into `rwkv6_time_mix.py`

Then Mamba:
15. `models/mamba_block.py` (pure PyTorch)
16. `models/mamba_encoder.py` (uni + bidir)

## Common Pitfalls (From Draft Experience)

1. **RWKV-6 decay must be negative in log-space:** `w_h = -torch.exp(w_raw)`.
   Forgetting the negation makes all decay positive → attention blows up.

2. **LION forward vs backward prefix sums differ:**
   - Forward: `cs = cumsum(w)`, backward: `cs_b = cumsum(w) - w`
   - Getting this wrong makes the diagonal coefficient != 1

3. **GroupNorm expects (B*T, D) not (B, T, D):** Reshape before `ln_x`.

4. **ConvSubsampling length calculation:**
   `new_len = ((lengths - 1) // 2 + 1)` applied twice for 2× stride-2 convs.

5. **CTC requires `log_probs.permute(1, 0, 2)`** — time-first, not batch-first.

6. **SpecAugment time masks can exceed sequence length:** Clamp
   `t = random.randint(0, min(time_mask_param, T))`.

7. **Carry-state BiWKV6 is broken without carry-state training:**
   The backward branch expects future-summarizing state at inference but gets
   past-summarizing state from previous chunks.

8. **RWKV-7 stock init is broken at small scale:** If we ever compare,
   apply all three fixes (decay +2.0, v_first disable, k_a = 0).

## Research Context

The thesis aims to show that:
1. LION (bidirectional RWKV-6) is the best lightweight offline ASR encoder
2. The multi-scale depth hierarchy is a robust structural property
3. LUCID and/or Delta Rule can improve LION (with corrected implementations)
4. The combination of mechanisms yields a state-of-the-art lightweight encoder

The MAIN GOAL is not just to confirm LION works, but to **improve it** or
**beat it** with mechanism combinations inspired by LUCID, Delta Rule, or
novel approaches. The corrected LUCID and causal-only Delta Rule are the
most promising paths.
