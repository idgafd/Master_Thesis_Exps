# Formal Experiments v1 — Stage 1: Foundation Setup

## Context

This is the formal experiment codebase for the master thesis, replacing the
draft `asr_backbone_comparison/` which served as an exploratory phase (20 runs,
35h Ukrainian Common Voice). The draft identified:

- **LION (bidirectional RWKV-6)** as the best offline encoder (CER 0.176)
- **LUCID preconditioner** as the strongest unidirectional mechanism (-15% CER)
- **Delta Rule** as a solid unidirectional mechanism (-10% CER)
- **Multi-scale depth hierarchy** confirmed by 5 independent mechanisms
- Neither LUCID nor Delta Rule transferred to LION yet (implementation issues identified)

The formal version starts clean with unified code, a professional dataset,
matched baselines, and corrected mechanism implementations.

---

## Stage 1 Goal

**Build a clean, tested, unified experimental framework** that can run all
planned experiments reproducibly. No new research findings at this stage — just
infrastructure that works correctly.

**Success criterion:** Run 5-epoch debug training for all 6 core backbones
(Transformer, RWKV-6, Mamba, LION, LION+LUCID, LION+Delta) on LibriSpeech
train-clean-100, verify they all converge, and confirm parameter counts are
matched within 5%.

---

## Task Breakdown

### 1. Project Setup & Dependencies
**Priority: P0 | Estimate: Small**

- [x] Create `formal_v1/` directory
- [ ] Create `pyproject.toml` with all dependencies:
  - torch, torchaudio (CUDA 12.4)
  - datasets (HuggingFace) for LibriSpeech
  - jiwer for WER/CER computation
  - matplotlib for plots
  - pyyaml for configs
  - triton (for custom RWKV kernels if needed)
  - NO external `rwkv-block` dependency — all RWKV code is self-contained
  - NO external `mamba-ssm` for core logic — rewrite Mamba from scratch for
    modifiability (use `mamba-ssm` only as optional fast backend for training)
- [ ] Create canonical `configs/default.yaml`

### 2. Data Pipeline — LibriSpeech via HuggingFace
**Priority: P0 | Estimate: Medium**

Switch from Common Voice Ukrainian to LibriSpeech English (clean):
- `train-clean-100` (100h) for training — 3× more than draft's 35h
- `dev-clean` for validation
- `test-clean` for final evaluation
- Later: `train-clean-360` or `train-other-500` for scaling experiments

**Implementation:**
```python
from datasets import load_dataset
ds = load_dataset("openslr/librispeech_asr", "clean")
# ds["train.100"], ds["validation"], ds["test"]
```

- [ ] `src/data/librispeech.py` — HuggingFace dataset loading + caching
- [ ] `src/data/vocab.py` — English character vocab (rewrite: a-z + space + apostrophe)
- [ ] `src/data/dataset.py` — `ASRDataset`, `DurationBatchSampler`, `collate_fn`
- [ ] `src/data/augment.py` — `SpecAugment` (same as draft, validated)
- [ ] Verify: consistent train/dev/test splits, deterministic ordering

**Key differences from draft:**
- English characters instead of Ukrainian Cyrillic
- Longer utterances available (LibriSpeech has up to 30s)
- Professional recording quality (read speech, studio conditions)
- Standard benchmark — results are directly comparable to literature

### 3. Shared Components
**Priority: P0 | Estimate: Small**

- [ ] `src/models/components.py`:
  - `ConvSubsampling` (2x stride-2 Conv2d, 4× downsample) — reuse from draft
  - `CTCHead` (LayerNorm + Linear) — reuse from draft
  - `SinusoidalPE` — reuse from draft

### 4. Unified RWKV-6 Block (From Scratch)
**Priority: P0 | Estimate: Large — this is the core work**

**The problem:** Draft has 10+ copy-pasted model files with minor variations.
The new version has ONE configurable RWKV-6 block that supports all mechanisms
via composition.

- [ ] `src/models/rwkv6_time_mix.py` — The unified TimeMix:

  ```python
  class RWKV6TimeMix(nn.Module):
      """Unified RWKV-6 time-mixing block.

      Supports three modes:
        mode="recurrent"  — causal chunked WKV (unidirectional, carry-state capable)
        mode="lion"       — LION parallel full attention (bidirectional, O(T²))
        mode="bidir_serial" — VIM-style serial bidir (forward+backward recurrent)

      Optional mechanisms (stackable):
        conv_shift: bool  — learned DWConv token shift (replaces fixed shift)
        headscale: bool   — per-head decay bias
        delta_rule: bool  — selective state erasure (causal-only on LION)
        lucid: bool       — key decorrelation preconditioner
        temperature: bool — per-head learnable attention temperature
      """
  ```

  **Shared parameter structure** (identical across all modes):
  - Token shift: `time_maa_*` params + LoRA mixing (`time_maa_w1/w2`)
  - Decay: `time_decay` + data-dependent LoRA (`time_decay_w1/w2`)
  - Projections: receptance, key, value, gate, output
  - GroupNorm: `ln_x`

  **Mode-specific forward paths:**
  - `_forward_recurrent()` — chunked WKV with carry state
  - `_forward_lion()` — full T×T attention matrix
  - `_forward_bidir_serial()` — forward + backward recurrent

  **Mechanism implementations** (modular, applied inside forward):
  - ConvShift: `DWConvShift` module replaces `_bidirectional_token_shift()`
  - Headscale: `head_decay_bias` parameter, applied to `w_h` before attention
  - Delta Rule: extra params (`a0/a1/a2, k_k, k_a`), correction to attention/state
  - LUCID: `lucid_temperature` parameter, preconditioner on attention output
  - Temperature: `attention_temperature` parameter, sharpens attention matrix

- [ ] `src/models/rwkv6_channel_mix.py` — Unified ChannelMix:
  ```python
  class RWKV6ChannelMix(nn.Module):
      """RWKV-6 FFN: sigmoid(r) * value(relu²(key(x)))
      Supports both causal and bidirectional token shift.
      """
  ```

- [ ] `src/models/rwkv6_block.py` — Layer block:
  ```python
  class RWKV6Block(nn.Module):
      """ln0 (layer 0 only) → ln1 → TimeMix → drop → ln2 → ChannelMix → drop"""
  ```

### 5. LION Attention Kernel (Corrected)
**Priority: P0 | Estimate: Medium**

- [ ] `src/models/lion_attention.py` — Clean LION kernel:
  - `lion_parallel_attention(r, k, v, w)` — standard bidirectional
  - `lion_parallel_attention_with_delta(r, k, v, w, kk, iclr, causal_only=True)`
  - `lion_parallel_attention_with_lucid(r, k, v, w, lucid_temp, chunk_size=None)`

  **LUCID fix (critical):**
  The draft applied LUCID as `A @ P^{-1} @ V` (precondition values, then attend).
  The correct formulation decorrelates the *output*: `P^{-1} @ (A @ V)`.
  Additionally implement chunked LUCID: apply preconditioner within `chunk_size`
  windows rather than full T×T.

  **Delta Rule fix:**
  Only apply causal delta correction on LION. No anticausal extrapolation.
  ```python
  A_delta_fwd = -torch.tril(A_fwd @ kk_corr_causal)
  # A_delta_bwd is REMOVED — no anticausal delta
  ```

### 6. Transformer Baseline (Fixed)
**Priority: P0 | Estimate: Small**

- [ ] `src/models/transformer.py`:
  - `norm_first=True` (pre-norm, matches LION's pre-LN architecture)
  - FFN dim matched to RWKV-6: `int((d_model * 3.5) // 32 * 32)` = 256
    (draft used `d_model * ffn_mult` = 1280 which is WAY too large)
    Actually: RWKV-6 uses `hidden_size_ffn = int((hidden_size * 3.5) // 32 * 32)`
    = `int(896 // 32 * 32)` = 896. So Transformer FFN should also be 896.
  - `ln0` at layer 0 — matches LION's input normalization
  - Same `SinusoidalPE`, same `src_key_padding_mask`
  - Count parameters and verify within 5% of LION

### 7. Mamba Block (Rewritten for Modifiability)
**Priority: P1 | Estimate: Large**

- [ ] `src/models/mamba_block.py` — Pure PyTorch Mamba:
  - Self-contained SSM block (no external `mamba-ssm` dependency for logic)
  - Supports carry-state natively
  - Can be wrapped in bidirectional (forward+backward) mode
  - Optional: use `mamba-ssm` CUDA kernels as fast backend during training
    when available, fall back to PyTorch for portability

  **Why rewrite:** The draft's Mamba uses `mamba_ssm.Mamba` as a black box.
  To test bidirectional Mamba or add LION-inspired mechanisms, we need control
  over the SSM internals.

  **Minimum viable Mamba:**
  - Selective scan (S6) with data-dependent A, B, C matrices
  - 1D depthwise conv for local context
  - Gated output (SiLU)
  - Token-by-token step for carry-state

### 8. Encoder Factory
**Priority: P0 | Estimate: Small**

- [ ] `src/models/encoder.py`:
  ```python
  def build_encoder(cfg: EncoderConfig) -> nn.Module:
      """Single entry point for all encoder types."""
  ```

  Configuration-driven, not string-matching. Example:
  ```yaml
  encoder:
    type: rwkv6
    mode: lion          # recurrent | lion | bidir_serial
    conv_shift: true
    headscale: false
    delta_rule: false
    lucid: false
    lucid_chunk_size: null  # null = full-sequence, 64 = chunked
    temperature: false
  ```

- [ ] `src/models/asr_model.py` — Same structure as draft:
  `ConvSubsampling → Encoder → CTCHead`

### 9. Training Pipeline
**Priority: P0 | Estimate: Medium**

- [ ] `src/training/train.py` — Training loop (mostly reuse from draft)
- [ ] `src/training/evaluate.py` — Full-utterance + chunked eval
- [ ] `src/training/decode.py` — Greedy CTC decode + CER/WER
- [ ] `src/training/schedulers.py` — Cosine + warmup (canonical, no WSD)

  **Canonical training config (fixed for all runs):**
  ```yaml
  # Model
  d_model: 256
  n_layers: 6
  n_heads: 4
  head_size: 64
  dropout: 0.1        # reduced from 0.15 — LibriSpeech is cleaner + more data

  # Training
  lr: 3e-4
  weight_decay: 0.01
  warmup_steps: 1000
  grad_clip: 5.0
  batch_max_seconds: 300

  # Scheduler: cosine with warmup (single canonical choice)
  scheduler: cosine
  num_epochs: 80       # with early stopping patience=15

  # SpecAugment (LibriSpeech LD policy, scaled for our model size)
  freq_mask_param: 27
  time_mask_param: 100
  num_freq_masks: 2
  num_time_masks: 2

  # Evaluation
  chunk_sizes_sec: [2, 5, 10]
  ```

### 10. Debug/Validation Pipeline
**Priority: P0 | Estimate: Small**

- [ ] `scripts/debug_run.py` — Run 5 epochs for each backbone:
  - Verify all backbones converge (loss decreasing)
  - Print parameter counts per module
  - Verify parameter counts are matched (within 5%)
  - Verify forward/backward pass shapes
  - Test carry-state (where applicable)
  - Save training curves for visual inspection

- [ ] `scripts/run_experiment.py` — Full experiment runner (adapted from draft)

### 11. Utilities
**Priority: P1 | Estimate: Small**

- [ ] `src/utils/misc.py` — Seeding, param counting, serialization (reuse)
- [ ] `src/utils/plots.py` — Result plotting (reuse + extend)

---

## Directory Structure

```
experiments/formal_v1/
├── PLAN.md                      # This file
├── CLAUDE.md                    # Instructions for AI assistant
├── pyproject.toml               # Dependencies
├── configs/
│   └── default.yaml             # Single canonical config
├── scripts/
│   ├── debug_run.py             # 5-epoch validation for all backbones
│   └── run_experiment.py        # Full experiment runner
└── src/
    ├── __init__.py
    ├── config.py                # ExperimentConfig dataclass
    ├── data/
    │   ├── __init__.py
    │   ├── librispeech.py       # HuggingFace dataset loading
    │   ├── vocab.py             # English char vocab
    │   ├── dataset.py           # ASRDataset, sampler, collate, SpecAugment
    │   └── augment.py           # SpecAugment
    └── models/
        ├── __init__.py
        ├── components.py        # ConvSubsampling, CTCHead, SinusoidalPE
        ├── asr_model.py         # Top-level ASR model
        ├── encoder.py           # Encoder factory
        ├── transformer.py       # Matched Transformer baseline
        ├── mamba_block.py       # Pure PyTorch Mamba (rewritten)
        ├── mamba_encoder.py     # Mamba encoder (uni + bidir)
        ├── rwkv6_time_mix.py    # Unified RWKV-6 TimeMix (all modes+mechanisms)
        ├── rwkv6_channel_mix.py # RWKV-6 ChannelMix
        ├── rwkv6_block.py       # RWKV-6 layer block
        ├── rwkv6_encoder.py     # RWKV-6 encoder (uni + lion + bidir_serial)
        ├── lion_attention.py    # LION parallel attention kernels
        └── mechanisms/
            ├── __init__.py
            ├── conv_shift.py    # DWConvShift
            ├── delta_rule.py    # Delta rule params + corrections
            ├── lucid.py         # LUCID preconditioner (corrected)
            ├── headscale.py     # Per-head decay bias
            └── temperature.py   # Per-head attention temperature
    └── training/
        ├── __init__.py
        ├── train.py
        ├── evaluate.py
        ├── decode.py
        └── schedulers.py
    └── utils/
        ├── __init__.py
        ├── misc.py
        └── plots.py
```

---

## Implementation Order

**Phase 1 — Core infrastructure (must complete before any experiments):**

1. Project setup: `pyproject.toml`, directory structure, `config.py`
2. Data pipeline: LibriSpeech loading → vocab → dataset → augment
3. Shared components: `ConvSubsampling`, `CTCHead`, `SinusoidalPE`
4. Transformer baseline (fixed: `norm_first`, matched FFN, `ln0`)
5. RWKV-6 TimeMix (recurrent mode only, from scratch, no external deps)
6. RWKV-6 ChannelMix + Block + Encoder (unidirectional)
7. LION attention kernel (clean rewrite of `_lion_parallel_attention`)
8. RWKV-6 Encoder in LION mode (bidirectional)
9. Training loop + evaluation + schedulers
10. Debug run: verify Transformer, RWKV-6, LION all converge in 5 epochs

**Phase 2 — Mechanisms (after Phase 1 is validated):**

11. ConvShift mechanism
12. LUCID preconditioner (CORRECTED: output decorrelation + chunked option)
13. Delta Rule (causal-only for LION)
14. Headscale
15. Debug run: verify LION+ConvShift, LION+LUCID, LION+Delta, RWKV6+LUCID

**Phase 3 — Extended architectures:**

16. Mamba rewrite (pure PyTorch)
17. Bidirectional Mamba wrapper
18. Full experiment runner

---

## Key Design Decisions

### Why rewrite RWKV-6 from scratch (not using rwkv-block)?

1. The draft's unidirectional RWKV-6 uses the external `rwkv-block` library,
   but all LION variants are hand-rolled. This creates two parallel
   implementations that can diverge silently.
2. We need to modify the WKV kernel internals for Delta Rule and LUCID.
   The external library treats the kernel as a black box.
3. Self-contained code is easier to debug, test, and reproduce.
4. The RWKV-6 block is not complex — it's ~200 lines of PyTorch.

### Why rewrite Mamba?

1. The draft wraps `mamba_ssm.Mamba` which is a compiled CUDA kernel.
   We can't add bidirectional mode or modify the SSM internals.
2. Bidirectional Mamba (untested in draft) requires running the SSM in both
   directions — impossible with the black-box wrapper.
3. Pure PyTorch Mamba is slower but fully transparent and modifiable.
4. Can optionally use `mamba-ssm` as fast backend when available.

### Why cosine scheduler (not WSD)?

The draft tested both. WSD gave negligible difference on Mamba (runs 003-004)
and was never better than cosine on any other architecture. Cosine is simpler,
has one fewer hyperparameter, and is the standard in ASR literature.

### Why `norm_first=True` for Transformer?

Pre-norm (apply LayerNorm before attention/FFN) matches LION's architecture
where `ln1` is applied before `att` and `ln2` before `ffn`. This ensures
any performance difference is due to the attention mechanism, not the
normalization placement.

### Why FFN dim = `int((d_model * 3.5) // 32 * 32)` for Transformer?

RWKV-6's ChannelMix uses this formula, giving FFN dim = 896 for d_model=256.
The Transformer should use the same FFN capacity. The draft used
`d_model * ffn_mult` = 1280, giving the Transformer an unfair capacity
advantage in the FFN path.

### LUCID correction: output vs value decorrelation

**Draft (wrong):** `Y = A @ P^{-1} @ V` — decorrelates values before attention
**Correct:** `Y = P^{-1} @ (A @ V)` — decorrelates the attention output

The preconditioner should act on the *output space* where correlated keys
produce similar outputs. Pre-multiplying V changes the value representations
before attention, which is a different (and less motivated) operation.

Additionally: chunked LUCID (64-frame windows) should be tested as the
default for LION, since full T×T preconditioner may average out local
correlations.

---

## Experiments Planned (After Stage 1)

### Core Baselines (all on LibriSpeech clean-100)
1. Transformer (matched)
2. RWKV-6 (unidirectional recurrent)
3. Mamba (unidirectional)
4. LION (bidirectional RWKV-6)
5. LION + ConvShift

### Mechanism Transfer to LION
6. LION + LUCID (corrected, full-sequence)
7. LION + LUCID (corrected, chunked-64)
8. LION + Delta Rule (causal-only)
9. LION + ConvShift + Headscale

### Mechanism Stacking on RWKV-6
10. RWKV-6 + LUCID
11. RWKV-6 + Delta Rule
12. RWKV-6 + LUCID + Delta Rule

### Novel Directions
13. Bidirectional Mamba
14. Bidirectional Mamba + LUCID
15. LION + LUCID + Delta Rule (if both individually help)

### Statistical Validation
16. Top-5 configs × 3 seeds

---

## Non-Goals for Stage 1

- No new research ideas or mechanism experiments
- No hyperparameter tuning
- No large-scale training (>10 epochs)
- No paper writing
- No analysis of results beyond "does it converge?"

---

## Next Steps — RWKV Improvements (post-Stage 1, post lucid_comparison)

Based on the `lucid_*` 30-epoch results (LION baseline 0.107 dev CER, ConvShift
the only mechanism that clearly beats LION at 0.104, LUCID neutral on LION,
LUCID self-reg actively regressing, and `lion_delta`/`lion_headscale` still
untrained), the following directions are prioritized for the next experimental
campaign on RWKV/LION.

### 1. Combine the winners — `lion_convshift + {delta, headscale}`
ConvShift is the only mechanism that gave a clear win on LION (~3% rel. CER
improvement) at no compute cost. Stacking another orthogonal mechanism on top
is the cheapest path to a new best point.
- New backbones: `lion_convshift_delta`, `lion_convshift_headscale`
- Requires one extra entry per backbone in `encoder.py`'s `mode_map`; mechanism
  flags are already derived via substring matching.

### 2. Train `lion_delta` standalone
Currently missing from the formal results table. Without this run we cannot
claim Delta Rule is or isn't transferable to LION (the corrected causal-only
formulation has never been measured end-to-end at 30 epochs).

### 3. `lion_lucid_chunked` chunk-size sweep
Full T×T LUCID washes out local correlations and was approximately neutral on
LION. A small window may recover the −15% draft gain.
- Sweep `chunk ∈ {16, 32, 64}` as separate runs.

### 4. Headscale on causal RWKV-6
Cheap, never tried. New backbone `rwkv6_headscale`.

### 5. Drop `lucid_self_reg` from the campaign (or rederive it)
Currently a confirmed regression: rwkv6_lucid_sr at 0.148 dev CER vs rwkv6
baseline at 0.126 (+17% rel. CER). The RKHS erase-then-write formulation in
`_wkv_subchunk` plausibly over-erases in the small-T regime. Either remove it
from `experiments.yaml` or rederive the update rule before re-running.

### 6. Hybrid layer schedule — recurrent bottom + lion top
Bottom 3 layers `recurrent`, top 3 `lion`. The multi-scale depth hierarchy
claim from the draft predicts this should match LION accuracy at lower
streaming memory cost.
- Requires `RWKV6Encoder` to accept a per-layer `mode` list.

### 7. Multi-seed validation (42, 123, 777) for the top-2 configs
Once a winner is identified, run 3 seeds. The current LION vs lion_lucid gap
(0.0014 dev CER) is inside seed noise and cannot be relied on without
statistical validation.

### Suggested registry additions

```yaml
- {id: rwkv_imp01_lion_delta,             backbone: lion_delta,                 seed: 42, epochs: 30, tags: [rwkv_imp]}
- {id: rwkv_imp02_lion_convshift_delta,   backbone: lion_convshift_delta,       seed: 42, epochs: 30, tags: [rwkv_imp, stacking]}
- {id: rwkv_imp03_lion_convshift_hs,      backbone: lion_convshift_headscale,   seed: 42, epochs: 30, tags: [rwkv_imp, stacking]}
- {id: rwkv_imp04_lion_lucid_chunked16,   backbone: lion_lucid_chunked,         seed: 42, epochs: 30, tags: [rwkv_imp, lucid_sweep]}
- {id: rwkv_imp05_lion_lucid_chunked32,   backbone: lion_lucid_chunked,         seed: 42, epochs: 30, tags: [rwkv_imp, lucid_sweep]}
- {id: rwkv_imp06_lion_lucid_chunked64,   backbone: lion_lucid_chunked,         seed: 42, epochs: 30, tags: [rwkv_imp, lucid_sweep]}
- {id: rwkv_imp07_rwkv6_headscale,        backbone: rwkv6_headscale,            seed: 42, epochs: 30, tags: [rwkv_imp]}
- {id: rwkv_imp08_lion_recurrent_hybrid,  backbone: lion_recurrent_hybrid,      seed: 42, epochs: 30, tags: [rwkv_imp, hybrid]}
```

---

## Stage 2 — Higher-Order Discretization for Causal Linear-State Recurrences

### 1. Refocused research question

The unifying claim of the discretization study:

> *Higher-order discretization (trapezoidal, generalized 2-step, Adams-Bashforth)
> is a free expressivity boost in causal linear-state recurrences (RWKV-6,
> Mamba, linear attention), orthogonal to other mechanism choices.*

Why **causal first**, LION later:
1. Trapezoidal/AB3 are concepts about how to integrate a continuous-time ODE
   step-by-step. Bidirectional LION already attends across the full sequence,
   so the "ZOH expressivity gap" argument is structurally weaker there.
2. The streaming-practical-relevance narrative (`INFRASTRUCTURE_PLAN.md §3.4`)
   lives entirely in Group A. Discretization improvements directly speak to
   that axis.
3. The clean baseline → variant story per family is more falsifiable than
   "stack one more bidir mechanism on LION".

### 2. Mathematical formulation

Treat every causal linear recurrence as a discretized continuous-time system:

  dS/dt = -A(t) S(t) + b(t),       b(t) = k(t) v(t)^T

with `A(t) = exp(w_t)` for RWKV-6, the SSM `A`-matrix for Mamba, and
`A = 0` for linear attention.

**Generalized linear multistep state update** (the unifying object — covers
ZOH, RWKV's bonus term `u`, trapezoidal, and Adams-Bashforth as special
cases of one coefficient family):

  S_t = W_t ⊙ S_{t-1}  +  Σ_{i=0}^{p-1}  α_i · W̃_t^{i/p} ⊙ b_{t-i}

| Scheme | p | Coefficients | Identification |
|---|---|---|---|
| ZOH (vanilla RWKV-6) | 1 | α₀ = 1 | current code |
| ZOH + bonus `u` | 1 | α₀ = 1, extra `u·b_t` in readout only | current code |
| **Pure trapezoidal** | 2 | α₀ = α₁ = ½ | new |
| **Generalized 2-step** | 2 | α₀, α₁ learnable per head | new |
| **Adams-Bashforth-3** | 3 | (23/12, −16/12, 5/12) | new |

#### Two refinements that matter

1. **Variable-decay correction.** RWKV-6's `w_t` is data-dependent, so the
   standard trapezoidal derivation (which assumes constant `A` on `[t−Δ, t]`)
   is biased. Use the **geometric-mean decay** on the lookback term:
       W̃_t = sqrt(W_t ⊙ W_{t-1}),    S_t = W_t S_{t-1} + ½(b_t + W̃_t ⊙ b_{t-1})
   This is the proper "exponential-trapezoidal" rule for time-varying A.
   Tested as a separate variant (`rwkv6_trap_var`) so its contribution can
   be isolated from the basic trapezoidal correction.

2. **Per-channel learnable order.** Instead of fixing the scheme, let the
   network choose its discretization per channel via learnable mix
   parameters:
       S_t = W_t S_{t-1} + α₀(θ) · b_t + α₁(θ) · W̃_t · b_{t-1}
   This subsumes ZOH (α₀ → 1, α₁ → 0) and trapezoidal (α₀ = α₁ = ½) and
   lets ablation reveal what the model actually wants per channel.

#### Relation to existing RWKV-6 components

- The bonus term `u` (`time_faaaa`) is mathematically a **rescaled α₀**
  applied at the readout (not the state update). When testing pure
  trapezoidal we therefore **ablate `u`**: keeping it would conflate a
  state-side and a readout-side α₀ contribution.
- ConvShift is a learned 3-tap depthwise convolution on `x` BEFORE the
  k/v projection — i.e., a learned discretization filter on the **input**
  side. Stage 2's discretization variants act on the **state-update**
  side. The `rwkv6_convshift_trap` cell tests whether the two sides are
  complementary or redundant.

### 3. Implementation strategy (efficient, single-kernel)

The recurrence S_t = W_t S_{t-1} + drive_t is **linear in the drive**, so
the multi-step drive Σ α_i W̃_t^{i/p} ⊙ b_{t-i} can be implemented as the
sum of `p` parallel scans with the same `W_t` schedule:

- Scan 0: drive = α₀ · k_t v_t^T            → standard chunked WKV
- Scan 1: drive = α₁ · W̃_t · k_{t-1} v_{t-1}^T → shifted-by-1 chunked WKV
- Scan 2 (AB3 only): drive = α₂ · W̃_t² · k_{t-2} v_{t-2}^T

The decay factor W̃_t is per-channel (K-dim) and absorbs cleanly into the
shifted `k_lag` row scaling: `k_b[i] = α₁ · W̃_t[i] · k_{t-1}[i]`. Output
and state of the parallel scans are summed (linearity).

This keeps the existing fast `_chunked_wkv` (SmerkyG algorithm) intact
and costs only a 2× scan budget for trapezoidal (3× for AB3). Expected
epoch time: ~150 s (vs 80 s baseline) at 30 epochs ≈ 75 min/run.

For inference / carry-state, the encoder state extends from `S` only to
`(S, k_{t-1}, v_{t-1})` for trapezoidal and `(S, k_{t-1}, v_{t-1}, k_{t-2}, v_{t-2})`
for AB3. The extra cache is `O(p · H · K)` per layer (KB scale, not MB).

### 4. Potential dangers (must be addressed in the implementation)

These three concerns are real and the implementation must explicitly
handle them. Each variant's success criterion includes "did training
remain stable" alongside CER.

1. **AB3 absolute-stability region is small.** Adams-Bashforth-3 is an
   explicit higher-order multistep; its region of absolute stability is
   roughly half that of AB1/ZOH. Combined with data-dependent step sizes
   (`exp(w_t)`) this can produce exploding gradients or oscillatory
   training. **Mitigations:**
   - Clamp decay strictly: `w_clamp = w.clamp(min=ln(0.5))` so step size
     `Δ ≤ 0.7` per token.
   - Train AB3 in fp32 only (no autocast on the WKV scan) — already true
     for the current `_chunked_wkv` which casts to fp64 for chunk_len > 24.
   - Monitor `grad_norm` per epoch; auto-halve LR if it exceeds 50 for
     two consecutive epochs.
   - If unstable: drop AB3 from the campaign and report it as a negative
     result — that itself is a useful contribution.

2. **Inference memory cost of the lookback cache.** During autoregressive
   decoding (and chunked carry-state evaluation), the recurrent state
   grows from `(S)` to `(S, k_lag, v_lag, [k_lag2, v_lag2])`. For RWKV-6
   with H=4, K=64, this adds 4 × 64 × 4 bytes = 1 KB per layer for
   trapezoidal, 2 KB for AB3. **Negligible vs. the S matrix (16 KB/layer)
   but must be reported alongside any expressivity gain.** Add these
   numbers to the streaming-memory plot
   (`outputs/_plots/streaming_memory_vs_duration.png`).

3. **`gen2` initialization must start at ZOH, not uniform.** A uniform
   softmax init (θ₀ = θ₁) puts the model at α₀ = α₁ = ½ — already the
   trapezoidal prior — and makes the "pure ZOH" attractor harder to
   reach. **Initialize θ such that `softplus(θ₀)/Σ = 1, softplus(θ₁)/Σ ≈ 0`
   (i.e. ZOH start), and let gradient pull α₁ up if needed.** The
   alternative (init at trapezoidal) is also valid; we run both as a
   tiny ablation (`rwkv6_gen2_zoh_init`, `rwkv6_gen2_trap_init`).

### 5. Experiment matrix (causal first, then transfer)

**Phase 1 — causal RWKV-6 (the proper testbed):**

| ID | Backbone | Coefficients | Question |
|----|----------|--------------|----------|
| disc01 | `rwkv6` | baseline ZOH+u | reference (~0.126 dev CER, 30 ep) |
| disc02 | `rwkv6_trap` | α₀=α₁=½, no `u`, current decay | does pure trapezoidal beat ZOH+u? |
| disc03 | `rwkv6_trap_var` | α₀=α₁=½, no `u`, geometric-mean W̃ | does the variable-decay correction matter? |
| disc04 | `rwkv6_gen2` | learnable α₀, α₁, ZOH init, keep `u` | what does the model want? |
| disc05 | `rwkv6_ab3` | (23/12, −16/12, 5/12), no `u`, decay clamped | does higher order help (or explode)? |
| disc06 | `rwkv6_convshift_trap` | trap_var + ConvShift | input-side + state-side filtering: complementary? |

**Phase 2 — transfer the winning scheme to LION (only if Phase 1 shows signal):**

| ID | Backbone | Notes |
|----|----------|-------|
| disc07 | `lion_trap` | symmetric trapezoidal on forward and backward sweeps |
| disc08 | `lion_convshift_trap` | with ConvShift, the current Group B winner |

**Routing note:** `lion_trap` and `lion_convshift_trap` route through
`mode="bidir_serial"` (not `mode="lion"`). LION's parallel T×T attention
has no explicit recurrence and therefore no discretization concept to
modify. `bidir_serial` runs the recurrent scan twice (forward + flipped),
so the trapezoidal update applies cleanly on each sweep and the two are
summed. This is a slower path than parallel LION but it is the only
mathematically meaningful way to extend a discretization study to a
bidirectional model in this codebase.

**Phase 3 — Mamba (separate, after Phase 1):**
- `mamba_trap`: replace ZOH with exponential-trapezoidal in selective scan.
  Cleanest test of the discretization claim because Mamba has explicit `A`,
  `B`, `Δ`. Implementation lives in `mamba_block.py`.

**Phase 4 — Linear attention (control):**
- Add a `linatt` backbone (already in `blocks.py` — needs a Phase-D-style
  factory entry) and a `linatt_trap` variant. Predicted to be marginal
  (no decay → trapezoidal collapses to a 2-tap moving average that the
  input projection already represents). A null result here *strengthens*
  the claim that the gain comes from the discretization-of-decay
  interaction.

### 6. Registry entries

```yaml
# ── Stage 2: Discretization study (causal first) ─────────────────
- {id: disc01_rwkv6_baseline,        backbone: rwkv6,                 seed: 42, epochs: 30, tags: [disc, baseline, recurrent]}
- {id: disc02_rwkv6_trap,            backbone: rwkv6_trap,            seed: 42, epochs: 30, tags: [disc, trap, recurrent]}
- {id: disc03_rwkv6_trap_var,        backbone: rwkv6_trap_var,        seed: 42, epochs: 30, tags: [disc, trap, recurrent]}
- {id: disc04_rwkv6_gen2,            backbone: rwkv6_gen2,            seed: 42, epochs: 30, tags: [disc, gen2, recurrent]}
- {id: disc05_rwkv6_ab3,             backbone: rwkv6_ab3,             seed: 42, epochs: 30, tags: [disc, ab3, recurrent]}
- {id: disc06_rwkv6_convshift_trap,  backbone: rwkv6_convshift_trap,  seed: 42, epochs: 30, tags: [disc, stacking, recurrent]}
- {id: disc07_lion_trap,             backbone: lion_trap,             seed: 42, epochs: 30, tags: [disc, trap, bidir]}
- {id: disc08_lion_convshift_trap,   backbone: lion_convshift_trap,   seed: 42, epochs: 30, tags: [disc, stacking, bidir]}
```

### 7. 2-GPU launch plan

8 runs split across 2 GPUs as 4-and-4 sequential per device,
~75 min/run ≈ 5 h per GPU:

  GPU 0 (causal core):    disc01 → disc02 → disc03 → disc04
  GPU 1 (causal+transfer): disc05 → disc06 → disc07 → disc08

`scripts/launch_discretization.sh` implements this. Logs land in
`outputs/logs/disc_gpu{0,1}.log`. Each run writes to
`outputs/disc{NN}_<backbone>_seed42/` and (on success) appends one row
to `outputs/_index.csv`. AB3 is intentionally placed on GPU 1 first so
that if it explodes the rest of the GPU 1 queue still proceeds.

### 8. Success criteria

- Phase 1 finishes without manual intervention on AB3 (no NaN, no LR
  rollback). Trapezoidal variants reach a final dev CER within ±10% of
  the `rwkv6` baseline (sanity check that we did not break the model).
- At least one of {`rwkv6_trap`, `rwkv6_trap_var`, `rwkv6_gen2`} beats
  the `rwkv6` baseline by ≥ 1.5% relative dev CER.
- The `rwkv6_convshift_trap` cell reveals whether input-side (ConvShift)
  and state-side (trapezoidal) discretization filters are complementary
  (gain stacks) or redundant (gain dominated by one).
- The `gen2` learned coefficients are inspected post-hoc: histogram of
  α₁/α₀ across heads/layers tells us whether the model uses the
  flexibility or stays close to ZOH.
