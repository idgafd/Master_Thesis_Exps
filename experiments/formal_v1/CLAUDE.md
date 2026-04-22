# CLAUDE.md — Formal Experiments v1

## Project Overview

Master-thesis codebase studying **causal linear-time RNNs as an alternative
architectural family to Transformer causal attention**. Short-range CTC
ASR on LibriSpeech clean-100 at 7 M / 30 ep is used as a controlled
measurement probe for mechanism-level expressivity claims, not as an
engineering target.

Architectures compared (causal, unless otherwise noted): RWKV-6, Mamba-2,
Linear Attention. Transformer serves as a matched reference baseline.
LION (bidirectional RWKV-6) is a separate parallel chapter on bidirectional
adaptation using the same mechanism vocabulary.

The draft phase in `../asr_backbone_comparison/` established infrastructure
feasibility on Common Voice Ukrainian. This formal version rewrites the
codebase cleanly, runs on LibriSpeech clean-100 English, and produces the
thesis results.

## Current State (2026-04-22)

**Stages 2–10 complete.** Causal RWKV-6 mechanism-discovery chain has
established the winning mechanism catalog (RSE-family transition,
multi-dilation ConvShift input-side, viscosity coupling) and the
cross-experiment invariant (dense per-token freedom without task prior
does not convert into CER). Authoritative summary: `STAGE10_SUMMARY.md`.

**Next: Stage 11 — causal architecture transfer.** Tests whether the
discovered mechanisms generalise to causal Mamba-2 and causal Linear
Attention, with pre-registered differential predictions keyed to each
architecture's structural deficit. See `STAGE10_PLAN.md` §5 Phase III.

The LION bidirectional chapter runs in parallel on existing infrastructure
(vanilla LION at 0.0712 dev / 80 ep already on disk).

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

**Thesis question.** Can causal linear-time RNNs — RWKV-6, Mamba-2, Linear
Attention — be systematically understood and improved as an alternative
architectural family to Transformer causal attention, by identifying where
their expressivity bottlenecks are, discovering mechanism-level solutions,
and showing those solutions generalise across the family?

CTC ASR on LibriSpeech clean-100 at 7 M / 30 ep is the controlled probe —
chosen because it gives a clean, reproducible spine for mechanism
comparison. It is not the research goal in itself.

**Contributions.**

1. **Mechanism-discovery on causal RWKV-6** (Stages 2–10). Catalogs where
   per-channel-decay linear recurrence saturates and what mechanism
   classes move CER. Establishes the empirical invariant:

   > *Function-class extensions aligned with a task-structural prior
   > convert into CER. Dense per-token freedom without such a prior
   > engages SGD but does not convert.*

   Invariant corroborated across five independent mechanisms (A1′, T1,
   T2, S9A, S9B) on top of the anchor, and again across three
   compositions (CB-1, CB-3, CB-7) on top of `multidil_sym` in Stage 10.

2. **Two transferable mechanism solutions discovered:**
   - **RSE-family** — block-complex transition geometry ($SO(2)^{K/2}
     \times (\mathbb{R}_+)^{K/2}$) with Rayleigh-viscosity coupling.
     Aligns with damped-oscillator / formant task prior.
   - **Multi-dilation ConvShift (symmetric)** — input-side temporal
     hierarchy with dilations {1, 2, 4, 8}. Aligns with the
     phoneme-to-syllable acoustic window (40–160 ms post-subsampling).

3. **Causal transfer study** (Stage 11). Tests whether the discovered
   mechanisms generalise to causal Mamba-2 and causal Linear Attention,
   with pre-registered differential predictions keyed to each
   architecture's structural deficit. Also tests Log-Linear Attention
   (Guo et al.) on its natural LA / Mamba-2 home, where the mechanism's
   multi-scale state claim is not absorbed by pre-existing decay
   diversity.

4. **Bidirectional adaptation — parallel chapter.** LION deploys the same
   mechanism vocabulary (ConvShift, multidil_sym, RSE) on the parallel
   bidirectional form. Quantifies the causality penalty (+0.008 CER on
   input-side mechanisms, measured cleanly via 10.3-sym vs 10.3-causal)
   and what bidirectional access recovers.

**Why causal-first is deliberate.** Causal and bidirectional RNNs serve
different deployment regimes: causal for streaming, online inference,
real-time, and autoregressive generation; bidirectional for offline
full-document encoding. The causal-RNN expressivity claim is the thesis
contribution. LION extends the mechanism vocabulary to bidirectional — the
same template the LION paper itself used to extend RWKV — but is not
positioned as the main result.

## Methodology

### Generating and filtering candidate mechanisms

Candidate mechanisms come from three primary sources, all legitimate:

- **Literature review.** Reading recent linear-time / sub-quadratic
  attention work, the RWKV / Mamba / LION / DeltaNet / Log-Linear
  lineages, and architecture-comparison papers. Documented in
  `TODO_FUTURE_IDEAS.md` with paper-by-paper mechanism decompositions
  and the Phase-4 cross-paper synthesis. **This is how we found the
  multi-dilation ConvShift win (Paper 7), how RSE's complex-pole
  framing was grounded in prior work, and how Log-Linear entered the
  queue as the natural-home test for LA.** Paper reviews are a *primary
  generator* of candidates and are not themselves an error path.
- **Internal diagnostics.** Mobility probes, depth-graded parameter
  patterns, spectral observations on trained checkpoints. Example:
  Stage-2 `gen2` α₁ depth pattern motivated Stage-4 `rse_depth`;
  Stage-3 θ budget saturation motivated `rse_strong`.
- **First-principles analysis.** Function-class extension arguments
  (RSE as Lie group replacement), physical priors (Rayleigh viscosity),
  task-structural priors (phoneme-scale RF for multidil).

The discipline is the **filtering step before running**: cross-reference
every candidate against `stages_2_9_summary.md`, `STAGE10_SUMMARY.md`,
and the Phase-4 synthesis table. If the candidate's mechanism family
and deployment shape match an already-characterised closed cell, require
a specific argument for why *this* run will differ from the closed
precedent before committing compute. If no such argument exists, do
not run — regardless of whether the candidate came from a fresh arXiv
paper, an internal idea, or a composition of known wins.

### Run-level discipline

- **30-epoch schedule is deliberate.** It exists for rapid iteration on
  mechanism discovery. RSE (Stage 3), depth/strong budget refinement
  (Stage 4), and Rayleigh viscosity coupling (Stage 5) were all
  discovered and validated within days because of this schedule. Full
  80-ep reference numbers exist for select runs (vanilla LION at 0.0712)
  but are not the iteration schedule.
- **Single-seed is the discovery default.** Multi-seed (3 seeds) is
  required only for BREAK-band claims per `STAGE10_PLAN.md` §8.2.
  Single-seed patterns outside noise are treated as evidence but flagged.
- **Exact-reduction-at-init discipline.** Every new mechanism must reduce
  to an already-characterised baseline at step 0 (bit-exact or fp32
  noise floor). This keeps attribution clean when gains are 1–3σ. Stages
  7A, 8, and 9 repeatedly demonstrated this matters for distinguishing
  mechanism-engagement from reparameterisation-absorption.
- **Matched-epoch tracking.** Every run logs dev CER at epochs
  {5, 10, 15, 20, 25, 30} against its comparison cohort. Matched-epoch 15
  halt criterion fires if the run is ≥ +0.006 behind its primary reference.
- **Pre-registered diagnostic probes** per mechanism (`diagnostics.json`),
  logged at ep 15 and ep 30 minimum. Mandatory per `STAGE10_PLAN.md` §7.5.

## Thesis Structure

| Chapter | Scope | Schedule | Main reference |
|---|---|---|---|
| 1 — Background + probe setup | Transformer reference, vanilla baselines, dataset, CTC pipeline | 30 ep | `RESULTS.md` |
| 2 — Causal RWKV-6 mechanism discovery | Stages 2–10; the invariant; RSE and multidil as the two wins | 30 ep | `stages_2_9_summary.md`, `STAGE10_SUMMARY.md` |
| 3 — Causal architecture transfer | Stage 11; Mamba-2 and LA transfers with differential predictions | 30 ep | `STAGE10_PLAN.md` §5 Phase III |
| 4 — Bidirectional adaptation | LION as parallel bidirectional form; same mechanism vocabulary | 80 ep for final numbers | LION outputs in `outputs/exp09_lion_seed42/`, `outputs/lucid_exp05_lion_convshift_seed42/` |
| 5 — Synthesis | The "function-class extension + task-structural prior" principle; what transfers across architectures; what is specific to each | — | this file + summaries |

Stage 12+ LION-specific experiments (lion_lucid_chunked, lion_loglinear
mirror-Fenwick) are the extended chapter-4 work, gated on chapter 2 + 3
core mechanisms landing cleanly.

## Lessons Learned (live record — update as new evidence lands)

### Methodological errors recognised

*Framing note: the source of a candidate mechanism (paper review, internal
diagnostic, composition idea) is never the error. Literature review in
particular has been a primary source of wins, including the Stage-10
`multidil_sym` result, RSE's complex-pole framing, and the upcoming
Log-Linear Stage-11 transfer. The errors below are all about what happens
**after** the candidate exists — in how candidates are filtered and how
results are interpreted.*

1. **Failure to cross-reference candidates against accumulated evidence
   before running — the core recurring error.** Prior stages had
   already established that certain mechanism classes, deployment
   shapes, and composition patterns were closed at this spine. New
   candidates whose signature matched a closed cell were run anyway,
   without an explicit argument for why their outcome would differ.
   Each of the bullets below is the same meta-pattern instantiated on
   different stages:

   - **Stage 6 closed the quadratic-lift Family-D cluster**
     (`hadamard_n2`, `qtail`, `qtail_lowrank_all` all at ~0.124).
     Stage 10 ran `pom_vlift` — a different parametrisation of the
     same quadratic function class — and got the same saturation
     number. Directly predicted by Stage-6 evidence.
   - **Stage 6 closed the channel-reweight axis** via the `rmsnorm` /
     `hadamard_n2` null. Stage 10.4 `chanmix_bypass` was the same axis
     with fewer parameters; PLATEAU as predicted.
   - **Stages 7A / 8 / 9 established the cross-experiment invariant**
     (*dense per-token freedom on top of the anchor does not convert
     into CER*) across five independent mechanisms (A1′, T1, T2, S9A,
     S9B). Stage 10 CB-sprint (CB-1, CB-3, CB-7) ran the same
     deployment shape on top of a new base (`multidil_sym`) and
     reproduced the same null. Invariant extension is useful thesis
     material; re-confirming it with three more runs was not the
     highest-value allocation of Stage-10 compute.
   - **Phase-4 cross-paper synthesis** (`TODO_FUTURE_IDEAS.md` §4.9)
     had already done the filtering work — it pre-registered that
     only 3 of 9 reviewed papers targeted axes not already
     characterised in Stages 2–9. The error was running several of
     the flagged-as-redundant candidates despite the synthesis saying
     their prior was PLATEAU-band.
   - **Stage 6 paired-pole line plateaued** at `p2rse_softmax` 0.1220.
     Phase 2b independent-λ was a predictable null from that
     evidence; it regressed 15σ.

   **The discipline that prevents this:** before approving a new run,
   check whether its mechanism family and deployment shape match an
   already-characterised closed cell in `stages_2_9_summary.md`, the
   Phase-4 synthesis, or `STAGE10_SUMMARY.md`. If yes, require a
   specific argument for why this run's outcome will differ from the
   closed precedent. If no argument, do not run — regardless of source.

2. **Framing drift from mechanism study to ceiling hunt.** "0.115
   ceiling" language made it easy to forget that vanilla LION was
   already at 0.071 in the same codebase. The causal RWKV-6 ceiling
   is a chapter-2 data point, not a spine-absolute minimum. Don't
   confuse architecture-scoped ceilings with spine-scoped ones. This
   compounded error 1: once the ceiling narrative took hold, it
   motivated additional runs aimed at "finding the mechanism that
   breaks 0.115" — which led back into the closed cells.

3. **Mechanism-specific diagnostic probes not saved.** `STAGE10_PLAN.md`
   §7.5 made diagnostics a prerequisite for any §9 verdict. No
   Stage-10 run produced a `diagnostics.json`. Every §3 "mechanism
   engaged / absorbed" claim is an unverified narrative until the
   probe is backfilled from checkpoint. Procedural gap to close.

4. **Reference baselines contaminated by schedule mismatch.** Stage-10
   plots used `exp02_rwkv6_seed42` (80-ep schedule) as the vanilla
   reference, biasing Δ-to-vanilla by +0.0085 at ep 30. The analysis
   document names this caveat but doesn't regenerate. Fix before
   thesis write-up.

5. **Single-seed verdicts treated as conclusions for MARGINAL+ results.**
   `STAGE10_PLAN.md` §8.2 requires multi-seed for BREAK claims.
   PLATEAU-band results are defensible single-seed; MARGINAL claims
   (10.3-sym at 0.1153 ties abs-best) require multi-seed before
   thesis-grade citation.

### What actually drives progress on this spine

1. **Exact reduction at init is the load-bearing discipline.** Every
   winning mechanism (RSE, depth/strong budget, viscosity, ConvShift,
   multidil_sym) had a clean zero-regression contract. Every
   AMBIGUOUS-ENGAGED null (A1′, T1, T2, S9A, S9B) also had one — which
   is how we can attribute their null to "mechanism activated but didn't
   help," not "implementation was wrong." Discipline distinguishes
   engaged-null from bugs.

2. **Matched-epoch tracking exposes what final CER hides.** Stage 9
   halt-at-ep-15 was possible only because per-epoch panels flagged
   sustained underperformance vs cohort. CB-1's saturation-tie with
   `multidil_sym` shows in the mid-training crossover, not the endpoint.
   Always plot per-epoch.

3. **Task-structural prior alignment, not parameter capacity, is the
   axis.** Wins: RSE (complex poles ≈ formants), multidil_sym
   (dilations = phoneme window), viscosity (Rayleigh physical prior).
   Losses: quadratic lifts (more parameters, same function class),
   dense per-token transition freedom (more parameters, unaligned).
   Parameter count is not predictive of whether a mechanism converts.

4. **Constraint removal is as valid as freedom addition.** Stage 4's
   depth-graded and strong budgets both improved on Stage 3 — not by
   adding parameters, but by undoing the π/4 under-allocation at deep
   layers. When a previous stage imposes a mistaken constraint, lifting
   it is the productive move.

5. **Formal novelty ≠ productive novelty.** M²RNN (Paper 9) crosses
   TC⁰→NC¹ formally. Deployed sparingly at one layer with λ_h zero-init,
   it's optimisation-inaccessible. Novelty of the function class is
   necessary but not sufficient — the deployment must be identifiable
   and the prior must match the task.

6. **The cross-experiment invariant is a thesis-grade finding, not a
   failure.** A1′ + T1 + T2 + S9A + S9B + CB-1 + CB-3 + CB-7 converging
   to the same engaged-null band across four mechanism families and two
   base mechanisms is stronger evidence than any single null. Cite as
   such.

### Real bottlenecks on causal RWKV-6 (as of Stage 10 close)

**Bottlenecks that, when addressed, produce CER gain:**

1. **Input-side local temporal receptive field.** RWKV-6's diagonal WKV
   scan has no explicit local mixing. Default ±1 frame (single-dilation
   ConvShift) is insufficient. Expanding to ±8 frames via multi-dilation
   {1, 2, 4, 8} matches phoneme-to-syllable acoustic scale and moves
   CER by ~−0.01. Whether expansion beyond ±8 frames continues to help
   is **unknown** (CB-2 wide4 never ran).

2. **Transition operator Lie group.** Real diagonal decay $(\mathbb{R}_+)^K$
   underserves complex-pole (formant) dynamics. Extension to
   $SO(2)^{K/2} \times (\mathbb{R}_+)^{K/2}$ via RSE is identifiable and
   productive. Higher-order extensions (polar non-normal, Cayley
   orthogonal, non-linear state) do not help at this scale.

3. **Rotation budget and damping coupling.** Uniform π/4 clip (Stage 3
   default) is too tight at deep layers. Remedies: uniform π/2 (strong),
   depth-graded (π/8 → π/2), Rayleigh viscosity $\lambda_{eff} = \lambda
   + \eta\theta^2$. Any one lifts the mechanism; combined
   (strong+viscosity) is the current anchor at 0.1185 dev.

**Axes that tested as NOT binding at this scale (contrary to paper priors):**

1. **Feature-side quadratic expressivity.** Four independent parametrisations
   (`hadamard_n2`, `qtail`, `qtail_lowrank_all`, `pom_vlift`) saturate at
   dev ~0.124. Cross-channel Kronecker gives ~1σ lift over diagonal;
   everything beyond is noise-band.

2. **Per-token state non-linearity.** M²RNN's tanh-state is redundant with
   RWKV-6's ChannelMix σ·ReLU² token-local non-linearity.

3. **Dense per-token transition freedom.** Tested in five forms (readout
   gauge A1′, recurrent delta T1, dense non-normal T2, sparse
   non-normal S9A/B); all engaged-null.

4. **Structural multi-scale readout (direct-sum decomposition).** Log-Linear
   on RWKV-6 at 30 ep tied vanilla (PLATEAU). RWKV-6's per-channel decay
   diversity already provides continuous multi-scale; discrete buckets
   don't add a new axis on this architecture. **NOTE:** the mechanism's
   native home is LA / Mamba-2 where decay diversity doesn't exist
   structurally; Stage 11 retests it there.

5. **Content-conditional input-side mixing.** CB-3 (softmax-gated α_d)
   tied `multidil_sym` within σ. Fixed per-layer α_d is sufficient.

6. **Composition of temporal mechanisms.** RSE × multidil_sym ties
   multidil_sym. Input-side and transition-side improvements target the
   same temporal-context deficit by different routes; they do not add
   orthogonally on RWKV-6 causal.

**Open unknowns at Stage 10 close:**

- Input-side RF expansion beyond ±8 frames (CB-2 wide4 / dense) — cheap probe, unrun.
- Frontend architecture (CB-5 v2 did not converge cleanly; rescue diagnosis mid-flight).
- Modern ChannelMix activations (SwiGLU / GeGLU) — untested on this spine.
- Whether any of these bottlenecks transfer differentially to Mamba-2 / LA (Stage 11).
