
---

# Master Plan v2.1 — Causal Linear-Time RNN Expressivity Thesis

*Locked 2026-04-25. Executable. Edit only via explicit decision.*

---

## 1. Thesis claim

> *Mechanism-prior alignment is the predictive criterion for when causal linear-time RNN extensions convert into measurable gain. Mechanisms transfer in proportion to architectural deficit, with architecture-specific adaptation in how they engage. The cross-experiment invariant — dense per-token freedom does not convert without prior alignment — extends across architectures, modes, base mechanisms, and tasks.*

---

## 2. Architecture × mode matrix (6 cells)

| # | Architecture | Mode | Decay character |
|---|---|---|---|
| 1 | RWKV-6 | causal | per-channel data-dependent (WKV) |
| 2 | RWKV-6 | LION | bidirectional, parallel T×T |
| 3 | Mamba-2 | causal | selective Δt-modulated continuous |
| 4 | Mamba-2 | LION | bidirectional via unified LION wrapper |
| 5 | Linear Attention | causal | none (pure accumulator) |
| 6 | Linear Attention | LION | bidirectional via unified LION wrapper |

Engineering: existing `lion` mode in `RWKV6TimeMix` is the canonical bidirectional wrapper. Same wrapper extended to Mamba-2 and LA — single unified bidirectional implementation.

---

## 3. Mechanism inventory — three winners, one variant per architecture

| Axis | Mechanism | RWKV-6 | Mamba-2 | LA |
|---|---|---|---|---|
| 1 | **multidil_v2** (ConvShift, k=3, dilations {1,2,4,8}, symmetric, init-fix α_{d>1}=0.01) | unified | unified | unified |
| 2 | **LUCID** | `lucid_chunked` (standard chunked preconditioner) | `lucid_c` (Mamba-2 C-correlation adaptation per `MAMBA2_LUCID_ADAPTATION.md`) | `lucid` (parallel attention native form) |
| 3-sub | **rse_strong_viscosity** (uniform π/2 budget, LoRA 48, Rayleigh η·θ²) | unified form | applied to Mamba-2 SSM transition | applied to LA decay |

multidil_v2 and rse_strong_viscosity are unified across architectures. LUCID has architecture-specific deployment because the kernel structure differs.

---

## 4. Cell structure per (architecture, mode, scale)

5 cells:

1. **vanilla**
2. **multidil_v2** (axis-1)
3. **LUCID** (architecture-specific variant per §3, axis-2)
4. **rse_strong_viscosity** (axis-3-sub)
5. **Composition** (pre-registered per §5)

---

## 5. Pre-registered compositions (cell 5)

| Architecture | Composition | Rationale |
|---|---|---|
| RWKV-6 causal | LUCID × multidil_v2 | P7 ceiling |
| RWKV-6 LION | LUCID × multidil_v2 | LUCID's native parallel home |
| Mamba-2 causal | LUCID × multidil_v2 | LUCID productive on Mamba-2; RSE NULL |
| Mamba-2 LION | LUCID × multidil_v2 | same logic |
| LA causal | RSE × multidil_v2 | RSE was LA's BREAK |
| LA LION | RSE × multidil_v2 | extends LA's productive composition |

**Rule:** multidil_v2 = universal axis-1 base; second mechanism = whichever transferred best on that architecture's causal results.

---

## 6. Probes

| Probe | Modality | Length regime | Role |
|---|---|---|---|
| LibriSpeech clean-100 | audio (CTC) | short | Main mechanism-attribution spine |
| Common Voice EN 100h | audio (CTC) | short | Pilot-gated within-audio robustness |
| MQAR | synthetic recall | $T \in \{64, 256, 1024\}$ | Axis-2 isolation + length scaling |

State-tracking probes ($S_n$, Dyck-$k$) are **out of scope** — none of the three mechanisms target the diagonal-vs-rank-1-update transition. Future-work paragraph in writeup.

---

## 7. Training schedule (single matched budget)

| Parameter | Value |
|---|---|
| Epochs | **50** (single budget across all ASR runs) |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01 |
| Scheduler | Cosine + 1000-step warmup |
| SpecAugment | LD policy (freq=27, time=100, 2+2 masks) |
| Batch | max 300 s total duration |
| Grad clip | 5.0 |
| Patience | 15 epochs |
| Seed | 42 |

MQAR uses step-based budget per existing convention.

---

## 8. Scale schedule

| Scale | When | Cells |
|---|---|---|
| 7M | Always | All 60 base cells |
| 14M | Always | All 60 base cells |
| 30M | **Conditional** — triggered if 7M→14M shows ranking shift among single mechanisms on any architecture, OR if mechanism gains grow with capacity | Best mechanism per causal architecture only (3 cells) |

---

## 9. LibriSpeech main matrix (sequential first step)

| Slice | Runs |
|---|---:|
| 7M × 6 archs × 5 cells | 30 |
| 14M × 6 archs × 5 cells | 30 |
| **LibriSpeech base** | **60** |
| 30M conditional | +3 |

---

## 10. Common Voice — pilot-gated

**Run after LibriSpeech matrix completes.**

### Step A: pilot (4 runs, 7M, 50 ep)

| Run |
|---|
| `rwkv6` vanilla on CV |
| `rwkv6_lucid_multidil_v2` on CV |
| `linear_attn` vanilla on CV |
| `linear_attn_rse_strong_viscosity` on CV |

### Step B: scope decision

| Pilot outcome | CV scope | Runs |
|---|---|---:|
| CV Δ >30% larger relative to LibriSpeech Δ | Full CV matrix mirrors LibriSpeech | +60 |
| CV Δ within 25% of LibriSpeech | Targeted: vanilla + best mechanism + composition per (arch, mode) × 2 scales | +24 |
| CV Δ smaller | Pilot-only as cross-distribution check | +0 |

---

## 11. MQAR — length sweep

### Engineering prerequisite
- `mamba-ssm` CUDA backend integration
- Delta-rule forward-path audit vs Zoology reference

### Cohort per length (seed 42)

| Backbone |
|---|
| transformer_causal |
| rwkv6 |
| rwkv6_lucid |
| rwkv6_multidil_v2 |
| rwkv6_lucid_multidil_v2 |
| mamba2 (CUDA) |
| mamba2_lucid_c |
| linear_attn |
| linear_attn_multidil_v2 |
| linear_attn_rse |

**10 backbones × 3 lengths = 30 runs.**

---

## 12. Cross-cutting requirements (mandatory on every run)

Diagnostics at ep {1, 5, 15, 30, best, final}.

Four-cell engagement classifier (engaged ✓/✗ × helpful ✓/✗) auto-computed from mobility + CER delta at end of every run.

---

## 13. Per-run output directory specification

Every run writes to `outputs/<exp_name>_seed42/` with the following exact structure. **All files mandatory unless noted.**

```
outputs/<exp_name>_seed42/
├── config.yaml
├── meta.json
├── history.csv
├── steps.parquet
├── diagnostics_ep{1,5,15,30,best,final}.json   (6 files)
├── eval_full.json
├── predictions_sample.txt
├── best_model.pt
└── run.log
```

### `config.yaml`
Full `ExperimentConfig` snapshot. No deltas — the complete config that produced this run, suitable for direct replay.

### `meta.json`
Run-level metadata. Single JSON object with these fields:

```json
{
  "run_id": "<uuid>",
  "exp_name": "<exp_name>",
  "backbone_name": "<backbone string>",
  "architecture": "rwkv6 | mamba2 | linear_attn",
  "mode": "causal | lion",
  "scale_M": 7 | 14 | 30,
  "mechanism_flags": {
    "multidil_v2": bool,
    "lucid": bool,
    "rse_strong_viscosity": bool
  },
  "composition_label": "vanilla | single:<mech> | composition:<P7|CB1v2|...>",
  "dataset": "librispeech_clean100 | common_voice_en_100h",
  "epochs_planned": 50,
  "epochs_completed": int,
  "seed": 42,
  "start_time_iso": "<ISO 8601>",
  "end_time_iso": "<ISO 8601>",
  "duration_seconds": float,
  "git_commit_hash": "<hash>",
  "git_branch": "<branch>",
  "gpu_model": "<e.g. RTX PRO 6000 Blackwell>",
  "gpu_count": int,
  "python_version": "<x.y.z>",
  "torch_version": "<x.y.z>",
  "cuda_version": "<x.y>",
  "final_status": "completed | early_stopped | crashed",
  "final_dev_cer": float,
  "final_test_cer": float,
  "best_dev_cer": float,
  "best_dev_cer_epoch": int,
  "param_count_total": int,
  "param_count_encoder": int,
  "param_count_mechanism_extras": int
}
```

### `history.csv`
One row per epoch. Columns (exact names, order doesn't matter):

| Column | Type | Description |
|---|---|---|
| `epoch` | int | epoch index, 1-based |
| `train_loss_avg` | float | mean batch loss this epoch |
| `train_loss_min` | float | min batch loss this epoch |
| `train_loss_max` | float | max batch loss this epoch |
| `dev_loss` | float | dev set CTC loss |
| `dev_cer` | float | dev set character error rate (primary) |
| `dev_wer` | float | dev set word error rate |
| `dev_per_utt_cer_p50` | float | median per-utterance CER on dev |
| `dev_per_utt_cer_p90` | float | 90th percentile per-utterance CER on dev |
| `dev_per_utt_cer_p99` | float | 99th percentile (tail) per-utterance CER on dev |
| `learning_rate_avg` | float | mean LR across this epoch's steps |
| `epoch_wallclock_s` | float | seconds for the epoch |
| `grad_norm_avg` | float | mean gradient L2-norm across steps |
| `grad_norm_max` | float | max gradient L2-norm across steps |
| `param_l2_norm_total` | float | total model L2 norm at end of epoch |
| `samples_seen` | int | cumulative training samples seen |
| `time_iso` | string | epoch completion timestamp (ISO 8601) |

### `steps.parquet`
Sampled per-step metrics, one row per 100 training steps. Parquet for compact storage:

| Column | Type | Description |
|---|---|---|
| `step` | int64 | global step counter |
| `epoch` | int | current epoch |
| `train_loss` | float | step loss |
| `learning_rate` | float | LR at this step |
| `grad_norm` | float | gradient L2-norm at this step |
| `iter_ms` | float | wall-clock ms for this iteration |
| `peak_memory_mb` | float | GPU peak memory in MB |

### `diagnostics_ep{1,5,15,30,best,final}.json`

Six files per run. Each file structure:

```json
{
  "run_id": "<uuid>",
  "epoch": int,
  "phase": "ep1 | ep5 | ep15 | ep30 | best | final",
  "model_state_path": "<path to checkpoint at this point>",

  "param_mobility": {
    "<param_name>": {
      "frobenius_norm": float,
      "delta_from_init_frobenius": float,
      "max_abs": float,
      "mean_abs": float,
      "frac_zero": float,
      "per_layer_stats": [...]    // length n_layers
    }
  },

  "gradient_signal": {
    "<param_name>": {
      "grad_norm_running_avg": float
    }
  },

  "activation_magnitudes": {
    "<branch_name>": {
      "mean_l2": float,
      "max_l2": float
    }
  },

  "spectral": {
    "max_spectral_radius_per_layer": [n_layers],
    "frac_lambda_above_0.99_per_layer": [n_layers],
    "frac_lambda_above_1.0_per_layer": [n_layers],
    "non_normality_score_per_layer": [n_layers]   // null if not applicable
  },

  "mechanism_specific": {
    "rse": {                        // present iff rse_strong_viscosity active
      "theta_clip": float,
      "theta_realised_p50_per_layer": [n_layers],
      "theta_realised_p99_per_layer": [n_layers],
      "theta_budget_saturation_frac_per_layer": [n_layers],
      "im_re_ratio_per_layer": [n_layers],
      "viscosity_eta_per_layer_per_head": [[n_layers][n_heads]],
      "lambda_eff_minus_lambda_raw_per_layer": [n_layers]
    },
    "multidil": {                   // present iff multidil_v2 active
      "alpha_d_per_layer": [[n_layers][4]],   // 4 dilations
      "alpha_engagement": "single | multi",   // computed: any α_d>1 > 0.05?
      "branch_filter_l2_per_layer_per_dilation": [[n_layers][4]]
    },
    "lucid": {                      // present iff LUCID active
      "tau_per_head": [n_heads],
      "preconditioner_eigenvalue_p50": float,
      "preconditioner_eigenvalue_p99": float,
      "ill_conditioning_score": float,
      "epsilon_floor_frac": float
    }
  },

  "engagement_summary": {
    "mechanism_active": [...],      // list of mechanism names from mechanism_flags
    "engaged": bool,                // |delta_from_init_frobenius| > threshold for any active mechanism param
    "helpful": bool,                // dev_cer < vanilla dev_cer - σ at this epoch
    "verdict": "absorbed | engaged-helpful | engaged-null"
  }
}
```

### `eval_full.json`
Final test set evaluation. Single JSON object:

```json
{
  "test_cer": float,
  "test_wer": float,
  "test_loss": float,
  "test_n_utterances": int,
  "per_utterance_cer": [...],         // length test_n_utterances
  "per_utterance_wer": [...],
  "per_utterance_length_frames": [...],
  "per_utterance_id": [...],          // dataset utterance IDs
  "wer_by_speaker": {...},            // present only for Common Voice
  "wer_by_accent": {...}              // present only for Common Voice (if accent labels exist)
}
```

### `predictions_sample.txt`
First 100 dev set predictions. Tab-separated, one utterance per line:

```
<utterance_id>\t<target>\t<prediction>\t<cer_for_this_utterance>
```

### `best_model.pt`
PyTorch checkpoint at best dev CER. Contains: model state dict, optimizer state (optional), epoch, dev_cer, full config.

### `run.log`
Training log, rotated if it exceeds 100 MB. Captures stdout/stderr from training process.

---

## 14. Supplementary evidence (no new compute, all from existing on-disk runs)

Reported in a "Closed Cells" section. Each strengthens the cross-experiment invariant via what didn't work and why:

- **RSE ablation chain** (Stages 3–5): rse → rse_depth / rse_strong → +viscosity. Establishes budget-refinement and physical-prior contributions independently.
- **Stage 6 quadratic-lift cluster** (hadamard_n2, qtail, qtail_lowrank, pom_vlift): four parametrisations saturate at dev ~0.124. Axis-5a ceiling.
- **Engaged-null catalog** (Stages 7A, 8 T1, 8 T2, 9 A, 9 B, 10 CB-3, 10 CB-7): seven instances of dense per-token freedom engaging without converting.
- **Stage 11 Mamba-2 novelty gate**: structurally inert at architecture's operating point. Distinguishes scale-mismatch null from task-prior null.
- **P8 same-axis composition saturation** (LUCID × RSE × multidil_v2 = 0.0969 vs P7 0.0921, ~3σ worse): brief mention in the closed-cells section. Expanded paragraph **conditional** on writeup time slack.

---

## 15. Engineering tasks (parallelisable with compute)

| Task |
|---|
| Unified LION wrapper extension to Mamba-2 and LA |
| `mamba-ssm` CUDA backend integration |
| Common Voice EN 100h subset data pipeline |
| Diagnostics module (per-mechanism probes + four-cell engagement classifier) per §13 |
| 14M and (conditional) 30M configs across 3 architectures |
| Delta-rule forward-path audit vs Zoology |

---

## 16. Run count totals

| Bucket | Mandatory | Conditional |
|---|---:|---:|
| LibriSpeech base matrix | 60 | — |
| LibriSpeech 30M (ranking-shift trigger) | — | 3 |
| Common Voice pilot | 4 | — |
| Common Voice expansion | — | 24 / 60 |
| MQAR length sweep | 30 | — |
| **Total mandatory** | **94** | |
| **+ targeted CV + 30M** | | **+27** |
| **+ full CV + 30M** | | **+63** |

---

## 17. Compute estimate

50 ep training, ~2 GPU-h average per LibriSpeech / CV run, MQAR negligible:

| Bucket | GPU-h |
|---|---:|
| LibriSpeech base matrix (60 × ~2 h) | ~140 |
| LibriSpeech 30M conditional | ~15 |
| CV pilot | ~10 |
| CV expansion targeted | ~50 |
| CV expansion full | ~120 |
| MQAR | ~6 |
| **Mandatory** | **~156 GPU-h** |
| **+ targeted CV + 30M** | **~221 GPU-h** |
| **+ full CV + 30M** | **~291 GPU-h** |

---

## 18. Locked decisions

| # | Decision |
|---|---|
| 1 | 50-epoch matched budget across all ASR runs |
| 2 | 7M and 14M as main scales |
| 3 | 30M conditional on ranking-shift trigger |
| 4 | 6 architecture-mode cells (3 causal + 3 unified-LION-bidirectional) |
| 5 | 3 main mechanisms: multidil_v2, LUCID (arch-specific), rse_strong_viscosity |
| 6 | Pre-registered compositions per cell (§5) |
| 7 | LibriSpeech full matrix first; CV pilot-gated after |
| 8 | MQAR length sweep $T \in \{64, 256, 1024\}$ |
| 9 | $S_n$ / state-tracking probes — out of scope, future-work paragraph |
| 10 | Diagnostics mandatory at ep {1, 5, 15, 30, best, final} per §13 |
| 11 | Per-run output directory spec per §13, mandatory on every run |
| 12 | Supplementary evidence drawn from existing on-disk runs |
| 13 | P8 saturation: brief mention; expanded paragraph conditional on time |

---

## 19. Open / conditional items

| # | Item | Trigger |
|---|---|---|
| A | 30M scale runs | 7M→14M ranking shift among single mechanisms, or capacity-conditional gain growth |
| B | Common Voice expansion scope | CV pilot Δ vs LibriSpeech Δ |
| C | MQAR T=4096 | T=1024 cohort produces clean separation |
| D | P8 saturation paragraph expansion | time slack in writeup phase |
| E | LION 80-ep reference run on best composition | one-off after main matrix; chapter-4 community-comparison |

---

## 20. What this thesis claims at the end

A single matrix:

|  | Causal 7M / 14M / *30M* | LION 7M / 14M | LibriSpeech | Common Voice |
|---|---|---|---|---|
| RWKV-6 | ✓ | ✓ | ✓ | pilot-gated |
| Mamba-2 | ✓ | ✓ | ✓ | pilot-gated |
| Linear Attention | ✓ | ✓ | ✓ | pilot-gated |

Three single mechanisms × six architecture-mode cells × two scales × (one or two) datasets, plus axis-2 (MQAR) length sweep.

The findings the thesis stakes:
1. **Three productive mechanisms** with three distinct transfer patterns (uniform, conditional, asymmetric).
2. **Architecture-deficit-proportional ordering** across multidil_v2 and architecture-specific adaptation across RSE+viscosity.
3. **Cross-experiment invariant** with eight independent confirmations.
4. **Same-axis composition saturation** (P8) extending invariant to composition setting.
5. **Bidirectional benefit per mechanism** quantified via causal-vs-LION delta.
6. **Cross-task validation** via MQAR (axis-2 isolation, length scaling).
7. **Cross-distribution validation** via Common Voice (pilot-gated).
8. **Cross-scale persistence** via 7M→14M (and conditional 30M) on identical mechanism cells.

Future work paragraph in writeup: state-tracking axis ($S_n$, Dyck-$k$) deferred to mechanisms targeting diagonal-vs-rank-1-update transition (delta rule, DeltaNet).

---

*End master plan v2.1. Five days. We've got this.*
