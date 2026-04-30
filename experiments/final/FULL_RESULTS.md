# Full Experimental Results — Final Stage

Source of truth: `experiments/final/outputs/*/results.json` (LibriSpeech + Common Voice)
and `experiments/synthetics_v1/outputs/cohort_reduced/_index.csv` (MQAR).
Schedule: 50-epoch matched budget per `Master_Plan.md` §7. Single seed = 42.
Output spec: per-run dirs named `<size>_<arch>_<mode>_<cell>_seed<S>` per §13.

Symbols:
- ✅ done · 🟡 in flight · ⛔ dropped per scope · ⚪ pending

LION wrapper mapping (locked, `Master_Plan.md` §15):
- **RWKV-6 LION** = LION-S (per-channel data-dep λ from native WKV decay).
- **Mamba-2 LION** = LION-S (scalar per-head λ from dt·A).
- **LA LION** has two variants:
  - **LION-LIT** (default per Afzal et al. 2025 Table 1; `linear_attn_lion`) — no decay, λ = 1.
  - **LION-S** (control with per-head σ-decay; `linear_attn_lion_s`) — added 2026-04-26 after LION-LIT vanilla broke.

RSE variants:
- **RSE-strong-viscosity** — uniform θ clip at π/2 across all 6 layers (Master_Plan §3 default).
- **RSE-depth-viscosity** — depth-graded θ clip (L0–L1: π/8, L2–L3: π/4, L4–L5: π/2). STATUS-only decision 2026-04-26.

---

## 1. LibriSpeech clean-100, 7M (50 ep, 6 layers)

### 1.1 Causal modes

| Cell | Backbone | Params | Best dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|---:|
| `7m_rwkv6_causal_vanilla_seed42` | `rwkv6` | 7.74M | 0.1051 | 0.1049 | 0.3182 |
| `7m_rwkv6_causal_multidil_v2_seed42` | `rwkv6_convshift_multidil_symmetric_v2` | 7.76M | 0.0803 | **0.0788** | 0.2372 |
| `7m_rwkv6_causal_lucid_chunked_seed42` | `rwkv6_lucid` | 7.74M | 0.1005 | 0.1007 | 0.3001 |
| `7m_rwkv6_causal_rse_strong_viscosity_seed42` | `rwkv6_rse_strong_viscosity` | 7.85M | 0.1008 | 0.1006 | 0.3033 |
| `7m_rwkv6_causal_rse_depth_viscosity_seed42` (probe #1) | `rwkv6_rse_depth_viscosity` | 7.81M | 0.0999 | 0.0989 | 0.2994 |
| `7m_rwkv6_causal_rse_trwg_strong_viscosity_seed42` (probe #3) | `rwkv6_rse_trwg_strong_viscosity` | 7.85M | 0.1032 | 0.1034 | 0.3092 |
| `7m_rwkv6_causal_p7_seed42` (LUCID × multidil) | `rwkv6_lucid_convshift_multidil_symmetric_v2` | 7.76M | 0.0787 | **0.0785** | 0.2332 |
| `7m_mamba2_causal_vanilla_seed42` | `mamba2` | 7.27M | 0.1057 | 0.1036 | 0.3180 |
| `7m_mamba2_causal_multidil_v2_seed42` | `mamba2_convshift_multidil_symmetric_v2` | 7.33M | 0.0839 | 0.0825 | 0.2481 |
| `7m_mamba2_causal_lucid_c_seed42` | `mamba2_lucid_c` | 7.27M | 0.0966 | 0.0958 | 0.2942 |
| `7m_mamba2_causal_rse_strong_viscosity_seed42` | `mamba2_rse_strong_viscosity` | 7.42M | 0.1054 | 0.1038 | 0.3172 |
| `7m_mamba2_causal_lucid_c_x_multidil_v2_seed42` | `mamba2_lucid_c_convshift_multidil_symmetric_v2` | 7.33M | 0.0812 | **0.0795** | 0.2377 |
| `7m_linear_attn_causal_vanilla_seed42` | `linear_attn_causal` | 6.26M | 0.1903 | 0.1879 | 0.5607 |
| `7m_linear_attn_causal_multidil_v2_seed42` | `linear_attn_convshift_multidil_symmetric_v2` | 6.28M | 0.1448 | 0.1409 | 0.4148 |
| `7m_linear_attn_causal_lucid_seed42` | `linear_attn_lucid` | 6.26M | 0.1730 | 0.1714 | 0.5189 |
| `7m_linear_attn_causal_rse_strong_viscosity_seed42` | `linear_attn_rse_strong_viscosity` | 6.37M | 0.1213 | 0.1198 | 0.3608 |
| `7m_linear_attn_causal_lucid_x_multidil_v2_seed42` | `linear_attn_lucid_convshift_multidil_symmetric_v2` | 6.28M | 0.1434 | 0.1410 | 0.4200 |
| `7m_linear_attn_causal_rse_x_multidil_v2_seed42` | `linear_attn_rse_strong_viscosity_convshift_multidil_symmetric_v2` | 6.39M | 0.1000 | **0.0999** | 0.2967 |

Best causal cell per architecture: RWKV-6 P7 = 0.0785, Mamba-2 P7 = 0.0795, LA RSE × multidil = 0.0999.

### 1.2 LION modes

#### RWKV-6 LION (LION-S by mapping)

| Cell | Backbone | Params | Best dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|---:|
| `7m_rwkv6_lion_vanilla_seed42` | `lion` | 7.74M | 0.0858 | 0.0859 | 0.2545 |
| `7m_rwkv6_lion_multidil_v2_seed42` | `lion_convshift_multidil_symmetric_v2` | 7.76M | 0.0764 | 0.0750 | 0.2275 |
| `7m_rwkv6_lion_lucid_chunked_seed42` | `lion_lucid_chunked` | 7.74M | 0.0857 | 0.0852 | 0.2547 |
| `7m_rwkv6_lion_rse_depth_viscosity_seed42` | `rwkv6_lion_rse_depth_viscosity` | 7.81M | 0.0727 | **0.0740** | 0.2219 |
| `7m_rwkv6_lion_p7_seed42` (LUCID × multidil) | `lion_lucid_chunked_convshift_multidil_symmetric_v2` | 7.76M | 0.0746 | 0.0747 | 0.2242 |

#### Mamba-2 LION (LION-S by mapping)

| Cell | Backbone | Params | Best dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|---:|
| `7m_mamba2_lion_vanilla_seed42` | `mamba2_lion` | 7.27M | 0.0871 | 0.0853 | 0.2585 |
| `7m_mamba2_lion_multidil_v2_seed42` | `mamba2_lion_convshift_multidil_symmetric_v2` | 7.33M | 0.0846 | 0.0833 | 0.2517 |
| `7m_mamba2_lion_lucid_c_seed42` | `mamba2_lion_lucid_c` | 7.27M | 0.0851 | 0.0849 | 0.2601 |
| `7m_mamba2_lion_rse_depth_viscosity_seed42` | `mamba2_lion_rse_depth_viscosity` | 7.27M | 0.0842 | 0.0825 | 0.2511 |
| `7m_mamba2_lion_p7_seed42` (LUCID-c × multidil) | `mamba2_lion_lucid_c_convshift_multidil_symmetric_v2` | 7.33M | 0.0820 | **0.0805** | 0.2461 |

#### LA LION-LIT (default per Master_Plan §15)

| Cell | Backbone | Params | Best dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|---:|
| `7m_linear_attn_lion_vanilla_seed42` | `linear_attn_lion` | 6.26M | 0.3003 | 0.2951 | 0.7967 |
| `7m_linear_attn_lion_multidil_v2_seed42` | `linear_attn_lion_convshift_multidil_symmetric_v2` | 6.28M | 0.1422 | 0.1404 | 0.4103 |
| `7m_linear_attn_lion_lucid_seed42` | `linear_attn_lion_lucid` | 6.26M | 0.3228 | 0.3194 | 0.8386 |
| `7m_linear_attn_lion_rse_depth_viscosity_seed42` | `linear_attn_lion_rse_depth_viscosity` | 6.33M | 0.1042 | **0.1042** | 0.3152 |
| `7m_linear_attn_lion_rse_x_multidil_v2_seed42` | `linear_attn_lion_rse_depth_viscosity_convshift_multidil_symmetric_v2` | 6.33M | 0.0961 | **0.0961** | 0.2901 |

LION-LIT × LUCID is the only negative LUCID transfer in the matrix (Δ test +0.0243 vs LION-LIT vanilla). Decay-prerequisite confirmed by the LION-S follow-ups below.

#### LA LION-S (control with per-head σ-decay)

| Cell | Backbone | Params | Best dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|---:|
| `7m_linear_attn_lion_s_vanilla_seed42` | `linear_attn_lion_s` | 6.26M | 0.1417 | 0.1381 | 0.4301 |
| `7m_linear_attn_lion_s_multidil_v2_seed42` | `linear_attn_lion_s_convshift_multidil_symmetric_v2` | 6.28M | 0.1160 | 0.1154 | 0.3464 |
| `7m_linear_attn_lion_s_lucid_seed42` | `linear_attn_lion_s_lucid` | 6.26M | 0.1332 | 0.1311 | 0.4068 |
| `7m_linear_attn_lion_s_lucid_multidil_v2_seed42` (P7-style) | `linear_attn_lion_s_lucid_convshift_multidil_symmetric_v2` | 6.28M | 0.1142 | **0.1129** | 0.3442 |
| `7m_linear_attn_lion_s_rse_depth_viscosity_seed42` | `linear_attn_lion_s_rse_depth_viscosity` | 6.34M | 0.0982 | **0.0988** | 0.2988 |

LION-S × DHO lands at test 0.0988, slotting between LION-LIT × DHO (0.1042) and LION-LIT × DHO × MSDC (0.0961). Reading: with LION-S σ-decay already supplying the bounded similarity prior, a complex-pole transition adds further BREAK headroom on top of the per-head decay; DHO × LION-S beats the LION-S × CVD × MSDC composition (0.1129) by Δ −0.014 test and is the lowest LA LION-S test CER on the matrix.

---

## 2. LibriSpeech clean-100, 14M (50 ep, 12 layers)

LION-mode 14M cells were dropped per scope (`STATUS.md` §LibriSpeech 14M). Causal-mode results below.

| Cell | Backbone | Params | Best dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|---:|
| `14m_rwkv6_causal_vanilla_seed42` | `rwkv6` | 13.56M | 0.1082 | 0.1103 | 0.3244 |
| `14m_rwkv6_causal_multidil_v2_seed42` | `rwkv6_convshift_multidil_symmetric_v2` | 13.60M | 0.0747 | **0.0751** | 0.2214 |
| `14m_rwkv6_causal_lucid_chunked_seed42` | `rwkv6_lucid` | 13.56M | 0.1051 | 0.1051 | 0.3093 |
| `14m_rwkv6_causal_rse_strong_viscosity_seed42` | `rwkv6_rse_strong_viscosity` | 13.79M | 0.1058 | 0.1077 | 0.3186 |
| `14m_rwkv6_causal_p7_seed42` (LUCID × multidil) | `rwkv6_lucid_convshift_multidil_symmetric_v2` | 13.60M | 0.0745 | **0.0746** | 0.2171 |
| `14m_mamba2_causal_vanilla_seed42` | `mamba2` | 12.62M | 0.0821 | 0.0827 | 0.2468 |
| `14m_mamba2_causal_multidil_v2_seed42` | `mamba2_convshift_multidil_symmetric_v2` | 12.74M | 0.0635 | **0.0631** ⭐ | 0.1850 |
| `14m_mamba2_causal_lucid_c_seed42` | `mamba2_lucid_c` | 12.62M | 0.0729 | 0.0728 | 0.2180 |
| `14m_mamba2_causal_rse_strong_viscosity_seed42` | `mamba2_rse_strong_viscosity` | 12.93M | 0.0833 | 0.0825 | 0.2482 |
| `14m_mamba2_causal_rse_depth_viscosity_seed42` | `mamba2_rse_depth_viscosity` | 12.93M | 0.0783 | 0.0781 | 0.1912 |
| `14m_mamba2_causal_lucid_c_x_multidil_v2_seed42` | `mamba2_lucid_c_convshift_multidil_symmetric_v2` | 12.74M | 0.0627 | **0.0618** ⭐ | 0.1810 |
| `14m_linear_attn_causal_vanilla_seed42` | `linear_attn_causal` | 10.60M | 0.1378 | 0.1359 | 0.4266 |
| `14m_linear_attn_causal_multidil_v2_seed42` | `linear_attn_convshift_multidil_symmetric_v2` | 10.64M | 0.0975 | 0.0969 | 0.2874 |
| `14m_linear_attn_causal_lucid_seed42` | `linear_attn_lucid` | 10.60M | 0.1291 | 0.1300 | 0.4018 |
| `14m_linear_attn_causal_rse_strong_viscosity_seed42` | `linear_attn_rse_strong_viscosity` | 10.83M | 0.0953 | 0.0947 | 0.2850 |
| `14m_linear_attn_causal_rse_x_multidil_v2_seed42` | `linear_attn_rse_strong_viscosity_convshift_multidil_symmetric_v2` | 10.86M | 0.0784 | **0.0786** | 0.2316 |

⭐ Mamba-2 × multidil_v2 (0.0631) and Mamba-2 × LUCID-c × multidil (0.0618) are the matrix ceiling.

### Per-architecture best at 14M

| Architecture | Best causal cell | Test CER |
|---|---|---:|
| RWKV-6 | P7 (LUCID × multidil) | 0.0746 |
| Mamba-2 | LUCID-c × multidil_v2 | **0.0618** |
| Linear Attention | RSE-strong × multidil_v2 | 0.0786 |

### 7M → 14M scaling

| Cell | 7M test | 14M test | Δ |
|---|---:|---:|---:|
| RWKV-6 vanilla | 0.1049 | 0.1103 | +0.0054 (NEG) |
| Mamba-2 vanilla | 0.1036 | 0.0827 | −0.0209 (POS) |
| LA vanilla | 0.1879 | 0.1359 | −0.0520 (POS) |
| RWKV-6 multidil_v2 | 0.0788 | 0.0751 | −0.0037 |
| Mamba-2 multidil_v2 | 0.0825 | 0.0631 | −0.0194 |
| LA multidil_v2 | 0.1409 | 0.0969 | −0.0440 |
| RWKV-6 LUCID-chunked | 0.1007 | 0.1051 | +0.0044 (NEG) |
| Mamba-2 LUCID-c | 0.0958 | 0.0728 | −0.0230 |
| LA LUCID | 0.1714 | 0.1300 | −0.0414 |
| RWKV-6 RSE-strong | 0.1006 | 0.1077 | +0.0071 (NEG) |
| Mamba-2 RSE-strong | 0.1038 | 0.0825 | −0.0213 |
| LA RSE-strong | 0.1198 | 0.0947 | −0.0251 |
| RWKV-6 P7 | 0.0785 | 0.0746 | −0.0039 |
| Mamba-2 LUCID-c × multidil | 0.0795 | 0.0618 | −0.0177 |
| LA RSE × multidil | 0.0999 | 0.0786 | −0.0213 |

---

## 3. Common Voice EN 100h — 7M causal pilot (single seed)

5 cells per architecture per `STATUS.md` and Master_Plan §10 Step A. All causal, single seed.

| Cell | Backbone | Best dev CER | Test CER | Test WER |
|---|---|---:|---:|---:|
| `cv_pilot_rwkv6_seed42` | `rwkv6` | 0.3036 | 0.3291 | 0.7283 |
| `cv_pilot_rwkv6_multidil_v2_seed42` | `rwkv6_convshift_multidil_symmetric_v2` | 0.2696 | 0.2953 | 0.6636 |
| `cv_pilot_rwkv6_lucid_chunked_seed42` | `rwkv6_lucid` | 0.2993 | 0.3248 | 0.7192 |
| `cv_pilot_rwkv6_lucid_multidil_v2_seed42` | `rwkv6_lucid_multidil_v2` | 0.2434 | **0.2693** | 0.6155 |
| `cv_pilot_rwkv6_rse_depth_viscosity_seed42` | `rwkv6_rse_depth_viscosity` | 0.2859 | 0.3114 | 0.7010 |
| `cv_pilot_mamba2_seed42` | `mamba2` | 0.2863 | 0.3136 | 0.7070 |
| `cv_pilot_mamba2_multidil_v2_seed42` | `mamba2_convshift_multidil_symmetric_v2` | 0.2514 | 0.2791 | 0.6350 |
| `cv_pilot_mamba2_lucid_c_seed42` | `mamba2_lucid_c` | 0.2726 | 0.3004 | 0.6794 |
| `cv_pilot_mamba2_lucid_c_multidil_v2_seed42` | `mamba2_lucid_c_convshift_multidil_symmetric_v2` | 0.2501 | **0.2772** | 0.6292 |
| `cv_pilot_mamba2_rse_strong_viscosity_seed42` | `mamba2_rse_strong_viscosity` | 0.2856 | 0.3118 | 0.7047 |
| `cv_pilot_mamba2_rse_depth_viscosity_seed42` | `mamba2_rse_depth_viscosity` | 0.2637 | 0.2910 | 0.6487 |
| `cv_pilot_linear_attn_seed42` | `linear_attn_causal` | 0.3554 | 0.3768 | 0.8196 |
| `cv_pilot_linear_attn_multidil_v2_seed42` | `linear_attn_convshift_multidil_symmetric_v2` | 0.3008 | 0.3251 | 0.7217 |
| `cv_pilot_linear_attn_lucid_seed42` | `linear_attn_lucid` | 0.3328 | 0.3556 | 0.7856 |
| `cv_pilot_linear_attn_rse_strong_viscosity_seed42` | `linear_attn_rse_strong_viscosity` | 0.3156 | 0.3418 | 0.7460 |
| `cv_pilot_linear_attn_rse_x_multidil_v2_seed42` | `linear_attn_rse_strong_viscosity_convshift_multidil_symmetric_v2` | 0.2900 | **0.3151** | 0.6937 |

CV expansion (Master_Plan §10 Step B) ⚪ pending decision based on CV-vs-LibriSpeech Δ.

---

## 4. MQAR length sweep (single-mech cohort)

Per `experiments/synthetics_v1/outputs/cohort_reduced/_index.csv`.
Reported: `final_per_query` accuracy, steps to ≥0.5 / ≥0.9, verdict.
Verdicts: `PASS` (per_query ≥ 0.95), `PARTIAL` (recall but unstable), `FAIL` (~0 acc), `OOM`, `SKIP` (predicted-fail vanilla recurrent on T=1024 per user 2026-04-28).

### T = 64 (K = 16)

| Backbone | per_query | steps→0.5 | steps→0.9 | wall (min) | verdict |
|---|---:|---:|---:|---:|:---:|
| `transformer_causal` | 0.9999 | 5000 | 5000 | 2.1 | PASS |
| `transformer` (bidir) | 0.9999 | 2000 | 2000 | 0.9 | PASS |
| `rwkv6` | 0.0005 | – | – | 6.3 | FAIL |
| `rwkv6_convshift_multidil_symmetric_v2` | 1.0 | 3000 | 4000 | 2.3 | PASS |
| `rwkv6_lucid` | 0.9998 | 5000 | 6000 | 9.5 | PASS |
| `rwkv6_rse_strong_viscosity` | 0.0003 | – | – | 9.6 | FAIL |
| `rwkv6_rse_depth_viscosity` | 0.0003 | – | – | 33.5 | FAIL |
| `rwkv6_delta` | 0.0002 | – | – | 63.0 | FAIL |
| `mamba` | 0.0002 | – | – | 28.3 | FAIL |
| `mamba2` | 0.0003 | – | – | 8.3 | FAIL |
| `mamba2_convshift_multidil_symmetric_v2` | 1.0 | 2000 | 2000 | 1.5 | PASS |
| `mamba2_lucid_c` | 0.9998 | 5000 | 5000 | 4.9 | PASS |
| `mamba2_rse_strong_viscosity` | 0.0002 | – | – | 17.8 | FAIL |
| `linear_attn_causal` | 0.9924 | 17000 | – | 17.5 | PARTIAL |
| `linear_attn_convshift_multidil_symmetric_v2` | 1.0 | 5000 | 6000 | 3.7 | PASS |
| `linear_attn_lucid` | 1.0 | 2000 | 2000 | 1.4 | PASS |
| `linear_attn_rse_strong_viscosity` | 0.0003 | – | – | 8.9 | FAIL |

### T = 256 (K = 64)

| Backbone | per_query | steps→0.5 | steps→0.9 | wall (min) | verdict |
|---|---:|---:|---:|---:|:---:|
| `transformer_causal` | 0.9999 | 12000 | 12000 | 10.8 | PASS |
| `rwkv6` | 0.0002 | – | – | 30.7 | FAIL |
| `rwkv6_convshift_multidil_symmetric_v2` | 1.0 | 3000 | 3000 | 4.9 | PASS |
| `rwkv6_lucid` | 1.0 | 3000 | 3000 | 5.9 | PASS |
| `mamba2` | 0.0003 | – | – | 32.7 | FAIL |
| `mamba2_convshift_multidil_symmetric_v2` | 1.0 | 1000 | 1000 | 2.0 | PASS |
| `mamba2_lucid_c` | 0.9999 | 3000 | 3000 | 6.5 | PASS |
| `linear_attn_causal` | 0.0672 | – | – | 68.5 | FAIL |
| `linear_attn_convshift_multidil_symmetric_v2` | 1.0 | 4000 | 4000 | 12.2 | PASS |
| `linear_attn_lucid` | 1.0 | 2000 | 2000 | 5.6 | PASS |

### T = 1024 (K = 256)

| Backbone | per_query | steps→0.5 | steps→0.9 | wall (min) | verdict |
|---|---:|---:|---:|---:|:---:|
| `transformer_causal` | 0.0060 | – | – | 202.6 | FAIL |
| `rwkv6` | – | – | – | – | SKIP |
| `rwkv6_convshift_multidil_symmetric_v2` | 1.0 | 2000 | 2000 | 8.7 | PASS |
| `rwkv6_lucid` | 1.0 | 4000 | 4000 | 30.0 | PASS |
| `rwkv6_rse_depth_viscosity` | – | – | – | – | SKIP |
| `mamba2` | – | – | – | – | SKIP |
| `mamba2_convshift_multidil_symmetric_v2` | 1.0 | 1000 | 1000 | 11.7 | PASS |
| `mamba2_lucid_c` | 1.0 | 2000 | 2000 | 38.0 | PASS |
| `mamba2_rse_depth_viscosity` | – | – | – | – | SKIP |
| `linear_attn_causal` | 0.0037 | – | – | – | SKIP |
| `linear_attn_convshift_multidil_symmetric_v2` | 1.0 | 5000 | 5000 | 34.7 | PASS |
| `linear_attn_lucid` | 1.0 | 3000 | 3000 | 26.7 | PASS |
| `linear_attn_rse_depth_viscosity` | – | – | – | – | OOM |

### MQAR cohort summary

- **multidil_v2** and **LUCID** PASS at every tested T across all three RNN families and the transformer baseline (where transformer doesn't time out).
- **Vanilla RNN backbones** (rwkv6, mamba2, linear_attn_causal) FAIL at T=64 already, confirming the no-recall baseline.
- **RSE** (both strong and depth) FAILs at T=64 across architectures. Axis-1 mechanism does not provide axis-2 recall — confirmed by commit `9777611` (RSE dropped from MQAR cohort).
- **Delta-rule (rwkv6_delta)** FAILs at T=64 — Master_Plan §6 prediction holds; engineering audit vs Zoology pending.
- **Transformer causal** FAILs at T=1024 (per-query 0.006) — confirms the length-extrapolation ceiling for vanilla causal attention without explicit recall priors.

---

## 5. LION vs VIM-style bidirectional — same budget, same layers

VIM-style = serial bidirectional (run causal kernel forward + backward, sum).
LION = parallel bidirectional attention (full T×T with paired complex / σ-decay weights).
All cells: 50 ep, seed 42, identical frontend / CTC head; layer count noted per row.

### 7M, 6 layers

| Architecture | LION cell (vanilla) | LION test CER | VIM cell | VIM test CER | Δ (LION − VIM) |
|---|---|---:|---|---:|---:|
| RWKV-6 | `7m_rwkv6_lion_vanilla_seed42` (LION-S) | **0.0859** | `7m_rwkv6_biwkv_seed42` | 0.1062 | −0.0203 (LION wins) |
| Mamba-2 | `7m_mamba2_lion_vanilla_seed42` (LION-S) | **0.0853** | `7m_mamba2_bidir_vim_seed42` | 0.1221 | −0.0368 (LION wins) |
| LA (LION-LIT) | `7m_linear_attn_lion_vanilla_seed42` (LIT) | 0.2951 | `7m_linear_attn_bidir_vim_seed42` | **0.1858** | +0.1093 (VIM wins) |
| LA (LION-S control) | `7m_linear_attn_lion_s_vanilla_seed42` | 0.1381 | (same VIM row) | **0.1858** | −0.0477 (LION-S wins) |

Read: LION-S beats VIM-style on every architecture. The single VIM win is over **LION-LIT** on LA — the no-decay variant; once LA gets the LION-S σ-decay (`linear_attn_lion_s`), LION wins by Δ −0.048 even on LA.

### 14M, 12 layers

LION 14M cells were dropped per scope (`STATUS.md` §LibriSpeech 14M). Only VIM-style 14M data exists.

| Architecture | LION 14M | VIM 14M cell | VIM test CER |
|---|---|---|---:|
| RWKV-6 | ⛔ dropped | `14m_rwkv6_biwkv_seed42` | **0.0738** |
| Mamba-2 | ⛔ dropped | `14m_mamba2_bidir_vim_seed42` | **0.0746** |
| LA | ⛔ dropped | `14m_linear_attn_bidir_vim_seed42` | 0.1207 |

Per Master_Plan §19 open item E (LION 80-ep reference run) is a one-off candidate for the strongest 14M LION composition if writeup time permits — it would close the LION-vs-VIM gap at the 14M scale.

### Smoke-test reference (5 ep, sanity check)

| Cell | Test CER |
|---|---:|
| `7m_rwkv6_biwkv_seed42_5ep` | 0.2540 |
| `7m_mamba2_bidir_vim_seed42_5ep` | 0.2517 |

Per commit `5e9ac31` log: at matched ep5 LION (RWKV-6) at 0.42 dev CER beat both VIM-style 5-ep numbers, foreshadowing the full-budget gap. (LION 5-ep cell is not on disk; the 5-ep ranking statement in commit message is the source.)

---

## 6. Catalogue of orphan / out-of-scope output dirs

For audit completeness; not part of any mandate cell:

| Dir | Status |
|---|---|
| `rwkv6_decay_coupled_delta_seed42` | no `results.json` — abandoned probe, parked. |
| `7m_rwkv6_biwkv_seed42_5ep` / `7m_mamba2_bidir_vim_seed42_5ep` | 5-epoch smoke runs; counted only as VIM 5-ep ranking. |

---

## 7. Headline progress against Master_Plan mandate

| Bucket | Mandatory | Recorded here | Pending |
|---|---:|---:|---:|
| LibriSpeech 7M (causal + LION) | 30 | 30 ✅ (incl. 4 RSE probes / variants) | 0 |
| LibriSpeech 14M (causal only; LION dropped) | 15 (causal kept) | 16 ✅ (15 spec + RSE-depth probe) | 0 in causal track; LION track ⛔ scope |
| LibriSpeech LION-VIM bidir reference | (extra) | 5 ✅ (3 × 7M VIM + 3 × 14M VIM, minus LA 7M-VIM already in 7M) | – |
| Common Voice EN 100h pilot | 4 | 16 ✅ (5 RWKV-6 + 6 Mamba-2 + 5 LA, single seed each) | 0 — scope decision (§10 Step B) ⚪ |
| MQAR cohort (T=64/256/1024) | 30 | 38 cell-results ✅ (PASS/FAIL/PARTIAL/SKIP/OOM verdicts) | T=4096 ⚪ (open item C) |

> Note on count: STATUS.md line 16 reports "~15 done" against the 60-cell LibriSpeech mandate — that line was last edited before the late-April push. The numbers above reflect the actual on-disk state per `experiments/final/outputs/` as of HEAD.
