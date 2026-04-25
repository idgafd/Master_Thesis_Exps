# STATUS — Final Stage Execution Tracker

Live tracker for the final-stage run matrix. Update as runs land.
Cross-reference with `Master_Plan.md` for cell definitions.

**Deadline:** 1 May 2026, end of day.

Legend: ✅ done · 🟡 in progress · ⚪ pending · ❌ blocked · 🔁 needs rerun · 🚫 out of scope

---

## Headline progress

| Bucket | Mandatory | Done | Pending |
|---|---:|---:|---:|
| LibriSpeech base matrix (7M + 14M) | 60 | ~15 | ~45 |
| LibriSpeech 30M (conditional) | 0–3 | 0 | tbd |
| Common Voice pilot | 4 | 0 | 4 |
| Common Voice expansion (conditional) | 0–60 | 0 | tbd |
| MQAR length sweep | 30 | 0 | 30 |
| **Mandatory total** | **94** | **~15** | **~79** |

Current best results on the spine (test CER, single seed, 7M, 30 ep):
- **Causal RWKV-6:** 0.0921 (LUCID × multidil_v2, P7) — new ceiling
- **Causal Mamba-2:** 0.0967 (multidil_v2 alone) / 0.0993 (× LUCID)
- **Causal LA:** ~0.170 (multidil_v2)
- **LION RWKV-6:** 0.0712 dev (vanilla, 80 ep reference)

---

## LibriSpeech base matrix — 7M (30 cells)

### Causal cells (15 cells)

| Architecture | vanilla | multidil_v2 | LUCID variant | rse_strong_visc | composition |
|---|:---:|:---:|:---:|:---:|:---:|
| RWKV-6 causal | ✅ | ✅ | ✅ `lucid_chunked` | ✅ anchor 0.1185 | ✅ P7 = 0.0921 |
| Mamba-2 causal | ✅ 0.1192 | ✅ 0.0967 | ✅ `lucid_c` 0.1109 | ✅ NULL 0.1183 | ✅ 0.0993 |
| Linear Attention causal | ✅ | ✅ | ✅ 0.2057 | ✅ −0.078 vs vanilla | ✅ 0.1718 |

15 / 15 cells on disk. Verify all have full diagnostics + `eval_full.json`
per `Master_Plan.md §13`.

### LION cells (15 cells)

| Architecture | vanilla | multidil_v2 | LUCID variant | rse_strong_visc | composition |
|---|:---:|:---:|:---:|:---:|:---:|
| RWKV-6 LION | ✅ 0.0712 (80ep) | ⚪ | ✅ `lucid_chunked` | ⚪ | ⚪ P7 |
| Mamba-2 LION | ✅ vanilla | ⚪ | ⚪ `lucid_c` | ⚪ | ⚪ |
| LA LION | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |

3 / 15 cells confirmed on disk. **12 LION cells pending — biggest 7M block.**

Engineering prerequisite: unified LION wrapper extension to Mamba-2
and LA per `Master_Plan.md §15`. Existing `lion` mode in
`RWKV6TimeMix` is the canonical wrapper; the same pattern needs to be
applied to Mamba-2 and LA blocks.

---

## LibriSpeech base matrix — 14M (30 cells)

All 30 cells pending. Same matrix shape as 7M.

| Architecture × mode | vanilla | multidil_v2 | LUCID | rse_strong_visc | composition |
|---|:---:|:---:|:---:|:---:|:---:|
| RWKV-6 causal | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| Mamba-2 causal | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| LA causal | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| RWKV-6 LION | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| Mamba-2 LION | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |
| LA LION | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ |

Engineering prerequisite: 14M configs across the three architectures.

---

## LibriSpeech 30M — conditional (3 cells)

Triggers per `Master_Plan.md §8`: 7M→14M ranking shift among single
mechanisms on any architecture, OR mechanism gains grow with capacity.

| Architecture × mode | best mechanism | status |
|---|---|:---:|
| RWKV-6 causal | (best per 7M/14M result) | ⚪ |
| Mamba-2 causal | (best per 7M/14M result) | ⚪ |
| LA causal | (best per 7M/14M result) | ⚪ |

Decision pending 14M completion.

---

## MQAR length sweep (30 runs)

10 backbones × 3 lengths × seed 42. Engineering prerequisites first:
- ⚪ `mamba-ssm` CUDA backend integration
- ⚪ Delta-rule forward-path audit vs Zoology reference

| Backbone | T=64 | T=256 | T=1024 |
|---|:---:|:---:|:---:|
| transformer_causal | ⚪ | ⚪ | ⚪ |
| rwkv6 | ⚪ | ⚪ | ⚪ |
| rwkv6_lucid | ⚪ | ⚪ | ⚪ |
| rwkv6_multidil_v2 | ⚪ | ⚪ | ⚪ |
| rwkv6_lucid_multidil_v2 | ⚪ | ⚪ | ⚪ |
| mamba2 (CUDA) | ⚪ | ⚪ | ⚪ |
| mamba2_lucid_c | ⚪ | ⚪ | ⚪ |
| linear_attn | ⚪ | ⚪ | ⚪ |
| linear_attn_multidil_v2 | ⚪ | ⚪ | ⚪ |
| linear_attn_rse | ⚪ | ⚪ | ⚪ |

Conditional (open item C): **MQAR T=4096** — only if T=1024 cohort
produces clean separation.

---

## Common Voice EN 100h — pilot (4 runs)

Run after the LibriSpeech 7M+14M base matrix completes.

| Run | status | result |
|---|:---:|---|
| `rwkv6` vanilla on CV | ⚪ | — |
| `rwkv6_lucid_multidil_v2` on CV | ⚪ | — |
| `linear_attn` vanilla on CV | ⚪ | — |
| `linear_attn_rse_strong_viscosity` on CV | ⚪ | — |

Scope decision per `Master_Plan.md §10` Step B based on Δ vs LibriSpeech:
- Full mirror (+60 runs)
- Targeted (+24 runs)
- Pilot-only (+0 runs)

Engineering prerequisite: ⚪ Common Voice EN 100h subset data pipeline.

---

## Engineering tasks (parallelisable with compute)

| Task | Status |
|---|:---:|
| Unified LION wrapper extension to Mamba-2 and LA | ⚪ |
| `mamba-ssm` CUDA backend integration | ⚪ |
| Common Voice EN 100h subset data pipeline | ⚪ |
| Diagnostics module per `Master_Plan.md §13` | 🟡 partial |
| 14M configs across 3 architectures | ⚪ |
| 30M configs (conditional) | ⚪ |
| Delta-rule forward-path audit vs Zoology | ⚪ |

---

## Writeup tasks

| Task | Status |
|---|:---:|
| Three-mechanism overview document | ✅ `Mechanisms_Overview.md` |
| Master plan v2.1 locked | ✅ `Master_Plan.md` |
| Top-level CLAUDE.md / STATUS.md | ✅ |
| Chapter 1 — Background + probe setup | ⚪ |
| Chapter 2 — Causal RWKV-6 mechanism discovery | ⚪ |
| Chapter 3 — Causal architecture transfer | ⚪ |
| Chapter 4 — Bidirectional adaptation (LION) | ⚪ |
| Chapter 5 — Synthesis | ⚪ |
| Send draft to advisor for review | ⚪ |
| Get advisor signature image for PDF | ⚪ |

---

## Open / conditional items

Per `Master_Plan.md §19`:

| # | Item | Trigger |
|---|---|---|
| A | 30M scale runs | 7M→14M ranking shift OR capacity-conditional gain |
| B | Common Voice expansion scope | CV pilot Δ vs LibriSpeech Δ |
| C | MQAR T=4096 | T=1024 cohort produces clean separation |
| D | P8 saturation paragraph expansion | time slack in writeup phase |
| E | LION 80-ep reference run on best composition | one-off after main matrix |

---

## Update log

- **2026-04-25** — Master plan v2.1 locked. Three-mechanism overview
  written. Top-level CLAUDE.md and STATUS.md added. 7M causal matrix
  (15 cells) confirmed on disk.
