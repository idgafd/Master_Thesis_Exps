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

> **Schedule note (2026-04-25):** the ~15 done cells above were trained
> on the 30-epoch discovery schedule (Stages 2–11). Per Master_Plan §7
> the locked matched budget is **50 epochs**. All 7M + 14M LibriSpeech
> cells must be re-run at 50 ep with full §13 output spec before they
> count against the mandate. The 30-ep results are kept as a reference
> floor for sanity-checking the 50-ep reruns. See the dedicated 50-ep
> section below the 30-ep tables.

Current best results on the spine (test CER, single seed, 7M, **30 ep — legacy reference, not the 50-ep target**):
- **Causal RWKV-6:** 0.0921 (LUCID × multidil_v2, P7) — new ceiling
- **Causal Mamba-2:** 0.0967 (multidil_v2 alone) / 0.0993 (× LUCID)
- **Causal LA:** ~0.170 (multidil_v2)
- **LION RWKV-6:** 0.0712 dev (vanilla, 80 ep reference)

---

## LibriSpeech base matrix — 7M (30 cells) — 30-ep legacy reference

> Tables in this section are the prior 30-ep schedule (pre-§7-lock).
> Kept verbatim for reference and as a sanity-check floor for the
> 50-ep rerun matrix below. None of these cells count against the
> 50-ep mandate.

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

## LibriSpeech base matrix — 7M, 50-ep rerun (current focus)

Per Master_Plan §7 (50-ep matched budget) and §13 (output spec).
**First batch — causal only, 12 cells**: 3 architectures × {vanilla
+ 3 single mechanisms}. Compositions (cell 5 per §4) and LION
variants land in subsequent batches once their prerequisites are met.

### Causal singles (12 cells)

| Architecture | vanilla | + multidil_v2 | + LUCID | + rse_strong_viscosity |
|---|:---:|:---:|:---:|:---:|
| RWKV-6 causal | ⚪ | ⚪ | ⚪ `lucid_chunked` | ⚪ |
| Mamba-2 causal | ✅ 0.1036 | ⚪ | ⚪ `lucid_c` | ⚪ |
| Linear Attention causal | ⚪ | ⚪ | ⚪ `lucid` | ⚪ |

**Output dirs**: `outputs/7m_<arch>_causal_<cellname>_seed42/` per
Master_Plan §13, where:
- `<arch>` ∈ {rwkv6, mamba2, linear_attn}
- `<cellname>` ∈ {vanilla, multidil_v2, lucid_chunked | lucid_c | lucid, rse_strong_viscosity}

**Backbone identifier (codebase) ↔ cell mapping** (locked):

| Cell output dir | Codebase backbone identifier |
|---|---|
| `7m_rwkv6_causal_vanilla_seed42` | `rwkv6` |
| `7m_rwkv6_causal_multidil_v2_seed42` | `rwkv6_convshift_multidil_symmetric_v2` |
| `7m_rwkv6_causal_lucid_chunked_seed42` | `rwkv6_lucid` |
| `7m_rwkv6_causal_rse_strong_viscosity_seed42` | `rwkv6_rse_strong_viscosity` |
| `7m_mamba2_causal_vanilla_seed42` | `mamba2` |
| `7m_mamba2_causal_multidil_v2_seed42` | `mamba2_convshift_multidil_symmetric_v2` |
| `7m_mamba2_causal_lucid_c_seed42` | `mamba2_lucid_c` |
| `7m_mamba2_causal_rse_strong_viscosity_seed42` | `mamba2_rse_strong_viscosity` |
| `7m_linear_attn_causal_vanilla_seed42` | `linear_attn_causal` |
| `7m_linear_attn_causal_multidil_v2_seed42` | `linear_attn_convshift_multidil_symmetric_v2` |
| `7m_linear_attn_causal_lucid_seed42` | `linear_attn_lucid` |
| `7m_linear_attn_causal_rse_strong_viscosity_seed42` | `linear_attn_rse_strong_viscosity` |

**Subsequent batches in this 50-ep cycle** (out of scope for current launch — listed for awareness only):
- 7M causal compositions (3 cells, cell 5 per §4 / §5)
- 7M LION (15 cells, gated on unified-LION-wrapper engineering)
- 14M (30 cells, gated on 14M-config engineering)

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
- **2026-04-25 (post-lock decision)** — Final-stage policy: per-run
  `best_model.pt` is committed to git under `experiments/final/`
  (counter to `formal_v1/.gitignore`). Per-epoch diagnostic snapshot
  checkpoints stay local. See `Master_Plan.md §13` and §18 entry 14.
  `experiments/final/.gitignore` added.
- **2026-04-25 (rerun mandate)** — All previously-✅ 7M cells were on
  the 30-ep discovery schedule. Per Master_Plan §7 (locked 50-ep
  matched budget) they don't satisfy the final-stage spec. New
  dedicated `LibriSpeech base matrix — 7M, 50-ep rerun` section
  added. First batch = 12 causal singles (3 archs × {vanilla + 3
  single mechanisms}), no compositions, no LION. Compositions and
  LION/14M follow once their prereqs are met. The 30-ep tables stay
  in place as a sanity-check floor.
- **2026-04-25 19:52 UTC** — `7m_mamba2_causal_vanilla_seed42` landed.
  Best dev CER 0.1057 @ ep49, **test CER 0.1036**. 30-ep prior was
  0.1192 — 50-ep schedule buys ~Δ −0.014. 1h 5min wall on GPU 1.
