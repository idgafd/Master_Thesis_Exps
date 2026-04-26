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
| RWKV-6 causal | ✅ 0.1049 | ✅ 0.0788 | ✅ 0.1007 `lucid_chunked` (rescued) | ✅ 0.1006 |
| Mamba-2 causal | ✅ 0.1036 | ✅ 0.0825 | ✅ 0.0958 `lucid_c` | 🟡 ep~30 |
| Linear Attention causal | ✅ 0.1879 | ✅ 0.1409 | ✅ 0.1714 `lucid` | ✅ 0.1198 |

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

**Subsequent batches in this 50-ep cycle**:
- 7M causal compositions (now in flight — see new table below)
- 7M LION non-vanilla / RSE (composition cells; tracked below)
- 14M (30 cells, gated on 14M-config engineering)

### Causal compositions + RWKV-6 RSE probes

| Architecture | composition (§5) | extra |
|---|---|---|
| RWKV-6 causal | ✅ P7 (LUCID × multidil_v2) **0.0785** | 🟡 `rwkv6_rse_depth_viscosity` (probe #1, queued); 🟡 `rwkv6_rse_split_strong_viscosity` (probe #2, queued) |
| Mamba-2 causal | 🟡 `mamba2_lucid_c × multidil_v2` (running on GPU 3) | — |
| Linear Attention causal | 🟡 `linear_attn_rse × multidil_v2` (§5-aligned, on GPU 2); 🟡 `linear_attn_lucid × multidil_v2` (extends 30-ep precedent, on GPU 0) | — |

**Composition saturation finding**: at 50 ep RWKV-6 P7 (0.0785) is essentially **tied** with multidil_v2 alone (0.0788). The 30-ep P7 ceiling (0.0921 vs multidil-alone 0.1000, Δ −0.008) was a schedule artifact; with the matched 50-ep budget, multidil_v2 + extra training time absorbs everything LUCID was adding. Extends the Master_Plan §14 P8 saturation observation from 3-mechanism to 2-mechanism.

---

## LibriSpeech base matrix — 7M LION, 50-ep (current focus)

Per Master_Plan §2 modes 2/4/6.  LION wrapper unified across architectures:
- RWKV-6 LION uses the canonical `lion_attention.lion_parallel_attention` (mode="lion" of `RWKV6TimeMix`) — LION-S (per-channel data-dep λ).
- Mamba-2 LION uses `mamba2_kernels.ssd_scan_lion` — LION-S (scalar per-head λ from dt·A).
- LA LION uses `linear_attn_lion.LIONLinearAttentionEncoder` (new) — **LION-LIT** (no decay, λ=1) per Afzal et al. 2025 Table 1's natural mapping for Katharopoulos LA.
  - Bit-exact verified: `lion_parallel_attention(phi_q, phi_k, v, w=0)` = bidirectional `phi(Q) phi(K)^T V`.

### LION singles (9 cells: 3 archs × {vanilla, multidil_v2, rse_strong_viscosity})

| Architecture | vanilla | + multidil_v2 | + rse_strong_viscosity |
|---|:---:|:---:|:---:|
| RWKV-6 LION | ✅ 0.0858 dev / 0.0859 test | ✅ 0.0764 dev / 0.0750 test | ❌ engineering blocker |
| Mamba-2 LION | ✅ 0.0871 dev / 0.0853 test | ✅ 0.0846 dev / 0.0833 test | ❌ engineering blocker |
| LA LION (LION-LIT) | ✅ 0.3003 dev / 0.2951 test | ✅ 0.1422 dev / 0.1404 test | ❌ engineering blocker |
| LA LION (LION-S, control) | ✅ 0.1417 dev / 0.1381 test | — | — |

**LION-S as a control** — LION-LIT vanilla landed dev ~0.30 / test ~0.30, well below causal LA (test 0.19).  Hypothesis: bidirectional content-similarity attention without decay smears across all positions on CTC ASR.  LA LION-S adds per-head selective σ-decay (mirrors Gated RFA → LION-S in Afzal et al. 2025 Table 1) as a falsification test for the "no decay is the missing piece" reading.

**LUCID LION cells** (3 cells, would complete the §4 5-cell shape per arch) — explicitly deferred per agent instruction (LUCID is special; start with single convshift and rse first).

**Backbone identifier (codebase) ↔ cell mapping**:

| Cell output dir | Codebase backbone identifier |
|---|---|
| `7m_rwkv6_lion_vanilla_seed42` | `lion` |
| `7m_rwkv6_lion_multidil_v2_seed42` | `lion_convshift_multidil_symmetric_v2` |
| `7m_mamba2_lion_vanilla_seed42` | `mamba2_lion` |
| `7m_mamba2_lion_multidil_v2_seed42` | `mamba2_lion_convshift_multidil_symmetric_v2` |
| `7m_linear_attn_lion_vanilla_seed42` | `linear_attn_lion` |
| `7m_linear_attn_lion_multidil_v2_seed42` | `linear_attn_lion_convshift_multidil_symmetric_v2` |

**RSE LION cells** are blocked on extending `rwkv6_time_mix.py` (assert at L381),
`mamba2_rse.py` (mode hardcoded to "recurrent"), and `linear_attn_rse.py`
(no LION dispatch) to support a complex-valued `lion_complex_attention`
kernel.  Scope decision deferred until vanilla + multidil_v2 LION cells
land (per agent instruction, plan B).

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
- **2026-04-25 (LION batch 1)** — Added `linear_attn_lion` backbone
  (LION-LIT for LA per Afzal et al. 2025 Table 1) and
  `mamba2_lion_convshift_multidil_symmetric_v2` to the encoder
  factory. Launched 3 vanilla LION cells across 2 GPUs.
  `7m_rwkv6_lion_vanilla_seed42` landed at dev 0.0858 / test 0.0859.
  `7m_mamba2_lion_vanilla_seed42` landed at dev 0.0871 / test 0.0853.
- **2026-04-25 (LA LION SCALE fix)** — Pre-fix `linear_attn_lion`
  omitted the L1 SCALE normalization. Without it, the bidirectional
  `phi(Q) phi(K)^T` row-sum scales as O(T·head_dim); LA LION vanilla
  landed at dev 0.4551 / test 0.4501 (broken). Added SCALE per LION
  paper Eq. 8; smoke-tested encoder magnitudes O(1) post-fix. Pre-fix
  artifact stashed locally under
  `7m_linear_attn_lion_vanilla_PRE_SCALE_FIX_seed42/` (gitignored).
  Rerun queued for next free GPU.
- **2026-04-25 19:52 UTC** — `7m_mamba2_causal_vanilla_seed42` landed.
  Best dev CER 0.1057 @ ep49, **test CER 0.1036**. 30-ep prior was
  0.1192 — 50-ep schedule buys ~Δ −0.014. 1h 5min wall on GPU 1.
- **2026-04-25 20:02 UTC** — `7m_rwkv6_causal_vanilla_seed42` landed.
  Best dev CER 0.1051, **test CER 0.1049** (7.74M params). 30-ep
  prior was 0.1263 — Δ −0.0212, the bigger gain. RWKV-6 vanilla and
  Mamba-2 vanilla now essentially tied at 50 ep (0.1049 vs 0.1036),
  vs the 0.007 Mamba-2 lead at 30 ep. 73 min wall on GPU 0.
- **2026-04-25 20:06 UTC** — `7m_rwkv6_causal_multidil_v2_seed42`
  landed. Best dev 0.0803, **test CER 0.0788** (7.76M params, +18k
  vs vanilla). Δ −0.0261 single-mechanism gain over 50-ep RWKV-6
  vanilla. Crosses an interesting threshold: the **single mechanism
  at 50 ep (0.0788) already beats the 30-ep P7 composition** (LUCID
  × multidil_v2 = 0.0921 test). Will revisit composition value once
  50-ep LUCID and RSE cells land. 80 min wall on GPU 3.
- **2026-04-25 20:14 UTC** — `7m_linear_attn_causal_vanilla_seed42`
  landed. Best dev 0.1903, **test CER 0.1879** (6.26M params; note
  LA is structurally smaller than RWKV-6/Mamba-2 in this codebase).
  30-ep prior was ~0.220 — Δ ≈ −0.032, largest vanilla gain from
  the longer schedule, consistent with LA being most under-fit at
  30 ep. **Round 1 complete (4 vanillas + multidil_v2/RWKV).** 88
  min wall on GPU 2.
- **2026-04-25 21:23 UTC** — `7m_mamba2_causal_multidil_v2_seed42`
  landed. Best dev 0.0839 @ ep50 (still improving at end), **test
  CER 0.0825** (7.33M params, +57k vs vanilla). Δ −0.0211 over
  50-ep Mamba-2 vanilla — mirrors the 30-ep mechanism gain
  (−0.0225) almost exactly. Cross-arch multidil_v2 transfer
  (RWKV-6 −0.026, Mamba-2 −0.021) is reproducing the
  deficit-proportional ordering predicted by Mechanisms_Overview.
  ~75 min wall on GPU 3.
- **2026-04-25 21:26 UTC** — `7m_mamba2_causal_lucid_c_seed42`
  landed. Best dev 0.0966 @ ep49, **test CER 0.0958** (7.27M params,
  +48 params — single τ per head). Δ −0.0078 over 50-ep Mamba-2
  vanilla, matching the 30-ep mechanism Δ (−0.0083) within ~0.0005.
  Both Mamba-2 single mechanisms preserve their Δ across schedules
  cleanly: multidil_v2 ~3× the gain of LUCID-c, matching closed-cell
  ordering. LUCID overhead ~40% epoch time on Mamba-2. ~92 min
  wall on GPU 1.
- **2026-04-25 22:16 UTC** — `7m_linear_attn_causal_lucid_seed42`
  landed. Best dev 0.1730 @ ep46, **test CER 0.1714** (6.26M params,
  +24 params). Δ −0.0165 over 50-ep LA vanilla — **bigger LUCID
  gain than on Mamba-2** (−0.0078) at 50 ep. Notable narrative
  shift: at 30 ep Mechanisms_Overview called LA + LUCID "weak"
  / "the anomaly"; at 50 ep LUCID converts on LA more than on
  Mamba-2. Asymmetric-transfer claim in §LUCID needs a 50-ep
  update once rwkv6_lucid_chunked rescue lands. ~120 min wall
  on GPU 2.
- **2026-04-25 ~21:55 UTC** — `7m_rwkv6_causal_lucid_chunked_seed42`
  training succeeded (best dev 0.1007 @ ep49 saved to best_model.pt)
  but **chunked-streaming eval crashed** on `torch.linalg.solve` →
  singular P at trained τ. Patched `_apply_lucid_recurrent` to
  use `solve_ex` with element-wise fallback (commit `048073d`)
  and made `eval_only.py` tolerant of per-chunk failures. Recovery
  rescue (test eval + tolerant chunked) pending GPU availability.
  Cell will flip to ✅ once results.json is written.
- **2026-04-25 22:56 UTC** — `7m_linear_attn_causal_multidil_v2_seed42`
  landed. Best dev 0.1448 @ ep49, **test CER 0.1409** (6.28M params,
  +18k vs LA vanilla). **Δ −0.0470 over LA vanilla** — the largest
  mechanism gain in the matrix so far. **Cross-architecture multidil_v2
  matrix complete at 50 ep with deficit-proportional ordering
  reproduced**: LA −0.047 > RWKV-6 −0.026 > Mamba-2 −0.021. Mechanism
  Δ across all three archs is invariant within ~0.002 between 30 ep
  and 50 ep — clean confirmation. ~91 min wall on GPU 3. **GPU 3 now
  idle**, kicking off rwkv6_lucid_chunked rescue there next.
- **2026-04-25 23:28 UTC** — `7m_rwkv6_causal_lucid_chunked_seed42`
  rescue completed (54 sec on GPU 3). best_dev 0.1005, **test CER
  0.1007**, all 6 chunked-streaming windows produced clean numbers
  (no fallback triggered visibly — patched `solve_ex` handled the
  singular case). **LUCID transfer matrix complete at 50 ep**:
  LA −0.0165 > Mamba-2 −0.0078 > RWKV-6 −0.0042. **Narrative
  inversion vs 30 ep**, where Mechanisms_Overview called LA + LUCID
  "the anomaly / weak". At 50 ep LUCID is biggest on LA. The
  architecture-specific deployment story (parallel on LA,
  lucid_chunked on RWKV-6, lucid_c on Mamba-2) still holds, but
  the magnitudes shifted with the longer schedule. Worth a
  writeup paragraph.
- **2026-04-26 01:22 UTC** — `7m_linear_attn_causal_rse_strong_viscosity_seed42`
  landed. Best dev 0.1213 @ ep50 (still improving), **test CER 0.1198**
  (6.37M params, +113k vs LA vanilla). **Δ −0.0681 vs LA vanilla** —
  **biggest single-mechanism gain in the matrix**, BREAK-band as
  predicted (30-ep had Δ −0.078; "RSE was LA's BREAK"). ~3h wall
  on GPU 2.
- **2026-04-26 01:23 UTC** — `7m_rwkv6_causal_rse_strong_viscosity_seed42`
  landed. Best dev 0.1008 @ ep49, **test CER 0.1006** (7.85M params,
  +112k vs vanilla). Δ −0.0043 vs RWKV-6 vanilla — small but the
  mechanism is engaged: per-layer θ_base mean ≈ 0.13 rad, peaks
  ≈ 0.5 rad (~30% of π/2 budget); LoRA W1/W2 moved from zero-init
  (mean ≈ 0.02), viscosity η moved (mean ≈ 0.05) — engaged-helpful
  but small. Reading: WKV's per-channel data-dep decay subsumes
  RSE's exponential-decay piece; only the complex-pole oscillation
  is novel, and ASR's binding axis-1 deficit on RWKV-6 is local
  mixing (multidil_v2 Δ −0.026), not damped oscillation. ~3h25m
  wall on GPU 0.
- **2026-04-26 01:23 UTC** — RSE_visc cross-arch transfer pattern
  matches Mechanisms_Overview prediction cleanly: LA Δ −0.0681
  (BREAK, no native decay) ≫ RWKV-6 Δ −0.0043 (WKV absorbs decay
  piece) ≫ Mamba-2 (heading for NULL, Δt absorbs both). Three
  RSE compositions (la_rse × multidil on GPU 2, la_lucid × multidil
  on GPU 0, mamba2_lucid_c × multidil on GPU 3 once rwkv6_p7 done)
  now in flight via the WATCH-A/B/C chain.
- **2026-04-26 01:38 UTC** — `7m_rwkv6_causal_p7_seed42` landed.
  Best dev 0.0787 @ ep50, **test CER 0.0785** (7.76M params).
  **Tied with multidil_v2 alone** (0.0788) within noise — composition
  gain that existed at 30 ep (P7 0.0921 vs multidil-alone 0.1000,
  Δ −0.008) **disappeared at 50 ep**. Reading: 30-ep P7 ceiling was
  a schedule artifact; matched 50-ep budget has multidil_v2 + extra
  training time absorb everything LUCID adds on top. Extends
  Master_Plan §14 P8 saturation argument from 3-mechanism to
  2-mechanism — composition Δ over the strongest single mechanism
  is small or zero at the matched budget on RWKV-6.
- **2026-04-26 01:38 UTC** — Engineering: registered
  `rwkv6_rse_split_strong_viscosity` (commit 0bb0954) for RWKV-6
  RSE probe #2: half blocks per head real-only (θ frozen at 0
  via mask), half forced-complex with init θ ∈ [-π/4, π/4] (4×
  larger than standard π/16). Smoke-tested. Watcher armed; will
  launch on second free GPU after probe #1.
