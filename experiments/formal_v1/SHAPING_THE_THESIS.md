# Shaping the Thesis — Priority and Structure Notes

Standalone thesis-positioning record. Written after the multi-axis
framing settled and before the final chapters are drafted. Summarises
*which axes the thesis should lead on*, *what each axis actually
contributes*, and *where to invest remaining GPU-h and writing effort*.

**Not a replacement for**:
- `CLAUDE.md` — operating manual, methodology, lessons learned
- `EXPRESSIVITY_AXES.md` — axis definitions, paper index, task suite
- `STAGE10_SUMMARY.md` — Stage 10 empirical summary
- `STAGE10_PLAN.md` — pending-run specifications

This document is the *meta-layer* above those four: the argument for
which axes land in the thesis and in what priority.

---

## The winning thesis structure: three axes, not one

The thesis is strongest when axes 1, 2, and 3 are presented as
**complementary evidence types**, not as competing claims. No single
axis is load-bearing alone; the three together form a claim structure
that's hard to attack from any single angle.

| Axis | What it contributes | Evidence type |
|---|---|---|
| **1 — Feature extraction / short-range temporal** | Cross-domain empirical robustness: 5 independent instances across 4 domains (2D vision × 2 architecture families, 2D spectrogram audio, 1D language, 1D audio) | **Extensive** (many works confirm) |
| **2 — Associative memory capacity** | Mechanism-task matching (differential): mechanism wins one task, loses another; the orthogonal-axis pattern is the claim | **Differential** (paper evidence + our nulls) |
| **3 — State-tracking / TC⁰ → PNC¹** | Formal theory ↔ empirical correspondence: two circuit-complexity theorems pre-register specific empirical predictions we test | **Formal + empirical** (strongest novelty) |

Axes 4 (long-range flow) and 5 (feature-map richness) are weaker on our
spine (Log-Linear PLATEAU on RWKV-6; Family-D saturated) — they should
appear in the thesis as negative-result evidence for the axis
decomposition (mechanisms that targeted those axes produced null
results on a task that doesn't exercise them), not as load-bearing
chapters.

---

## Why the three-axis frame beats any single-axis frame

Each axis contributes a *different type* of evidence. The thesis claim
lands only if all three converge:

> *Different mechanism families target different expressivity axes,
> with empirical gain determined by alignment between the mechanism's
> function-class extension and the task's structural prior. This
> decomposition is validated across domains (axis 1), across tasks
> (axis 2), and against formal complexity theory (axis 3).*

**Without axis 1**: the cross-domain robustness argument disappears;
one could accuse the work of being ASR-specific.

**Without axis 2**: the mechanism-task matching is a conceptual claim,
not an empirically measured differential.

**Without axis 3**: the decomposition is purely empirical with no
formal grounding; a reviewer can reasonably ask "why these specific
axes and not some other decomposition?"

Each axis answers an objection the other two cannot.

---

## Priority order for remaining GPU-h and writing effort

**Updated 2026-04-23** post-v2 init-fix results. Previous priority
order re-ranked given:
- P1 v2 (`multidil_sym_v2`) landed dev 0.1013 / test 0.1000, 10σ gain.
- CB-1 v2 landed dev 0.0973 / test 0.0961 — **first sub-0.10 causal
  RWKV-6**. RSE × multidil compose orthogonally within axis 1.
- Stage 11.0a LA baseline complete (0.2235 test vanilla). Stage 11.2
  RSE transfer complete on both Mamba-2 (ambiguous-engaged, predicted)
  and LA (BREAK, −0.081 vs vanilla, predicted).
- Stage 11.5 single-dil controls complete (11.5a/b tie broken-init
  multidil within σ; 11.5c LA is init-confounded).

Ordered by marginal contribution to the thesis, post-update:

1. **P5 + P6 on RWKV-6 — close the v2 composition matrix.**
   ~1.5 h each (parallel on 2 GPUs → 1.5 h wallclock).
   - **P5** `rwkv6_convshift_multidil_symmetric_gated_v2` (CB-3 v2) —
     tests whether content-conditional α matters now that multidil
     actually engages.
   - **P6** `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2`
     (CB-7 v2) — tests whether channel-side Kronecker composes
     orthogonally with working multidil (mirror of CB-1 v2 on a
     different axis-pair).
   - Either result is thesis-relevant: confirms or revises the scope
     of the broken-init Group-3 invariant extension (see
     `STAGE10_SUMMARY.md` §3 revision note).

2. **Dispatch wiring for multidil_v2 on Mamba-2 / LA** +
   **P2, P3 cross-architecture v2 transfer.** Engineering ~30–60 min;
   runs ~1.5 h each (parallel on 2 GPUs → ~2 h wallclock total).
   - **P2** `mamba2_convshift_multidil_symmetric_v2`,
     **P3** `linear_attn_convshift_multidil_symmetric_v2`.
   - Pre-registered prediction (updated post-v2): LA > Mamba-2 > RWKV-6
     in absolute multidil gain. Tests whether the differential-transfer
     pattern from broken-init (confirmed for single-dil + scalar) also
     holds with working multi-dilation.

3. **CB-2 wide4 / dense (axis 1, now reopened).**
   ~3 GPU-h. Multi-dilation's widest existing branch (d = 8, α = 1.23
   at L5 in P1 v2) is non-trivially engaged → pushing to d = 16 has a
   clean prior now. See `STAGE10_PLAN.md` §6 CB-2.

4. **11.5c LA identity-init retest.** ~50 min. Resolves the Stage 11.5
   LA init-mismatch confound — tells us whether α_1 scalar itself helps
   on LA or whether 11.5c's regression was entirely init-driven.

5. **$S_3$ permutation composition benchmark (axis 3).** Highest
   novelty per unit effort remaining. ~50-line synthetic-data generator
   + reuse of existing training harness. Directly tests the formal
   NC¹ vs PNC¹ separation proved in arXiv:2603.01959 + arXiv:2603.03612.
   See `EXPRESSIVITY_AXES.md` §Task suite implication / §Two task
   options.

6. **MQAR benchmark completion (axis 2).** Already in progress on
   a separate track. Produces the sharpest mechanism-task differential
   in the thesis: DeltaNet / DeltaProduct predicted major win;
   multidil_sym / RSE predicted flat. The prediction shape *is* the
   axis-2 claim.

7. **Dyck-$k$ (axis 3 secondary).** Only after $S_3$ completes. Adds
   robustness evidence; doesn't carry load-bearing claim.

8. **Stage 12 LION-chapter — multidil_v2 and CB-1 v2 on LION.**
   Direct transfer of the two post-tonight wins to bidirectional form.
   Load-bearing for the causal-vs-bidirectional gap analysis in the
   synthesis chapter. 80-ep reference runs.

### One-pass rationale for the re-ordering

Priorities 1 and 2 close the v2 matrix cleanly (RWKV-6 compositions + cross-architecture transfer), which is the most direct next-step yield from the init-fix result. Priority 3 (CB-2) and Priority 4 (11.5c LA) tie up the single remaining Stage-10 item and the one confound in Stage 11, respectively. Priority 5 ($S_3$) remains the highest-novelty axis-3 test. Priorities 6–8 are the axis-2, axis-3-secondary, and bidirectional-extension chapters. **Stage 11.0a (LA baseline) is removed from the priority list — it has landed.**

---

## The axis-3-specific argument (highest novelty-per-effort)

If you have to pick ONE remaining investment for maximum thesis
differentiation, axis 3 wins on five grounds:

1. **The benchmark doesn't exist in our codebase yet.** Implementing
   $S_3$ composition puts the thesis in a class of work most
   ASR-focused theses don't do. Distinctive positioning.
2. **The formal theorems pre-register the outcome.**
   arXiv:2603.01959 proves diagonal SSMs must fail on non-Abelian
   state-tracking at sufficient $T$; arXiv:2603.03612 proves DPLR
   is $\mathsf{PNC}^1$-complete. An empirical test at our regime
   either confirms the theorem-to-empirical correspondence or
   reveals an interesting finite-precision gap. **Both outcomes are
   publishable; there is no "boring" result.**
3. **It differentiates us from pure-benchmark papers.** Most papers in
   linear-time-RNN space either propose a mechanism and benchmark it
   on standard datasets, or test an existing mechanism in a new
   setting. Running a synthetic benchmark explicitly designed to
   probe a proved complexity-class separation is a methodological
   move few papers in this area make.
4. **It makes the axis-3 chapter possible.** Without running $S_3$,
   axis 3 is a theoretical citation. With $S_3$, it becomes an
   empirical chapter where pre-registered predictions meet measured
   behaviour.
5. **It's cheap relative to ASR-scale training.** $S_3$ at $T = 1024$,
   $d = 256$, 6 layers runs in minutes-to-hours, not days. A full
   mechanism × length sweep fits in a single afternoon of compute.

---

## Risk assessment — variance, not just mean

Different axes have different outcome-variance profiles:

- **Axis 1**: *lowest variance*. Already well-validated by 5 prior
  papers. CB-2 will produce a clean either-way result. Guaranteed
  publishable, but marginal novelty.
- **Axis 2**: *middle variance*. MQAR has a moderate prior of
  confirming DeltaNet dominance; if it does, the axis-2 differential
  is sharp. If MQAR doesn't separate as predicted, interpretation
  becomes more complex.
- **Axis 3**: *highest variance*. $S_3$ confirming the theorem gives
  the thesis its single strongest chapter. $S_3$ showing messy
  finite-precision behaviour requires more writing effort but is
  still a contribution. The upside is larger than the other two;
  so is the interpretive complexity in the downside case.

**Strategic reading:** axis 1 is the safe-floor investment, axis 3 is
the high-ceiling investment, axis 2 is in between. The thesis wants
all three: a safe floor, a sharp differential, and a high-ceiling
novelty piece.

---

## What to cut if compute / time forces a cut

Cut axes 4 and 5 as primary chapters. Keep them only as:

- Supporting evidence in the synthesis chapter for the "axes we
  characterised but do not exercise on our spine" framing.
- Paper-index entries in `EXPRESSIVITY_AXES.md` explaining why those
  axes' mechanisms landed null on our ASR benchmark (axis-mismatch,
  not mechanism failure).

Do **not** cut any of axes 1, 2, 3. Each provides a distinct
type of evidence the thesis needs.

Do **not** prioritise additional axis-1 mechanism variants on
RWKV-6 causal beyond CB-2. The cross-experiment invariant and the
Stage-10 CB-sprint already showed that further RWKV-6-causal
within-axis-1 composition tends to saturate. Remaining axis-1
thesis value is in Stage 11 cross-architecture transfer.

---

## One-sentence thesis core

> *The expressivity of causal linear-time RNNs decomposes into
> orthogonal axes, each addressed by specific mechanism families,
> and empirical gain is determined by alignment between the
> mechanism's function-class extension and the task's structural
> prior — validated across domains, across tasks, and against
> formal complexity theory.*

Every chapter of the thesis is an instantiation of this sentence
at a different scope: Chapter 2 on axis 1 via causal RWKV-6 mechanism
discovery, Chapter 3 on cross-architecture transfer, Chapter 4 on
LION bidirectional extension, Chapter 5 on multi-axis synthesis with
axes 2 and 3 from MQAR and $S_3$.

---

## Final call

Run all three axes. Don't cut 1, 2, or 3. Invest remaining GPU-h
in priority order above, starting with Stage 11.0a (LA baseline),
then MQAR completion, then $S_3$ implementation. That sequence lands
the thesis with a defensible claim structure supported by three
complementary evidence types, with the highest-novelty chapter
(axis 3) positioned as the distinctive contribution.
