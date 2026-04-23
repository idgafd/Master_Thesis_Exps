# Stage 11 Agent — Next-Experiment Queue

**Audience:** the Stage-11 agent on this instance. Both GPUs are free now.
**Last updated 2026-04-23 after commits `3af846d`, `e9f6d10`, `848c3fb`.**

**Current state:**
- **Init fix** for `MultiDilationDWConvShift` is in `src/models/mechanisms/conv_shift.py` (Option-B). `_v2` dispatch is wired for RWKV-6, Mamba-2, and LA. Unit tests passing.
- **RWKV-6 ceiling:** CB-1 v2 at dev **0.0973 / test 0.0961** (RSE × multidil_v2).
- **Mamba-2 single-mechanism leader:** P2 v2 at test **0.0967** (Mamba-2 + multidil_v2). Within σ of CB-1 v2.
- **Axis-2 gap identified:** `rwkv6_lucid` at test **0.1216** (previously overlooked in `outputs/lucid_exp03_rwkv6_lucid_seed42/`) — measurable axis-2 signal on ASR, differential from Delta-rule null. See `EXPRESSIVITY_AXES.md` §Axis 2.
- **Priority order driven by `SHAPING_THE_THESIS.md` §Priority order.**
- **MQAR / axis-2-benchmark work is on a separate track — not your scope.**

**Binding discipline** (carry-over from Stage-10):
- `_v2` suffix for every fixed-init rerun; never overwrite an existing output directory.
- `diagnostics.json` is mandatory per `STAGE10_PLAN.md` §7.5. Log α_d per layer per epoch + mechanism-specific probes listed below each experiment.
- Matched-epoch 15 halt criterion: ≥ +0.006 behind stage-specific primary reference → halt.
- Seed 42, 30 ep, canonical LibriSpeech clean-100 spine.

---

## ✅ COMPLETED (earlier sessions)

- **Priority 1 (original)** — P5 (CB-3 v2) + P6 (CB-7 v2): done, commit `848c3fb`.
  - P5 landed dev 0.1150 / test 0.1136 — **REGRESSION**; gating hurts on top of working multidil.
  - P6 landed dev 0.0989 / test 0.0988 — **MARGINAL**; axis-5 absorbed on top of axis-1.
- **Priority 2 (original)** — P2 (Mamba-2 multidil_v2) + P3 (LA multidil_v2): done, commit `848c3fb`.
  - P2 test 0.0967 — Mamba-2 single-backbone leader.
  - P3 test 0.1700 — largest absolute multidil gain (cross-arch differential confirmed).
- **P1 v2, P4 v2** (RWKV-6 multidil_v2, CB-1 v2): done on a separate instance, commits `3af846d` + `e9f6d10`.

Results surfaced in `STAGE10_PLAN.md §9.2` and `STAGE10_SUMMARY.md`.

---

## Priority 1 — LUCID cross-architecture sweep + LUCID × multidil composition (~6 h serial / ~3 h parallel on 2 GPUs)

Open the previously-overlooked axis-2 track. `rwkv6_lucid` (already on
disk at `outputs/lucid_exp03_rwkv6_lucid_seed42/`, test CER 0.1216)
shows **measurable axis-2 signal on ASR** — −0.0047 vs vanilla RWKV-6,
which directly differs from the Delta-rule T1 null on the same axis.
The gap suggests LUCID may partially overlap axis 5 (content-adaptive
decorrelation / normalization). Three experiments close the axis-2
track cross-architecture and test its composition with axis 1.

**Dispatch status:** `lucid` substring is already recognised in
`encoder.py` (lines ~357, 560 — verify). Existing backbones
`rwkv6_lucid`, `lion_lucid`, `lion_lucid_chunked`, `rwkv6_lucid_delta`,
`rwkv6_lucid_sr` are already wired. For LA, check whether the substring
triggers on the LA path too; if not, add minimal dispatch (same as the
multidil_v2 dispatch pattern for Mamba-2 / LA).

### P7 — `rwkv6_lucid_convshift_multidil_symmetric_v2` (LUCID × multidil_v2, axis-2 × axis-1 composition)

- **Backbone:** `rwkv6_lucid_convshift_multidil_symmetric_v2`
  - Triggers: `lucid` + `convshift` + `multidil` + `symmetric` + `_v2`
  - If the existing substring dispatcher doesn't compose these, add a single substring-check — all three mechanism flags are already independently recognised.
- **Output dir:** `outputs/rwkv6_lucid_convshift_multidil_symmetric_v2_seed42/`
- **GPU:** 0 (run first)
- **Runtime:** ~1.5 h

**What it tests — the highest-EV composition remaining.** CB-1 v2 at
0.0961 was axis-1 × axis-1 (RSE × multidil, both axis 1 in our
placement). This is the first **genuine cross-axis composition** on the
v2 baseline: axis-2 (LUCID) × axis-1 (multidil). LUCID alone adds
~0.005 on RWKV-6; multidil_v2 alone adds ~0.026. If they compose
additively, predicted dev ~0.070-0.085 (large BREAK). If they compose
partially (LUCID's axis-5 overlap redundant on top of working
multidil), predicted dev ~0.095-0.100 (tied P1 v2 / below).

**Pre-registered decision rule:**
- **BREAK** (dev < 0.090): axis-2 × axis-1 compose orthogonally and strongly. Paper-worthy cross-axis claim.
- **MARGINAL** (dev 0.090 – 0.100): composition adds on top of P1 v2 but axis-2 contribution partly redundant. Still thesis-positive.
- **TIED** (dev 0.100 – 0.108): LUCID's axis-5 overlap absorbs its contribution on top of working multidil. Axis-2 matters only when axis-1 is not already maxed out.
- **REGRESSION** (dev > 0.108): destructive composition; investigate preconditioner interaction with depth-graded α_d.

**Decision for the composition if LUCID alone helps on LA (P9 below):**
if P9 shows LUCID gain on LA, queue a LA LUCID × multidil_v2 after P7
(P10 below) — predicted largest cross-arch cross-axis composition gain.

**Mandatory diagnostics:**
- α_d per layer per epoch (standard multidil probe)
- LUCID preconditioner statistics per layer at ep 15, 30:
  - $\tau$ temperature value per head
  - $\|P - I\|_F$ / $\|I\|_F$ per layer (how far from identity)
  - Condition number of $P$ at ep 30
- LUCID-multidil interaction: does α_d differ between P1 v2 (no LUCID) and P7 (with LUCID)?

### P8 — `rwkv6_lucid_rse_convshift_multidil_symmetric_v2` (LUCID × RSE × multidil_v2, three-mechanism composition) — CONDITIONAL

- **Launch only if P7 lands MARGINAL or BREAK** (dev ≤ 0.100). Otherwise skip.
- **Backbone:** `rwkv6_lucid_rse_convshift_multidil_symmetric_v2`
- **Output dir:** `outputs/rwkv6_lucid_rse_convshift_multidil_symmetric_v2_seed42/`
- **Runtime:** ~1.5 h
- **Tests:** whether LUCID adds to CB-1 v2 (0.0961) — stacking axis-1-sub-axis + axis-1-sub-axis + axis-2 in one backbone. If P7 already lands below P1 v2, this tests whether RSE also contributes on top.

### P9 — `linear_attn_lucid` (cross-architecture axis-2 transfer)

- **Backbone:** `linear_attn_lucid`
  - LA has explicit attention (parallel Katharopoulos form), which is LUCID's natural home.
  - If substring dispatch doesn't cover the LA path: add one line in `encoder.py` mode_map + verify `lucid` flag propagates to the LA module instantiation (pattern as multidil_v2 dispatch for LA).
- **Output dir:** `outputs/linear_attn_lucid_seed42/`
- **GPU:** 1 (parallel with P7)
- **Runtime:** ~1.5 h

**What it tests:** does LUCID transfer to LA? LA has explicit attention
matrix $A = \phi(Q) \phi(K)^\top$; LUCID's preconditioner directly
applies. LA has no native decorrelation in the accumulator (unlike
Mamba-2's Δt which doesn't substitute for decorrelation specifically),
so LUCID's gain should be substantial. Pre-registered prediction: gain
comparable to or larger than RSE+viscosity on LA (−0.078 at 11.2b);
final dev likely in 0.15-0.18 range.

**Mandatory diagnostics:**
- LUCID τ per head + $\|P - I\|_F$ per layer at ep 15, 30
- LA-specific: does LUCID preconditioner stabilise LA's L1 denominator?
  (The 11.5c LA init-confound suggests LA's attention matrix is
  poorly-conditioned by default; LUCID should help precisely there.)

### P10 — `linear_attn_lucid_convshift_multidil_symmetric_v2` (LA LUCID × multidil_v2) — CONDITIONAL

- **Launch only if P9 lands MARGINAL or BREAK (meaningful gain over LA vanilla 0.2201).**
- **Backbone:** `linear_attn_lucid_convshift_multidil_symmetric_v2`
- **Output dir:** `outputs/linear_attn_lucid_convshift_multidil_symmetric_v2_seed42/`
- **Runtime:** ~1.5 h
- **Tests:** cross-architecture cross-axis composition. LA is the most mechanism-hungry architecture (biggest deficit on both axes); the composition should therefore give the largest absolute gain. LA + RSE+viscosity alone: 0.1422. LA + multidil_v2 alone: 0.1700. LA + LUCID: expected 0.15-0.18. Three-way composition predicted: 0.11-0.14 range.

### Mamba-2 LUCID — feasibility check first (DO NOT launch blind)

Mamba-2's SSM path does NOT have an explicit attention matrix that
LUCID's $P^{-1} A V$ form directly applies to. Before attempting a run:

1. Read `src/models/mamba2_block.py` and `mamba2_encoder.py` to identify whether there's an analogue of the attention-output path where LUCID's preconditioner would fit.
2. If yes and the port is ≤30 min engineering: queue `mamba2_lucid` as P11 on the Mamba-2 spine.
3. If structurally not applicable: document the incompatibility in a brief note in `EXPRESSIVITY_AXES.md §Axis 2` ("LUCID does not apply to pure-SSM architectures because there is no explicit attention matrix to precondition"), and skip.

This is genuinely an open question — don't force a square-peg port. Flag the result either way.

### Joint decision after P7 + P9 finish

- **P7 BREAK + P9 BREAK:** axis-2 transfers and composes across architectures. Major thesis finding. Queue P8 + P10.
- **P7 MARGINAL + P9 BREAK:** LUCID is genuinely axis-2-primary (works on LA where it's the only mechanism) but has limited within-RWKV-6 composition value. Still informative. Queue P10.
- **P7 TIED + P9 BREAK:** LUCID's axis-2 contribution on RWKV-6 overlaps axis-5 (redundant with multidil's structural gains); on LA its pure-axis-2 effect is clean. Queue P10 but skip P8.
- **P7 REGRESSION:** destructive composition on RWKV-6. Skip P10; investigate.

Commit + push after P7 + P9 both finish. Add four rows to `STAGE10_PLAN.md §9.2` or §9.3 (appropriate subsection) — RWKV-6 lucid× composition + LA lucid alone + LA lucid×composition + any conditional follow-ups.

---

## Priority 2 — Cross-architecture CB-1 v2 equivalents on Mamba-2 + LA (~3 h parallel on 2 GPUs, +15 min engineering)

Tests whether CB-1 v2's within-axis-1 composition (RSE × multidil_v2, which gave 0.0961 on RWKV-6) transfers to other architectures.

### P11 — `mamba2_rse_convshift_multidil_symmetric_v2`

- **Prediction:** likely minimal additional gain over P2 v2 (Mamba-2 multidil_v2 at 0.0967). RSE alone on Mamba-2 was null (11.2a). Composition of (null mechanism) × (working mechanism) may stay tied the working one.
- **Still worth running** because: (a) confirms that Mamba-2 RSE is null regardless of composition partner, (b) provides cross-architecture CB-1-matrix completion, (c) single-seed at our spine cost is low.
- Output dir: `outputs/mamba2_rse_convshift_multidil_symmetric_v2_seed42/`
- Runtime: ~1.5 h

### P12 — `linear_attn_rse_convshift_multidil_symmetric_v2`

- **Prediction:** large gain. LA + RSE (11.2b): −0.078 gain. LA + multidil_v2 (P3): −0.050 gain. Composition plausibly stacks partially: expected dev 0.115-0.140 range, potentially **below vanilla RWKV-6 (0.1263)** which would be a strong cross-architecture claim.
- Output dir: `outputs/linear_attn_rse_convshift_multidil_symmetric_v2_seed42/`
- Runtime: ~1.5 h
- Mandatory diagnostics: RSE θ mobility per head per layer + multidil α_d + LA running-sum Z stability.

**Launch as serial pair (P11 then P12) or parallel if compute allows.** After both finish: commit + push; update `STAGE10_PLAN.md §9.3` with two new rows.

---

## Priority 3 — CB-2 `multidil_wide4` / `multidil_dense` (~3 h combined)

Now that multi-dilation engages (α₈ at L5 = 1.23 in P1 v2), the
RF-expansion hypothesis has a clean prior. Full spec in `STAGE10_PLAN.md`
§6 CB-2. Use the fixed-init mechanism — both variants inherit the
Option-B init automatically.

**Launch order (serial on one GPU if priorities 1–2 still running, or
parallel on 2 GPUs if both free):**

1. `rwkv6_convshift_multidil_symmetric_wide4_v2` — dilations {1, 2, 4, 8, 16}
2. `rwkv6_convshift_multidil_symmetric_dense_v2` — dilations {1, 2, 3, 4, 6, 8}

**Pre-registered prediction** (from `STAGE10_PLAN.md §6 CB-2`):
- BREAK (< 0.0960): RF wasn't at ceiling at ±8 frames; push further.
- TIED (0.0960–0.1005): ±8 frames saturates; axis-1-on-RWKV-6-causal closed.
- REGRESSION (> 0.1010): extra branches net-harmful; informative negative.

---

## Priority 4 — 11.5c LA identity-init retest (~50 min)

11.5c `linear_attn_convshift_symmetric` landed at dev 0.2287 / test
0.2245, **+0.0052 worse than LA vanilla**. The existing note in
`STAGE10_SUMMARY.md §9` flags this as confounded by init mismatch
(11.1b's branch-1 was overridden to center-tap identity for bit-exact
zero-regression; 11.5c's was default). The retest uses matched
identity init to isolate whether single-dilation ConvShift itself
helps on LA or whether 11.5c's regression was entirely init-driven.

- **Backbone:** `linear_attn_convshift_symmetric_identityinit`
  (new substring-dispatch path)
- **Output dir:** `outputs/linear_attn_convshift_symmetric_identityinit_seed42/`
- **Runtime:** ~50 min

Lightweight single-axis confound resolution. Worth running once
priorities 1–3 are complete.

---

## Priority 5 — $S_3$ permutation composition benchmark (axis 3)

Separate synthetic-benchmark infrastructure. See
`EXPRESSIVITY_AXES.md §Task suite implication` for the task spec and
`SHAPING_THE_THESIS.md §The axis-3-specific argument` for the thesis
rationale. ~50-line data generator + reuse of existing training harness;
runs ≪ 1 h per mechanism.

**Not yet your scope** unless you want to pick it up after Priorities
1–4 close. If MQAR-track agent doesn't pick it up, flag for
coordination.

---

## Priority 6 — MQAR (axis 2)

**Not your scope.** Another agent is handling this track. Do not run
MQAR experiments here.

---

## Priority 7 — Dyck-$k$ (axis 3 secondary)

After $S_3$. Not your scope until $S_3$ lands.

---

## Priority 8 — Stage 12 LION-chapter v2 transfer

After the full axis-1 + axis-2 + axis-3 causal matrix is closed.
Direct transfer of P1 v2 + CB-1 v2 to LION bidirectional form at
80-ep reference schedule. Thesis Chapter 4 material. Parked until
priorities 1–3 complete.

---

## Don't-do list

- **Don't re-run any experiment in the ✅ COMPLETED list** (P1 v2, P4 v2, P5 v2, P6 v2, P2 v2, P3 v2, 11.0/11.1/11.2/11.5). They're done, committed at `3af846d`, `e9f6d10`, `848c3fb`.
- **Don't modify `src/models/mechanisms/conv_shift.py`.** The init fix is in. Read to understand, don't edit.
- **Don't touch any committed `_v2` output directory.** They're on disk and on main.
- **Don't run MQAR or any axis-2/axis-3 synthetic benchmark.** Another agent handles those tracks.
- **Don't overwrite existing broken-init result directories.** Use `_v2` / explicit suffix naming for all new reruns.
- **Don't skip `diagnostics.json`.** α_d per layer + mechanism-specific probes per experiment spec. The Stage-10 procedural gap (no mechanism probes logged) must not repeat.
- **Don't force LUCID into Mamba-2** if the SSM path has no attention-matrix analog. Document the incompatibility and skip rather than port poorly.

---

## Workflow summary

1. **Now:** Launch **P7** (RWKV-6 LUCID × multidil_v2) on GPU 0 + **P9** (LA LUCID) on GPU 1 in parallel. ~1.5 h wallclock.
2. **+1.5 h:** Check results; commit with matrix rows added to `STAGE10_PLAN.md §9.2` / §9.3 and a brief note in `STAGE10_SUMMARY.md` / `EXPRESSIVITY_AXES.md §Axis 2`. Push.
3. **+1.5 h:** Conditional on P7/P9 outcomes (decision tree above):
   - If P7 BREAK/MARGINAL: launch **P8** (LUCID × RSE × multidil_v2 triple composition) on GPU 0.
   - If P9 BREAK/MARGINAL: launch **P10** (LA LUCID × multidil_v2 cross-axis on LA) on GPU 1.
4. **+3 h:** Feasibility-check Mamba-2 LUCID (~15 min). If feasible, queue as **P11** on next free GPU; if not, document incompatibility.
5. **+4–5 h:** Launch **P11 / P12** (Priority 2 — cross-architecture CB-1 v2 equivalents on Mamba-2 and LA).
6. **+6–7 h:** CB-2 wide4 + dense (Priority 3) on RWKV-6 v2.
7. **+8 h:** 11.5c LA identity-init retest (Priority 4).
8. **At end:** coordinated doc revision — update `SHAPING_THE_THESIS.md`, `STAGE10_SUMMARY.md`, `EXPRESSIVITY_AXES.md` to reflect the full v2 matrix including LUCID axis. Push thesis-framework state up to $S_3$-benchmark-ready.

Total wallclock from start to end of priorities 1–4: **~8–10 h** on
two GPUs. All single-seed.

Any blocker, missing infrastructure, or ambiguous result — escalate by
documenting the state in the run's `diagnostics.json` plus a terminal
`train.log` tail, and pause before proceeding. Do not force through
unclear results.
