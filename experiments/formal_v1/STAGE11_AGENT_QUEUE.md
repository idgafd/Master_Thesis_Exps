# Stage 11 Agent — Next-Experiment Queue

**Audience:** the Stage-11 agent on this instance (the one that ran
11.0, 11.1, 11.2, 11.5). You have both GPUs free now. Another agent
on a separate instance completed P1 v2 + P4 v2 (init-fix + CB-1 v2);
their work is already committed to main at `3af846d` and `e9f6d10`.
**Do not run P1 or P4 again — they are done.**

**Current state (2026-04-23):**
- Init fix for `MultiDilationDWConvShift` is in `src/models/mechanisms/conv_shift.py`. Verify by reading the `__init__` — α_{d>1} should init to 0.01 and non-main branches should have `nn.init.normal_(std=0.01)`.
- New causal RWKV-6 ceiling: dev **0.0973 / test 0.0961** (CB-1 v2). See `STAGE10_PLAN.md` §9.2.
- Priority order driven by `SHAPING_THE_THESIS.md` §Priority order (updated 2026-04-23).
- MQAR / axis-2 work is on a separate track — **not your scope**.

**Binding discipline** (carry-over from Stage-10):
- All new fixed-init reruns use the `_v2` suffix in backbone name and output directory. Do not overwrite any existing directory.
- `diagnostics.json` is mandatory for each run per `STAGE10_PLAN.md` §7.5. Log α_d per layer per epoch at minimum; mechanism-specific probes listed below.
- Matched-epoch 15 halt criterion fires if dev CER is ≥ +0.006 behind the stage-specific primary reference.
- Seed 42, 30 ep, canonical LibriSpeech clean-100 spine unless noted.

---

## Priority 1 — P5 and P6 parallel on 2 GPUs (~1.5 h wallclock)

Close the v2 composition matrix on RWKV-6. Both runs use the fixed-init
`MultiDilationDWConvShift` (already in `conv_shift.py`); no new code
changes in the mechanism file.

### P5 — `rwkv6_convshift_multidil_symmetric_gated_v2` (CB-3 v2)

- **Backbone:** `rwkv6_convshift_multidil_symmetric_gated_v2`
- **Output dir:** `outputs/rwkv6_convshift_multidil_symmetric_gated_v2_seed42/`
- **Launch on:** GPU 0 (or whichever is free first)
- **Runtime:** ~1.5 h
- **Substring-dispatch requirement:** the substring `gated` must continue to trigger the content-conditional softmax-α path (existing in broken-init CB-3). Verify by reading `encoder.py` / `rwkv6_time_mix.py` for the `gated_alpha` branch.

**What it tests:** content-conditional softmax-α (from CB-3) on top of *working* multi-dilation. CB-3 broken-init tied multidil_sym within σ; the gate had nothing to gate because d>1 branches were inert. With fixed init, the gate has four real branches to mix. Two outcomes:
- Gate learns to shift α toward specific (layer, token) dilation preferences → content-conditional RF is an axis beyond static α_d.
- Gate stays near uniform / matches the fixed α_d pattern of P1 v2 → content-conditional RF adds nothing; static α_d is sufficient.

**Mandatory diagnostics in `diagnostics.json`:**
- α_d per layer at epochs {5, 10, 15, 20, 25, 30}
- Gate output distribution per layer (mean + percentiles) at ep 15 and 30
- Per-token α variance — does the gate produce materially different α at different tokens?

**Decision rule:**
- **BREAK** (dev < 0.0960): gated α is a genuinely productive sub-axis; content-conditional RF selection matters.
- **MARGINAL** (dev 0.0960–0.1010): within σ of P1 v2 0.1013; gate adds marginal value, confirm with second seed.
- **TIED** (dev 0.1010–0.1030): content-conditional α adds nothing beyond static multi-dilation; close the sub-axis.
- **REGRESSION** (dev > 0.1030): gate destabilises training relative to P1 v2; investigate.

### P6 — `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2` (CB-7 v2)

- **Backbone:** `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2`
- **Output dir:** `outputs/rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2_seed42/`
- **Launch on:** GPU 1
- **Runtime:** ~1.5 h
- **Substring-dispatch requirement:** `qtail_lowrank_all` triggers the Kronecker feature-side lift (Stage 6 mechanism).

**What it tests:** cross-axis composition of channel-side Kronecker (axis 5, mild Stage-6 ~1σ gain) × working multi-dilation (axis 1, P1 v2 at 0.1013). CB-7 broken-init tied multidil_sym at 0.1159. With fixed multidil, CB-7 v2 tests whether axis-5 and axis-1 compose orthogonally or remain independent.

**Pre-registered prediction:** the two axes are structurally orthogonal (position-wise temporal mixing × channel-wise quadratic), so additive composition is plausible. Expected dev CER band: 0.0955–0.1005, with BREAK possible.

**Mandatory diagnostics:**
- α_d per layer at {5, 10, 15, 20, 25, 30}
- qtail branch magnitude (Kronecker contribution norm) per layer at ep 15, 30
- Effective rank of Kronecker feature lift via SVD of `qtail_W` at ep 30

**Decision rule:**
- **BREAK** (dev < 0.0960): axis-1 × axis-5 composition works. Strong evidence for cross-axis additivity; thesis-paper-worthy.
- **MARGINAL** (dev 0.0960–0.1010): tied P1 v2; qtail adds within σ; multi-seed gate.
- **TIED** (dev 0.1010–0.1030): qtail adds nothing on top of working multidil; axis-5 contribution absorbed.
- **REGRESSION** (> 0.1030): destructive composition; investigate.

### After P5 + P6 finish

Decision tree:
- If both land ≤ P1 v2 (≤ 0.1013): the broken-init CB-sprint "invariant extension" narrative is fully retracted. Compositions on top of working multidil produce further gain. Flag for coordinated rewrite of `STAGE10_SUMMARY.md` §3 and §4.
- If both tie P1 v2 within σ: CB-1 v2 was the *only* productive composition; gated α and Kronecker don't compound further. Update composition-candidates interpretation.
- If one BREAKs and one doesn't: the axis-pair structure of CB-1 (axis 1 × axis 1 sub-axes) and CB-7 (axis 1 × axis 5) differentiate; the result guides Stage 12 LION composition queue.

**Commit + push after P5, P6 finish.** Two new entries in `STAGE10_PLAN.md §9.2` table; brief caveat in `STAGE10_SUMMARY.md` §3; update `SHAPING_THE_THESIS.md` §Priority order to mark P5/P6 complete.

---

## Priority 2 — Dispatch wiring + P2/P3 cross-architecture v2 transfer (~2 h wallclock, +30–60 min engineering)

Current state: only `rwkv6_*_v2` backbones dispatch to the fixed-init
`MultiDilationDWConvShift`. Mamba-2 and LA paths have their own
convshift wiring (existing broken-init runs `mamba2_convshift_multidil_symmetric`
and `linear_attn_convshift_multidil_symmetric`). You need to wire the
`_v2` substring into those architectures' dispatch paths too.

### Engineering step — wire `_v2` into Mamba-2 and LA dispatch (30–60 min)

In `src/models/encoder.py` (or wherever the dispatch logic lives):

1. Register new backbone names in `mode_map` (or equivalent):
   ```python
   "mamba2_convshift_multidil_symmetric_v2": "recurrent",
   "linear_attn_convshift_multidil_symmetric_v2": "causal",
   ```
2. Substring dispatch for `_v2` → fixed-init mode. Check whether the
   fix-agent already added a config flag (e.g., `conv_shift_multidil_fixed_init`).
   If yes, propagate the flag into the Mamba-2 and LA module
   instantiations. If no, simply check the `_v2` substring in the
   architecture-specific convshift wiring (Mamba-2's `mamba.conv1d.alpha`
   path, LA's `premix.alpha` path) and pass `fixed_init=True`.
3. Write a unit test (mirror of `test_multidil_init_gradient_flow_symmetric`
   in `tests/test_mechanisms.py`) that exercises Mamba-2 and LA `_v2`
   paths for gradient flow through the d>1 branches. This must pass
   before training launches.

Keep the change minimal: the `MultiDilationDWConvShift` class itself
is already fixed. You're only adding dispatch paths on two new
architectures.

### P2 — `mamba2_convshift_multidil_symmetric_v2`

- **Backbone:** `mamba2_convshift_multidil_symmetric_v2`
- **Output dir:** `outputs/mamba2_convshift_multidil_symmetric_v2_seed42/`
- **Launch:** as soon as dispatch wiring is verified and test passes
- **Runtime:** ~1.5 h

**What it tests:** whether multi-dilation actually engages on Mamba-2
when the gradient trap is fixed. Broken-init Mamba-2 multidil
(`mamba2_convshift_multidil_symmetric` at dev 0.1079) tied the
single-dilation control (`mamba2_convshift_symmetric` at 0.1074) —
confirming the broken-init multidil on Mamba-2 was effectively
single-dilation. **The open question:** does multi-dilation actually
help on Mamba-2, or does Mamba-2's native short DWConv substitute
for what wider dilations would add?

**Pre-registered prediction:** gain smaller than on RWKV-6 because
Mamba-2's built-in short DWConv already covers the d=1 scale.
Expected dev: 0.093–0.100 range. If the gain matches P1 v2's
magnitude (~0.014), the differential-transfer prediction weakens.
If the gain is substantially smaller (~0.005), the
architecture-deficit-proportional prediction holds.

**Mandatory diagnostics:**
- α_d per layer at {5, 10, 15, 20, 25, 30}
- Mamba-2 native DWConv weight statistics at ep 30 — do they change
  when the external multi-dilation is active?
- Δt selective-gate distribution — does multi-dilation interact with
  Mamba-2's selective decay?

### P3 — `linear_attn_convshift_multidil_symmetric_v2`

- **Backbone:** `linear_attn_convshift_multidil_symmetric_v2`
- **Output dir:** `outputs/linear_attn_convshift_multidil_symmetric_v2_seed42/`
- **Launch:** in parallel with P2 on the other GPU
- **Runtime:** ~1.5 h

**What it tests:** whether multi-dilation transfers cleanly to LA
(no native local-bias path at all) with the init fix applied. Broken-init
LA multidil at 0.1978 was massive improvement over LA vanilla (0.2235),
entirely driven by the d=1 single-dilation component. If multi-dilation
engages on LA too, the gain should be substantially larger than on
RWKV-6 (since LA has the most structural room for local-bias
restoration).

**Pre-registered prediction:** largest absolute multi-dilation gain of
the three architectures. Expected dev: 0.16–0.19 range (vs 0.1978
broken-init, vs vanilla LA 0.2235). If the improvement is ~0.03 or
more, the axis-1 cross-architecture transfer claim is empirically
validated at its sharpest.

**Mandatory diagnostics:**
- α_d per layer at {5, 10, 15, 20, 25, 30}
- LA running-sum normaliser Z_t stability — does ConvShift pre-mixing
  affect the L1 denominator convergence?

### After P2 + P3 finish

Commit + push. Two rows added to `STAGE10_PLAN.md §9.3` master matrix.
Brief caveat in `STAGE10_SUMMARY.md §8` (Stage 11.1 reinterpretation)
noting the cross-architecture v2 transfer results. Update
`SHAPING_THE_THESIS.md §Priority order` to mark priority 2 complete.

**Decision tree:**
- If P2 and P3 gain ≈ P1 v2 gain across all architectures: architecture-
  independent axis-1 mechanism. Simpler thesis framing.
- If gain ordering LA > Mamba-2 > RWKV-6 matches the pre-registered
  differential prediction: confirms axis-deficit-proportional transfer
  *with working multi-dilation*, strengthening both the cross-architecture
  claim AND the axis-decomposition framework.
- If any architecture shows near-zero gain: that architecture's native
  structure substitutes for multi-dilation; informative null.

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

- **Don't re-run P1 or P4.** They're done, committed at `3af846d`
  and `e9f6d10`.
- **Don't modify `src/models/mechanisms/conv_shift.py`.** The init
  fix is already in. Read it to understand, don't edit.
- **Don't touch `outputs/rwkv6_convshift_multidil_symmetric_v2_seed42/`
  or `outputs/rwkv6_rse_convshift_multidil_symmetric_v2_seed42/`.**
  Those are the other instance's committed results.
- **Don't run MQAR or any synthetic benchmark** until Priorities 1–4
  close. Another agent handles axis 2.
- **Don't overwrite existing broken-init result directories.** Use
  `_v2` naming for all fixed-init reruns.
- **Don't skip `diagnostics.json`.** The Stage-10 procedural gap
  (no mechanism probes logged) must not repeat. Extract α_d values
  at minimum; log to `diagnostics.json` via the existing probe
  infrastructure.

---

## Workflow summary

1. **Now:** Launch P5 on GPU 0, P6 on GPU 1 in parallel. ~1.5 h wallclock.
2. **+1.5 h:** Check results; commit with `_v2` rows added to `STAGE10_PLAN.md §9.2` and a brief caveat in `STAGE10_SUMMARY.md §3`. Push.
3. **+1.5 h:** Engineering — wire `_v2` dispatch for Mamba-2 and LA (~30–60 min). Write unit tests. Verify gradient flow.
4. **+2.5 h:** Launch P2 on GPU 0, P3 on GPU 1 in parallel. ~1.5 h wallclock.
5. **+4 h:** Commit P2/P3 results. Push.
6. **Then:** CB-2 wide4 + dense (priority 3), 11.5c retest (priority 4).
7. **At end of all priorities 1–4:** coordinated doc revision — update `SHAPING_THE_THESIS.md`, `STAGE10_SUMMARY.md`, `EXPRESSIVITY_AXES.md` to reflect the full v2 matrix. Then push the thesis framework up to $S_3$-benchmark-ready state.

Total wallclock from start to end of priorities 1–4: **~7–8 h** on
two GPUs. All single-seed.

Any blocker, missing infrastructure, or ambiguous result — escalate
by documenting the state in the run's `diagnostics.json` plus a
terminal `train.log` tail, and pause before proceeding. Do not force
through unclear results.
