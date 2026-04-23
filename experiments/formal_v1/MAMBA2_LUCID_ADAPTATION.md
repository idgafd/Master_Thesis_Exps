# Mamba-2 Analysis and LUCID Adaptation

**Scope.** Two connected threads: (a) what the Mamba-2 experiments have
actually established about Mamba-2's native axis coverage on our spine,
and (b) how LUCID — previously marked "architecturally incompatible"
with Mamba-2 — can in fact be adapted cleanly without materialising the
implicit full-sequence attention matrix.

**Status (2026-04-23).** Mamba-2 results are all in `STAGE10_PLAN.md`
§9.3 and `STAGE10_SUMMARY.md` §§6–8. This document is the Mamba-2-
specific synthesis + the LUCID adaptation spec the Stage-11 agent can
execute as P11.

---

## 1. Mamba-2 results on our spine — what we know

Causal Mamba-2 backbone, 7 M encoder, 6 layers, seed 42, 30 ep,
LibriSpeech clean-100. All test CER unless noted.

| Run | Test CER | Δ vs Mamba-2 vanilla | Interpretation |
|---|---:|---:|---|
| `mamba2` (vanilla) | **0.1192** | ref | Strongest single-arch baseline (beats RWKV-6 vanilla 0.1263) |
| `mamba2_convshift_multidil_symmetric` (broken init) | 0.1055 | −0.0137 | Single-dil + per-layer α effective mechanism |
| `mamba2_convshift_symmetric` (11.5b, single-dil) | 0.1044 | −0.0148 | Ties broken-init multidil within σ; scalar inert on Mamba-2 |
| `mamba2_rse_strong_viscosity` (11.2a) | 0.1183 | −0.0009 | **NULL / ambiguous-engaged** — RSE is null on Mamba-2 |
| **`mamba2_convshift_multidil_symmetric_v2` (P2 v2)** | **0.0967** | **−0.0225** | **Best single-backbone result on spine** — within σ of RWKV-6 CB-1 v2 composition |

### Axis-coverage reading of Mamba-2

Mamba-2's native primitives and what they cover:

| Native primitive | What it provides | Axis coverage |
|---|---|---|
| Selective $\Delta t$ | Content-dependent continuous-decay per channel per token | **Axis 1** (continuous-decay diversity, partial axis 5 for the content-dependence) |
| Short DWConv (k=4 by default) on input | Local temporal mixing, ±3 frames | **Axis 1** (short-range temporal) |
| SSD dual form (chunk-parallel scan) | Efficient training + inference | Not an axis — efficiency property |

What this predicts:

- **Axis 1 is partially pre-installed.** Multi-dilation multidil_v2 still
  helps (P2 v2 gain −0.0225) because Mamba-2's native DWConv is k=4
  short-range, not multi-scale; multidil adds {1,2,4,8}-scale coverage.
  But the marginal axis-1 gain is smaller than on RWKV-6 (−0.0263) and
  LA (−0.0501), consistent with Mamba-2 having the most native axis-1
  coverage of the three architectures.
- **RSE is null on Mamba-2** (11.2a, tied vanilla within σ) because RSE's
  complex-pole rotation is axis 1 — specifically, the oscillatory
  decay-dynamics sub-axis — and Mamba-2's selective $\Delta t$ already
  provides per-channel content-dependent decay. RSE's contribution on
  Mamba-2 is reparameterisation-absorbable by $\Delta t$.
- **Axis 2 (associative memory / key decorrelation) is NOT covered**
  by any Mamba-2 native primitive. Selective $\Delta t$ modulates
  decay rates — it does not decorrelate keys. Short DWConv smooths
  locally — it does not manage interference. **This is the standing
  gap in Mamba-2's native coverage.**

### Confirmed architecture-deficit-proportional transfer

| Axis | RWKV-6 gain | Mamba-2 gain | LA gain | Notes |
|---|---:|---:|---:|---|
| **Axis 1** (multidil_v2) | −0.0263 | −0.0225 | −0.0501 | Mamba-2 smallest gain (most native coverage); LA largest (no native coverage) |
| **Axis 1** (RSE + viscosity) | −0.0086 | −0.0009 (null) | −0.0779 | Same ordering; Mamba-2 null because $\Delta t$ substitutes |
| **Axis 2** (LUCID) | −0.0047 (`rwkv6_lucid`) | **untested** (this doc proposes the adaptation) | **untested** (P9 in queue) | Axis-2 gap on Mamba-2 unfilled |

Mamba-2's axis-1 and axis-5 are well-covered by native primitives; its
axis-2 is not. That's the gap LUCID (or a Delta-rule variant) could
fill, and the reason to test LUCID on Mamba-2 rather than pre-declaring
it incompatible.

---

## 2. Why "architecturally incompatible" was wrong

The previous argument (in `EXPRESSIVITY_AXES.md` §Axis 2 interim note):

> Mamba-2's SSD scan does NOT materialise an explicit attention matrix
> of the form $Y = AV$ that LUCID's $Y = P^{-1} AV$ preconditioner
> would apply to. [...] Skipped as structurally incompatible.

This conflates two separate claims. Reading the actual LUCID
implementation in `src/models/rwkv6_time_mix.py:2470` reveals the
conflation:

### Operational LUCID (what the code actually does)

LUCID is applied as a **value-preconditioner within chunks**, then the
recurrence runs on preconditioned values. The existing function
`_apply_lucid_recurrent(k, v, τ, chunk_size=64)` takes keys and values
and returns preconditioned values — no explicit attention matrix is
ever touched:

```python
for start in range(0, T, chunk_size):
    k_c = k[:, :, start:end, :]
    v_c = v[:, :, start:end, :]
    k_rn = sqrt_d * F.normalize(k_c, dim=-1)             # RMS-normalised keys
    gram = k_rn @ k_rn.transpose(-2, -1)                 # T_c × T_c Gram matrix
    P    = exp((τ * (gram / sqrt_d - sqrt_d)).clamp(-30, 30))
    P    = P + 1e-6 * I                                  # stability
    v_out = solve(P, v_c)                                # within-chunk linear solve
```

Algebraic identity: $\mathbf{O} = A \cdot P^{-1} \cdot V = A \cdot (P^{-1} V) = A \cdot \tilde{V}$.

Matrix multiplication is associative — you can apply $P^{-1}$ to $V$
first and never materialise $A$ explicitly. The code exploits exactly
this identity. Both `rwkv6_lucid` (pure recurrent scan, no explicit
attention) and `lion_lucid_chunked` (parallel chunked attention) use
this same $P$ + solve pattern; the *recurrent* variant specifically
works without any materialised attention matrix across the sequence.

### What Mamba-2's SSD dual form actually materialises

Mamba-2's chunk-parallel SSD dual form computes within a chunk of size $T_c$:

$$\mathbf{Y}_c = (\mathbf{L}_c \odot (\mathbf{C}_c \mathbf{B}_c^\top)) \, \mathbf{x}_c$$

where:
- $\mathbf{L}_c$ is the structured lower-triangular decay mask (from
  $\Delta t$ and $A$), chunk-local, deterministic at evaluation time;
- $\mathbf{C}_c \in \mathbb{R}^{T_c \times d_{\text{state}}}$ is the
  selective-output projection (**query analog**);
- $\mathbf{B}_c \in \mathbb{R}^{T_c \times d_{\text{state}}}$ is the
  selective-input projection (**key analog**);
- $\mathbf{x}_c \in \mathbb{R}^{T_c \times d_{\text{inner}}}$ is the
  input signal (value analog).

Mamba-2 **does** materialise $\mathbf{C}_c \mathbf{B}_c^\top$ within each
chunk — that is the computational trick that makes SSD training fast
via matrix multiplication rather than serial scan. Full $T \times T$
attention is never materialised; chunk-local $T_c \times T_c$
attention-like structure is, and it is exactly at the scale and form
LUCID operates on.

### Resolution

The claim "Mamba-2 doesn't materialise full attention" is correct. The
conclusion "therefore LUCID doesn't apply" is wrong. LUCID's chunked
formulation works at chunk-local granularity on keys and values —
exactly the granularity Mamba-2 already operates at. The two are
structurally compatible at the chunked granularity, which is the
granularity both the existing `_apply_lucid_recurrent` and the existing
Mamba-2 SSD dual-form already use.

---

## 3. Mamba-2-LUCID — mathematical spec

### Identification of roles

- **Keys (LUCID's $\mathbf{K}$) ↔ Mamba-2's $\mathbf{B}$.**
  In the SSD dual form, attention weight $A[t, s]$ depends on
  $\mathbf{C}_t^\top \mathbf{B}_s \cdot L[t,s]$. $\mathbf{B}$ is the
  side being compared against at read time — the key analog.
- **Values (LUCID's $\mathbf{V}$) ↔ Mamba-2's $\mathbf{x}$.**
  The signal being integrated into the state. Note: Mamba-2 also has a
  separate pre-projection $W_v$ applied to the input; LUCID can either
  precondition the pre-projection output or the post-projection $\mathbf{x}$.
  The post-projection version is the cleanest analog because it matches
  where LUCID inserts in the RWKV-6 time-mix path (after $W_v$).
- **Queries (LUCID's $\mathbf{Q}$) ↔ Mamba-2's $\mathbf{C}$.**
  Not used directly in LUCID's preconditioner construction — LUCID
  builds $P$ from $\mathbf{K}$ only.

### The adaptation (Alt 0 — faithful port)

Within each SSD chunk of size $T_c$ (default 64 in Mamba-2):

1. **Normalise the B projection (RMS-norm)**
   $$\mathbf{B}_c^{\text{RN}} = \sqrt{d_{\text{state}}} \cdot \frac{\mathbf{B}_c}{\|\mathbf{B}_c\|_2}$$
   per-row normalisation. $\mathbf{B}_c^{\text{RN}}$ has shape
   $T_c \times d_{\text{state}}$ with row norms $\sqrt{d_{\text{state}}}$.

2. **Gram matrix (B-correlation)**
   $$\mathbf{G}_c = \mathbf{B}_c^{\text{RN}} (\mathbf{B}_c^{\text{RN}})^\top$$
   shape $T_c \times T_c$. Diagonal equals $d_{\text{state}}$ by construction.

3. **Preconditioner (unit-diagonal by construction)**
   $$\mathbf{P}_c = \exp\left(\tau_h \cdot \left(\frac{\mathbf{G}_c}{\sqrt{d_{\text{state}}}} - \sqrt{d_{\text{state}}}\right)\right) + \varepsilon \mathbf{I}$$
   where $\tau_h$ is a learnable per-head scalar (softplus-positive, as in LUCID original), and $\varepsilon = 10^{-6}$ for numerical stability. The `- √d` term makes diagonal $\exp(0) = 1$.

4. **Value-side solve**
   $$\tilde{\mathbf{x}}_c = \mathbf{P}_c^{-1} \mathbf{x}_c$$
   chunk-local $T_c \times T_c$ linear solve. Use `torch.linalg.solve`
   in fp32 for numerical conditioning (as in the existing RWKV-6 LUCID
   code path), cast back to the working dtype.

5. **Feed preconditioned $\tilde{\mathbf{x}}_c$ into the SSD scan unchanged.**
   The rest of Mamba-2 (decay $\mathbf{L}_c$, state read via
   $\mathbf{C}_c$, inter-chunk state transport) is untouched.

6. **Inter-chunk state update.** The recurrent state that carries across
   chunk boundaries uses the chunk's original $\mathbf{B}, \mathbf{C}$
   and the preconditioned $\tilde{\mathbf{x}}_c$. The state equation
   shape is unchanged; only $\mathbf{x}_c$ is replaced by $\tilde{\mathbf{x}}_c$
   in the chunk-local contribution to the state. This is identical to how
   `lion_lucid_chunked` passes state across chunks.

### Computational cost per chunk per head

- Gram matrix: $O(T_c^2 \cdot d_{\text{state}})$
- Preconditioner build (elementwise exp): $O(T_c^2)$
- Linear solve: $O(T_c^3)$
- Total added: $O(T_c^3 + T_c^2 \cdot d_{\text{state}})$

At $T_c = 64, d_{\text{state}} = 64$: $\sim 0.5 \text{M}$ FLOPs per chunk
per head. Mamba-2's SSD scan within the same chunk costs roughly
$O(T_c^2 \cdot d_{\text{state}} \cdot d_{\text{inner}}) \sim 16\text{M}$ FLOPs.
LUCID adds ~3% overhead. Well within Mamba-2's efficiency envelope.

### Engineering scope

Estimated 60–100 lines plus tests. Specific insertion points:

1. **`src/models/mamba2_block.py`** — find the SSD dual-form chunk block
   (where $\mathbf{C}_c \mathbf{B}_c^\top$ is computed). Insert the
   preconditioner spec above before the $(\ldots) \, \mathbf{x}_c$
   multiplication. Add kwargs `lucid: bool`, `lucid_chunk_size: int`.
2. **`src/models/mamba2_block.py` (init)** — add parameter
   `self.lucid_temperature = nn.Parameter(torch.zeros(n_heads))` if
   `lucid`; use `F.softplus(self.lucid_temperature)` when applying.
3. **`src/models/mamba2_encoder.py`** — propagate `lucid` and
   `lucid_chunk_size` kwargs through to the block.
4. **`src/models/encoder.py`** — add `mamba2_lucid` to mode_map; verify
   the `lucid` substring already triggers the right flag (it does in the
   RWKV-6 path; may need to wire it into the Mamba-2 construction path
   similarly).
5. **`tests/test_mechanisms.py`** (or new `tests/test_mamba2_lucid.py`) —
   add gradient-flow test: verify the preconditioner parameters receive
   non-zero gradient under a dummy loss; verify forward pass is
   numerically stable at fp32 and fp16; verify the `lucid=False` path
   reproduces vanilla Mamba-2 output bit-exactly.

No changes to the mechanism class itself (LUCID is just a mathematical
operation applied per-chunk; no new file needed).

---

## 4. Alternative formulations — named but not recommended as default

**Alt 1 — State-covariance preconditioner.** Maintain a running
state-covariance estimate $\mathbf{\Sigma}_t = \lambda \mathbf{\Sigma}_{t-1} + \mathbf{h}_t \mathbf{h}_t^\top$,
precondition the state at readout with $(\mathbf{I} + \tau \mathbf{\Sigma}_t)^{-1}$.
Decorrelates state dimensions rather than output contributions.
**Different semantics** from LUCID (state-side decorrelation, not
key-similarity-aware value decorrelation). Worth trying only if Alt 0
is null; not the default.

**Alt 2 — C-correlation instead of B-correlation.** Build $\mathbf{P}_c$
from $\mathbf{C}_c \mathbf{C}_c^\top$ rather than $\mathbf{B}_c \mathbf{B}_c^\top$.
Non-standard LUCID variant — the paper's formulation uses keys, not
queries. Skip unless Alt 0 is null AND we have a specific argument for
why query-side decorrelation would help.

**Alt 3 — Combined B+C correlation.** $\mathbf{G}_c = \alpha \mathbf{B}_c \mathbf{B}_c^\top + (1-\alpha) \mathbf{C}_c \mathbf{C}_c^\top$.
Arbitrary interpolation; no principled motivation. Skip.

**Alt 4 — Pre-projection preconditioning.** Apply $\mathbf{P}^{-1}$ to
the input before the Mamba-2 input projection $W_v$, not after.
Algebraically different from Alt 0 because of the linear projection
interaction. Worth noting exists; Alt 0 (post-projection) is the
cleaner mathematical analog of the RWKV-6 LUCID path.

**Default: Alt 0 only.** Run Alt 1 as a follow-up if Alt 0 is null and
the thesis needs the coverage; otherwise skip the alternatives.

---

## 5. Pre-registered predictions and decision rule

### Single-mechanism test: `mamba2_lucid`

Pre-registered cohort:
- **Primary reference:** Mamba-2 vanilla test CER 0.1192.
- **Axis-2 neighbour:** `rwkv6_lucid` test CER 0.1216 (−0.0047 vs RWKV-6 vanilla, on LibriSpeech).
- **Hypothetical analog for scale:** LA + RSE+viscosity gain of −0.078 vs LA vanilla — an *upper-bound* reference for "large axis-deficit mechanism provides large gain."

Predicted band:
- **BREAK:** dev < 0.109 (−0.010 vs vanilla). Would indicate axis-2
  is a real gap in Mamba-2's coverage and LUCID fills it meaningfully.
- **MARGINAL:** dev 0.109 – 0.116 (−0.003 to −0.010 vs vanilla). Axis-2
  signal present but smaller than on RWKV-6 (−0.0047 analog) — could
  indicate Mamba-2's selective $\Delta t$ provides partial axis-2 coverage
  after all.
- **TIED:** dev 0.116 – 0.120 (within σ of vanilla). Mamba-2 has
  effective axis-2 coverage via native primitives; LUCID is redundant.
- **REGRESSION:** dev > 0.120. Destructive; investigate preconditioner
  conditioning + numerical stability.

The TIED outcome would be genuinely surprising and informative — it
would suggest either (a) some Mamba-2 primitive we haven't identified
provides implicit key decorrelation, or (b) the particular LUCID
adaptation doesn't capture what's needed on Mamba-2 (try Alt 1).

### Composition test: `mamba2_lucid_convshift_multidil_symmetric_v2`

Pre-registered cohort:
- **Primary reference:** P2 v2 Mamba-2 + multidil_v2 test CER 0.0967.
- **Axis-2 alone:** `mamba2_lucid` result from the single-mechanism test.

Predicted bands:
- **BREAK:** test < 0.090. Axis-2 × axis-1 compose additively on Mamba-2.
  Largest causal single-arch result in the thesis. Paper-worthy.
- **MARGINAL:** test 0.090 – 0.097. Within σ of P2 v2; axis-2
  contribution partially redundant with axis-1 gains on Mamba-2.
- **TIED:** test 0.097 – 0.100. LUCID absorbed by working multidil.
- **REGRESSION:** test > 0.100. Destructive composition; rare but
  possible (see P5 v2 CB-3 result on RWKV-6).

### Cross-architecture differential: the thesis-scale claim

With LUCID on all three architectures, we'd have a three-way axis-2
transfer matrix mirroring the axis-1 multidil transfer and axis-2/3
RSE transfer:

| Architecture | Predicted LUCID gain | Basis |
|---|---:|---|
| RWKV-6 | ~−0.005 (measured) | Partial decay diversity + some key structure |
| Mamba-2 | unknown, predicted ~−0.005 to −0.015 | No axis-2 coverage; moderate gain |
| LA | unknown, predicted large (~−0.03 to −0.08) | No axis-2 coverage and no decay diversity; largest gap |

If Mamba-2's gain lands between RWKV-6's and LA's, the
architecture-deficit-proportional transfer claim is confirmed on
axis 2 as well as axes 1 and 2/3. **Three-axis × three-architecture
differential transfer matrix with pre-registered predictions — a
thesis-grade result.**

---

## 6. Queue integration

Proposed update to `STAGE11_AGENT_QUEUE.md` Priority 1 §Mamba-2 LUCID
section:

> **P11 (was: feasibility-check-first) → definite run after P7 + P9
> land.** Implement the Alt-0 B-correlation adaptation per
> `MAMBA2_LUCID_ADAPTATION.md` §3. ~60–100 lines + unit tests.
> Run order:
>
> 1. After P7 (RWKV-6 LUCID × multidil_v2) and P9 (LA LUCID) show the
>    LUCID family's behavior on two architectures.
> 2. Implement the Mamba-2 SSD chunk-local preconditioner per §3.
> 3. Unit tests pass.
> 4. Run `mamba2_lucid` — single-mechanism test, pre-registered decision rule in §5 above.
> 5. Conditional on MARGINAL+ result: run `mamba2_lucid_convshift_multidil_symmetric_v2` as the Mamba-2 cross-axis composition.

The "skipped as architecturally incompatible" note in
`EXPRESSIVITY_AXES.md` §Axis 2 should be updated to cite this doc and
retract the incompatibility claim, replaced by: *"LUCID adapts to
Mamba-2 via chunk-local B-correlation preconditioning; spec in
`MAMBA2_LUCID_ADAPTATION.md`. Priority 1 P11 in the agent queue."*

---

## 7. Thesis-level implications if Mamba-2-LUCID works

1. **Axis-2 cross-architecture transfer confirmed across three architectures**
   — RWKV-6, LA, Mamba-2. Differential prediction set becomes
   testable across three cells on axis 2, matching the axis-1 and
   axis-2/3 (RSE) three-cell matrices already confirmed.

2. **Mamba-2's "native-primitive coverage" story sharpens.** Current
   reading: "Mamba-2 covers axis 1 via $\Delta t$ + DWConv."
   After Mamba-2-LUCID: "Mamba-2 covers axes 1 and 5 natively; axis 2
   requires an added mechanism." The differential between RSE null
   (axis-1-redundant with $\Delta t$) and LUCID gain (axis-2 fills a
   real gap) is exactly the kind of clean mechanism-architecture
   matching the thesis is built around.

3. **A new best Mamba-2 result is plausible.** P2 v2 at 0.0967 is the
   current single-backbone leader on the spine. Mamba-2-LUCID × multidil
   could produce a sub-0.09 test CER, potentially matching or beating
   LION in 30-ep causal vs 80-ep bidirectional regimes.

If Mamba-2-LUCID is null — also informative: it would suggest
Mamba-2's selective $\Delta t$ provides implicit key decorrelation we
haven't yet analytically characterised. That's a thesis-discussable
finding in its own right.

---

## 8. Summary

- **Mamba-2's axis coverage:** 1 (mostly, via $\Delta t$ + DWConv),
  5 (partially, via selective content-dependence). **NOT axis 2.**
- **The previous "LUCID is structurally incompatible with Mamba-2"
  conclusion was wrong.** It conflated "no full-T attention materialised"
  with "no LUCID applies." LUCID operates on chunk-local values via
  key-correlation preconditioners; Mamba-2's SSD dual form already
  chunk-materialises attention-like structure at matching granularity.
- **Concrete adaptation (Alt 0):** chunk-local preconditioner built from
  B-correlation ($T_c \times T_c$), value-side solve for $\tilde{\mathbf{x}}_c$,
  unchanged SSD scan on preconditioned values, unchanged inter-chunk state
  transport. ~3% computational overhead. Mathematically faithful to LUCID.
- **Engineering scope:** 60–100 lines in `mamba2_block.py` + kwargs
  propagation + unit tests. Same complexity as the existing
  multidil_v2 dispatch wiring done in commit `848c3fb`.
- **Pre-registered decision rule** in §5; predicted MARGINAL band for
  single-mechanism Mamba-2-LUCID, BREAK band reasonable for
  composition with multidil_v2.
- **Thesis value if successful:** three-architecture × three-axis
  differential transfer matrix — axis-1 (multidil), axis-1 (RSE+visc),
  axis-2 (LUCID) — with pre-registered predictions and clean measured
  ordering LA > RWKV-6 > Mamba-2 within each axis proportional to
  native-primitive coverage.
- **Thesis value if null:** Mamba-2 has implicit key decorrelation via
  some primitive we haven't analytically identified yet. Different
  result, still informative.

Either outcome is worth the ~60–100 lines and one run.
