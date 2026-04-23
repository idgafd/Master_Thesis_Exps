# Mamba-2 Write-Novelty Gate — Design Doc + Results

**Status:** implemented, two-run ablation complete. **Verdict: H2 confirmed — structurally inert at this probe.** See §12 Results (2026-04-23).

**Authors:** thesis discussion, 2026-04-23.

**Relation to existing work:**
- Extends `MAMBA2_LUCID_ADAPTATION.md` (chunk-local LUCID, now confirmed as +0.0079 test CER gain, ~6σ, single seed 42).
- Targets a different Mamba-2 bottleneck than chunk-local LUCID. See §2 below.
- Composable with B-side LUCID (both mechanisms are orthogonal in their targets).

---

## 1. Why this idea exists

Chunk-local LUCID (`mamba2_lucid` B-side) gives a clean Stage-11 result:

| Mechanism | Test CER | Δ vs vanilla |
|---|---|---|
| vanilla mamba2 | 0.1192 | — |
| mamba2_lucid (B-side) | 0.1113 | −0.0079 (~6σ) |

The diagnostic triad (ep15 checkpoint of seed-42) established that:

- Intra-chunk B-correlation stays at ratio 0.91–0.98 throughout training — Mamba-2's in_proj **does not** learn to orthogonalise B (reparam-absorption hypothesis falsified).
- Inter-chunk B-correlation is moderate (ratio 0.67–0.83, cos ≈ 0.2–0.25) and SGD modestly self-decorrelates it.
- τ engages modestly (1.0 → 1.28 mean, max 1.48 at L4). Gap stable at +0.005–0.006 through ep18, final ep30 test −0.0079.

**The open question after chunk-local LUCID:** is the +0.0079 ceiling set by LUCID's scope (chunk-local only), or by what LUCID-on-inputs can achieve at all?

The analysis that led to this doc: chunk-local LUCID addresses the **within-chunk correlation** bottleneck, but **not** the "write-blind-to-state" bottleneck of Mamba-2's recurrence. Those are structurally different bottlenecks, and the kernel machinery underlying LUCID can be reinterpreted to target the second one directly.

Rejected alternatives before landing here:

- **Cross-chunk LUCID (Nyström summary of past B):** diagnostic showed inter-chunk cos ≈ 0.2; predicted gain +0.002–0.005, not worth infrastructure.
- **Multi-scale LUCID (T_c ∈ {32,64,128,256}):** rejected — this is axis-1 multidil relabelled; does not touch state update.
- **Increasing chunk size:** rejected — no new mathematics; compute-for-gain tradeoff only.

---

## 2. The Mamba-2 state-update bottlenecks

From Mamba-3 (arXiv:2603.15569) and the linear-attention lineage:

| # | Bottleneck | Mamba-3 fix | Current address |
|---|---|---|---|
| 1 | Rank-1 per-token write ($B_t \otimes X_{d,t}$) | MIMO (rank R) | — |
| 2 | **Write-blind-to-state** (additive accumulation ignores existing state) | — (not fixed) | **this doc** |
| 3 | Real-only exponential decay (no rotation) | Complex SSM / SO(2) blocks | RSE on RWKV-6 |
| 4 | Point-evaluation discretisation (single-term input) | Exp-trapezoidal (2-band) | multidil ConvShift |

Bottleneck (2) — write-blind-to-state — is the one neither Mamba-3 nor our prior mechanism catalog cleanly addresses. Delta rule (DeltaNet, RWKV-7) touches it with rank-1 projection-based replacement; novelty gating is a different mathematical route to the same bottleneck.

**Precise statement of bottleneck (2):** Mamba-2's update $s_t = \alpha_t s_{t-1} + B_t \otimes X_{d,t}$ adds $B_t \otimes X_{d,t}$ regardless of whether the direction $B_t$ is already saturated in $s_{t-1}$. When $B$ stays correlated (which the diagnostic confirms it does), the same directions get written repeatedly, causing **accumulation collapse** — later writes conflate with earlier ones along the dominant key subspace.

---

## 3. The mechanism — state-write novelty gate

### 3.1 Recurrence

Replace the standard SSM update with:

$$s_t = \alpha_t s_{t-1} + \omega_t \cdot (B_t \otimes X_{d,t})$$

where $\omega_t \in (0, 1]$ is a scalar **novelty gate** per token per head.

### 3.2 Gate computation

$$\omega_t = \frac{1}{1 + \gamma_h \big/ (B_t^\top \Sigma_c^{-1} B_t + \epsilon)}$$

- $\Sigma_c$: per-head running key covariance, frozen within chunk $c$, updated at chunk boundaries.
- $\gamma_h$: per-head learnable gate strength. Parametrised as $\gamma_h = \text{softplus}(\gamma_{h,\text{raw}})$.
- $\epsilon = 10^{-6}$: numerical safety.

**Why this form:**

- $B_t^\top \Sigma_c^{-1} B_t$ is the **Mahalanobis norm** of $B_t$ against the state's accumulated key distribution. Large when $B_t$ is novel (orthogonal to already-covered directions), small when redundant.
- $\omega_t$ is a bounded saturation function in $(0, 1]$: → 1 when Mahalanobis is large (novel), → 0 when small (redundant).
- $\gamma_h$ controls the threshold where attenuation kicks in.

### 3.3 $\Sigma$ update

At chunk boundary:

$$\Sigma_c = \bar{\alpha}_{c-1} \cdot \Sigma_{c-1} + B_{c-1}^\top B_{c-1} + \epsilon_{\text{reg}} I$$

where:
- $B_{c-1}$ is the $(T_c, N)$ key tensor for chunk $c-1$.
- $\bar{\alpha}_{c-1} = \prod_{t \in c-1} \alpha_t$ is the chunk-product decay.
- $\epsilon_{\text{reg}} = 10^{-3}$: regulariser ensuring invertibility.

Then compute $\Sigma_c^{-1}$ once per chunk.

### 3.4 Initialisation (exact reduction to vanilla Mamba-2)

- $\gamma_{h,\text{raw}} = -30$ → $\gamma_h \approx 10^{-13}$.
- First-chunk $\Sigma_0 = \epsilon_{\text{reg}} I$.
- At init: $\gamma_h / (B^\top \Sigma^{-1} B + \epsilon) \approx 10^{-16}$ → $\omega_t \equiv 1$ within fp32 noise floor.
- Recurrence is exactly vanilla Mamba-2 at init. SGD grows $\gamma_h$ only if the gate is productive.

---

## 4. Connection to LUCID

Both mechanisms share the kernel-preconditioner machinery:

| | chunk-local LUCID | novelty gate |
|---|---|---|
| Target | value stream $X_d$ | state write $\omega_t$ |
| Kernel object | $P = \exp(\kappa(G - \sqrt{d}I))$ | $\Sigma = \sum \alpha B B^\top + \epsilon I$ |
| Scope | chunk-local (T_c tokens) | running (full sequence) |
| Operation | $P^{-1} X_d$ (whiten values) | $\omega_t = f(B_t, \Sigma_c^{-1})$ (gate write) |
| Bottleneck | within-chunk value redundancy | write-blind-to-state |
| Touches recurrence? | no (preprocessing) | **yes** (modifies state update) |

Abstractly: *use kernel-Gram-derived operators to modulate contributions based on key-similarity structure*. Chunk-local LUCID applies this at the value-preprocessing stage; novelty gate applies it at the write-gate stage. Same machinery, different point in the pipeline.

Composition: both mechanisms address different bottlenecks. Expected to be ~additive when composed.

---

## 5. Complexity

Per-layer FLOPs for $T = 1024, D = 1024, H = 16, N = P = 64, T_c = 64$:

| Component | FLOPs | Scaling |
|---|---|---|
| Transformer attention (for reference) | $\sim 2 \cdot 10^9$ | $T^2 \cdot D$ |
| Vanilla Mamba-2 SSD scan | $\sim 6.7 \cdot 10^7$ | $T \cdot H \cdot N \cdot P$ |
| + chunk-local LUCID | $+ \sim 6.7 \cdot 10^7$ | $T \cdot H \cdot T_c^2$ |
| **+ novelty gate (chunked)** | $+ \sim 6.7 \cdot 10^7$ | $T \cdot H \cdot N^2$ |

Per chunk per head:
- Σ update (matmul $B^\top B$): $O(T_c \cdot N^2)$
- Σ inverse: $O(N^3)$
- Per-token ω: $O(T_c \cdot N^2)$

Total across all chunks: $O(T \cdot N^2 + (T/T_c) \cdot N^3)$, still linear in $T$.

**Parallelism:** within-chunk ω computation is fully parallel (batched matmul); Σ sequential across chunks matches SSD's own chunk-sequential structure. No regression in GPU-friendliness.

**Memory:** adds $O(H \cdot N^2)$ per carry state (2 × 16 × 64² × 4B ≈ 512 KB in fp32 per batch per layer).

**Asymptotic advantage over Transformer preserved:** linear vs quadratic. At $T = 8192$, novelty-gated Mamba-2 is ~100× cheaper than Transformer attention.

---

## 6. First-run priority

**Chunked Σ + per-head scalar γ.** Rationale:

- **Chunked > per-token:** chunked is ~100 LoC, parallel-friendly, one matrix inverse per chunk. Per-token Sherman-Morrison is ~300 LoC, inherently sequential. If chunked shows no signal, per-token almost certainly won't either.
- **Per-head scalar γ > per-layer γ:** 16 extra scalars vs 96 scalars. Enough to detect "does SGD want the gate at all." Per-layer γ is a cheap follow-up if signal lands.

Deferred until first-run results:
- Per-token Sherman-Morrison Σ update.
- Per-layer γ.
- Weighted intra-chunk Σ accumulation (currently unweighted sum of $B B^\top$ within chunk).
- Composition with C-side LUCID.

---

## 7. Implementation instruction (self-contained, for agent)

### Files to change

1. `src/models/mamba2_kernels.py` — add `novelty_gate_strength: Optional[torch.Tensor] = None` and `sigma_state: Optional[torch.Tensor] = None` to `ssd_scan_causal`. Return updated sigma in the state output.
2. `src/models/mamba2_block.py` — add `use_novelty_gate: bool = False`. Create `self.novelty_gamma_raw = nn.Parameter(torch.full((self.nheads,), -30.0, dtype=dtype))`. Plumb through to `ssd_scan_causal` via `F.softplus(self.novelty_gamma_raw)`.
3. `src/models/encoder.py` or equivalent backbone registry — register `mamba2_novelty_gate` (novelty gate only) and `mamba2_lucid_novelty` (LUCID + novelty gate composed).
4. Carry-state `init_state` dict — add `"sigma": torch.zeros(B, H, N, N)` and stream across chunks.

### Core change in `ssd_scan_causal`

After existing chunking logic (around line 200 in current kernel), before `Y_diag` / `produced` einsums:

```python
EPS_REG = 1e-3
EPS_NUM = 1e-6

if novelty_gate_strength is not None:
    if sigma_state is None:
        Sigma = EPS_REG * torch.eye(N, device=X.device, dtype=torch.float32)
        Sigma = Sigma.view(1, 1, N, N).expand(Bsz, H, N, N).contiguous()
    else:
        Sigma = sigma_state.float()  # (B, H, N, N)

    omega_chunks = []
    for c in range(nC):
        # Per-chunk Sigma inverse
        Sigma_reg = Sigma + EPS_REG * torch.eye(N, device=X.device, dtype=torch.float32)
        Sigma_inv = torch.linalg.inv(Sigma_reg)           # (B, H, N, N)

        # Per-token Mahalanobis within chunk
        B_chunk_h = B_c[:, c].permute(0, 2, 1, 3).float() # (B, H, T_c, N)
        tmp = torch.einsum('bhtn,bhnm->bhtm', B_chunk_h, Sigma_inv)
        mahal = (tmp * B_chunk_h).sum(dim=-1)              # (B, H, T_c)

        gamma_h = novelty_gate_strength.view(1, H, 1)      # (1, H, 1)
        omega_c = 1.0 / (1.0 + gamma_h / (mahal + EPS_NUM))  # (B, H, T_c)
        omega_chunks.append(omega_c.permute(0, 2, 1))      # (B, T_c, H)

        # Update Sigma for next chunk
        alpha_chunk = torch.exp(A_cumsum[:, :, c, -1])     # (B, H)
        B_outer = torch.einsum('bhtn,bhtm->bhnm', B_chunk_h, B_chunk_h)
        Sigma = alpha_chunk.view(Bsz, H, 1, 1) * Sigma + B_outer

    omega = torch.stack(omega_chunks, dim=1)               # (B, nC, T_c, H)
    # Apply gate to X_c — scalar multiply broadcasts over P dim
    X_c = X_c * omega.unsqueeze(-1)

    final_sigma = Sigma
else:
    final_sigma = None
```

Return `final_sigma` alongside `final_state` in the scan output.

### Init-reduction smoke test (mandatory before training)

Add to `tests/` or as a one-off script:

1. With `novelty_gamma_raw = -30` (default init): `mamba2_novelty_gate(x)` produces output within `atol=1e-5` of `mamba2(x)` on the same input. Numerical reduction to vanilla holds.
2. With `novelty_gamma_raw = 0`: output differs from vanilla by more than noise floor. Confirms the mechanism is reachable.
3. Shape, dtype, device parity with vanilla.
4. Carry-state parity: running in two halves with state-carry produces same output as single-forward.

### Training runs

Seed 42, 30 epochs, otherwise identical config to `mamba2_lucid`.

**Run 1 — `mamba2_novelty_gate`** (novelty only, no LUCID). Primary experiment.

**Run 2 — `mamba2_lucid_novelty`** (LUCID + novelty composed). Launch only after Run 1 shows ep5–ep10 signal. Tests additivity.

### Logging

Per epoch: dev CER.

At ep0, ep5, ep15, ep30: per-layer per-head:
- `novelty_gamma_raw` values
- Mean Mahalanobis score $B^\top \Sigma^{-1} B$ over a fixed dev batch
- Mean $\omega_t$ over a fixed dev batch

### Report

1. Per-epoch dev CER: `mamba2_novelty_gate` vs `mamba2_lucid` vs `mamba2` vanilla.
2. Did $\gamma$ grow per layer? By how much?
3. Mean $\omega_t$ at ep30 — close to 1 (gate dormant) or substantially below (gate active)?
4. Final test CER for both runs.
5. One-line verdict: BREAK-band (≥ +0.015 over vanilla), MARGINAL (+0.005–0.015), null.

---

## 8. Predicted outcomes

**Expected best case (Run 2 composed):** dev CER at ep30 ≈ 0.108, test ≈ 0.105. That's roughly +0.014 over vanilla Mamba-2 test, or +0.006 over chunk-local LUCID alone. Would be BREAK-band on a composition claim.

**Plausible case (Run 1 alone):** dev CER ≈ 0.112, test ≈ 0.110. Comparable to LUCID's +0.0079; composition adds orthogonally.

**Null case:** $\gamma$ doesn't grow; $\omega_t$ stays at 1; dev CER matches vanilla. Would mean either (a) Mamba-2's decay naturally handles write-blind-to-state, or (b) the specific $\omega_t$ form is wrong and other formulations are worth trying (tanh, exponential, learned MLP).

---

## 9. What this unlocks for the thesis

If Run 1 lands ≥ MARGINAL and Run 2 shows additivity with LUCID:

> The LUCID kernel idea, generalised from input-value preprocessing to a state-write novelty gate, addresses the write-blind-to-state bottleneck that Mamba-2 inherits from linear attention. The mechanism is rank-1-preserving, linear-time (preserves Mamba-2's asymptotic advantage over Transformer), and reduces to vanilla Mamba-2 at init exactly. Chunk-local LUCID and novelty gating address orthogonal bottlenecks and compose.

This is a **recurrence-level mechanism contribution**, not just "another axis-2 transfer." The chapter-3 story becomes: "the kernel-preconditioner idea, instantiated at two different points in the computation, closes two of the four Mamba bottlenecks enumerated by Mamba-3 (plus the axis-1 closure from multidil ConvShift, total three of four)."

The remaining bottleneck (rank-1 per-token write) is the one MIMO structurally addresses and that this work deliberately does not touch.

---

## 10. Open follow-ups (only if Run 1 is positive)

- **Per-token Sherman-Morrison Σ update.** Higher-resolution gating, at cost of sequential compute.
- **Per-layer γ.** Tests whether the gate is productive at a specific depth.
- **Weighted intra-chunk Σ accumulation.** Currently chunk Σ adds tokens uniformly; weight by running decay for theoretical consistency.
- **Ablation against delta rule.** Delta rule (DeltaNet, RWKV-7) addresses the same bottleneck via rank-1 projection-replace. Side-by-side comparison would be thesis-grade.
- **Form of ω_t.** Alternative saturation functions: $\tanh$, $\exp(-\gamma/\text{mahal})$, learned scalar MLP. The specific form of the gate is a hyperparameter of the mechanism.

---

## 12. Results (2026-04-23)

### 12.1 Final CER — two-run ablation

| Run | Backbone | γ regime | Test CER | Δ vs vanilla mamba2 (0.1192) |
|---|---|---|---|---|
| Run 1 | `mamba2_novelty_gate` | γ trainable, softplus(γ_raw − 5), init γ_raw=0 → γ init ≈ 6.7e-3; grew to 1.1e-2 max by ep30 | **0.1186** | **−0.0006 (tied)** |
| Run 2 | `mamba2_novelty_fixed_g05` | γ = 0.5 fixed (buffer, no grad, no softplus) | **0.1187** | **−0.0005 (tied)** |

Reference points from the same training spine:
- `mamba2_lucid` (B-side): 0.1113 (−0.0079, ~6σ) — chunk-local LUCID preconditioner, the mechanism this doc was designed to extend.
- `mamba2_lucid_c` (C-side): 0.1109 (−0.0083, ~6σ).
- `mamba2_convshift_multidil_symmetric_v2` (axis-1 anchor on Mamba-2): 0.0969 (for scale reference).

The two novelty-gate runs differ in γ by 50× yet produce test CER within 0.0001 of each other, both **tied with vanilla** at the noise floor.

### 12.2 Parameterisation history (for reproducibility)

1. **Initial proposal (§4):** γ = softplus(γ_raw), γ_raw init = −30 → γ ≈ 9.4e-14 (near-bit-exact vanilla at init). **Abandoned pre-launch** — σ(−30) ≈ 9.4e-14 collapses Adam chain-rule scaling to ~1e-13; γ_raw would be Adam-unreachable (validated via gradient probe).
2. **Shifted-softplus (launched as Run 1):** γ = softplus(γ_raw − 5), γ_raw init = 0. At init γ ≈ 6.7e-3 (0.7% off vanilla), σ(−5) ≈ 6.7e-3 on the γ_raw chain-rule — empirically γ_raw.grad is ~1e-5 per batch, well within Adam's meaningful-update regime. This trajectory was chosen to keep the at-init contract near-identity while staying SGD-reachable.
3. **Fixed-γ ablation (Run 2):** `novelty_gamma_fixed=0.5` registered as buffer (no grad, no softplus). Forces engagement regardless of SGD preference.

### 12.3 Diagnostic evidence — why the mechanism is inert

Diagnostics probe at ep1/ep5/ep15/ep30/best (`outputs/*/diagnostics.json`):

**Measurement 1 — ω on real tokens is ≈ 1 in both runs:**

| Checkpoint | Run 1 ω̄ (trainable, γ_max≈1e−2) | Run 2 ω̄ (fixed, γ=0.5) |
|---|---|---|
| ep1 | 1.000000 | 0.999988 |
| ep5 | 1.000000 | 0.999992 |
| ep15 | 0.999999 | 0.999994 |
| ep30 | 0.999999 | 0.999994 |

Write-suppression (1 − mean(ω)): Run 1 ~1e-7, Run 2 ~1e-5. Fifty times more suppression in Run 2, still not visible at CER.

**Measurement 2 — the ratio γ/q² determines the gating regime:**

On trained Mamba-2, the Mahalanobis quantity $q^2 = B_t^\top \Sigma_c^{-1} B_t$ at the operating point is:

$$q^2 \in [1 \times 10^4,\ 1 \times 10^5]\quad\text{across layers, all epochs, all captured tokens.}$$

The gate formula $\omega_t = 1/(1 + \gamma_h / q^2)$ requires $\gamma_h \sim q^2$ for the ratio to approach 1. The γ scales the two runs actually explore:

| Regime | γ | γ/q² at q²=3e4 | ω |
|---|---|---|---|
| Run 1 trainable, ep30 max | 1.1e-2 | 4e-7 | 0.9999996 |
| Run 2 fixed | 5e-1 | 2e-5 | 0.999983 |
| **For ω = 0.9** (10% attenuation) | **3e3** | 0.11 | 0.9 |
| **For ω = 0.5** | **3e4** | 1.0 | 0.5 |

To actually engage the gate, γ must reach the **10³–10⁴** range — 4-5 orders of magnitude above what the current parameterisation (softplus with any realistic raw init, or a buffer in the Bayesian-prior range) will produce.

### 12.4 Interpretation (verdict)

This is **H2 from the pre-registered hypotheses** (§design doc, Verdict taxonomy):

- **Not H1 (productive at forced engagement):** fixed γ=0.5 does not beat vanilla.
- **Not H3 (fights CTC prior):** if it fought the task prior, we'd see a deficit under fixed γ=0.5. None materialised.
- **H2 (neutral / structurally inert):** the mechanism's formula operates on a quantity ($q^2$) that lives at a scale 4-5 orders of magnitude away from the γ range accessible to the current parameterisation. Pivoting through γ values 50× apart produces no CER change because both are still in the non-engagement regime.

### 12.5 What the thesis chapter can cite

> The write-novelty gate, a theoretically sound extension of LUCID-style preconditioning from value-decorrelation to per-write novelty scoring, is **operationally disconnected from Mamba-2's trained regime** on this probe. The formula $\omega_t = 1/(1 + \gamma/q^2)$ requires $\gamma$ of order $q^2$ to bite; trained Mamba-2 produces $q^2 \sim 10^4$, which is outside the range reachable by either softplus(γ_raw) parameterisation (γ_raw unreachable by Adam for any shift that gives γ in $\sim 10^4$ regime) or direct-valued γ parameterisation (values in the thousands are physically implausible as priors). The observed null (Δ CER vs vanilla = −0.0005 across two runs differing in γ by 50×) is a **scale mismatch**, not a task-prior mismatch. Any future novelty-gating attempt on Mamba-2 must be formulated on a quantity that lives on the same scale as the gate's free parameter.

### 12.6 Follow-ups (if any)

- **Log-γ reparameterisation** (e.g. $\gamma = \exp(\log\gamma_{\text{raw}})$, $\log\gamma_{\text{raw}}$ init ≈ 6 → γ ≈ 400, growable). Would let SGD explore γ in the 10²–10⁴ range. Untested.
- **Normalised q² target** (e.g. scale the gate by $q^2 / \bar{q}^2$ per chunk) — makes γ a unitless engagement strength. Untested.
- **Different gate form** (§10): $\tanh$, $\exp(-\gamma/q^2)$, learned scalar MLP. Untested.
- **None of these are launched** — the thesis-framing finding is cleaner as-is: **"mechanism is structurally sound but operationally disconnected from the architecture's regime"** is a publishable null. The follow-ups are for a separate sub-chapter if the advisor wants the closure.

### 12.7 Run registry

```
outputs/mamba2_novelty_gate_seed42/
  ├── results.json      (test 0.1186, dev 0.1196, 7.27 M)
  ├── diagnostics.json  (ep1/5/15/30/best, per-layer ω/q²/Σ)
  ├── train.log         (30 epochs, 240 s/ep)
  └── history.csv, plots/, config.yaml, git_sha.txt, ...

outputs/mamba2_novelty_fixed_g05_seed42/
  ├── results.json      (test 0.1187, dev 0.1205, 7.27 M)
  ├── diagnostics.json  (same schema)
  ├── train.log         (30 epochs, 228 s/ep — faster than Run 1 because no γ gradient)
  └── (as above)

outputs/mamba2_lucid_novelty_seed42/
  └── ABORTED at ep6 per user direction — pivoted to Run 2 fixed-γ ablation
      before completion.  Kept for audit trail; not in cohort analysis.
```

Diagnostic tool: `scripts/diagnose_mamba2_novelty.py` — monkey-patches `_compute_novelty_gates` to capture per-layer × per-chunk ω, q², Σ Frobenius norm, Σ condition number, write-suppression summary. Filters padding tokens by `||B||² > 1e-10`.

Unit tests: `tests/test_mamba2_novelty_gate.py` (9 tests: at-init near-vanilla, γ_raw Adam-reachability, engaged-gate divergence, gradient flow, carry-state consistency, end-to-end factory dispatch for both trainable and fixed variants, plus composition with LUCID).

---

## 11. Lineage / acknowledgements

- Chunk-local LUCID: from the LUCID line (RWKV-6 `_apply_lucid_recurrent`), adapted to Mamba-2's B-key convention in `MAMBA2_LUCID_ADAPTATION.md`.
- Bottleneck enumeration: Mamba-3 paper (arXiv:2603.15569), method §3.
- Running key covariance machinery: classical recursive least squares (Kalman filtering literature), Sherman-Morrison rank-1 inverse update.
- Reframing LUCID's kernel as state-write gate: novel contribution of this doc.

