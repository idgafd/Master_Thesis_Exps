# Run 006 Analysis — bidir_rwkv6_conv_nogate / bidir_rwkv6_conv

---

## What Changed vs Run 005

| | Run 005 | Run 006 |
|---|---|---|
| Backbones | bidir_rwkv6 (baseline) | bidir_rwkv6_conv_nogate, bidir_rwkv6_conv |
| Architecture delta | fixed (x[t-1]+x[t+1])/2 token shift | learned DWConv1d(k=3, groups=D) token shift, init [0.5,0,0.5] |
| Gate | none | nogate: none / conv: G=σ(W_gate(xres)), init bias=3.0 |
| Params | 7.74M | 7.75M (nogate) / 8.14M (conv+gate) |
| supports_carry_state | no | no |
| Scheduler | cosine 60ep | cosine 60ep (same) |

---

## Key Observations (Facts)

| Metric | bidir_rwkv6 (005) | bidir_rwkv6_conv_nogate | bidir_rwkv6_conv |
|---|---|---|---|
| Params | 7.74M | 7.75M | 8.14M |
| BestDevCER | 0.1676 | **0.1587** | 0.1635 |
| BestEpoch | 59 | 55 | 57 |
| TestLoss | 0.7988 | **0.7585** | 0.7785 |
| TestCER | 0.1790 | **0.1760** | 0.1813 |
| TestWER | 0.6704 | **0.6563** | 0.6845 |
| R@2s CER | 0.2161 | **0.2084** | 0.2123 |
| R@5s CER | 0.1857 | **0.1804** | 0.1848 |
| R@10s CER | 0.1790 | **0.1744** | 0.1791 |

- `bidir_rwkv6_conv_nogate` is the **best model in the entire experiment series to date**.
- `bidir_rwkv6_conv` (with gate) is worse than both the no-gate variant and the run-005 LION baseline.
- nogate converged at epoch 55 vs LION epoch 59 — faster convergence with ConvShift.
- nogate train loss at epoch 60: 0.522 vs conv 0.556 — nogate finds a better loss landscape.
- Test loss improvement nogate vs baseline: 0.7585 vs 0.7988 — substantial (~5%).
- The gap is consistent on dev, test, and all chunk sizes: nogate dominates uniformly.
- Gate adds 394K parameters (8.14M vs 7.75M) with negative net effect.

---

## Anomalies

1. **Gate hurts consistently.** Adding the xres-conditioned gate pushes CER from 0.1760 to 0.1813
   (+3.0% relative). This is not noise — it persists across dev CER (0.1587 vs 0.1635),
   test loss (0.7585 vs 0.7785), and all chunked reset windows.

2. **ConvShift (nogate) beats the fixed shift.** Even though the init replicates the original
   fixed shift exactly, the learned deviation improves CER by 1.7% relative (0.1760 vs 0.1790).
   This means the model actively moves away from the symmetric (x[t-1]+x[t+1])/2 prior.

3. **Gate variant has higher train loss.** 0.556 vs 0.522 at epoch 60 — the gate is not just
   failing to help generalization, it is actually making optimization harder.

---

## Hypotheses

### Why does ConvShift (no gate) help?

The fixed symmetric token shift (x[t-1]+x[t+1])/2 treats past and future context equally.
A learned DWConv1d(k=3) with per-channel weights can break this symmetry and learn
asymmetric temporal weighting per feature dimension. For Ukrainian speech, different
feature bands (low/high frequency Mel bins) may benefit from different temporal receptive
fields. The DWConv1d acts as a soft learned interpolation that refines the input to the
RWKV-6 WKV attention — a benign inductive bias that leaves the attention mechanism itself
unchanged.

### Why does the gate hurt?

The gate is `G = σ(W_gate(xres))` where `xres = conv_shift(x) - x`.
At init: bias=3.0, so G≈0.95 (near-identity, attention output passes through almost fully).

Problem: `xres` is a **local 3-frame residual signal** (what the ConvShift changed vs
the input frame). This is being used to gate a **global full-sequence attention output**.
These are semantically incompatible:
- Local: "how much did my 3-frame neighborhood deviate from center frame"
- Global: "what is the full-sequence attention-weighted output"

The gate picks up local textural patterns from xres and uses them to partially suppress
globally-computed representations. Since LION's parallel attention already models all
long-range dependencies, any gate perturbation is net negative. The 394K gate parameters
have enough capacity to fit spurious correlations that hurt generalization.

Additionally: the gate injects a nonlinear xres->sigmoid pathway into the residual stream,
which increases gradient complexity without a corresponding expressiveness benefit for ASR.

---

## Next Actions

1. **Confirm ConvShift is the active ingredient (control experiment):**
   Run `bidir_rwkv6` with a **frozen** DWConv1d (weights fixed at [0.5, 0, 0.5], no grad).
   If frozen conv ≈ 0.179 → improvement comes from *learned* shift, not structural.
   If frozen conv already helps → structure (e.g. groupnorm interaction) is the cause.

2. **Try a simpler gate:**
   Replace xres-conditioned gate with a **per-layer learnable scalar** `G = σ(b)` (no input
   dependence, one parameter per layer). This tests whether the problem is "input-conditioned
   gating" vs "gating at all." Expected: scalar gate ≈ nogate (converges b → large positive).

3. **Try gating the ConvShift output, not the attention output:**
   `x_shifted = (1-G)*x + G*conv_shift(x)` where G is per-channel scalar.
   This constrains the gate to blend only within the shift operation — semantically coherent.

4. **Extend training epochs:**
   nogate dev CER was still marginally declining at epoch 55 (plateau ~0.1587). Try 80-90 epochs
   with cosine restart or linear decay tail to see if more is extractable.
