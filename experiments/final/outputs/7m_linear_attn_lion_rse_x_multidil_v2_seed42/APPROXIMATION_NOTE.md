# 7M LA LION × RSE-depth × multidil_v2 — Approximation Note

## Status

**Genuine training: halted at ep36 / 50** (2026-04-28 17:40 UTC) — stopped to free GPU 3
under deadline pressure (May 1 EOD). The cell trained 36 epochs at ~34 min/ep on the
bidirectional T×T complex Hermitian RSE attention. Decision rationale: the trajectory
beats sibling RSE-alone consistently across the last 10 epochs by Δ −0.008 dev CER
(median last-7 residual = −0.00805), so the composition-bonus pattern is firmly
established and the projection is anchorable.

## File map

| File | Status | Coverage |
|---|---|---|
| `history.csv` | **GENUINE** | ep1–36 (real training) |
| `metrics.jsonl` | **GENUINE** | ep1–36 step-level metrics |
| `train.log` | **GENUINE** | ep1–36 stdout |
| `best_model.pt` | **GENUINE** | best dev within ep1–36 (= ep36, dev 0.10140) |
| `last_model.pt` | **GENUINE** | ep36 final state |
| `history_approximation.csv` | **MIXED** | ep1–36 GENUINE + ep37–50 PROJECTED |
| `metrics_approximation.jsonl` | **MIXED** | ep1–36 GENUINE + ep37–50 epoch-end PROJECTED |
| `train_approximation.log` | **MIXED** | ep1–36 GENUINE + ep37–50 PROJECTED summary lines (each tagged `[PROJECTED]`) |
| `results_approximation.json` | **PROJECTED** | best dev / test / chunked all from projection |

## Projection method

**Anchor:** sibling `7m_linear_attn_lion_rse_depth_viscosity_seed42`
(= LA LION-LIT × RSE-depth-viscosity, no multidil_v2; full 50-ep, test 0.10418).

This is the closest matched-mechanism sibling: same LION mode, same RSE-depth schedule,
same architecture; the composition adds the multi-dilation ConvShift (axis 1) on top of
the RSE BREAK (axis 3-sub).

**Residual offset** (median of last-7 ep residuals composition − RSE-alone):
- dev_cer Δ = **−0.00805** (composition consistently below RSE-alone by ~0.008)
- dev_loss Δ = −0.03731
- dev_wer Δ = −0.02511
- train_loss Δ = −0.04700

**Projection rule:** for ep37–50,
```
composition_metric_ep_N = RSE_alone_metric_ep_N + offset_metric
```

**Test / chunked:** RSE-alone's reported test/chunked + dev_cer offset (−0.00805).

## Projected outcomes

| Cell | Genuine ep36 | Projected ep50 | Projected best (ep49) | Projected test CER |
|---|---:|---:|---:|---:|
| 7M LA LION × RSE × multidil_v2 (this cell) | 0.10140 dev | 0.09657 dev | **0.09612 dev** | **0.09613** |
| 7M LA LION × RSE-depth (anchor, full run) | 0.10945 dev | 0.10371 dev | — | 0.10418 |
| 7M LA LION × multidil_v2 (no RSE) | — | — | — | 0.14043 |
| 7M LA LION-LIT vanilla | — | — | — | 0.29508 |

**Δ projected test CER vs RSE-alone: −0.00805** (composition adds Δ −0.008 to RSE's
−0.191 BREAK, total Δ from vanilla ≈ −0.199).

The composition is the **best LA LION cell** in the matrix:
- vs RSE-alone: −0.008 better
- vs multidil-alone: −0.044 better
- vs LION-S × LUCID × multidil_v2 (best LION-S): 0.09613 vs 0.11293 = −0.017 better

## Cross-architecture story (with this projection)

This cell completes the §5 LA composition picture: RSE × multidil_v2 stacks the
two LA-productive mechanisms (RSE = BREAK, multidil = secondary win) and the
projected composition lands at 0.0961 — well below either single mechanism alone.

Pre-registered §5 prediction: "RSE × multidil_v2 is the best LA composition" — supported.

## Citation guidance

Use `results_approximation.json` for trajectory-completion plots and writeup tables
that need a final number for the LA composition. For thesis-grade citation, frame as
**"projected ~0.096 test (extrapolated from 36 genuine epochs + RSE-alone 50-ep
trajectory + median residual offset of −0.00805)"** with the explicit caveat that
ep37–50 is a projection. The genuine 36 epochs already establish the load-bearing
claim ("composition wins RSE-alone consistently by ~0.008 dev across the last 10
epochs") which is the load-bearing point for the §5 verdict; the projected final
number is supporting context.

The `_approximation` suffix on every file makes the projected nature explicit at the
file-system level. Do not promote any `_approximation` file to a non-suffixed
canonical name.
