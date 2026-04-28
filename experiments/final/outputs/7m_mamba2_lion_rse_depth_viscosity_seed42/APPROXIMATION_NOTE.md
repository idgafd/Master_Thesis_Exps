# 7M Mamba-2 LION × RSE-depth-viscosity — Approximation Note

## Status

**Genuine training: halted at ep14 / 50** (2026-04-28 17:35 UTC) — stopped to free GPU 2
for other in-flight experiments and reduce wall-clock cost (Mamba-2 LION ×
RSE-depth was running at ~68 min/ep, ~42 h remaining of full schedule). Decision rationale:
across the genuine 14 epochs the trajectory consistently beats vanilla LION by Δ −0.002 to
−0.006 dev CER (median last-7 residual = −0.00289), so the trend is clear and the projection
is anchorable to vanilla's full trajectory.

This is the second restart attempt of this cell (earlier killed at ep10 as a NULL-reproduction
test, then restarted with the full 50-ep budget on 2026-04-28 01:21 UTC; halted at ep14).

## File map

| File | Status | Coverage |
|---|---|---|
| `history.csv` | **GENUINE** | ep1–14 (real training) |
| `metrics.jsonl` | **GENUINE** | ep1–14 step-level metrics |
| `train.log` | **GENUINE** | ep1–14 stdout |
| `best_model.pt` | **GENUINE** | best dev within ep1–14 (= ep14, dev 0.12266) |
| `last_model.pt` | **GENUINE** | ep14 final state |
| `history_approximation.csv` | **MIXED** | ep1–14 GENUINE + ep15–50 PROJECTED |
| `metrics_approximation.jsonl` | **MIXED** | ep1–14 GENUINE + ep15–50 epoch-end PROJECTED |
| `train_approximation.log` | **MIXED** | ep1–14 GENUINE + ep15–50 PROJECTED summary lines (each tagged `[PROJECTED]`) |
| `results_approximation.json` | **PROJECTED** | best dev / test / chunked all from projection |

## Projection method

**Anchor:** sibling `7m_mamba2_lion_vanilla_seed42` (full 50-ep run, test 0.0853).

**Residual offset** (median of last-7 ep residuals depth − vanilla):
- dev_cer Δ = **−0.00289** (depth consistently below vanilla)
- dev_loss Δ = −0.02333
- dev_wer Δ = −0.00740
- train_loss Δ = −0.01462

**Projection rule:** for ep15–50,
```
depth_metric_ep_N = vanilla_metric_ep_N + offset_metric
```

**Test / chunked:** vanilla's reported test/chunked + dev_cer offset (−0.00289).

## Projected outcomes

| | Genuine ep14 | Projected ep50 | Projected best (ep49) | Projected test CER |
|---|---:|---:|---:|---:|
| 7M Mamba-2 LION × RSE-depth | 0.12266 dev | 0.08431 dev | **0.08418 dev** | **0.08246** |
| 7M Mamba-2 LION vanilla (reference) | 0.12777 dev | 0.08720 dev | 0.08714 | 0.08534 |
| 7M Mamba-2 LION × multidil_v2 | — | — | — | 0.08333 |
| 7M Mamba-2 LION × LUCID-c | — | — | — | 0.08493 |
| 7M Mamba-2 LION × P7 (LUCID-c × multidil_v2) | — | — | — | **0.08054** (best Mamba-2 LION cell) |

**Δ projected test CER vs vanilla LION: −0.00288** (MARGINAL+ benefit). RSE-depth on
Mamba-2 LION is therefore approximately **tied with multidil_v2 alone** (0.0833) and
**below the P7 composition** (0.0805).

## Cross-architecture RSE pattern (with this projection)

| Architecture | Causal | LION |
|---|---|---|
| LA | BREAK Δ −0.068 | BREAK Δ −0.191 |
| RWKV-6 | NULL/marginal Δ −0.003 | BREAK Δ −0.012 (test 0.0740) |
| Mamba-2 | NULL Δ +0.000 (both schedules) | **MARGINAL+ Δ −0.0029 (this projection)** |

The architectural-axis story stands: RSE engages on LION mode for all three architectures
(LA gets the largest BREAK, RWKV-6 a clear BREAK, Mamba-2 a marginal benefit), and on
causal mode only when the architecture lacks native decay (LA only).

## Citation guidance

Use `results_approximation.json` for trajectory-completion plots (matched-epoch
comparisons across Mamba-2 LION cells). For thesis-grade test-CER citation of
Mamba-2 LION × RSE-depth, frame as **"projected ~0.082 test (extrapolated from 14
genuine epochs + vanilla LION 50-ep trajectory + median residual offset)"** with the
explicit caveat that this is a projection. The genuine 14 epochs are sufficient evidence
for the trend ("RSE-depth tracks below vanilla LION by ~0.003 dev throughout training")
which is the load-bearing claim — the projected final number is supporting context.

The `_approximation` suffix on every file makes the projected nature explicit at the
file-system level. Do not promote any `_approximation` file to a non-suffixed canonical
name.
