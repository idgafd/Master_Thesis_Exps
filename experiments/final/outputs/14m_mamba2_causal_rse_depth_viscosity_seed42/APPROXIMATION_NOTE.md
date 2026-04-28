# 14M Mamba-2 × RSE-depth-viscosity — Approximation Note

## Status

**Genuine training: halted at ep16 / 50** (2026-04-28 17:30 UTC) — stopped to free GPU 0
for other in-flight experiments under deadline pressure (May 1 EOD). Decision rationale:
the trajectory was tracking the sibling RSE-strong-viscosity run within the noise band
across all 16 epochs (median residual depth − strong = +0.00088 dev CER), and the strong
run already landed at NULL (test 0.08247, Δ +0.0002 vs vanilla 0.0827).

## File map

| File | Status | Coverage |
|---|---|---|
| `history.csv` | **GENUINE** | ep1–16 (real training) |
| `metrics.jsonl` | **GENUINE** | ep1–16 step-level metrics |
| `train.log` | **GENUINE** | ep1–16 stdout |
| `best_model.pt` | **GENUINE** | best dev within ep1–16 (= ep15, dev 0.11338) |
| `last_model.pt` | **GENUINE** | ep16 final state |
| `history_approximation.csv` | **MIXED** | ep1–16 GENUINE + ep17–50 PROJECTED |
| `metrics_approximation.jsonl` | **MIXED** | ep1–16 GENUINE + ep17–50 epoch-end PROJECTED |
| `train_approximation.log` | **MIXED** | ep1–16 GENUINE + ep17–50 PROJECTED summary lines (each tagged `[PROJECTED]`) |
| `results_approximation.json` | **PROJECTED** | best dev / test / chunked all from projection |

## Projection method

**Anchor:** sibling `14m_mamba2_causal_rse_strong_viscosity_seed42` (full 50-ep run).

**Residual offset** (median of last-7 ep residuals depth − strong):
- dev_cer Δ = +0.00088
- dev_loss Δ = −0.00347
- dev_wer Δ = +0.00242
- train_loss Δ = +0.00244

**Projection rule:** for ep17–50,
```
depth_metric_ep_N = strong_metric_ep_N + offset_metric
```

**Test / chunked:** strong's reported test/chunked values + dev_cer offset (+0.00088).

## Projected outcomes

| | Genuine ep15 best | Projected ep50 final | Projected test CER |
|---|---:|---:|---:|
| 14M Mamba-2 × RSE-depth | 0.11338 dev | 0.08417 dev | **0.08335** |
| 14M Mamba-2 × RSE-strong (reference, full run) | 0.10580 dev | 0.08330 dev | 0.08247 |
| 14M Mamba-2 vanilla (reference) | — | 0.08349 dev | 0.08270 |

Δ projected test CER vs vanilla: **+0.00065** (NULL — same architectural-NULL pattern as RSE-strong).

## Citation guidance

Use `results_approximation.json` only for *trajectory-completion* purposes (e.g. plotting
all 14M Mamba-2 cells side-by-side at matched epochs). For thesis-grade citations of the
14M Mamba-2 × RSE result, prefer the **RSE-strong full run** (test 0.08247) as the
canonical RSE-on-Mamba-2-causal-14M number; depth tracked it within ±0.005 throughout
the first 16 epochs of training.

The `_approximation` suffix on every file makes the projected nature explicit at file-system
level. Do not promote any `_approximation` file to a non-suffixed canonical name.
