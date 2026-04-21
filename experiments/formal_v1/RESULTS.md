# Formal v1 — Results

Causal RWKV-6 experiment chain on LibriSpeech train-clean-100. Seed 42,
30 epochs, RTX PRO 6000. Seed noise σ ≈ 0.0014. Full chronology,
per-epoch trajectories, mechanism-level diagnostics and implementation
links are in [stages_2_9_summary.md](stages_2_9_summary.md).

## Training config (shared spine)

| Parameter | Value |
|---|---|
| d_model | 256 |
| n_layers | 6 |
| n_heads / head_size | 4 / 64 |
| FFN dim | 896 |
| Dropout | 0.1 |
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Scheduler | cosine + 1000-step warmup |
| Epochs | 30 (patience=15) |
| SpecAugment | LD policy (freq=27, time=100, 2+2 masks) |
| Batch | max 300 s total duration |
| Grad clip | 5.0 |

## Dataset

LibriSpeech clean via HuggingFace: `train.100` (~28 539 utterances),
`validation`, `test`. English characters, CTC blank index 0.

## Main final results (dev / test CER, seed 42, 30 ep)

| Stage | Backbone | Dev | Test |
|---|---|---:|---:|
| 2 | `rwkv6` (vanilla baseline) | 0.1258 | 0.1263 |
| 2 | `rwkv6_trap` / `trap_var` / `gen2` | ~0.1261 | ~0.1254 |
| 2 | `rwkv6_ab3` | 0.1299 | 0.1285 |
| 2 | `rwkv6_convshift_trap` (cross-axis) | 0.1150 | 0.1150 |
| 3 | `rwkv6_rse` | 0.1251 | 0.1238 |
| 3 | `rwkv6_rse_convshift` (cross-axis) | 0.1145 | 0.1126 |
| 4 | `rwkv6_rse_depth` | 0.1207 | 0.1200 |
| 4 | `rwkv6_rse_strong` | 0.1192 | 0.1188 |
| 5 | `rwkv6_rse_depth_viscosity` | 0.1198 | 0.1198 |
| **5** | **`rwkv6_rse_strong_viscosity` — anchor** | **0.1185** | **0.1177** |
| 5 Ph1 | `rwkv6_p2rse` / `p2rse_softmax` | 0.1250 / 0.1220 | 0.1241 / 0.1215 |
| 6 | `rwkv6_p2rse_strong_viscosity` (shared-λ) | 0.1190 | 0.1196 |
| 6 | `rwkv6_p2rse_indeplam_strong_viscosity` (indep-λ) | 0.1394 | 0.1383 |
| 6 | `rwkv6_rmsnorm` / `hadamard_n2` | 0.1264 / 0.1253 | 0.1252 / 0.1251 |
| 6 | `rwkv6_qtail` (K², top-2) | 0.1260 | 0.1240 |
| 6 | `rwkv6_qtail_gamma` / `qtail_gamma_dbeta` | 0.1257 / 0.1247 | 0.1249 / 0.1245 |
| 6 | `rwkv6_qtail_lowrank` / `qtail_lowrank_all` | 0.1247 / 0.1238 | 0.1242 / 0.1240 |
| 7A | `rwkv6_rse_dphi_viscosity` (data-dep readout φ) | 0.1217 | 0.1207 |
| 8 T1 | `rwkv6_delta_warmstart_fixed` (recurrent delta, wired) | 0.1258 | 0.1256 |
| 8 T2 | `rwkv6_nonnormal_rse_viscosity` (dense polar non-normal) | 0.1202 | 0.1200 |
| 9 A | `rwkv6_sparse_nonnormal_rse_viscosity` (halted ep 15) | 0.1467 (ep 15) | — |
| 9 B | `rwkv6_sparse_nonnormal_rse_edge_only_viscosity` | 0.1218 | 0.1216 |

**Best causal result:** `rwkv6_rse_strong_viscosity` at dev **0.1185** /
test **0.1177**. See [stages_2_9_summary.md](stages_2_9_summary.md) for
per-epoch trajectories, hypothesis logic, diagnostic probes, and the
invariant across Stage 7A / 8 / 9 dense-vs-sparse variants.

## Sibling docs

- [stages_2_9_summary.md](stages_2_9_summary.md) — sequential narrative
  + per-epoch tables + implementation links.
- [TODO_FUTURE_IDEAS.md](TODO_FUTURE_IDEAS.md) — resolved-status index
  and parking lot for literature mechanisms not yet tested.
- [TODO_DELTA_RULE.md](TODO_DELTA_RULE.md) — dedicated delta-rule catalog.

## Notes

- CER computed character-by-character. WER computed word-by-word.
- "Δ vs X" in other docs = relative change `(new − base) / base · 100 %`.
