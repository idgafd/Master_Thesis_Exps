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
| 10.1 | `rwkv6_loglinear` | 0.1240 | 0.1226 |
| 10.2 | `rwkv6_m2rnn_sparse` | 0.1276 | 0.1264 |
| 10.3 causal | `rwkv6_convshift_multidil` | 0.1229 | 0.1224 |
| **10.3-sym** | **`rwkv6_convshift_multidil_symmetric`** | **0.1153** | **0.1145** |
| 10.4 | `rwkv6_chanmix_bypass` | 0.1251 | 0.1248 |
| 10.5 | `rwkv6_orthogonal` (still training, ep 15) | 0.1518 (ep 15) | — |
| 10.6 | `rwkv6_pom_vlift` | 0.1254 | 0.1253 |
| CB-1 | `rwkv6_rse_convshift_multidil_symmetric` | 0.1169 | 0.1156 |
| CB-3 | `rwkv6_convshift_multidil_symmetric_gated` | 0.1167 | 0.1157 |
| CB-5 | `rwkv6_frontend_v2` (lean + matched) | did not converge | — |
| CB-7 | `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric` | 0.1159 | 0.1150 |
| CB-7 v2 | `rwkv6_qtail_lowrank_all_convshift_multidil_symmetric_v2` | 0.0989 | 0.0988 |
| H1 | `rwkv6_qtail_lowrank_all_betapp_convshift_multidil_symmetric_v2` | 0.1003 | 0.0992 |
| H2 | `rwkv6_qtail_lowrank_all_gamma0_convshift_multidil_symmetric_v2` | **0.0977** | **0.0982** |

## H1 / H2 — Family-D bottleneck probes (seed 42, 30 ep)

Two single-seed probes on top of CB-7 v2 (= `qtail_lowrank_all × multidil_v2`,
dev 0.0989 / test 0.0988) testing why qtail saturates on ASR.

- **H1 — per-(head, channel-pair) β allocation (`betapp`).** Adds 6.1k params
  (β_pp shape `(n_head, K'²) = (4, 256)` per active layer), init 1.0; outer
  `beta_qtail` per-head scalar still zero-init for bit-exact reduction.
  Result: dev 0.1003 / test 0.0992 — **TIED within σ ≈ 0.0014**. Hypothesis
  "scalar β was the allocation bottleneck" falsified at this scale.
- **H2 — γ=0 init Kronecker (`gamma0`).** Initialises `qtail_gamma` at 0.0
  instead of 1.0 (undecayed-accumulator regime, paper's literal Taylor form).
  +24 params. Result: dev **0.0977** / test **0.0982** — **MARGINAL+** (dev
  −0.0012 vs CB-7 v2, test −0.0006). Per-epoch trajectory: H2 ahead of
  cohort at all six checkpoints from ep 10 (peak Δ = −0.0028 at ep 20),
  end-of-cosine narrowing to dev 1σ / test 0.4σ.

Differential reading: Family-D nulls on ASR were partly a *decay-handling*
limitation (cross-channel state needs longer memory horizon than per-channel
decay implies), not a *capacity* or *β-allocation* limitation. dev 0.0977
sits within 0.4σ of CB-1 v2's all-time-best 0.0973, with a different
mechanism family — first time Family-D × multidil_v2 ties RSE × multidil_v2
on dev. Multi-seed (seeds 43, 44) needed to lift H2 from MARGINAL+ to BREAK
per `STAGE10_PLAN.md §8.2`.

Both runs preserve the bit-exact-reduction-at-init contract via β_qtail = 0
gating; smoke-tested in `scripts/smoke_qtail_h1_h2.py`. Implementation:
`use_qtail_beta_per_pair` flag (substring `betapp`) and `qtail_gamma_init`
constructor knob (substring `gamma0` → 0.0) in `rwkv6_time_mix.py`.

**Best causal result:** `rwkv6_rse_convshift` at dev **0.1145** /
test **0.1126** (Stage 3); matched at test-CER by
`rwkv6_convshift_multidil_symmetric` (10.3-sym) at **0.1153** /
**0.1145** without the transition-side rotation.

**Best pure-transition result:** `rwkv6_rse_strong_viscosity` at dev
**0.1185** / test **0.1177**.

See [stages_2_9_summary.md](stages_2_9_summary.md) for Stages 2–9
per-epoch trajectories, and [STAGE10_SUMMARY.md](STAGE10_SUMMARY.md)
for the Stage-10 honest summary.

## Sibling docs

- [stages_2_9_summary.md](stages_2_9_summary.md) — sequential narrative
  + per-epoch tables + implementation links.
- [TODO_FUTURE_IDEAS.md](TODO_FUTURE_IDEAS.md) — resolved-status index
  and parking lot for literature mechanisms not yet tested.
- [TODO_DELTA_RULE.md](TODO_DELTA_RULE.md) — dedicated delta-rule catalog.

## Notes

- CER computed character-by-character. WER computed word-by-word.
- "Δ vs X" in other docs = relative change `(new − base) / base · 100 %`.
