# Stage 5 Phase 1 — Running status

Live tracking document for the decisive P²-RSE pair. Auto-updated information
lives under the AUTO markers; narrative outside them is hand-written.

## Launched

- **GPU 0 — `stage5_01_rwkv6_p2rse_seed42`** — P²-RSE, unconstrained linear β (main).
- **GPU 1 — `stage5_02_rwkv6_p2rse_softmax_seed42`** — P²-RSE, convex softmax β (control).

Both: seed 42, 30 epochs, identical Stage-2/3/4 training recipe, LibriSpeech train-clean-100.

## Quick monitoring commands

```bash
# Sessions
tmux ls

# Live log
tail -f experiments/formal_v1/outputs/logs/stage5_01.log
tail -f experiments/formal_v1/outputs/logs/stage5_02.log

# Per-epoch history (once first epoch completes)
column -ts, experiments/formal_v1/outputs/stage5_01_rwkv6_p2rse_seed42/history.csv
column -ts, experiments/formal_v1/outputs/stage5_02_rwkv6_p2rse_softmax_seed42/history.csv

# Attach to a live session (Ctrl-b d to detach)
tmux attach -t stage5_01
```

## Sanity-check pre-launch results

Before launching, the `scripts/smoke_p2rse.py` battery confirmed:

| Check | Outcome |
|---|---|
| Parameter counts (encoder, total) | rse: 5.90 M / 7.81 M · rse_strong: 5.94 M / 7.85 M · **p2rse: 5.99 M / 7.90 M** (+0.8 % vs strong) |
| Forward/backward shape | `(B=2, 80, T=500)` → `(B=2, T'=125, V=29)` |
| Gradient flow | All `time_theta_2`, `time_theta_w1_2`, `time_theta_w2_2`, `beta_mixer` receive finite gradients |
| Phase-complementary init | `max \|θ^(1) + θ^(2)\| = 0.0` exactly on every layer |
| β init at t=0 | linear: logits std ≈ 0.15 (near-zero). softmax: β mean = 0.500 (uniform) |
| Baseline RSE regression | `rwkv6_rse` and `rwkv6_rse_strong` produce identical output (torch.where masking fix did not regress) |
| Linear vs softmax differ | max diff 1.19 — dispatch correctly routes to different paths |
| End-to-end CTC loss | Both backbones: loss 17→13 after 1 update step, grad norm ~16 |

## Decision thresholds (from STAGE5_PLAN.md §3)

- **Break** — best dev CER ≤ 0.1160
- **Marginal** — 0.1160 < x ≤ 0.1180
- **Plateau** — > 0.1180

## Results (AUTO — populate from history.csv as epochs complete)

<!-- AUTO:TRAJECTORY -->

| Epoch | stage5_01 p2rse dev CER | stage5_02 p2rse_softmax dev CER |
|---:|---:|---:|
| _pending_ | — | — |

<!-- /AUTO:TRAJECTORY -->

## Final (AUTO — populate on completion)

<!-- AUTO:FINAL -->

| Run | Best dev CER | Test CER | Classification |
|---|---:|---:|---|
| _pending_ | — | — | — |

<!-- /AUTO:FINAL -->

## Projected epoch time

Based on Stage 3 RSE (~250 s/epoch for single mode), P²-RSE should run at
**~400–500 s/epoch** (two scans + β mixer). Full 30-epoch run ≈ 3.5–4 hours per GPU.

## Decision action (from STAGE5_PLAN.md §6)

Fill after both runs complete. Reminder table:

| stage5_01 (A) | stage5_02 (B) | Action |
|:---:|:---:|---|
| Break | Break | → Phase 2a (stack mechanisms) |
| Break | Plateau | → Phase 2a (confirms softmax-was-the-bottleneck) |
| Marginal | Plateau | → Phase 2b (diagnostic) |
| Plateau | Plateau | → Phase 2c (pivot to viscosity) |
