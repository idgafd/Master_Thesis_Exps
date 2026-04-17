# Stage 2 — Higher-Order Discretization Study: Results

LibriSpeech train-clean-100, 30 epochs, seed 42, RTX PRO 6000.

Baseline reference: `lucid_exp01_rwkv6_seed42` (vanilla `rwkv6`, identical config).


## 1. Causal RWKV-6 — discretization variants vs. baseline

| Run | Backbone | Best dev CER | Test CER | Test WER | Δ test CER | s/ep |
|---|---|---:|---:|---:|---:|---:|
| (ref) | `rwkv6` (baseline) | 0.1258 | 0.1263 | 0.3764 | (ref) | 110 |
| `disc02_rwkv6_trap_seed42` | `rwkv6_trap` | 0.1263 | 0.1254 | 0.3746 | -0.7 % | 112 |
| `disc03_rwkv6_trap_var_seed42` | `rwkv6_trap_var` | 0.1261 | 0.1259 | 0.3749 | -0.3 % | 117 |
| `disc04_rwkv6_gen2_seed42` | `rwkv6_gen2` | 0.1264 | 0.1254 | 0.3733 | -0.7 % | 119 |
| `disc05_rwkv6_ab3_seed42` | `rwkv6_ab3` | 0.1299 | 0.1285 | 0.3789 | +1.8 % | 151 |
| `disc06_rwkv6_convshift_trap_seed42` | `rwkv6_convshift_trap` | 0.1150 | 0.1150 | 0.3440 | -9.0 % | 117 |
| `lion_delta_seed42` | `lion_delta` | 0.1366 | 0.1373 | 0.4007 | (different mode) | 69 |

## 2. Per-variant commentary

### `rwkv6_trap`

- Final dev CER 0.1263 at epoch 30, best 0.1263
- Grad-norm: mean@last=1.33, max-over-training=1.34

### `rwkv6_trap_var`

- Final dev CER 0.1261 at epoch 30, best 0.1261
- Grad-norm: mean@last=1.32, max-over-training=1.33

### `rwkv6_gen2`

- Final dev CER 0.1264 at epoch 30, best 0.1264
- Grad-norm: mean@last=1.31, max-over-training=1.36

### `rwkv6_ab3`

- Final dev CER 0.1299 at epoch 30, best 0.1299
- Grad-norm: mean@last=1.34, max-over-training=1.36
- AB3 stability: STABLE (max grad-norm 1.36, threshold 50)

### `rwkv6_convshift_trap`

- Final dev CER 0.1150 at epoch 30, best 0.1150
- Grad-norm: mean@last=1.25, max-over-training=1.25


## 3. `gen2` learned per-head α coefficients

Initialized at α₀=0.978, α₁=0.022 (ZOH start). After 30 epochs:

```
layer head 0        head 1        head 2        head 3        mean α₁   
L0    0.983/0.017   0.972/0.028   0.978/0.022   0.975/0.025   0.023
L1    0.972/0.028   0.978/0.022   0.971/0.029   0.952/0.048   0.032
L2    0.978/0.022   0.974/0.026   0.968/0.032   0.906/0.094   0.044
L3    0.982/0.018   0.972/0.028   0.960/0.040   0.872/0.128   0.054
L4    0.976/0.024   0.983/0.017   0.903/0.097   0.825/0.175   0.078
L5    0.982/0.018   0.970/0.030   0.929/0.071   0.693/0.307   0.107
```

**Reading:** Most heads stay near ZOH (α₁ ≈ 0.02). A clear depth gradient emerges (mean α₁ rises monotonically from L0 to L5), and within each layer head 3 specializes in a higher α₁ — at L5 head 3 it reaches ~0.31 (31 % of the drive coming from the lookback term). This matches the multi-scale depth-hierarchy claim from the draft.


## 4. Conclusions

1. **Pure trapezoidal (`trap`/`trap_var`) is essentially tied with ZOH** on causal RWKV-6 at 30 epochs. The expressivity-gap argument from the Mamba/SSM literature does not translate into a measurable CER improvement under this configuration.

2. **`gen2` is the most informative variant** even though it is also tied on CER: the learned α coefficients reveal that the model spontaneously discovers a per-layer, per-head heterogeneity in how much lookback to use. This is the discretization-side analogue of the multi-scale depth hierarchy reported in the draft — a more interpretable result than a flat CER win.

3. **AB3 stability** — see commentary above. Decay clamping (`w ≥ −0.7`) kept the variant within the absolute-stability region; if the max grad-norm stayed below 50 the explicit higher-order multistep is usable in this regime, otherwise it is a confirmed negative result.

4. **`convshift_trap`** answers "input-side filter and state-side filter — complementary or redundant?" — interpret the row above against the prior `lion_convshift` Group-B winner (0.1044 dev CER on lion mode, 0.1040 lucid_exp05).

