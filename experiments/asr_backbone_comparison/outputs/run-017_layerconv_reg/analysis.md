# Run 017 Analysis — Strong Regularisation (ConvShift + LayerConv)

Counterpart to run-016: tests the same heavy regularisation regime on the
Conv-based backbones (ConvShift nogate and LayerConv). Provides the conv-specific
failure mode analysis.

**Key question:** Do ConvShift and LayerConv exhibit the same underfitting failure
as run-016, or do they respond differently to strong regularisation due to the
local inductive bias of depthwise convolution?

---

## Setup

| Parameter | Standard (runs 006, 014) | Strong Reg (run-017) |
|-----------|--------------------------|---------------------|
| Dropout | 0.15 | **0.25** |
| freq_mask_param | 15 | **27** |
| time_mask_param | 35 | **70** |
| num_time_masks | 2 | **5** |
| Epochs | 60 | **80** |
| early_stopping_patience | 15 | **20** |
| Scheduler | cosine | cosine |

Backbones: `bidir_rwkv6_conv_nogate` (ConvShift) and `bidir_rwkv6_layerconv`
(LayerConv). Both trained simultaneously.

---

## Results

| Backbone | Best Ep | Dev CER | Test CER | Test WER | Test Loss |
|----------|---------|---------|----------|----------|-----------|
| `bidir_rwkv6_conv_nogate` (strong reg) | 74/80 | 0.2977 | 0.3189 | 0.9132 | 1.2390 |
| `bidir_rwkv6_layerconv` (strong reg) | 74/80 | 0.2978 | 0.3205 | 0.9020 | 1.2341 |

For comparison — same backbones with standard regularisation:

| Backbone | Dev CER | Test CER | Test WER |
|----------|---------|----------|----------|
| `bidir_rwkv6_conv_nogate` (006) | 0.1587 | 0.1760 | 0.6563 |
| `bidir_rwkv6_layerconv` (014) | 0.1574 | 0.1768 | 0.6548 |

**Strong regularisation degraded performance by ~81% in CER for conv models**
(0.1760 → 0.3189 for ConvShift). This is far worse than run-016's non-conv models
(0.1790 → 0.2779, ~55% degradation). Conv models are more sensitive to heavy
SpecAugment than non-conv models.

---

## Conv Models Are More Sensitive to SpecAugment

Comparing run-016 vs run-017 under identical regularisation:

| Backbone type | Standard CER | Strong-reg CER | Degradation |
|---------------|-------------|----------------|-------------|
| No conv (`bidir_rwkv6`) | 0.1790 | 0.2779 | +55% |
| No conv (`bidir_rwkv6_temperature`) | 0.1792 | 0.2874 | +60% |
| Conv (`bidir_rwkv6_conv_nogate`) | 0.1760 | 0.3189 | +81% |
| Conv (`bidir_rwkv6_layerconv`) | 0.1768 | 0.3205 | +81% |

Both conv models show identical 81% degradation under strong reg, while
non-conv models show 55-60% degradation. The additional ~21% degradation for
conv models is explained by the SpecAugment-conv interaction:

**Why ConvShift is more sensitive to SpecAugment:**

1. **The depthwise conv sees masked frames as content.** SpecAugment zero-masks
   time frames. The ConvShift conv operates on the raw mel spectrogram features
   (post-frontend, pre-attention). When a frame is masked to zero, the conv
   mixes it with its neighbours — the zero is treated as a silent frame, not as
   an absence of information. The model must learn to ignore this zero-frame
   signal, which conflicts with the conv's purpose of exploiting local structure.

2. **Five 70-frame masks destroy conv's local context windows.** The ConvShift
   conv has kernel=3 (run-006) or kernel=7 (run-014, lower layers). A mask
   spanning 70 frames creates 70 positions where the conv window is dominated
   by zeros. With 5 such masks over a ~125-frame utterance, a large fraction of
   frames see only masked content in their receptive field.

3. **Learned kernels adapt to masking patterns, hurting real inputs.** Under
   heavy masking, the conv weights adapt to handle the frequent zero-segments.
   This means the weights are no longer optimal for real speech — they are
   instead tuned to the SpecAugment masking distribution.

4. **LION attention is not affected the same way.** The LION attention matrix
   A(i,j) is computed from queries and keys. If position i is masked (zero input),
   both r_i and k_i project to near-zero vectors. The attention weight A(i,j)
   becomes small. The model learns to downweight masked positions naturally
   through the attention mechanism. The conv has no equivalent self-suppression
   mechanism.

---

## Training Dynamics

```
bidir_rwkv6_conv_nogate (strong reg) train loss:
  ep 1: ~3.77   ep20: ~2.03  ep40: ~1.87  ep60: ~1.81
  ep10: ~2.52   ep30: ~1.92  ep50: ~1.83  ep80: ~1.83 (plateau)

bidir_rwkv6_conv_nogate (standard, run 006) train loss:
  ep 1: ~3.4    ep20: ~0.82  ep40: ~0.57  ep60: ~0.52
```

Strong-reg conv models plateau at train loss ~1.83 (epoch ~60-80). The standard
models reach ~0.52. Standard-to-strong-reg loss ratio at convergence: 3.5×.
The models cannot learn the training distribution.

**Both conv models achieve virtually identical results (dev CER 0.2977 vs 0.2978)**
despite ConvShift (kernel=3 uniform) and LayerConv ([7,7,5,5,5,3]) having
different inductive biases. Under heavy regularisation, the kernel schedule
advantage disappears completely — the models are all underfitting equally.

---

## Chunked Evaluation

| Backbone | Full CER | R@2s | R@5s | R@10s |
|----------|----------|------|------|-------|
| bidir_rwkv6_conv_nogate (strong reg) | 0.3189 | 0.3408 | 0.3192 | 0.3154 |
| bidir_rwkv6_layerconv (strong reg) | 0.3205 | 0.3449 | 0.3221 | 0.3170 |

The chunked degradation (R@2s − Full = +0.022 to +0.024) is similar in magnitude
to standard-reg runs. Context truncation is still painful — the models haven't
learned more robust local representations despite the local inductive bias of
ConvShift. Heavy reg prevented learning altogether.

---

## Cross-Run Comparison (016 vs 017)

Under standard regularisation, conv models outperform non-conv:
- ConvShift CER 0.1760 < LION CER 0.1790 (conv wins by −1.7%)

Under strong regularisation, conv models are significantly worse than non-conv:
- ConvShift CER 0.3189 > LION CER 0.2779 (conv loses by +14.7%)

The relationship inverts. This is a clear interaction effect: **ConvShift's
advantage depends on the regularisation regime.** Strong SpecAugment eliminates
the advantage and introduces additional harm.

---

## Conclusion

**Conv-based models are more sensitive to heavy SpecAugment than standard LION.**
The 81% CER degradation (vs 55% for non-conv) is caused by the SpecAugment-conv
interaction: masked frames disrupt the local context that ConvShift is designed
to exploit.

This confirms that the strong-reg experiment is doubly wrong:
1. The regularisation is too strong for the data scale (same failure as run-016).
2. Conv models additionally suffer from SpecAugment masking corrupting their
   local context windows.

**The standard training settings (dropout=0.15, freq=15, time=35, 2+2 masks)
remain the right configuration for all conv-based runs at this scale.**

**Interesting structural finding from the failure:** The SpecAugment sensitivity
difference reveals that ConvShift's gain over LION baseline comes from learning
local temporal patterns that SpecAugment masks destroy. This implies the conv
kernel is actually doing meaningful work — it's not just a slightly better token
shift, but is genuinely learning from local temporal structure.

---

## What to Do

- Drop all strong-regularisation experiments.
- Return to standard training settings for runs 018+.
- If regularisation is ever revisited: try a middle ground — dropout=0.20,
  time_mask=50, 3 time masks. Never go as far as LD policy (time_mask=70, 5 masks).
- Do NOT apply heavy SpecAugment to conv-based models specifically — the
  sensitivity asymmetry makes it harmful even before the underfitting threshold.
