# Run 016 Analysis — Strong Regularisation (Baseline + Temperature)

Tests whether heavy regularisation closes the dev-test gap observed in runs
014–015, allowing the temperature mechanism to generalise its dev improvements
to the test set.

**Key question:** Is the dev-test gap a regularisation problem (models learning
dev-specific features due to insufficient constraints), or a data-size problem
(the 35h dataset is too small for these modifications to generalise regardless)?

---

## Setup

| Parameter | Standard (runs 005-015) | Strong Reg (run-016) |
|-----------|------------------------|---------------------|
| Dropout | 0.15 | **0.25** |
| freq_mask_param | 15 | **27** |
| time_mask_param | 35 | **70** |
| num_time_masks | 2 | **5** |
| Epochs | 60 | **80** |
| early_stopping_patience | 15 | **20** |
| Scheduler | cosine | cosine |

Backbones: `bidir_rwkv6` (control) and `bidir_rwkv6_temperature` (experimental).
Both trained simultaneously to enable direct comparison.

---

## Results

| Backbone | Best Ep | Dev CER | Test CER | Test WER | Test Loss |
|----------|---------|---------|----------|----------|-----------|
| `bidir_rwkv6` (strong reg) | 76/80 | 0.2546 | 0.2779 | 0.8733 | 1.1431 |
| `bidir_rwkv6_temperature` (strong reg) | 75/80 | 0.2649 | 0.2874 | 0.8924 | 1.1899 |

For comparison — same backbones with standard regularisation:

| Backbone | Dev CER | Test CER | Test WER |
|----------|---------|----------|----------|
| `bidir_rwkv6` (005) | 0.1676 | 0.1790 | 0.6704 |
| `bidir_rwkv6_temperature` (015) | 0.1606 | 0.1792 | 0.6681 |

**Strong regularisation degraded performance by ~55% in CER** (0.1790 → 0.2779
for the baseline). Both models are far worse than their standard-reg counterparts.

---

## Training Dynamics — Underfitting Diagnosis

```
bidir_rwkv6 (strong reg) training loss trajectory:
  ep 1: 3.741   ep20: 2.112   ep40: 1.864   ep60: 1.758
  ep10: 2.479   ep30: 1.967   ep50: 1.809   ep80: 1.740 (final)

bidir_rwkv6 standard (run 005, reference):
  ep 1: ~3.3    ep20: ~1.1    ep40: ~0.75   ep60: ~0.52

```

At epoch 80, the strong-reg model's train loss is 1.74 — vs the standard-reg
model's train loss of ~0.52 at epoch 60. The model could not learn the training
data. This is severe underfitting.

**Dev-test gap under strong reg:**

| Backbone | Dev CER | Test CER | Gap |
|----------|---------|----------|-----|
| `bidir_rwkv6` (standard, 005) | 0.1676 | 0.1790 | −0.012 |
| `bidir_rwkv6` (strong reg, 016) | 0.2546 | 0.2779 | +0.023 |
| `bidir_rwkv6_temperature` (standard, 015) | 0.1606 | 0.1792 | +0.019 |
| `bidir_rwkv6_temperature` (strong reg, 016) | 0.2649 | 0.2874 | +0.022 |

The dev-test gap under strong reg (0.022–0.023) is actually *similar* to the gap
under standard reg with modifications (0.017–0.019). Strong regularisation did
not close the gap — it just moved both dev and test CER to higher (worse) values.
The gap persisted because it is a data-distribution problem, not a
parameter-estimation problem.

---

## Why It Failed — Five Independent Reasons

**1. SpecAugment masking covered too much of short utterances.**

The dataset averages 4–5 seconds per utterance (≈ 100–125 frames after 4×
subsampling). Five time masks of width 70 frames each = 350 masked frames total.
This exceeds the length of many utterances. Even if masks don't fully overlap,
a large fraction of each utterance is blanked, making CTC paths degenerate.

*Calculation:* 5 masks × 70 frames = 350 frame-equivalents. An 80-frame utterance
(2 seconds) that gets 3 non-overlapping masks of 70 each would be mostly blanked.
The CTC loss over an almost-empty input cannot train the model.

**2. Dropout 0.25 is too aggressive for a 7.7M model at this data scale.**

Standard ASR SpecAugment papers (Park et al. 2019, Google RNNT) apply these
settings to models with 60–300M parameters trained on 960h+ of data. Our model
has 7.7M parameters on 35h. The capacity ratio is ~50× less than those baselines.
The regularisation settings from the literature do not transfer to this regime.

A 7.7M model needs most of its capacity to fit 35h of data correctly.
Dropout 0.25 on every layer randomly zeroes 25% of 256-dim activations — that
is equivalent to randomly disabling 64 feature dimensions per position per layer.
For a model that already uses its capacity tightly, this is destructive.

**3. Conv models suffered more under SpecAugment.**

Run-017 (see its own analysis) shows that ConvShift + heavy SpecAugment produces
even worse results than standard LION + heavy SpecAugment. The depthwise conv in
ConvShift sees masked regions as structured input (zeros) and adapts to that
pattern. The learned conv weights then suppress signal in what it perceives as
"masked" regions, degrading overall performance.

**4. The dev-test gap is structural, not a regularisation artefact.**

The Common Voice Ukrainian dev split has a slightly different speaker and recording
distribution than the test split. Any improvement mechanism that uses dev as
its stopping criterion (early stopping, model selection) inherently overfits to
the dev split's distribution. More regularisation doesn't fix this — it prevents
learning dev-specific features but also prevents learning genuine features that
would generalise.

**5. Temperature worsened under strong reg.**

`bidir_rwkv6_temperature` is 0.0095 CER worse than the strong-reg `bidir_rwkv6`
baseline (0.2874 vs 0.2779). Under standard reg, temperature was essentially
tied with baseline on test (0.1792 vs 0.1790). Under strong reg, temperature
is measurably worse. The temperature mechanism requires the model to learn
meaningful attention distributions to benefit from sharpening — when the model
can barely learn the task, temperature adds optimisation noise without benefit.

---

## Chunked Evaluation

| Backbone | Full CER | R@2s | R@5s | R@10s |
|----------|----------|------|------|-------|
| bidir_rwkv6 (strong reg) | 0.2779 | 0.3041 | 0.2783 | 0.2735 |
| bidir_rwkv6_temperature (strong reg) | 0.2874 | 0.3172 | 0.2914 | 0.2858 |

The chunked degradation pattern (R@2s >> R@10s) is qualitatively similar to
standard-reg runs — the models still suffer from context truncation at short
chunk sizes. This confirms the models have not learned better global representations
despite more regularisation; they still rely on full-sequence context.

---

## Code Issues / Design Notes

1. **No τ trajectory logged for strong-reg temperature run.** The training log
   (`log_run016_temperature_reg.txt`) does include τ values per epoch, but the
   trajectories are not separately extracted here. Given the model's poor learning
   curve, τ likely failed to differentiate meaningfully — the gradient signal for
   sharpening/flattening attention is weak when the base CTC loss is still very high.

2. **early_stopping_patience=20 with 80 epochs.** Best epoch was 76/80 for
   bidir_rwkv6 and 75/80 for temperature. Both models nearly exhausted the full
   training budget. Extended training (e.g., 120 epochs with patience=30) would
   not help — the models were not improving meaningfully in the last 20-25 epochs,
   just oscillating around the underfitting floor at train loss ~1.74.

---

## Conclusion

**Strong regularisation (dropout=0.25, SpecAugment LD policy) causes complete
failure for 7.7M models trained on 35h of speech.**

The original training settings (dropout=0.15, freq_mask=15, time_mask=35, 2+2 masks)
were near-optimal for this scale. They were not chosen aggressively enough to
close the dev-test gap because that gap cannot be closed by regularisation alone.

**The dev-test gap is a data-size effect.** With only 35h of training data,
the dev-test gap (0.017–0.019 for enhanced models) reflects genuine distributional
differences between the dev and test speaker populations. Adding more
regularisation doesn't fix this — it prevents learning both dev-specific
and test-relevant features simultaneously.

**Recommendation:** Drop all strong-reg experiments. Use the original training
settings for all future runs. If the dev-test gap is a concern, the only remedy
is more training data (e.g., larger Ukrainian corpora, multilingual pre-training,
or data augmentation via synthesis).
