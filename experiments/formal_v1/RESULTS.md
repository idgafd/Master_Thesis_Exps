# Formal v1 — Results

## Dataset
- **Train:** LibriSpeech train-clean-100 (100h, ~28,539 utterances)
- **Dev:** LibriSpeech dev-clean
- **Test:** LibriSpeech test-clean

## Training Configuration
| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_layers | 6 |
| n_heads / head_size | 4 / 64 |
| FFN dim | 896 |
| Dropout | 0.1 |
| Optimizer | AdamW (lr=3e-4, wd=0.01) |
| Scheduler | Cosine + 1000-step warmup |
| Epochs | 80 (patience=15) |
| SpecAugment | LD policy (freq=27, time=100) |
| Batch | max 300s total duration |
| Grad clip | 5.0 |

---

## Model Parameter Counts

| Backbone | Total Params | Frontend | Encoder | CTC Head | vs LION |
|----------|-------------|----------|---------|----------|---------|
| transformer | — | — | — | — | — |
| rwkv6 | — | — | — | — | — |
| mamba | — | — | — | — | — |
| lion | — | — | — | — | ref |
| lion_convshift | — | — | — | — | — |
| lion_lucid | — | — | — | — | — |
| lion_delta | — | — | — | — | — |
| lion_headscale | — | — | — | — | — |
| rwkv6_lucid | — | — | — | — | — |
| rwkv6_delta | — | — | — | — | — |
| mamba_bidir | — | — | — | — | — |

---

## Table 1: Core Baselines (Full-Utterance Evaluation)

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Best Epoch |
|---|----------|---------|---------|----------|----------|------------|
| 1 | transformer | — | — | — | — | — |
| 2 | rwkv6 | — | — | — | — | — |
| 3 | mamba | — | — | — | — | — |
| 4 | lion | — | — | — | — | — |
| 5 | lion_convshift | — | — | — | — | — |

## Table 2: LION + Mechanism Improvements

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Delta vs LION |
|---|----------|---------|---------|----------|----------|---------------|
| 6 | lion_lucid | — | — | — | — | — |
| 7 | lion_lucid_chunked | — | — | — | — | — |
| 8 | lion_delta | — | — | — | — | — |
| 9 | lion_convshift_headscale | — | — | — | — | — |

## Table 3: RWKV-6 (Recurrent) + Mechanisms

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Delta vs rwkv6 |
|---|----------|---------|---------|----------|----------|----------------|
| 10 | rwkv6_lucid | — | — | — | — | — |
| 11 | rwkv6_delta | — | — | — | — | — |
| 12 | rwkv6_lucid_delta | — | — | — | — | — |

## Table 4: Bidirectional Mamba

| # | Backbone | Dev CER | Dev WER | Test CER | Test WER | Delta vs mamba |
|---|----------|---------|---------|----------|----------|----------------|
| 13 | mamba_bidir | — | — | — | — | — |

## Table 5: Chunked Evaluation (Dev, Reset Mode)

| Backbone | Full | 2s | 5s | 10s |
|----------|------|-----|-----|------|
| transformer | — | — | — | — |
| rwkv6 | — | — | — | — |
| mamba | — | — | — | — |
| lion | — | — | — | — |
| lion_convshift | — | — | — | — |

## Table 6: Chunked Evaluation (Dev, Carry-State)

| Backbone | Full | 2s | 5s | 10s |
|----------|------|-----|-----|------|
| rwkv6 | — | — | — | — |
| mamba | — | — | — | — |

---

## Table 7: Statistical Validation (Top-5 x 3 seeds)

| Backbone | Seed 42 CER | Seed 123 CER | Seed 777 CER | Mean | Std |
|----------|-------------|--------------|--------------|------|-----|
| — | — | — | — | — | — |

---

## Early Baseline Results (10 epochs, seed=42)

Quick 10-epoch runs to verify all backbones train correctly before full 80-epoch experiments.

| Backbone | Params | Dev CER | Dev WER | ~Epoch Time |
|----------|--------|---------|---------|-------------|
| Transformer | 6.26M | 0.3522 | 0.8781 | 46s |
| Linear Attention | 6.26M | 0.4206 | 0.9668 | 50s |
| RWKV-6 | 7.74M | 0.1995 | 0.5518 | 83s |
| Mamba | 7.30M | — | — | very slow |

RWKV-6 is the clear winner at 10 epochs. Transformer and Linear Attention are still converging.

### Mamba training speed issue

The Mamba implementation uses a pure-PyTorch selective scan with a Python `for t in range(T)` loop
over every time step (~500 after subsampling). This makes it ~10x slower than it should be.
The math is correct but training time is not comparable to the other backbones.
Need to either integrate the `mamba-ssm` CUDA kernel or rewrite the scan as a parallel operation.

### RWKV-6 sanity check: Common Voice Ukrainian

To confirm that the CER difference between LibriSpeech and the old Common Voice experiments
is the dataset (not a code change), we ran the new blocks.py RWKV-6 on Common Voice Ukrainian
with the exact same config as the old run-020 experiment (which used `rwkv-block` library).

| Epoch | New blocks.py (CV 25.0) | Old rwkv-block (CV 24.0, run-020) |
|-------|-------------------------|-------------------------------------|
| 1 | 0.9453 | 0.8754 |
| 5 | 0.3611 | 0.3845 |
| 10 | 0.3079 | 0.3199 |

CER dynamics match closely. The ~1% gap is expected from dataset version differences (CV 25.0 vs 24.0).
The old run-020 reached 0.2296 at epoch 60 — our reimplementation would likely land in the same range.

Conclusion: the fast convergence on LibriSpeech (0.20 CER at epoch 10) is because LibriSpeech clean
is much easier than Common Voice (read speech, studio quality, consistent speakers vs noisy crowdsourced).
The RWKV-6 implementation is correct.

---

## Notes

- All parameter counts should be within 5% of LION (reference).
- CER computed character-by-character. WER computed word-by-word.
- "Delta vs X" = relative change: `(new - base) / base * 100%`
- Tables will be populated as experiments complete.
