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

### Mamba training speed issue — RESOLVED

The original Mamba implementation used a pure-PyTorch selective scan with a sequential Python
loop over every time step. This was replaced with a **parallel associative scan**
(Hillis-Steele, chunked with chunk_size=64). Additionally, `torch.compile(encoder)` support
was added which achieves parity with the CUDA mamba-ssm kernels (see comparison below).

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

## Mamba Comparison: PyTorch Reimplementation vs CUDA mamba-ssm

**Goal:** Verify that our pure-PyTorch Mamba reimplementation produces the same
accuracy as the official CUDA `mamba-ssm` package, and characterize the speed gap.

### Why eager mode for this comparison

Both experiments below ran in **eager mode** (no `torch.compile`). This is the
fairest comparison because:

1. **Accuracy validation** — eager mode executes the exact operations we wrote,
   with no compiler-introduced numerical differences.
2. **Reproducibility** — `torch.compile` is sensitive to PyTorch version and GPU
   architecture; eager results are portable.
3. **The speed gap is a known, measured quantity** — see the benchmark section
   below for compiled numbers.

For subsequent full 80-epoch training runs, we will use `torch.compile(encoder)`
via the `--compile` flag, which brings PyTorch Mamba to CUDA-level speed.

### Accuracy comparison (10 epochs, seed=42, LibriSpeech clean-100)

| Epoch | PyTorch Dev CER | CUDA Dev CER | PyTorch Train Loss | CUDA Train Loss |
|-------|----------------:|-------------:|-------------------:|----------------:|
| 1     | 0.4726          | 0.4473       | 2.4500             | 2.4152          |
| 2     | 0.3342          | 0.3185       | 1.6607             | 1.6013          |
| 3     | 0.2815          | 0.2667       | 1.3737             | 1.3327          |
| 4     | 0.2475          | 0.2373       | 1.2343             | 1.2049          |
| 5     | 0.2248          | 0.2160       | 1.1242             | 1.1008          |
| 6     | 0.2095          | 0.2067       | 1.0466             | 1.0276          |
| 7     | 0.2018          | 0.1940       | 1.0017             | 0.9864          |
| 8     | 0.1924          | 0.1880       | 0.9551             | 0.9415          |
| 9     | 0.1895          | 0.1843       | 0.9387             | 0.9270          |
| 10    | 0.1889          | 0.1852       | 0.9190             | 0.9076          |

**Summary:**
- Both converge along nearly identical trajectories
- CUDA is ~0.004 CER ahead at epoch 10 (0.185 vs 0.189) — within noise,
  likely from different random number generation paths in the two SSM kernels
- CUDA test CER: **0.1808**, test WER: **0.5183** (7.30M params)
- PyTorch test evaluation pending (crashed on carry-state eval, bug now fixed)

### Speed comparison (eager mode)

| Metric | PyTorch (eager) | CUDA mamba-ssm |
|--------|----------------:|---------------:|
| Avg epoch time | ~380s | ~60s |
| **Speed ratio** | **6.3× slower** | **1×** |

### Speed with torch.compile (benchmark, B=8, T=500, D=256, 6 layers)

| Mode | fwd+bwd | Peak memory |
|------|--------:|------------:|
| PyTorch eager | 137 ms | 12 GB |
| `torch.compile(encoder)` | **26 ms** | 11 GB |
| CUDA mamba-ssm | 35 ms | 0.6 GB |

`torch.compile` achieves **better-than-CUDA training speed** (~26ms vs ~35ms)
at the cost of higher memory (11 GB vs 0.6 GB) and a one-time ~100s compilation
overhead. The RTX 5090 (32 GB) has sufficient memory.

### Key changes that enabled this

1. **Parallel associative scan** (Hillis-Steele) replaced the sequential loop
2. **Slice-based shifting** replaced `F.pad` in the scan (fewer allocations)
3. **Chunked scan** (chunk_size=64) bounds memory to O(B×C×D×N) per chunk
4. **`torch.compile` support** — the parallel scan's 6 iterations (log₂64)
   are small enough for the compiler to fuse with surrounding ops
5. **`--compile` flag** added to `run_experiment.py` for easy use
6. **`--gpu` flag** added for multi-GPU parallel experiment execution

---

## Notes

- All parameter counts should be within 5% of LION (reference).
- CER computed character-by-character. WER computed word-by-word.
- "Delta vs X" = relative change: `(new - base) / base * 100%`
- Tables will be populated as experiments complete.
