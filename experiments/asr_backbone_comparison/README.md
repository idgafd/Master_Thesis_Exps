# ASR Encoder Backbone Comparison

Minimal CTC-based ASR pipeline comparing five encoder backbones on ~32h Ukrainian speech from Mozilla Common Voice. Only the encoder block is swapped; frontend (ConvSubsampling), optimizer, and CTC head are identical across all runs.

## Hypothesis

1. Lightweight recurrent/SSM encoders (Mamba, RWKV-6) achieve CER competitive with a Transformer of comparable size on 32h data.
2. Under chunked inference (2–5s chunks), stateful architectures with carry-state achieve lower CER than reset-state.
3. Stateful encoders degrade less from full-utterance to chunked CER than the stateless Transformer.

## Backbones

| Backbone | Complexity | Stateful |
|---|---|---|
| Transformer | O(T²) | No |
| Linear Attention | O(T·D²) | No |
| Mamba | O(T) | Yes |
| RWKV-6 | O(T) | Yes |
| RWKV-7 | O(T) | Yes |

All: ~7M parameters, 6 layers, d_model=256.

## Project Structure

```
experiments/asr_backbone_comparison/
├── pyproject.toml          # uv project + dependencies
├── setup_rwkv.sh           # clone & patch RWKV-block, install it
├── configs/
│   └── default.yaml        # full experiment config
├── src/asr_exp/
│   ├── config.py           # ExperimentConfig dataclass + YAML loader
│   ├── data/               # text normalization, vocab, dataset, CV loader
│   ├── models/             # all 5 encoder backbones + ASRModel wrapper
│   ├── training/           # train loop, evaluators, CTC decode
│   └── utils/              # seeding, param count, plots
└── scripts/
    └── run_experiment.py   # CLI entry point (restartable)
```

## Setup

### 1. Prerequisites

- Python ≥ 3.10
- CUDA 12.4 (adjust `cu124` in `pyproject.toml` for other versions)
- [uv](https://github.com/astral-sh/uv) package manager

### 2. Create environment and install dependencies

```bash
cd experiments/asr_backbone_comparison

# Create venv and install all Python deps (torch via CUDA wheel)
uv sync

# Install RWKV-block (clones, patches, installs as editable)
bash setup_rwkv.sh

# mamba-ssm needs to be built against the installed torch/CUDA
# Run inside the uv venv:
uv pip install causal-conv1d mamba-ssm --no-build-isolation
```

### 3. Common Voice API key

Get your key at <https://commonvoice.mozilla.org/en/profile/settings> and export it:

```bash
export CV_API_KEY="your_key_here"
```

Or set `cv_api_key` in `configs/default.yaml` (do not commit credentials).

If the dataset tarball or extracted folder already exist in `data/cv_uk/`, the download step is skipped automatically.

## Running Experiments

```bash
# Full run with default config
uv run scripts/run_experiment.py

# Use a custom config
uv run scripts/run_experiment.py --config configs/default.yaml

# Run only specific backbones
uv run scripts/run_experiment.py --backbone transformer mamba

# Resume after interruption (default behavior — completed backbones are skipped)
uv run scripts/run_experiment.py --resume

# Re-run a backbone that already has results
uv run scripts/run_experiment.py --backbone rwkv7 --force

# Load saved checkpoints and evaluate only (no training)
uv run scripts/run_experiment.py --eval-only

# Override output directory
uv run scripts/run_experiment.py --output-dir outputs/run_v2
```

## Restart / Resume Behavior

The runner is designed to be safely interrupted and restarted:

| File | Purpose |
|---|---|
| `outputs/<run>/results.json` | Aggregated results for all completed backbones |
| `outputs/<run>/best_<backbone>.pt` | Best model checkpoint (lowest dev CER) |
| `outputs/<run>/history_<backbone>.json` | Per-epoch training history |
| `outputs/<run>/vocab.json` | Vocabulary (built once from train split) |

On restart, any backbone already present in `results.json` is skipped. Pass `--force` to re-run a specific backbone.

## Outputs

```
outputs/<run>/
├── config_snapshot.yaml          # fully-resolved config (incl. CLI overrides)
├── run_info.json                 # GPU, PyTorch version, Python, timestamp
├── results.json                  # aggregated metrics for all completed backbones
├── results_tables.txt            # formatted CER tables (Tables 1–3)
├── vocab.json                    # character vocabulary
├── best_<backbone>.pt            # best checkpoint per backbone (lowest dev CER)
├── history_<backbone>.json       # per-epoch loss/CER/WER/LR — written every epoch
├── samples_<backbone>.txt        # first 50 test ref/hyp pairs
└── plots/
    ├── convergence.{pdf,png}     # train/dev loss and CER curves
    ├── chunked_cer_reset.{pdf,png}  # grouped bar chart: full vs chunked (reset)
    └── carry_vs_reset.{pdf,png}  # carry vs reset CER for stateful models
```

### `results.json` schema (per backbone)

```jsonc
{
  "backbone": "mamba",
  "params": {
    "frontend":  412160,   // ConvSubsampling
    "encoder":  6291456,   // backbone-specific layers
    "ctc_head":   10240,   // LayerNorm + linear projection
    "total":    6713856
  },
  "n_params": 6713856,             // same as params.total (backwards compat)
  "best_epoch": 18,
  "best_dev_cer": 0.2134,
  "training_wall_time_s": 7240.5,  // sum of epoch times
  "test": {"loss": ..., "cer": ..., "wer": ...},
  "chunked_reset": {"2.0s": {"cer": ..., "wer": ..., "n_evaluated": ...}, ...},
  "chunked_carry": {"2.0s": {"cer": ..., "wer": ..., "n_evaluated": ...}, ...},
  "history": {
    "epoch": [...], "train_loss": [...], "dev_loss": [...],
    "dev_cer": [...], "dev_wer": [...], "lr": [...], "epoch_time_s": [...]
  }
}
```

## Config Reference

All parameters live in `configs/default.yaml` and map 1:1 to `ExperimentConfig` fields in `src/asr_exp/config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `max_train_hours` | 35.0 | Cap training data |
| `d_model` | 256 | Model hidden dim |
| `n_layers` | 6 | Encoder depth |
| `num_epochs` | 30 | Max training epochs |
| `lr` | 3e-4 | Peak learning rate |
| `batch_max_seconds` | 240 | Dynamic batch budget (seconds of audio) |
| `chunk_sizes_sec` | [2, 5, 10] | Chunk sizes for chunked inference eval |
| `early_stopping_patience` | 10 | Epochs without dev CER improvement |
| `backbones` | all 5 | Which backbones to run |

## Known Issues

- **RWKV-7**: Converges poorly in stock form due to decay/v_first/k_a init issues. Run-018 fixes bring it to Mamba-level (CER 0.2602) but still far below LION.
- **Carry-state eval subset**: carry-state evaluation is limited to `max_carry_eval_utterances=500` utterances (Mamba step-by-step is slow). This makes direct comparison with reset-state (full test set) unfair — noted in results.
- **mamba-ssm CUDA kernels**: The carry-state Mamba step uses a pure-PyTorch implementation to avoid `causal_conv1d` kernel version mismatches. Training uses the fused kernel normally.

---

## Experiment Index

All runs share the common setup above (6 layers, d=256, 4 heads, cosine+warmup, 60 epochs, SpecAugment) unless noted.

### Run Overview

| Run | Config / tag | What was tested | Test CER | Connects to |
|-----|-------------|-----------------|----------|-------------|
| 003 | `mamba-wsd12` | Mamba + WSD scheduler, decay start at epoch 12 | 0.2098 | → 004 |
| 004 | `mamba-wsd46` | Mamba + WSD scheduler, decay start at epoch 46 | 0.2125 | establishes Mamba ceiling |
| 005 | `bidir-cosine` | LION (bidir RWKV-6) vs bidir linear attention | 0.1790 / 0.2044 | LION becomes the base for runs 006–015 |
| 006 | `bidir-conv` | LION + ConvShift token-mix; with/without Xres gate | **0.1760** / 0.1813 | best result; nogate variant used in run-008 |
| 007 | `biwkv6` | BiWKV6 serial bidir (streaming-capable); with/without ConvShift+gate | 0.2201 / 0.2211 | → 009 (deeper + longer) |
| 008 | `lion12` | LION 12-layer (2× depth), 100 epochs | 0.1994 | scaling dead-end; confirms arch > capacity |
| 009 | `biwkv6-6L` | BiWKV6 6-layer, 100 epochs | 0.1894 | extended training helps; carry-state streaming explored |
| 010 | `complex_decay` | Complex decay poles: fixed θ=0.31 (cplx_b) and θ=0.90 (cplx_c) | 0.2140 / 0.2322 | oscillatory (negative) attention regions are harmful → 011 |
| 011 | `complex_d_cos2` | Learnable θ per layer (cplx_d) + cos² non-negative mask (cplx_b_cos2) | 0.2107 / 0.1955 | cos² idea carried to → 012 |
| 012 | `d_cos2_headscale` | Learnable θ + cos² combined (d_cos2); per-head decay bias (headscale, 24 params) | 0.1977 / **0.1839** | headscale is cheapest useful addition found |
| 013 | `gaussian_dual` | Gaussian attention modulation (48 params); dual-decay weighted output (1560 params) | 0.1861 / 0.1875 | both marginal; dual-decay expensive for similar gain |
| 014 | `layerconv` | Layer-dependent ConvShift: kernel 7 → 3 across layers | 0.1768 | extends run-006 ConvShift |
| 015 | `temperature` | Per-head learnable temperature τ (sharpens upper-layer attention) | 0.1792 | related to headscale (run-012) |
| 016 | `temperature_reg` | bidir_rwkv6 + temperature with strong reg (dropout=0.25, heavy SpecAug) | 0.2779 / 0.2874 | underfitting — reg too aggressive |
| 017 | `layerconv_reg` | conv_nogate + layerconv with strong reg (dropout=0.25, heavy SpecAug) | 0.3189 / 0.3205 | underfitting — reg too aggressive |
| 018 | `rwkv7_fix` | RWKV-7 decay init fix vs all fixes (decay + v_first + k_a) | 0.3776 / 0.2602 | decay-only still broken; all-fixes approach Mamba |

### Architecture Lineage

```
Mamba  (runs 003–004)         CER ~0.21   WSD scheduler ablation
  └── replaced by
LION / bidir_rwkv6 (run 005)  CER 0.1790  new baseline
  ├── + ConvShift  (run 006)  CER 0.1760  ← best overall
  ├── BiWKV6 fork  (run 007)  CER 0.220   streaming variant
  │     └── 6L 100ep (run 009) CER 0.189  longer training helps streaming
  ├── 12-layer     (run 008)  CER 0.199   depth does not help
  ├── complex decay (run 010) CER 0.21–0.23  oscillatory mask harmful
  │     └── cos²   (run 011) CER 0.196   non-negative mask recovers
  │           └── headscale (run 012) CER 0.184  cheapest win (24 params)
  ├── Gaussian mask (run 013) CER 0.186   non-monotone attention, marginal
  ├── layer ConvShift (run 014) CER 0.177  layer-dep kernels, ~same as uniform
  ├── temperature   (run 015) CER 0.179  per-head τ, ~same as baseline
  ├── strong reg    (runs 016–017)        underfitting — too aggressive
  └── RWKV-7 fixes  (run 018) CER 0.260  fix_all recovers partial convergence
```

### Best Results

| Rank | Run | Backbone | Test CER |
|------|-----|----------|----------|
| 1 | 006 | `bidir_rwkv6_conv_nogate` (LION + ConvShift) | **0.1760** |
| 2 | 014 | `bidir_rwkv6_layerconv` (LION + LayerConv) | 0.1768 |
| 3 | 005 | `bidir_rwkv6` (LION baseline) | 0.1790 |
| 4 | 015 | `bidir_rwkv6_temperature` (LION + τ) | 0.1792 |
| 5 | 012 | `bidir_rwkv6_headscale` | 0.1839 |
| 6 | 013 | `bidir_rwkv6_gaussian` | 0.1861 |
| 7 | 009 | `biwkv6_no_conv_no_gate` 6L/100ep | 0.1894 |
| 8 | 003 | `mamba` (WSD-12) | 0.2098 |
| 9 | 018 | `rwkv7_fix_all` (RWKV-7 all fixes) | 0.2602 |
