# Environment Setup — Stage 2 Discretization Study

Self-contained reproduction guide for running the 8-run discretization
campaign (`scripts/launch_discretization.sh`) on a 2-GPU node.

## 1. Hardware assumptions

- 2× CUDA-capable GPUs (the script pins one process per device with
  `--gpu 0` and `--gpu 1`).
- ≥ 24 GB VRAM per device. Trapezoidal/AB3 variants run two and three
  parallel WKV scans respectively, raising peak VRAM from the ~4–6 GB
  seen at baseline to ~9–12 GB. AB3 is intentionally placed first on
  GPU 1 so the queue still proceeds if it explodes (see PLAN.md §4.1).
- ~50 GB free disk for the LibriSpeech cache + 8 × ~150 MB run dirs.

## 2. One-time setup

```bash
cd experiments/formal_v1

# Pin Python deps (no mamba-ssm needed for Stage 2)
uv sync

# Pre-fetch the LibriSpeech clean splits (~13 GB) into the cache dir.
# This avoids a slow first-epoch download race between the two GPUs.
uv run python -c "
from src.data.librispeech import load_librispeech
load_librispeech('train.clean.100', cache_dir='./data/librispeech')
load_librispeech('validation.clean', cache_dir='./data/librispeech')
load_librispeech('test.clean', cache_dir='./data/librispeech')
"
```

If `uv sync` fails on PyTorch (CUDA mismatch on a non-CUDA-13 host),
install Torch matching the local CUDA explicitly first:

```bash
uv pip install torch==2.11.0+cu124 --index-url https://download.pytorch.org/whl/cu124
uv sync
```

## 3. Environment variables for parallel 2-GPU runs

The launch script does not set these by default; add them to your shell
profile (or prefix the launch command) if you observe the corresponding
issues.

```bash
# Avoid HuggingFace dataset cache contention between concurrent processes
export HF_DATASETS_CACHE="${PWD}/data/librispeech"
export HF_HUB_DISABLE_TELEMETRY=1

# Determinism — keeps cuBLAS reductions reproducible across the two
# GPU processes. Slightly slower but worth it for the discretization
# story (small effects must be distinguishable from numerical noise).
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Cap CPU threads so the two processes do not over-subscribe cores
# (8 cores per process is plenty for the data loader + collate).
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Fail fast on NaN — important for AB3 stability monitoring
export TORCH_USE_CUDA_DSA=1
```

## 4. Sanity check before launch

```bash
# 1. Each new backbone resolves through the encoder factory
uv run python -c "
from src.config import ExperimentConfig
from src.models.encoder import build_encoder
for bb in ['rwkv6', 'rwkv6_trap', 'rwkv6_trap_var', 'rwkv6_gen2',
          'rwkv6_ab3', 'rwkv6_convshift_trap', 'lion_trap',
          'lion_convshift_trap']:
    cfg = ExperimentConfig(backbone=bb)
    enc = build_encoder(cfg)
    n = sum(p.numel() for p in enc.parameters())
    print(f'{bb:30s} → {enc.__class__.__name__:18s} params={n:,}')
"

# 2. One-batch forward+backward on each new backbone (no real data needed)
uv run python -c "
import torch
from src.config import ExperimentConfig
from src.models.encoder import build_encoder
torch.manual_seed(0)
for bb in ['rwkv6_trap', 'rwkv6_gen2', 'rwkv6_ab3', 'lion_trap']:
    cfg = ExperimentConfig(backbone=bb)
    enc = build_encoder(cfg).cuda()
    x = torch.randn(2, 250, 256, device='cuda')
    L = torch.tensor([250, 200], device='cuda')
    y, _ = enc(x, L)
    y.sum().backward()
    print(f'{bb:25s} OK  out_shape={tuple(y.shape)}')
"
```

If both blocks print without errors the campaign is ready to launch.

## 5. Launch and monitor

```bash
bash scripts/launch_discretization.sh
# Live monitoring:
tail -f outputs/logs/disc_gpu0.log
tail -f outputs/logs/disc_gpu1.log
# Per-run training curves (overwritten each epoch):
ls outputs/disc01_rwkv6_baseline_seed42/plots/
```

## 6. AB3 instability watch

If `outputs/disc05_rwkv6_ab3_seed42/history.csv` shows `grad_norm_mean
> 50` for two consecutive epochs, or NaN dev loss, kill that single run
and report it as a negative result:

```bash
# Find the AB3 process and stop it
pkill -f "rwkv6_ab3"
```

The remaining GPU 1 jobs (disc06–disc08) will continue automatically.
Document the failure mode in RESULTS.md under a Stage 2 "Negative
Results" subsection — a confirmed instability is a useful finding, not
a wasted run.

## 7. Post-run reporting

```bash
uv run python -m src.reporting.collect    # rebuilds outputs/_index.csv
uv run python -m src.reporting.tables     # rewrites RESULTS.md AUTOGEN tables
uv run python -m src.reporting.plots      # rebuilds outputs/_plots/*.png
```

The `gen2` learned coefficients should be inspected:

```bash
uv run python -c "
import torch, glob
ckpt = torch.load('outputs/disc04_rwkv6_gen2_seed42/best_model.pt',
                  map_location='cpu', weights_only=False)
sd = ckpt['model'] if 'model' in ckpt else ckpt
import torch.nn.functional as F
for k in sorted(k for k in sd if 'disc_alpha' in k):
    a = F.softplus(sd[k])
    print(f'{k}: softplus={a.tolist()}')
"
```

This produces the per-head, per-layer histogram of α₁/α₀ that tells us
whether the model used the trapezoidal flexibility or stayed near ZOH.
