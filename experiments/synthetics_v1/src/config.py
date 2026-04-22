"""Synthetics-v1 experiment configuration.

Lightweight dataclass with only the fields the encoder factory and training
loop need. Audio-specific fields from `formal_v1.src.config.ExperimentConfig`
are intentionally absent.

`to_experiment_config()` produces a populated ExperimentConfig that
`formal_v1.src.models.encoder.build_encoder()` can consume — this is how we
reuse the formal_v1 encoder factory without modifying it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class SyntheticsConfig:
    # ── Task ─────────────────────────────────────────────────────────────
    # MQAR knobs — see src/tasks/mqar.py for semantics.
    task: str = "mqar"
    vocab_size: int = 8192
    key_vocab_size: int = 4096          # first half of vocab is key alphabet
    value_vocab_size: int = 4096        # second half is value alphabet
    seq_len: int = 256
    n_kv_pairs: int = 64                # K, scales as ~T/4 by convention
    n_queries: Optional[int] = None     # None → equal to n_kv_pairs

    # ── Data pipeline ────────────────────────────────────────────────────
    train_examples_per_epoch: int = 50_000
    eval_examples: int = 3_000
    batch_size: int = 64
    num_workers: int = 2

    # ── Model (encoder spine — matches formal_v1 envelope) ───────────────
    backbone: str = "transformer_causal"
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    head_size: int = 64
    dropout: float = 0.1

    # Mamba family
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba2_d_state: int = 64
    mamba2_headdim: int = 64
    mamba2_ngroups: int = 1
    mamba2_chunk_size: int = 64

    # ── Training ─────────────────────────────────────────────────────────
    max_steps: int = 30_000
    eval_every_steps: int = 1_000
    log_every_steps: int = 100
    early_stop_threshold: float = 0.99      # per-sequence accuracy
    early_stop_patience_evals: int = 5      # consecutive evals without improvement
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1_000
    grad_clip: float = 5.0
    seed: int = 42

    # ── Compilation / runtime ────────────────────────────────────────────
    compile_encoder: bool = False
    device: str = "cuda"

    # ── Paths ────────────────────────────────────────────────────────────
    output_dir: str = "./outputs/run_default"

    @property
    def ffn_dim(self) -> int:
        """Same formula as formal_v1 — preserves matched parameter envelope."""
        return int((self.d_model * 3.5) // 32 * 32)

    @property
    def resolved_n_queries(self) -> int:
        return self.n_queries if self.n_queries is not None else self.n_kv_pairs


def load_config(yaml_path: str, overrides: dict | None = None) -> SyntheticsConfig:
    """Load YAML, apply optional dict overrides, return SyntheticsConfig."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    cfg = SyntheticsConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


