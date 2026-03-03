"""Experiment configuration — loaded from YAML, overridable via CLI flags."""

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class ExperimentConfig:
    # ── Data ─────────────────────────────────────────────────────────────
    max_train_hours: float = 35.0
    max_val_hours: float = 5.0
    max_test_hours: float = 5.0
    max_audio_sec: float = 15.0
    min_audio_sec: float = 0.5
    sample_rate: int = 16000
    n_mels: int = 80
    win_length_ms: int = 25
    hop_length_ms: int = 10

    # ── Model ────────────────────────────────────────────────────────────
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    head_size: int = 64
    ffn_mult: int = 5
    dropout: float = 0.15
    conv_channels: int = 256

    # ── Training ─────────────────────────────────────────────────────────
    batch_max_seconds: float = 240.0
    num_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 5.0
    early_stopping_patience: int = 10
    seed: int = 42

    # ── SpecAugment ──────────────────────────────────────────────────────
    spec_augment: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 35
    num_freq_masks: int = 2
    num_time_masks: int = 2

    # ── Evaluation ───────────────────────────────────────────────────────
    chunk_sizes_sec: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0])
    max_carry_eval_utterances: int = 500

    # ── Paths ────────────────────────────────────────────────────────────
    output_dir: str = "./outputs/run_default"
    data_dir: str = "./data/cv_uk"

    # ── Common Voice download ─────────────────────────────────────────────
    cv_api_key: str = ""
    cv_dataset_id: str = "cmj8u3pys00t5nxxb56wugqgq"
    cv_tarball_name: str = "Common Voice Scripted Speech 24.0 - Ukrainian.tar.gz"

    # ── Backbones ────────────────────────────────────────────────────────
    backbones: List[str] = field(
        default_factory=lambda: [
            "transformer",
            "linear_attention",
            "mamba",
            "rwkv6",
            "rwkv7",
        ]
    )


def load_config(yaml_path: str, overrides: dict | None = None) -> ExperimentConfig:
    """Load config from YAML file with optional dict overrides."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    # Resolve CV_API_KEY from environment if not set in yaml
    if not data.get("cv_api_key"):
        data["cv_api_key"] = os.environ.get("CV_API_KEY", "")

    cfg = ExperimentConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg
