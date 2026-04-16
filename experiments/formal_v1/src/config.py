"""Experiment configuration — loaded from YAML, overridable via CLI flags."""

from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class ExperimentConfig:
    # ── Data ─────────────────────────────────────────────────────────────
    dataset: str = "librispeech_clean"  # librispeech_clean | librispeech_other
    max_audio_sec: float = 20.0
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
    dropout: float = 0.1
    conv_channels: int = 256

    # ── Encoder-specific ─────────────────────────────────────────────────
    backbone: str = "lion"
    # RWKV-6 / LION mode: "recurrent" | "lion" | "bidir_serial"
    rwkv_mode: str = "lion"
    # Mechanism flags (RWKV-6 / LION only)
    conv_shift: bool = False
    headscale: bool = False
    delta_rule: bool = False
    lucid: bool = False
    lucid_chunk_size: Optional[int] = None  # None = full-sequence
    lucid_self_reg: bool = False  # RKHS delta rule self-regulation in state
    temperature: bool = False
    # Mamba-specific
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # ── Training ─────────────────────────────────────────────────────────
    batch_max_seconds: float = 300.0
    num_epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    grad_clip: float = 5.0
    early_stopping_patience: int = 15
    seed: int = 42

    # ── SpecAugment (LibriSpeech LD policy) ──────────────────────────────
    spec_augment: bool = True
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2

    # ── Evaluation ───────────────────────────────────────────────────────
    chunk_sizes_sec: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0])
    max_carry_eval_utterances: int = 500
    max_reset_eval_utterances: int = 500
    chunked_eval_batch_size: int = 16

    # ── Compilation ──────────────────────────────────────────────────────
    compile_encoder: bool = False  # torch.compile the encoder for ~5× training speedup

    # ── Paths ────────────────────────────────────────────────────────────
    output_dir: str = "./outputs/run_default"
    data_cache_dir: str = "./data/librispeech"

    @property
    def ffn_dim(self) -> int:
        """RWKV-6 ChannelMix FFN dim formula, used by ALL architectures."""
        return int((self.d_model * 3.5) // 32 * 32)

    @property
    def hop_length_samples(self) -> int:
        return int(self.hop_length_ms * self.sample_rate / 1000)

    @property
    def win_length_samples(self) -> int:
        return int(self.win_length_ms * self.sample_rate / 1000)


def load_config(yaml_path: str, overrides: dict | None = None) -> ExperimentConfig:
    """Load config from YAML file with optional dict overrides."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})

    cfg = ExperimentConfig()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg
