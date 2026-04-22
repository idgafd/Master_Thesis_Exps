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
    # Stage 2 discretization scheme — overrides the standard ZOH state update.
    # Values: "zoh" | "trap" | "trap_var" | "gen2" | "ab3"
    #   zoh:      S_t = W S_{t-1} + k_t v_t^T            (current default)
    #   trap:     S_t = W S_{t-1} + ½(k_t v_t^T + W k_{t-1} v_{t-1}^T)
    #   trap_var: trap with geometric-mean decay W̃ = sqrt(W_t W_{t-1})
    #   gen2:     learnable α₀, α₁ per head (initialized to ZOH or trap)
    #   ab3:      Adams-Bashforth 3-step (23/12, -16/12, 5/12), decay clamped
    discretization: str = "zoh"
    discretization_init: str = "zoh"  # "zoh" | "trap" — only used by gen2
    discretization_drop_u: bool = False  # ablate the bonus term `u` (time_faaaa)
    # ── Stage 3 RSE (Rotational State Evolution) ─────────────────────────
    # Replaces scalar diagonal decay with 2x2 block-diagonal SO(2)×R+ transition.
    # Each head's K dim splits into K/2 blocks of 2 rows; each block has a
    # data-dependent rotation angle θ in addition to the scalar decay.
    # See ProposalA.md for the full mathematical specification.
    rse: bool = False
    # Multi-Rate RSE: M independent (lambda_m, theta_m) scales with
    # query-conditional softmax mixer.  M = 1 reproduces single-scale RSE.
    rse_n_scales: int = 1
    # ── Stage 5 P²-RSE (Paired-Pole RSE) ─────────────────────────────────
    # Two complex poles per 2x2 block with shared λ, independent θ, and
    # data-dependent real mixer (unconstrained or softmax).  Initialization:
    # θ^(2)_base = -θ^(1)_base (phase-complementary).
    # Mixer variants:
    #   "linear" — unconstrained real β ∈ R^2 (main proposal, Exp A)
    #   "softmax" — convex β (control, Exp B) — tests whether softmax was the
    #              m2 plateau cause
    p2rse: bool = False
    p2rse_mixer: str = "linear"  # "linear" | "softmax"
    # Stage 5 Phase 3: viscosity coupling (Rayleigh dissipation).
    # λ_eff = λ_raw + η_{h,b} · θ² inside the RSE complex scan, with
    # η ∈ R^(H×Bk) initialized to 0 (bit-identical to baseline at init).
    rse_viscosity: bool = False
    # ── Stage 10.1 — Log-Linear RWKV-6 (Fenwick bucket readout) ──────────
    # L ≈ ceil(log2 T) + 1 Fenwick bucket states partition the prefix at
    # log-scale; per-token per-scale mixer λ_t^(ℓ) = 1 + LoRA(x) (zero-init LoRA)
    # ⇒ at step 0, Σ_ℓ S^(ℓ) = S and Σ_ℓ λ·S^(ℓ) = S ≡ vanilla RWKV-6 readout.
    use_loglinear: bool = False
    loglinear_levels: int = 10
    # ── Stage 10.2 — M²RNN sparing-use (non-linear state, sole Family-C) ─
    # Parallel non-linear branch Z=tanh(SW+kv^T), gated forget-update, added
    # to the RWKV readout with scalar λ_h (zero-init). Active at one layer only.
    use_m2rnn: bool = False
    m2rnn_layer: int = 5  # top layer of a 6-layer stack (0-indexed)
    # ── Stage 10.3 — Multi-dilation ConvShift (Family A input-side) ──────
    # Parallel DWConv1d branches with dilations {1, 2, 4, 8}, learnable
    # per-layer α_d (α_1=1, α_{2,4,8}=0 at init). Causal padding in
    # mode=recurrent; symmetric in mode=lion/bidir_serial.
    use_conv_shift_multidilation: bool = False
    # CB-3: content-conditional α_d via softmax(W_α · x_t + b_d). Each
    # token selects its own dilation mix. b_d init = large log-one-hot
    # on d=1 → reduces to single-dilation at init.
    conv_shift_multidil_content_conditional: bool = False
    # ── Stage 10.4 — Avey partial-embedding ChannelMix bypass (Family D) ─
    # Split W_k output in half along FFN dim → [z_h, z_t]; tail gets ReLU²,
    # head passes linearly; α-gated bypass, α=0 at init ⇒ vanilla ChannelMix.
    use_chanmix_bypass: bool = False
    # ── Stage 10.5 — Cayley-orthogonal transition (Family B, NCGRU-style) ─
    # Full SO(K) transition via G_t = exp(-λ_t) · O_t where
    # O_t = (I - A_t)(I + A_t)^{-1} and A_t = U_t V_t^T - V_t U_t^T is a
    # rank-2·cayley_rank skew matrix. U=V=0 at init ⇒ A=0 ⇒ O=I ⇒ vanilla.
    use_cayley_orthogonal: bool = False
    cayley_rank: int = 1  # rank-1 keeps param parity; higher rank breaks parity.
    # ── Stage 10.6 — PoM polynomial value-lift (Family D) ────────────────
    # v̂_t = v_t + Σ_{p=2..k} γ_p ⊙ (W_h x_t)^⊙p; γ=0 at init ⇒ v̂=v.
    # Thin config: D_pom = 64, k = 2.
    use_pom_vlift: bool = False
    pom_order: int = 2
    pom_expansion: int = 64
    # ── CB-5 — Frontend v2 (3-stage Conv1d + pre-LN + SiLU) ──────────────
    # Replaces ConvSubsampling with ConvSubsamplingV2.  Tests whether the
    # acoustic feature extraction stage is the binding gap at this scale
    # (hypothesis from STAGE10_ANALYSIS §4.2).  NOT a mechanism flag for
    # the mixer — independent of all the Family A/B/C/D mechanism flags.
    use_frontend_v2: bool = False
    # Mamba-specific
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    # Mamba-2 specific (used by mamba2 backbone family only)
    mamba2_d_state: int = 64
    mamba2_headdim: int = 64
    mamba2_ngroups: int = 1
    mamba2_chunk_size: int = 64

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
