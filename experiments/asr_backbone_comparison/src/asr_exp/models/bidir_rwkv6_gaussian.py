"""Bidirectional RWKV v6 with Gaussian distance mask — LION parallel form.

Adds a learnable per-head Gaussian distance mask on top of the standard LION
bidirectional attention:

    effective_A[i,j] = LION_A[i,j] * exp(-(|i-j| - mu_h)^2 / (2*sigma_h^2))

Key properties:
  - Always non-negative (unlike cos mask from configs B/C/D)
  - Single peak, no secondary lobes (unlike cos² which has harmonics)
  - Learnable center mu_h: if mu > 0, attention peaks at a non-zero distance
  - Learnable width sigma_h: controls how sharply focused the distance mask is
  - At mu=0 and large sigma: degenerates to baseline LION (no modification)

This is the only experiment that can test whether monotone decay is fundamentally
sufficient, or whether the network wants to attend preferentially to specific
non-zero distances.

Compute overhead: one (H, T, T) element-wise multiply per layer on the
materialised attention matrix. At T~250, H=4: negligible vs matmul cost.

Extra params: 2 per head per layer = n_heads * n_layers * 2.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from asr_exp.models.bidir_rwkv6 import (
    _bidirectional_token_shift,
    BidirRWKV6ChannelMix,
)
from asr_exp.models.components import SinusoidalPE


# ── Gaussian-masked LION attention ───────────────────────────────────────────

def _lion_gaussian_parallel_attention(
    r: torch.Tensor,   # (B, H, T, K)
    k: torch.Tensor,   # (B, H, T, K)
    v: torch.Tensor,   # (B, H, T, K)
    w: torch.Tensor,   # (B, H, T, K) — log-space decay (negative)
    mu: torch.Tensor,  # (H,) — Gaussian center per head
    sigma: torch.Tensor,  # (H,) — Gaussian width per head (positive)
) -> torch.Tensor:
    """LION parallel full attention with per-head Gaussian distance mask.

    Computes the standard bidirectional LION attention matrix (forward tril +
    backward triu) and multiplies element-wise by a Gaussian mask:

        G_h[i,j] = exp(-(|i-j| - mu_h)^2 / (2 * sigma_h^2))

    The full attention matrix is materialised (T x T per head), which is fine
    for our sequence lengths (T ~ 250 after 4x subsampling).

    Returns: (B, H, T, K) float32.
    """
    B, H, T, K = r.shape

    r = r.float()
    k = k.float()
    v = v.float()
    w = w.float()

    # ── Standard LION prefix sums ──
    cs = torch.cumsum(w, dim=2)
    cs_b = cs - w

    shift_f = cs[:, :, T // 2 : T // 2 + 1, :]
    shift_b = cs_b[:, :, T // 2 : T // 2 + 1, :]
    cs_f_s = (cs - shift_f).clamp(-60, 60)
    cs_b_s = (cs_b - shift_b).clamp(-60, 60)

    # Forward (lower-triangular including diagonal)
    exp_cs_f = torch.exp(cs_f_s)
    A_fwd = torch.tril(
        (r * exp_cs_f) @ (k * torch.exp(-cs_f_s)).transpose(-2, -1)
    )

    # Backward (strictly upper-triangular)
    exp_cs_b = torch.exp(cs_b_s)
    A_bwd = torch.triu(
        (r * torch.exp(-cs_b_s)) @ (k * exp_cs_b).transpose(-2, -1),
        diagonal=1,
    )

    # Combined LION attention matrix: (B, H, T, T)
    A = A_fwd + A_bwd

    # ── Gaussian distance mask ──
    # Build distance matrix: (T, T)
    pos = torch.arange(T, device=r.device, dtype=torch.float32)
    dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()  # (T, T)

    # Per-head Gaussian: (H, T, T)
    mu_h = mu.view(H, 1, 1)          # (H, 1, 1)
    sigma_h = sigma.view(H, 1, 1)    # (H, 1, 1)
    gauss = torch.exp(-0.5 * ((dist.unsqueeze(0) - mu_h) / sigma_h) ** 2)

    # Apply mask: (B, H, T, T) * (1, H, T, T) → (B, H, T, T)
    A = A * gauss.unsqueeze(0)

    return A @ v  # (B, H, T, K)


# ── Time-mixing block ────────────────────────────────────────────────────────

class BidirRWKV6GaussianTimeMix(nn.Module):
    """Bidirectional RWKV6 time-mixing with Gaussian distance mask.

    Each head gets learnable (mu_h, sigma_h) controlling where in distance-space
    the attention peaks. Starts at mu=0 (local), sigma=large (flat) → baseline.
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        sigma_init: float = 15.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        hidden_size_att = hidden_size

        # ── Gaussian mask params ──
        # mu: init 0 (peak at distance 0 = local attention = baseline-like)
        self.gauss_mu_raw = nn.Parameter(torch.zeros(n_head, dtype=dtype))
        # sigma: init via softplus^{-1}(sigma_init). softplus(x) = log(1 + exp(x))
        # inverse: x = log(exp(sigma_init) - 1)
        sigma_raw_init = math.log(math.exp(sigma_init) - 1.0)
        self.gauss_sigma_raw = nn.Parameter(
            torch.full((n_head,), sigma_raw_init, dtype=dtype)
        )

        # ── Standard RWKV-6 params (identical to baseline) ──
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(num_hidden_layers - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)
            ddd = torch.ones(1, 1, hidden_size, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_DIM = 32
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_MIX_DIM * 5, dtype=dtype)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, D_MIX_DIM, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
            )

            decay_speed = torch.ones(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                decay_speed[n] = -6 + 5 * (n / (hidden_size_att - 1)) ** (
                    0.7 + 1.3 * ratio_0_to_1
                )
            self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, hidden_size_att))

            D_DECAY_DIM = 64
            self.time_decay_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_DECAY_DIM, dtype=dtype)
            )
            self.time_decay_w2 = nn.Parameter(
                torch.zeros(D_DECAY_DIM, hidden_size_att, dtype=dtype).uniform_(-0.01, 0.01)
            )

        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

    def get_gauss_info(self) -> dict:
        """Return mu and sigma per head for logging."""
        mu = F.softplus(self.gauss_mu_raw).detach().cpu()
        sigma = F.softplus(self.gauss_sigma_raw).detach().cpu()
        return {
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "mu_ms": (mu * 40.0).tolist(),      # convert frames → ms (40ms/frame)
            "sigma_ms": (sigma * 40.0).tolist(),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        dxprev = _bidirectional_token_shift(x) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, D)

        mw, mk, mv, mr, mg = xxx.unbind(dim=0)
        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = w.to(r.dtype)

        r_h = r.view(B, T, H, K).transpose(1, 2)
        k_h = k.view(B, T, H, K).transpose(1, 2)
        v_h = v.view(B, T, H, K).transpose(1, 2)
        w_h = w.view(B, T, H, K).transpose(1, 2)

        w_h = -torch.exp(w_h)

        # Resolve Gaussian params — keep on compute graph for gradient
        mu = F.softplus(self.gauss_mu_raw)       # (H,) always >= 0
        sigma = F.softplus(self.gauss_sigma_raw)  # (H,) always > 0

        y = _lion_gaussian_parallel_attention(r_h, k_h, v_h, w_h, mu, sigma)
        y = y.to(r.dtype)

        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)
        return y


# ── Block and Encoder ────────────────────────────────────────────────────────

class BidirRWKV6GaussianBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dropout: float,
        sigma_init: float = 15.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype) if layer_id == 0 else nn.Identity()

        self.att = BidirRWKV6GaussianTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id,
            sigma_init, dtype
        )
        self.ffn = BidirRWKV6ChannelMix(hidden_size, num_hidden_layers, layer_id, dtype)

        self.drop0 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.drop1 = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln0(x)
        x = self.drop0(x + self.att(self.ln1(x)))
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x


class BidirRWKV6GaussianEncoder(nn.Module):
    """Bidirectional RWKV v6 encoder with per-head Gaussian distance mask.

    Each head learns (mu_h, sigma_h) controlling distance-dependent attention:
      mu_h = 0   → peak at distance 0 (local, baseline-like)
      mu_h > 0   → peak at non-zero distance (non-monotone attention)
      sigma_h    → width of the attention window

    Starts at baseline behavior (mu=0, sigma=large) and learns deviations.
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        sigma_init: float = 15.0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % 32 == 0
        assert d_model % head_size == 0

        self.d_model = d_model
        self.n_layers = n_layers
        n_head = d_model // head_size

        extra = n_head * n_layers * 2  # mu + sigma per head per layer

        self.pos_enc = SinusoidalPE(d_model, max_len=8000)
        self.layers = nn.ModuleList([
            BidirRWKV6GaussianBlock(
                d_model, n_head, head_size, n_layers, i, dropout,
                sigma_init, dtype
            )
            for i in range(n_layers)
        ])

        sigma_init_ms = sigma_init * 40.0
        print(
            f"Bidirectional RWKV-6 GAUSSIAN encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"sigma_init={sigma_init:.1f} frames ({sigma_init_ms:.0f}ms), "
            f"extra_params={extra})"
        )

    def get_theta_info(self) -> Optional[str]:
        """Return per-layer Gaussian mu/sigma info for logging."""
        parts = []
        for i, layer in enumerate(self.layers):
            info = layer.att.get_gauss_info()
            mu_str = ",".join(f"{m:.1f}" for m in info["mu_ms"])
            sig_str = ",".join(f"{s:.0f}" for s in info["sigma_ms"])
            parts.append(f"L{i}:μ=[{mu_str}]ms σ=[{sig_str}]ms")
        return "  gauss: " + " | ".join(parts)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask_f = (
            (torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
            .unsqueeze(-1).float()
        )
        for layer in self.layers:
            x = layer(x) * mask_f
        return x, None
