"""Mamba SSM encoder backbone with pure-PyTorch carry-state stepping."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Pure-PyTorch Mamba single-token step ─────────────────────────────────────

def _mamba_step_pytorch(
    mamba_block,
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-token Mamba step in pure PyTorch.

    Replaces mamba_block.step() to bypass the causal_conv1d CUDA kernel,
    which has version-compatibility issues with some torch/CUDA combos.

    Args:
        mamba_block: a mamba_ssm.Mamba module
        hidden_states: (B, 1, D)
        conv_state:   (B, d_inner, d_conv)
        ssm_state:    (B, d_inner, d_state)

    Returns: output (B, D), updated conv_state, updated ssm_state
    """
    dtype = hidden_states.dtype
    x_in = hidden_states.squeeze(1)  # (B, D)

    xz = mamba_block.in_proj(x_in)
    x, z = xz.chunk(2, dim=-1)

    # Shift conv buffer and apply depthwise conv
    conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
    conv_state[:, :, -1] = x.to(conv_state.dtype)
    x = (conv_state * mamba_block.conv1d.weight.squeeze(1)).sum(dim=-1)
    if mamba_block.conv1d.bias is not None:
        x = x + mamba_block.conv1d.bias
    x = F.silu(x).to(dtype=dtype)

    x_db = mamba_block.x_proj(x)
    dt, B_ssm, C_ssm = torch.split(
        x_db,
        [mamba_block.dt_rank, mamba_block.d_state, mamba_block.d_state],
        dim=-1,
    )
    dt = mamba_block.dt_proj(dt)
    dt = F.softplus(dt)

    A = -mamba_block.A_log.float().exp()  # (d_inner, d_state)
    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
    dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(1).float()

    ssm_state = ssm_state * dA + dB * x.unsqueeze(-1).float()
    y = (ssm_state * C_ssm.unsqueeze(1).float()).sum(dim=-1).to(dtype=dtype)
    y = y + mamba_block.D.float() * x
    y = y * F.silu(z)

    out = mamba_block.out_proj(y)
    return out, conv_state, ssm_state


# ── Encoder ───────────────────────────────────────────────────────────────────

class MambaEncoder(nn.Module):
    """Mamba SSM encoder supporting carry-state streaming inference.

    Requires: mamba-ssm and causal-conv1d packages.
    """

    supports_carry_state = True

    def __init__(self, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        from mamba_ssm import Mamba  # hard dependency — no fallback

        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "mamba": Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2),
                        "norm2": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 4, d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.dropout = nn.Dropout(dropout)
        print(
            f"Mamba encoder initialized "
            f"({n_layers} layers, d_model={d_model}, d_state=16, d_conv=4, expand=2)"
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        if state is not None:
            return self._forward_carry(x, state)
        return self._forward_stateless(x)

    def _forward_stateless(self, x: torch.Tensor):
        """Fast fused forward used during training and reset-state eval."""
        for layer in self.layers:
            residual = x
            x = residual + self.dropout(layer["mamba"](layer["norm1"](x)))
            x = x + layer["ffn"](layer["norm2"](x))
        return x, None

    def _forward_carry(self, x: torch.Tensor, state: List):
        """Token-by-token carry-state stepping in pure PyTorch."""
        B, T, D = x.shape
        new_state = []
        for layer_idx, layer in enumerate(self.layers):
            conv_st, ssm_st = state[layer_idx]
            residual = x
            x_norm = layer["norm1"](x)
            outputs = []
            for t in range(T):
                out_t, conv_st, ssm_st = _mamba_step_pytorch(
                    layer["mamba"], x_norm[:, t : t + 1, :], conv_st, ssm_st
                )
                outputs.append(out_t)
            x = residual + self.dropout(torch.stack(outputs, dim=1))
            x = x + layer["ffn"](layer["norm2"](x))
            new_state.append((conv_st, ssm_st))
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device) -> List:
        """Allocate zero initial state, matching each Mamba block's weight dtype."""
        states = []
        for layer in self.layers:
            mb = layer["mamba"]
            conv_dtype = mb.conv1d.weight.dtype
            states.append(
                (
                    torch.zeros(batch_size, mb.d_inner, mb.d_conv, dtype=conv_dtype, device=device),
                    torch.zeros(batch_size, mb.d_inner, mb.d_state, dtype=torch.float32, device=device),
                )
            )
        return states
