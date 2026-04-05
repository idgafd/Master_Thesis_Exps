"""Pure PyTorch Mamba SSM block — self-contained, no mamba-ssm dependency.

Implements the Selective Scan (S6) mechanism:
  in_proj -> (x, z) split -> conv1d -> x_proj(dt, B, C) -> SSM -> gate(silu(z)) -> out_proj

Reference: mamba_ssm/modules/mamba_simple.py (Gu & Dao, 2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class MambaBlock(nn.Module):
    """Pure PyTorch Mamba SSM block.

    Supports carry-state via step() and sequential scan.
    No compiled CUDA kernels — fully transparent and modifiable.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize dt_proj
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias (softplus range)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: (B, T, D)
        state: optional (conv_state, ssm_state) for carry
        Returns: (B, T, D), new_state_or_None
        """
        B, T, D = x.shape

        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # Conv1d (causal)
        x_conv = x_inner.transpose(1, 2)  # (B, d_inner, T)
        x_conv = self.act(self.conv1d(x_conv)[..., :T])  # causal: trim padding
        x_conv = x_conv.transpose(1, 2)  # (B, T, d_inner)

        # Project to dt, B, C
        x_dbl = self.x_proj(x_conv)  # (B, T, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # dt projection and softplus
        dt = self.dt_proj(dt)  # (B, T, d_inner)
        dt = F.softplus(dt)

        # A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # Sequential SSM scan
        y, new_ssm_state = self._ssm_scan(
            x_conv, dt, A, B_ssm, C_ssm, state
        )

        # Gate and output
        y = y * self.act(z)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv  # skip connection
        out = self.out_proj(y)

        return out, new_ssm_state

    def _ssm_scan(
        self,
        x: torch.Tensor,      # (B, T, d_inner)
        dt: torch.Tensor,     # (B, T, d_inner)
        A: torch.Tensor,      # (d_inner, d_state) — negative
        B: torch.Tensor,      # (B, T, d_state)
        C: torch.Tensor,      # (B, T, d_state)
        state: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selective scan in pure PyTorch (sequential over T)."""
        batch, T, d_inner = x.shape
        d_state = A.shape[1]

        if state is not None:
            _, ssm_state = state
            ssm_state = ssm_state.float()
        else:
            ssm_state = torch.zeros(batch, d_inner, d_state, dtype=torch.float32, device=x.device)

        x = x.float()
        dt = dt.float()
        B = B.float()
        C = C.float()

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]       # (B, d_inner)
            dt_t = dt[:, t, :]     # (B, d_inner)
            B_t = B[:, t, :]       # (B, d_state)
            C_t = C[:, t, :]       # (B, d_state)

            # Discretize
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)           # (B, d_inner, d_state)

            # State update
            ssm_state = ssm_state * dA + x_t.unsqueeze(-1) * dB

            # Output
            y_t = (ssm_state * C_t.unsqueeze(1)).sum(-1)  # (B, d_inner)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, T, d_inner)
        return y.to(x.dtype), ssm_state

    def step(
        self,
        x: torch.Tensor,  # (B, 1, D)
        conv_state: torch.Tensor,  # (B, d_inner, d_conv)
        ssm_state: torch.Tensor,   # (B, d_inner, d_state)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step inference for carry-state."""
        xz = self.in_proj(x.squeeze(1))  # (B, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)

        # Conv step
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_inner
        x_conv = torch.sum(
            conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = self.act(x_conv)

        # Project
        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())

        # Discretize
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(1)

        ssm_state = ssm_state * dA + x_conv.unsqueeze(-1) * dB
        y = (ssm_state.to(x_conv.dtype) * C_ssm.unsqueeze(1)).sum(-1)
        y = y + self.D * x_conv
        y = y * self.act(z)

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
