"""Pure PyTorch Mamba SSM block — self-contained, no mamba-ssm dependency.

Implements the Selective Scan (S6) mechanism:
  in_proj -> (x, z) split -> conv1d -> x_proj(dt, B, C) -> SSM -> gate(silu(z)) -> out_proj

Two scan backends:
  1. parallel_scan — Hillis-Steele associative scan, O(T log T) work, no compilation.
     Good for eager-mode inference and as a fallback.
  2. sequential_scan — simple loop, designed for torch.compile fusion into a single
     CUDA kernel. When the *encoder* is compiled, this runs ~5× faster than (1).

Wrap the encoder with `torch.compile(encoder)` for training-speed parity with
the CUDA mamba-ssm kernels. See benchmark results in mamba_encoder.py docstring.

Reference: mamba_ssm/modules/mamba_simple.py (Gu & Dao, 2023)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# ---------------------------------------------------------------------------
# Scan backends
# ---------------------------------------------------------------------------

def parallel_scan(gates: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Hillis-Steele inclusive prefix-sum scan.

    Computes:  h_0 = tokens_0,  h_t = gates_t * h_{t-1} + tokens_t

    O(T log T) work, O(log T) depth.  No torch.compile needed — works in
    eager mode and inside gradient-checkpointed regions.

    Args:
        gates:  (B, T, D, N) — multiplicative coefficients (dA)
        tokens: (B, T, D, N) — additive terms (dB * x)
    Returns:
        h: (B, T, D, N) — hidden states at each time step
    """
    T = gates.shape[1]
    a = gates
    b = tokens

    num_steps = math.ceil(math.log2(max(T, 2)))
    for d in range(num_steps):
        stride = 1 << d
        a_shifted = F.pad(a[:, :-stride], (0, 0, 0, 0, stride, 0), value=1.0)
        b_shifted = F.pad(b[:, :-stride], (0, 0, 0, 0, stride, 0), value=0.0)
        new_a = a * a_shifted
        new_b = a * b_shifted + b
        a = new_a
        b = new_b

    return b


def sequential_scan(gates: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Sequential linear-recurrence scan — compile-friendly.

    Same semantics as `parallel_scan` but written as a plain loop so that
    `torch.compile` can fuse it into a single CUDA kernel.  In eager mode
    this is slower than the parallel variant; when the *enclosing module* is
    compiled it is ~4× faster.

    Args / Returns: same as `parallel_scan`.
    """
    B, T, D, N = gates.shape
    h = tokens[:, 0]
    outputs = [h]
    for t in range(1, T):
        h = gates[:, t] * h + tokens[:, t]
        outputs.append(h)
    return torch.stack(outputs, dim=1)


def selective_scan(
    x: torch.Tensor,      # (B, T, D)
    dt: torch.Tensor,     # (B, T, D)
    A: torch.Tensor,      # (D, N) — negative
    B: torch.Tensor,      # (B, T, N)
    C: torch.Tensor,      # (B, T, N)
    state: Optional[torch.Tensor] = None,  # (B, D, N)
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked selective scan — bounded memory, GPU-friendly.

    Splits the sequence into chunks of ``chunk_size`` frames.  Within each
    chunk, uses the parallel associative scan (O(log C) depth, 6 steps for
    C=64).  Across chunks, carries state sequentially.

    For best training speed, wrap the *encoder* with ``torch.compile(encoder)``
    — the compiler fuses the scan steps and surrounding ops into efficient
    kernels, achieving parity with the CUDA mamba-ssm package.

    Memory is bounded to O(B*C*D*N) per chunk.
    """
    batch, T, D = x.shape
    N = A.shape[1]

    x = x.float()
    dt = dt.float()
    B = B.float()
    C = C.float()

    A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, D, N)

    carry = state.float() if state is not None else torch.zeros(
        batch, D, N, device=x.device, dtype=torch.float32
    )

    y_chunks = []
    for t0 in range(0, T, chunk_size):
        t1 = min(t0 + chunk_size, T)
        dt_c = dt[:, t0:t1]
        x_c = x[:, t0:t1]
        B_c = B[:, t0:t1]
        C_c = C[:, t0:t1]

        # Discretize within this chunk only
        dA_c = torch.exp(dt_c.unsqueeze(-1) * A_expanded)
        dB_x_c = (dt_c.unsqueeze(-1) * B_c.unsqueeze(2)) * x_c.unsqueeze(-1)

        # Fold carry state into first timestep
        dB_x_c = dB_x_c.clone()
        dB_x_c[:, 0] = dA_c[:, 0] * carry + dB_x_c[:, 0]

        # Parallel scan within chunk
        h_c = parallel_scan(dA_c, dB_x_c)

        # Output for this chunk
        y_c = (h_c * C_c.unsqueeze(2)).sum(-1)
        y_chunks.append(y_c)

        carry = h_c[:, -1]
        del dA_c, dB_x_c, h_c

    y = torch.cat(y_chunks, dim=1)
    return y, carry


# Keep old name as alias for backward compatibility
selective_scan_parallel = selective_scan


class MambaBlock(nn.Module):
    """Pure PyTorch Mamba SSM block.

    Uses parallel associative scan for GPU-efficient training.
    Supports carry-state via step() for inference.
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

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        """Zero-initialized carry state for this block.

        Returns a dict with:
            conv : (B, d_inner, d_conv - 1) — tail of x_inner for conv context
            ssm  : (B, d_inner, d_state)    — S6 hidden state
        """
        return {
            "conv": torch.zeros(
                batch_size, self.d_inner, self.d_conv - 1,
                device=device, dtype=torch.float32,
            ),
            "ssm": torch.zeros(
                batch_size, self.d_inner, self.d_state,
                device=device, dtype=torch.float32,
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        x: (B, T, D)
        state: optional {"conv": (B, d_inner, d_conv - 1),
                         "ssm":  (B, d_inner, d_state)} for carry
        Returns: (B, T, D), new_state
        """
        B, T, D = x.shape

        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        # ── Causal Conv1d with carry-state support ────────────────────────
        # x_inner is (B, T, d_inner); conv1d wants (B, d_inner, T).
        x_inner_bdt = x_inner.transpose(1, 2)  # (B, d_inner, T)

        conv_prefix_len = 0
        if state is not None and state.get("conv") is not None:
            # Prepend the last (d_conv - 1) frames from the previous chunk
            # so the conv sees real history instead of zero padding.
            conv_prefix = state["conv"].to(x_inner_bdt.dtype)  # (B, d_inner, d_conv-1)
            x_inner_bdt = torch.cat([conv_prefix, x_inner_bdt], dim=-1)
            conv_prefix_len = conv_prefix.shape[-1]

        # Apply conv. With kernel=d_conv and padding=d_conv-1, a non-carry
        # call produces T+d_conv-1 outputs; slicing [:T] is the causal trim.
        # When carrying, we prepended d_conv-1 real frames and want exactly
        # the T outputs that correspond to the new chunk.
        conv_out_all = self.conv1d(x_inner_bdt)
        if conv_prefix_len > 0:
            # Skip the outputs that depend on prepended frames only.
            # With padding=d_conv-1 and kernel=d_conv, output length is
            # x_inner_bdt.shape[-1] + d_conv - 1 = (T + d_conv - 1) + d_conv - 1.
            # The outputs corresponding to the new chunk's frames are
            # indices [conv_prefix_len : conv_prefix_len + T].
            conv_out = conv_out_all[..., conv_prefix_len : conv_prefix_len + T]
        else:
            conv_out = conv_out_all[..., :T]  # causal trim, as before

        x_conv = self.act(conv_out).transpose(1, 2)  # (B, T, d_inner)

        # New conv state = last (d_conv - 1) frames of x_inner (pre-conv).
        # We take from the original x_inner (B, T, d_inner), tail-padded on
        # the left if T < d_conv - 1.
        tail_len = self.d_conv - 1
        if T >= tail_len:
            new_conv_state = x_inner[:, -tail_len:, :].transpose(1, 2).contiguous()
        else:
            pad = torch.zeros(
                B, tail_len - T, self.d_inner,
                device=x_inner.device, dtype=x_inner.dtype,
            )
            new_conv_state = torch.cat([pad, x_inner], dim=1).transpose(1, 2).contiguous()

        # ── Project to dt, B, C ───────────────────────────────────────────
        x_dbl = self.x_proj(x_conv)  # (B, T, dt_rank + 2*d_state)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, T, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # ── SSM scan (fused by torch.compile when encoder is compiled) ────
        ssm_state = state.get("ssm") if state is not None else None
        y, new_ssm_state = selective_scan(
            x_conv, dt, A, B_ssm, C_ssm, ssm_state
        )

        # ── Gate and output ───────────────────────────────────────────────
        y = y * self.act(z)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv  # skip connection
        out = self.out_proj(y)

        return out, {"conv": new_conv_state, "ssm": new_ssm_state}

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
