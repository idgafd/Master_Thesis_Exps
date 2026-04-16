"""Unified RWKV-6 TimeMix block — supports recurrent, LION, and bidir_serial modes.

Self-contained implementation (no external rwkv-block dependency).
All mechanism flags (conv_shift, headscale, delta_rule, lucid, temperature) are
compositional and can be combined freely.

Reference: RWKV-6 (Peng et al. 2024), LION (Afzal et al. 2025)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lion_attention import (
    lion_parallel_attention,
    lion_attention_with_delta,
    lion_attention_with_lucid,
)


def _bidirectional_token_shift(x: torch.Tensor) -> torch.Tensor:
    """Bidirectional shift: (x[t-1] + x[t+1]) / 2, zero-padded."""
    left = F.pad(x[:, :-1, :], (0, 0, 1, 0))
    right = F.pad(x[:, 1:, :], (0, 0, 0, 1))
    return (left + right) * 0.5


def _causal_token_shift(x: torch.Tensor) -> torch.Tensor:
    """Causal shift: x[t-1], zero at t=0."""
    return F.pad(x[:, :-1, :], (0, 0, 1, 0))


class RWKV6TimeMix(nn.Module):
    """Unified RWKV-6 time-mixing block.

    Modes:
        "recurrent"    — causal chunked WKV, carry-state capable
        "lion"         — LION parallel full T*T attention, bidirectional
        "bidir_serial" — forward + backward recurrent, merged

    Mechanism flags (all composable):
        conv_shift:  learned DWConv1d replacing fixed token shift
        headscale:   per-head learnable decay bias
        delta_rule:  selective state erasure (causal-only on LION)
        lucid:       LUCID preconditioner on attention output
        temperature: per-head learnable attention temperature
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        mode: str = "lion",
        conv_shift: bool = False,
        headscale: bool = False,
        delta_rule: bool = False,
        lucid: bool = False,
        lucid_chunk_size: Optional[int] = None,
        temperature: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8
        self.mode = mode
        self.use_conv_shift = conv_shift
        self.use_headscale = headscale
        self.use_delta_rule = delta_rule
        self.use_lucid = lucid
        self.lucid_chunk_size = lucid_chunk_size
        self.use_temperature = temperature
        self.layer_id = layer_id

        hidden_size_att = hidden_size

        # ── Token shift parameters ───────────────────────────────────────
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

            # ── Decay parameters ─────────────────────────────────────────
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

            # Bonus term (u in RWKV notation, time_faaaa in code)
            tmp = torch.zeros(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (hidden_size_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(n_head, head_size))

        # ── Linear projections ───────────────────────────────────────────
        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

        # ── Mechanism: ConvShift ─────────────────────────────────────────
        if self.use_conv_shift:
            from src.models.mechanisms.conv_shift import DWConvShift
            self.conv_shift_module = DWConvShift(hidden_size)

        # ── Mechanism: Headscale ─────────────────────────────────────────
        if self.use_headscale:
            self.head_decay_bias = nn.Parameter(torch.zeros(1, 1, n_head, head_size))

        # ── Mechanism: Delta Rule ────────────────────────────────────────
        if self.use_delta_rule:
            from src.models.mechanisms.delta_rule import DeltaRuleParams
            self.delta_params = DeltaRuleParams(hidden_size, n_head, head_size, dtype=dtype)

        # ── Mechanism: LUCID ─────────────────────────────────────────────
        if self.use_lucid:
            # Init so softplus(param) = 1.0, matching the paper's unit scaling
            # softplus(x) = 1.0 → x = ln(e - 1) ≈ 0.5413
            import math
            self.lucid_temperature = nn.Parameter(
                torch.full((n_head,), math.log(math.e - 1))
            )

        # ── Mechanism: Temperature ───────────────────────────────────────
        if self.use_temperature:
            self.attention_temperature = nn.Parameter(torch.ones(1, n_head, 1, 1))

    def _token_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute dxprev based on mode and conv_shift setting."""
        if self.use_conv_shift:
            return self.conv_shift_module(x) - x
        if self.mode in ("lion", "bidir_serial"):
            return _bidirectional_token_shift(x) - x
        return _causal_token_shift(x) - x

    def _compute_rkv_gw(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared computation: token shift -> r, k, v, g, w."""
        B, T, D = x.size()

        dxprev = self._token_shift(x)

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

        return r, k, v, g, w

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        r, k, v, g, w = self._compute_rkv_gw(x)

        # Reshape to (B, H, T, K)
        r_h = r.view(B, T, H, K).transpose(1, 2)
        k_h = k.view(B, T, H, K).transpose(1, 2)
        v_h = v.view(B, T, H, K).transpose(1, 2)
        w_h = w.view(B, T, H, K).transpose(1, 2)

        # Log-decay: actual decay = exp(-exp(w_raw))
        w_h = -torch.exp(w_h)

        # Apply headscale
        if self.use_headscale:
            w_h = w_h + self.head_decay_bias.to(w_h.dtype)

        # Apply temperature
        if self.use_temperature:
            r_h = r_h * self.attention_temperature

        # Dispatch to mode-specific attention
        new_state = None
        if self.mode == "lion":
            y = self._forward_lion(r_h, k_h, v_h, w_h)
        elif self.mode == "recurrent":
            y, new_state = self._forward_recurrent(r_h, k_h, v_h, w_h, state)
        elif self.mode == "bidir_serial":
            y = self._forward_bidir_serial(r_h, k_h, v_h, w_h)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        y = y.to(r.dtype)

        # Reshape back and apply GroupNorm + gate + output
        y = y.transpose(1, 2).reshape(B * T, D)
        y = self.ln_x(y).view(B, T, D)
        y = self.output(y * g)

        return y, new_state

    def _forward_lion(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """LION parallel full T*T attention."""
        if self.use_delta_rule:
            kk, iclr = self.delta_params.compute_kk_iclr(
                k, k.shape[0], k.shape[2], self.n_head, self.head_size
            )
            return lion_attention_with_delta(r, k, v, w, kk, iclr)

        if self.use_lucid:
            temp = F.softplus(self.lucid_temperature)
            return lion_attention_with_lucid(
                r, k, v, w, temp, chunk_size=self.lucid_chunk_size
            )

        return lion_parallel_attention(r, k, v, w)

    def _forward_recurrent(
        self,
        r: torch.Tensor,  # (B, H, T, K)
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,  # already log-decay (negative)
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chunked parallel WKV with carry state (SmerkyG algorithm)."""
        B, H, T, K = r.shape

        if state is None:
            wkv_state = torch.zeros(B, H, K, K, dtype=torch.float32, device=r.device)
        else:
            wkv_state = state.float()

        u = self.time_faaaa.view(1, H, 1, K).to(r.dtype)

        if self.use_lucid:
            # LUCID in recurrent mode: precondition values before WKV
            temp = F.softplus(self.lucid_temperature)
            v_precond = _apply_lucid_recurrent(k, v, temp, self.lucid_chunk_size or 64)
            y, wkv_state = _chunked_wkv(r, k, v_precond, w, u, wkv_state)
        else:
            y, wkv_state = _chunked_wkv(r, k, v, w, u, wkv_state)
        return y, wkv_state

    def _forward_bidir_serial(
        self,
        r: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        w: torch.Tensor,
    ) -> torch.Tensor:
        """Bidirectional serial: forward recurrent + backward recurrent, merged."""
        B, H, T, K = r.shape

        # Forward pass
        y_fwd, _ = self._forward_recurrent(r, k, v, w, state=None)

        # Backward pass (flip time, run forward, flip back)
        r_b = r.flip(2)
        k_b = k.flip(2)
        v_b = v.flip(2)
        w_b = w.flip(2)
        y_bwd, _ = self._forward_recurrent(r_b, k_b, v_b, w_b, state=None)
        y_bwd = y_bwd.flip(2)

        return y_fwd + y_bwd


def _apply_lucid_recurrent(
    k: torch.Tensor,   # (B, H, T, K)
    v: torch.Tensor,   # (B, H, T, K)
    temp: torch.Tensor, # (H,) — per-head temperature (already softplus'd)
    chunk_size: int = 64,
) -> torch.Tensor:
    """Paper-faithful LUCID preconditioner for recurrent RWKV-6.

    Applies the LUCID preconditioner within fixed-size chunks:
        P = exp(K_RN @ K_RN^T / sqrt(d) - sqrt(d))  (unit diagonal)
        Solve P · Y = V for Y

    This preconditions the values BEFORE they enter the WKV state,
    matching the paper's formulation: O = A · P^{-1} · V.

    The inter-chunk recurrent state dynamics are unchanged.
    """
    B, H, T, K = k.shape
    sqrt_d = K ** 0.5

    # Reshape temp for broadcasting: (H,) -> (1, H, 1, 1)
    if temp.dim() == 1:
        temp = temp.view(1, H, 1, 1)

    v_out = torch.zeros_like(v)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        k_c = k[:, :, start:end, :]  # (B, H, cs, K)
        v_c = v[:, :, start:end, :]

        # RMSNorm: k_rn = sqrt(d) * k / ||k||_2
        k_rn = sqrt_d * F.normalize(k_c, dim=-1)

        # Gram matrix: diagonal = d
        gram = k_rn @ k_rn.transpose(-2, -1)  # (B, H, cs, cs)

        # Paper: exp(gram / sqrt(d) - sqrt(d)) — unit diagonal
        scaled = (temp * (gram / sqrt_d - sqrt_d)).clamp(-30, 30)
        P = torch.exp(scaled.float())  # (B, H, cs, cs)

        # Regularize for numerical stability (ensures P is always invertible)
        P = P + 1e-6 * torch.eye(end - start, device=P.device, dtype=P.dtype)

        # Solve P · Y = V for preconditioned values
        v_out[:, :, start:end, :] = torch.linalg.solve(P, v_c.float()).to(v.dtype)

    return v_out


def _chunked_wkv(
    r: torch.Tensor,   # (B, H, T, K)
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,   # log-decay (negative)
    u: torch.Tensor,   # (1, H, 1, K) bonus term
    wkv_state: torch.Tensor,  # (B, H, K, K)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked parallel WKV with carry-state (SmerkyG algorithm).

    Processes T tokens using decreasing chunk sizes for GPU-parallel
    intra-chunk attention with sequential inter-chunk state updates.
    """
    B, H, T, K = r.shape

    processed = 0
    remaining = T
    out_parts = []
    state = wkv_state

    for chunk_len in [128, 16, 2, 1]:
        while remaining >= chunk_len:
            mul = remaining // chunk_len
            seg_len = chunk_len * mul

            out, state = _wkv_subchunk(
                r[:, :, processed:processed + seg_len],
                k[:, :, processed:processed + seg_len],
                v[:, :, processed:processed + seg_len],
                w[:, :, processed:processed + seg_len],
                u, state,
                chunk_len=chunk_len,
            )
            out_parts.append(out)
            processed += seg_len
            remaining -= seg_len

    return torch.cat(out_parts, dim=2), state


def _wkv_subchunk(
    r: torch.Tensor,   # (B, H, L, K) — L must be exact multiple of chunk_len
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,   # log-decay (negative)
    u: torch.Tensor,   # (1, H, 1, K)
    wkv_state: torch.Tensor,  # (B, H, K, K)
    chunk_len: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel intra-chunk + recurrent inter-chunk WKV.

    Direct port of RWKVx060_subchunk_torch_inner from RWKV-block.
    """
    B, H, L, K = k.shape
    V = v.size(-1)
    T = chunk_len

    # Single token: simple step
    if L == 1:
        kv = k.mT @ v
        out = r @ (wkv_state + u.mT * kv)
        wkv_state = torch.exp(w).mT * wkv_state + kv
        return out, wkv_state

    assert L % T == 0
    N = L // T

    # Numerical stability: clamp decay factor
    precision_min_val = 0.005
    precision_dtype = torch.float64 if T > 24 else torch.float32
    # w is log-decay (negative): decay = exp(w) in (0, 1)
    w_decay = torch.exp(w).clamp(min=precision_min_val)

    # Cumulative log-decay
    w_log = w_decay.float().log()
    wc_log = w_log.view(w.size(0), H, N, T, K)
    wc_log_cum = wc_log.cumsum(dim=-2)
    shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))

    # Pre-compute decay weights
    ws = wc_log.sum(dim=-2, keepdim=True)
    w_inter = ws - wc_log_cum
    w_intra = wc_log_cum - wc_log

    ws_list = list(ws.mT.exp().to(r.dtype).unbind(dim=-3))
    w_inter = w_inter.exp().to(r.dtype)
    w_intra = w_intra.exp().to(r.dtype)

    # Reshape to chunks
    r = r.view(B, H, N, T, K)
    k = k.view(B, H, N, T, K)
    v = v.view(B, H, N, T, V)
    u_c = u.unsqueeze(2).to(r.dtype)

    # Parallel intra-chunk attention
    wc_log_offset = shifted_wc_log_cum[..., T // 2:T // 2 + 1, :]
    r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp()
    k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp()
    a = ((r * r_decay) @ (k * k_inv_decay).mT).to(r.dtype).tril(-1)
    # Add bonus term on diagonal
    a = a + torch.einsum('bhntk,bhntk->bhnt', r, u_c * k).diag_embed()
    out = a @ v

    # Pre-compute chunked kv products for state update
    wkv = (k * w_inter).mT @ v
    wkv_list = list(wkv.unbind(dim=-3))

    # Recurrent inter-chunk state
    states = []
    for i in range(N):
        states.append(wkv_state)
        wkv_state = wkv_state * ws_list[i] + wkv_list[i]
    states = torch.stack(states, dim=2)

    # Apply state to output
    out = out + (r * w_intra) @ states
    out = out.view(B, H, L, V)

    return out, wkv_state
