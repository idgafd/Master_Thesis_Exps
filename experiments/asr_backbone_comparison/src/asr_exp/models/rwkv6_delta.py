"""RWKV-6 + Delta Rule encoder backbone.

Takes the stock RWKV-6 architecture (gated FFN, data-dependent token shift,
data-dependent decay) and adds RWKV-7's delta rule to the state update:

  Standard RWKV-6:  S = diag(w) * S + v^T * k
  Delta rule:       S = diag(w) * S + S @ ab + v^T * k_tilde

where ab = (-kk)^T @ (kk * iclr)^T is the selective erasure matrix.

New learnable parameters per layer (from RWKV-7):
  - a0, a1, a2: LoRA for in-context learning rate (ICLR)
  - k_k: key normalization scaling
  - k_a: key scaling factor for ICLR modulation

The FFN gate, data-dependent token shift (ddlerp via LoRA), and wide decay
range are all preserved from RWKV-6 — only the state update equation changes.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RWKV6DeltaTimeMix(nn.Module):
    """RWKV-6 TimeMix with delta rule state update.

    Keeps RWKV-6's data-dependent token shift (ddlerp via shared LoRA),
    data-dependent decay, SiLU gate, and GroupNorm output.
    Adds delta rule selective erasure from RWKV-7.
    """

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8

        hidden_size_att = hidden_size

        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(num_hidden_layers - 1, 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)
            ddd = torch.ones(1, 1, hidden_size, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            # ── RWKV-6 token shift (ddlerp via LoRA) ──
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            # Extra shift for the new 'a' input (ICLR)
            self.time_maa_a = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

            D_MIX_DIM = 32
            # 6 components: w, k, v, r, g, a
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_MIX_DIM * 6, dtype=dtype)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(6, D_MIX_DIM, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
            )

            # ── RWKV-6 decay (wide range, data-dependent via LoRA) ──
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

            # ── RWKV-6 bonus term ──
            tmp = torch.zeros(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (hidden_size_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(n_head, head_size))

            # ── Delta rule parameters (from RWKV-7) ──
            D_AAA_LORA = max(1, round(hidden_size_att ** 0.5 * 1.8 / 32)) * 32

            # ICLR: in-context learning rate
            self.a0 = nn.Parameter(torch.zeros(1, 1, hidden_size_att, dtype=dtype))
            self.a1 = nn.Parameter(torch.zeros(hidden_size_att, D_AAA_LORA, dtype=dtype))

            def ortho_init(x, scale):
                gain = math.sqrt(x.shape[0] / x.shape[1]) if x.shape[0] > x.shape[1] else 1
                nn.init.orthogonal_(x, gain=gain * scale)
                return x.to(dtype=dtype)

            self.a2 = nn.Parameter(
                ortho_init(torch.zeros(D_AAA_LORA, hidden_size_att), 0.1)
            )

            # Key normalization and scaling
            self.k_k = nn.Parameter(torch.ones(1, 1, hidden_size_att, dtype=dtype) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, hidden_size_att, dtype=dtype))

        # ── Linear projections (same as RWKV-6) ──
        self.receptance = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.key = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.gate = nn.Linear(hidden_size, hidden_size_att, bias=False, dtype=dtype)
        self.output = nn.Linear(hidden_size_att, hidden_size, bias=False, dtype=dtype)
        self.ln_x = nn.GroupNorm(
            n_head, hidden_size_att, dtype=dtype,
            eps=(1e-5) * (self.head_size_divisor ** 2),
        )

    def forward(
        self, x: Tensor, shift_state_in: Tensor, wkv_state_in: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, T, C = x.size()
        H = self.n_head
        K = self.head_size

        # ── Token shift (RWKV-6 ddlerp via shared LoRA) ──
        shift_state_out = x[:, -1]
        dxprev = torch.cat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 6, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(6, B, T, C)

        mw, mk, mv, mr, mg, ma = xxx.unbind(dim=0)
        xw = x + dxprev * (self.time_maa_w + mw)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xr = x + dxprev * (self.time_maa_r + mr)
        xg = x + dxprev * (self.time_maa_g + mg)
        xa = x + dxprev * (self.time_maa_a + ma)

        # ── Projections ──
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))  # RWKV-6 SiLU gate (preserved!)

        # ── Decay (RWKV-6 style, wide range) ──
        w = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(r.dtype)

        # ── Delta rule: ICLR and key modulation ──
        iclr = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # Normalized key for erasure (L2-normalized per head)
        kk = F.normalize(
            (k * self.k_k).view(B, T, H, -1), dim=-1, p=2.0
        ).view(B, T, C)

        # Scale key by ICLR (selective write strength)
        k = k * (1 + (iclr - 1) * self.k_a)

        # ── WKV with delta rule ──
        u = self.time_faaaa  # (H, K) bonus term

        y, wkv_state_out = _rwkv6_delta_wkv(
            B, T, C, H, K, r, k, v, w, u, kk, iclr, wkv_state_in
        )

        # ── Output (RWKV-6 style: GroupNorm + gate) ──
        y = y.view(B * T, C)
        y = self.ln_x(y).view(B, T, C)
        y = self.output(y * g)

        return y, shift_state_out, wkv_state_out


def _rwkv6_delta_wkv(
    B: int, T: int, C: int, H: int, K: int,
    r: Tensor, k: Tensor, v: Tensor, w: Tensor, u: Tensor,
    kk: Tensor, iclr: Tensor,
    wkv_state: Tensor,
) -> Tuple[Tensor, Tensor]:
    """RWKV-6 WKV kernel with delta rule selective erasure — chunked form.

    Uses the same chunked parallel approach as stock RWKV-6:
    - Intra-chunk: parallel attention matrix + delta rule correction
    - Inter-chunk: recurrent state passing with delta rule

    The delta rule correction within each chunk is computed as a first-order
    modification to the standard RWKV-6 intra-chunk attention matrix.
    State updates between chunks include the full delta rule.
    """
    # Reshape to (B, H, T, K)
    r = r.view(B, T, H, K).transpose(1, 2)
    k = k.view(B, T, H, K).transpose(1, 2)
    v = v.view(B, T, H, K).transpose(1, 2)
    w = -torch.exp(w.view(B, T, H, K).transpose(1, 2))  # log-decay (negative)
    kk = kk.view(B, T, H, K).transpose(1, 2)
    iclr = iclr.view(B, T, H, K).transpose(1, 2)
    u = u.view(1, H, 1, K).to(r.dtype)

    state = wkv_state.float()
    out_parts = []

    # Process in chunks (same chunk sizes as stock RWKV-6 kernel)
    chunk_size = 128
    pos = 0
    while pos < T:
        cs = min(chunk_size, T - pos)
        r_c = r[:, :, pos:pos+cs, :]
        k_c = k[:, :, pos:pos+cs, :]
        v_c = v[:, :, pos:pos+cs, :]
        w_c = w[:, :, pos:pos+cs, :]
        kk_c = kk[:, :, pos:pos+cs, :]
        iclr_c = iclr[:, :, pos:pos+cs, :]

        # ── Intra-chunk parallel attention (same as stock RWKV-6) ──
        w_log = w_c.float()
        w_cum = torch.cumsum(w_log, dim=2)
        w_cum_shifted = F.pad(w_cum[:, :, :-1, :], (0, 0, 1, 0))

        # Stabilization
        mid = cs // 2
        offset = w_cum_shifted[:, :, mid:mid+1, :]
        r_decay = torch.exp((w_cum_shifted - offset).clamp(-30, 30)).to(r.dtype)
        k_inv_decay = torch.exp((offset - w_cum).clamp(-30, 30)).to(r.dtype)

        # Standard intra-chunk attention
        A_intra = (r_c * r_decay) @ (k_c * k_inv_decay).transpose(-2, -1)
        A_intra = A_intra.tril(-1)
        # Bonus on diagonal
        A_intra = A_intra + torch.einsum('bhts,bhts->bht', r_c, u * k_c).diag_embed()

        # ── Delta rule intra-chunk correction ──
        # Key correlation matrix: how much each position's erasure key
        # correlates with other positions' keys (causal direction)
        kk_corr = (kk_c * iclr_c) @ kk_c.transpose(-2, -1)  # (B, H, cs, cs)
        kk_corr_causal = torch.tril(kk_corr, diagonal=-1)

        # First-order correction: A_delta ≈ -(A_intra @ tril(kk_corr))
        A_delta = -torch.tril(A_intra @ kk_corr_causal)

        A_total = A_intra + A_delta

        # Intra-chunk output
        y_intra = A_total @ v_c  # (B, H, cs, K)

        # ── Inter-chunk: apply state from previous chunks ──
        w_intra = torch.exp((w_cum - w_log).clamp(-30, 30)).to(r.dtype)
        y_state = (r_c * w_intra) @ state.to(r.dtype)
        y_chunk = y_intra + y_state
        out_parts.append(y_chunk)

        # ── Update state for next chunk ──
        # Standard RWKV-6 inter-chunk state update (vectorized):
        #   state = state * ws + (k * w_inter)^T @ v
        # Plus delta rule correction applied as aggregate erasure.
        ws = torch.exp(w_log.sum(dim=2, keepdim=True)).float()  # total chunk decay
        w_inter = torch.exp(
            (w_log.sum(dim=2, keepdim=True) - w_cum).clamp(-30, 30)
        ).to(r.dtype)

        # Standard kv accumulation (vectorized, no loop)
        kv_chunk = (k_c * w_inter).transpose(-2, -1) @ v_c  # (B, H, K, K)

        # Delta rule aggregate erasure for the chunk:
        # Sum of erasure effects across the chunk, weighted by decay
        # ab_sum ≈ sum_t (w_inter_t * (-kk_t)^T @ (kk_t * iclr_t))
        neg_kk_weighted = (-kk_c * w_inter).transpose(-2, -1)  # (B, H, K, cs)
        kk_iclr_c = kk_c * iclr_c                               # (B, H, cs, K)
        ab_sum = neg_kk_weighted @ kk_iclr_c                    # (B, H, K, K)

        state = (
            state * ws.squeeze(2).unsqueeze(-2)  # decay
            + state @ ab_sum.float()              # aggregate erasure
            + kv_chunk.float()                    # new associations
        )

        pos += cs

    out = torch.cat(out_parts, dim=2)
    out = out.transpose(1, 2).reshape(B, T, C)

    return out, state.to(dtype=wkv_state.dtype)


class RWKV6DeltaChannelMix(nn.Module):
    """RWKV-6 Channel Mix (FFN) — unchanged from stock RWKV-6.

    Keeps the sigmoid gate: output = sigmoid(r(xr)) * value(relu²(key(xk)))
    """

    def __init__(self, hidden_size: int, num_hidden_layers: int, layer_id: int,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        hidden_size_ffn = int((hidden_size * 3.5) // 32 * 32)

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)
            ddd = torch.ones(1, 1, hidden_size, dtype=dtype)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size
            self.time_maa_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(hidden_size, hidden_size_ffn, bias=False, dtype=dtype)
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.value = nn.Linear(hidden_size_ffn, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: Tensor, shift_state_in: Tensor) -> Tuple[Tensor, Tensor]:
        shift_state_out = x[:, -1]
        dxprev = torch.cat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, shift_state_out


class RWKV6DeltaBlock(nn.Module):
    """Single RWKV-6 + Delta Rule layer block."""

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int, dropout: float,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
        else:
            self.ln0 = nn.Identity()

        self.att = RWKV6DeltaTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype
        )
        self.ffn = RWKV6DeltaChannelMix(
            hidden_size, num_hidden_layers, layer_id, dtype
        )

        if dropout > 0.0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    def forward(
        self, x: Tensor, last_state: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x = self.ln0(x)

        att_out, tmix_shift, tmix_wkv = self.att(
            self.ln1(x), last_state[0], last_state[1]
        )
        x = self.drop0(x + att_out)

        ffn_out, ffn_state = self.ffn(self.ln2(x), last_state[2])
        x = self.drop1(x + ffn_out)

        return x, (tmix_shift, tmix_wkv, ffn_state)


class RWKV6DeltaEncoder(nn.Module):
    """RWKV-6 encoder with delta rule state update.

    Architecture: stock RWKV-6 + delta rule selective erasure from RWKV-7.
    Keeps: gated FFN, data-dependent token shift, wide decay range.
    Adds: selective state erasure, in-context learning rate, decoupled keys.
    """

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % 32 == 0
        assert d_model % head_size == 0

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = d_model // head_size
        self.head_size = head_size

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                RWKV6DeltaBlock(
                    d_model, self.n_head, head_size, n_layers, i, dropout, dtype
                )
            )

        print(
            f"RWKV-6+DeltaRule encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size})"
        )

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
        state: Optional[List] = None,
    ) -> Tuple[Tensor, Optional[List]]:
        if state is not None:
            return self._forward_carry(x, state)
        return self._forward_stateless(x)

    def _forward_stateless(self, x: Tensor):
        state = self.init_state(x.shape[0], x.device)
        for i, layer in enumerate(self.layers):
            x, state[i] = layer(x, state[i])
        return x, None

    def _forward_carry(self, x: Tensor, state: List):
        new_state = []
        for i, layer in enumerate(self.layers):
            x, new_layer_state = layer(x, state[i])
            new_state.append(new_layer_state)
        return x, new_state

    def init_state(self, batch_size: int, device: torch.device) -> List:
        return [
            (
                torch.zeros(batch_size, self.d_model, device=device),
                torch.zeros(
                    batch_size, self.n_head, self.head_size, self.head_size,
                    dtype=torch.float32, device=device,
                ),
                torch.zeros(batch_size, self.d_model, device=device),
            )
            for _ in range(self.n_layers)
        ]
