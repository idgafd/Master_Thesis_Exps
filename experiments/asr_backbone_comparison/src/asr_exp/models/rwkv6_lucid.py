"""RWKV-6 + LUCID preconditioner encoder backbone.

Takes the stock RWKV-6 architecture and adds LUCID's key decorrelation
preconditioner inside the chunked WKV computation.

Standard RWKV-6 intra-chunk:  Y = tril(Q @ K^T * decay_mask) @ V
LUCID modification:           Y = tril(Q @ K^T * decay_mask) @ P @ V

where P = (M * exp(K @ K^T))^{-1} decorrelates correlated keys within
each chunk, reducing noise accumulation in the state.

Inter-chunk recurrent state passing is unchanged: S = decay * S + K^T @ V.

This is a hybrid architecture: RWKV-6 state dynamics with LUCID readout.
Pure step-by-step recurrent inference is replaced by chunked inference
(chunk_size=64 by default).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RWKV6LucidTimeMix(nn.Module):
    """RWKV-6 TimeMix with LUCID preconditioner in chunked parallel form.

    Keeps RWKV-6's data-dependent token shift (ddlerp via shared LoRA),
    data-dependent decay, SiLU gate, and GroupNorm output.
    Adds LUCID preconditioner for key decorrelation within chunks.
    """

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int,
                 dtype: torch.dtype = torch.float32,
                 chunk_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8
        self.chunk_size = chunk_size

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

            D_MIX_DIM = 32
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_MIX_DIM * 5, dtype=dtype)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, D_MIX_DIM, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
            )

            # ── RWKV-6 decay ──
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

            # ── Bonus term ──
            tmp = torch.zeros(hidden_size_att, dtype=dtype)
            for n in range(hidden_size_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (hidden_size_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(n_head, head_size))

        # ── Learnable LUCID temperature ──
        # Controls how strongly the preconditioner acts
        self.lucid_temperature = nn.Parameter(torch.ones(1, n_head, 1, 1, dtype=dtype))

        # ── Linear projections ──
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

        # ── Token shift (RWKV-6 ddlerp) ──
        shift_state_out = x[:, -1]
        dxprev = torch.cat((shift_state_in.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, C)

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

        w = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(r.dtype)
        u = self.time_faaaa

        # ── Chunked WKV with LUCID preconditioner ──
        y, wkv_state_out = _rwkv6_lucid_wkv(
            B, T, C, H, K, r, k, v, w, u, wkv_state_in,
            self.chunk_size, self.lucid_temperature,
        )

        y = y.view(B * T, C)
        y = self.ln_x(y).view(B, T, C)
        y = self.output(y * g)

        return y, shift_state_out, wkv_state_out


def _rwkv6_lucid_wkv(
    B: int, T: int, C: int, H: int, K: int,
    r: Tensor, k: Tensor, v: Tensor, w: Tensor, u: Tensor,
    wkv_state: Tensor,
    chunk_size: int,
    lucid_temp: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Chunked RWKV-6 WKV with LUCID preconditioner.

    Within each chunk:
      A_intra = tril(r_decay @ k_inv_decay^T)  — standard RWKV-6 intra-chunk attention
      K_gram = k_norm @ k_norm^T                — key correlation matrix
      P = (M * exp(temp * K_gram))^{-1}         — LUCID preconditioner
      Y_intra = A_intra @ P @ V                 — decorrelated output

    Between chunks: standard recurrent state passing (unchanged).
    """
    # Reshape to (B, H, T, K)
    r = r.view(B, T, H, K).transpose(1, 2)
    k = k.view(B, T, H, K).transpose(1, 2)
    v = v.view(B, T, H, K).transpose(1, 2)
    w = -torch.exp(w.view(B, T, H, K).transpose(1, 2))  # log-decay
    u = u.view(1, H, 1, K).to(r.dtype)

    state = wkv_state.float()
    out_parts = []

    # Process in chunks
    pos = 0
    while pos < T:
        cs = min(chunk_size, T - pos)
        r_c = r[:, :, pos:pos+cs, :]
        k_c = k[:, :, pos:pos+cs, :]
        v_c = v[:, :, pos:pos+cs, :]
        w_c = w[:, :, pos:pos+cs, :]

        # Compute cumulative decay within chunk (log-space)
        w_log = w_c.float()
        w_cum = torch.cumsum(w_log, dim=2)
        w_cum_shifted = F.pad(w_cum[:, :, :-1, :], (0, 0, 1, 0))

        # Numerical stabilization
        mid = cs // 2
        offset = w_cum_shifted[:, :, mid:mid+1, :]
        r_decay = torch.exp((w_cum_shifted - offset).clamp(-30, 30)).to(r.dtype)
        k_inv_decay = torch.exp((offset - w_cum).clamp(-30, 30)).to(r.dtype)

        # Intra-chunk attention: A = tril(r_decay @ k_inv_decay^T)
        A_intra = (r_c * r_decay) @ (k_c * k_inv_decay).transpose(-2, -1)
        A_intra = A_intra.tril(-1)

        # Add bonus term on diagonal
        A_intra = A_intra + torch.einsum('bhts,bhts->bht', r_c, u * k_c).diag_embed()

        # ── LUCID preconditioner ──
        # L2-normalize keys per head for stable gram matrix
        k_for_gram = F.normalize(k_c, dim=-1, p=2.0)  # (B, H, cs, K)
        K_gram = k_for_gram @ k_for_gram.transpose(-2, -1)  # (B, H, cs, cs) in [-1, 1]

        # Temperature-scaled gram matrix (values stay bounded)
        temp = F.softplus(lucid_temp)  # ensure positive, starts ~1.3
        scaled_gram = (temp * K_gram).clamp(-20, 20)

        # Preconditioner: P = (I + temp * exp(K_gram))^{-1}
        # Using identity-based form for stability instead of M-based
        precond_matrix = torch.eye(
            cs, device=k_c.device, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0) + torch.exp(scaled_gram.float())

        # Solve P @ X = V for preconditioned values
        v_c_f = v_c.float()
        P_v = torch.linalg.solve(precond_matrix, v_c_f)  # (B, H, cs, K)

        # Apply attention with preconditioned values
        y_intra = (A_intra.float() @ P_v).to(r.dtype)

        # Inter-chunk: apply state from previous chunks
        w_intra = torch.exp((w_cum - w_log).clamp(-30, 30)).to(r.dtype)
        y_state = (r_c * w_intra) @ state.to(r.dtype)  # (B, H, cs, K)

        y_chunk = y_intra + y_state
        out_parts.append(y_chunk)

        # Update state for next chunk
        ws = torch.exp(w_log.sum(dim=2, keepdim=True)).float()  # total chunk decay
        w_inter = torch.exp((w_log.sum(dim=2, keepdim=True) - w_cum).clamp(-30, 30)).to(r.dtype)
        state_update = (k_c * w_inter).transpose(-2, -1) @ v_c  # (B, H, K, K)
        state = state * ws.squeeze(2).unsqueeze(-2) + state_update.float()

        pos += cs

    out = torch.cat(out_parts, dim=2)  # (B, H, T, K)
    out = out.transpose(1, 2).reshape(B, T, C)

    return out, state.to(dtype=wkv_state.dtype)


class RWKV6LucidChannelMix(nn.Module):
    """RWKV-6 Channel Mix — identical to stock RWKV-6."""

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


class RWKV6LucidBlock(nn.Module):
    """Single RWKV-6 + LUCID layer block."""

    def __init__(self, hidden_size: int, n_head: int, head_size: int,
                 num_hidden_layers: int, layer_id: int, dropout: float,
                 dtype: torch.dtype = torch.float32, chunk_size: int = 64):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
        else:
            self.ln0 = nn.Identity()

        self.att = RWKV6LucidTimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype,
            chunk_size=chunk_size,
        )
        self.ffn = RWKV6LucidChannelMix(
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


class RWKV6LucidEncoder(nn.Module):
    """RWKV-6 encoder with LUCID preconditioner.

    Hybrid architecture: RWKV-6 state dynamics (inter-chunk recurrent)
    with LUCID key decorrelation (intra-chunk parallel).
    """

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        dtype: torch.dtype = torch.float32,
        chunk_size: int = 64,
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
                RWKV6LucidBlock(
                    d_model, self.n_head, head_size, n_layers, i, dropout, dtype,
                    chunk_size=chunk_size,
                )
            )

        print(
            f"RWKV-6+LUCID encoder initialized "
            f"({n_layers} layers, d_model={d_model}, head_size={head_size}, "
            f"chunk_size={chunk_size})"
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
