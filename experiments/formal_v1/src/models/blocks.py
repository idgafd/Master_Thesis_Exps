"""
Pure base encoder blocks — Transformer, Linear Attention, RWKV-6, Mamba.

Clean, self-contained implementations based on the original papers and
reference repositories:
  - Transformer: Vaswani et al. 2017 (pre-norm variant)
  - Linear Attention: Katharopoulos et al. 2020 (ELU+1 feature map)
  - RWKV-6: Peng et al. 2024 (ref: RWKV-block/v6_finch)
  - Mamba: Gu & Dao 2023 (ref: state-spaces/mamba)

No LION, no mechanism modifications, no bidirectional extensions.
Each block follows the same interface:
  forward(x, lengths, state=None) -> (output, new_state_or_None)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Shared components
# =============================================================================

class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        x = x + self.pe[:, offset:offset + x.size(1)]
        return self.dropout(x)


class ConvSubsampling(nn.Module):
    """Two stride-2 Conv2d layers -> 4x temporal downsampling."""

    def __init__(self, n_mels: int, d_model: int, channels: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        freq_out = math.ceil(math.ceil(n_mels / 2) / 2)
        self.proj = nn.Linear(channels * freq_out, d_model)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, n_mels, T), lengths: (B,) -> (B, T', d_model), new_lengths"""
        x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x = self.proj(x)
        new_lengths = ((lengths - 1) // 2 + 1)
        new_lengths = ((new_lengths - 1) // 2 + 1)
        new_lengths = torch.clamp(new_lengths, min=1)
        return x, new_lengths


class CTCHead(nn.Module):
    """LayerNorm + Linear -> vocab logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


# =============================================================================
# 1. TRANSFORMER (Pre-Norm, Vaswani et al. 2017)
# =============================================================================

class TransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder.

    Fixed from draft:
    - norm_first=True (pre-norm architecture)
    - FFN dim matched to RWKV-6: int((d_model * 3.5) // 32 * 32)
    - ln0 at layer 0 (input normalization)
    """

    supports_carry_state = False

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        self.ln0 = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, state=None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.ln0(x)
        x = self.pos_enc(x)
        B, T, _ = x.shape
        padding_mask = ~(torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x, None


# =============================================================================
# 2. LINEAR ATTENTION (Katharopoulos et al. 2020)
# =============================================================================

class LinearAttentionLayer(nn.Module):
    """Single linear attention layer with ELU+1 feature map.

    Non-causal (bidirectional) attention: O(T*D^2) instead of O(T^2*D).
    Uses the kernel trick: attn(Q,K,V) = phi(Q) @ (phi(K)^T @ V) / normalizer
    where phi(x) = elu(x) + 1.

    Reference: Katharopoulos et al. "Transformers are RNNs", ICML 2020.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _elu_feature(x: torch.Tensor) -> torch.Tensor:
        """ELU+1 feature map: ensures non-negative kernel values."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, D), mask: (B, T) True=padded positions."""
        B, T, D = x.shape
        residual = x
        x = self.norm1(x)

        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, K)
        K = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        Q = self._elu_feature(Q)
        K = self._elu_feature(K)

        # Zero out padded positions
        if mask is not None:
            m = (~mask).float().unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)
            K = K * m
            V = V * m

        # Linear attention via kernel trick:
        # out_i = Q_i @ (sum_j K_j^T V_j) / (Q_i @ sum_j K_j)
        KV = torch.einsum("bhnd,bhne->bhde", K, V)      # (B, H, K, K)
        Z = 1.0 / (torch.einsum("bhnd,bhd->bhn", Q, K.sum(dim=2)) + 1e-6)  # (B, H, T)
        out = torch.einsum("bhnd,bhde,bhn->bhne", Q, KV, Z)  # (B, H, T, K)

        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.dropout(self.o_proj(out))
        x = residual + out
        x = x + self.ffn(self.norm2(x))
        return x


class LinearAttentionEncoder(nn.Module):
    """Stack of linear attention layers. Bidirectional, no carry-state."""

    supports_carry_state = False

    def __init__(self, d_model: int, n_layers: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        self.ln0 = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            LinearAttentionLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, state=None,
    ) -> Tuple[torch.Tensor, None]:
        x = self.ln0(x)
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x, None


# =============================================================================
# 3. RWKV-6 (Peng et al. 2024)
# =============================================================================
# Self-contained pure PyTorch implementation based on RWKV-block/v6_finch.
# Uses the chunked parallel WKV from rwkv5_optimized_ops.py (SmerkyG).
# Causal (unidirectional) — supports carry-state.

class RWKV6TimeMix(nn.Module):
    """RWKV-6 Time Mixing (causal recurrent attention).

    State update: S_t = diag(exp(w_t)) * S_{t-1} + k_t * v_t^T
    Output:       y_t = r_t^T @ S_t

    Where w is data-dependent log-decay computed via LoRA.
    Token shift is causal: x_{t-1} (standard RWKV).

    Parameters match the official RWKV-6 (v6_finch) exactly:
    - time_maa_x/w/k/v/r/g: token shift mixing ratios
    - time_maa_w1/w2: LoRA for data-dependent mixing (hidden->32*5->5*hidden)
    - time_decay + time_decay_w1/w2: base + LoRA for data-dependent decay
    - time_faaaa: bonus term (per-head, per-dim)
    - receptance/key/value/gate/output: linear projections
    - ln_x: GroupNorm on output

    Reference: RWKV-block/rwkv_block/v6_finch/block/rwkv6_time_mix.py
    WKV kernel: RWKV-block/rwkv_block/v5_eagle/block/rwkv5_optimized_ops.py
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dtype: torch.dtype = torch.float32,
    ):
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

            # Token shift mixing ratios
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(
                1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            )
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # LoRA for data-dependent mixing (5 branches: w, k, v, r, g)
            D_MIX_DIM = 32
            self.time_maa_w1 = nn.Parameter(
                torch.zeros(hidden_size, D_MIX_DIM * 5, dtype=dtype)
            )
            self.time_maa_w2 = nn.Parameter(
                torch.zeros(5, D_MIX_DIM, hidden_size, dtype=dtype).uniform_(-0.01, 0.01)
            )

            # Decay: base + data-dependent LoRA
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

        # Linear projections
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
        self, x: torch.Tensor, shift_state: torch.Tensor, wkv_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) input embeddings
            shift_state: (B, D) last token from previous chunk
            wkv_state: (B, H, K, K) WKV state matrix

        Returns:
            output: (B, T, D)
            new_shift_state: (B, D)
            new_wkv_state: (B, H, K, K)
        """
        B, T, D = x.size()
        H = self.n_head
        K = self.head_size

        # Causal token shift: prepend last token from previous chunk
        shift_state_out = x[:, -1]
        dxprev = torch.cat([shift_state.unsqueeze(1), x[:, :-1]], dim=1) - x

        # Data-dependent mixing via LoRA
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

        # Data-dependent decay
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = w.to(r.dtype)

        # Reshape to (B, H, T, K)
        u = self.time_faaaa.view(1, H, 1, K).to(r.dtype)
        r = r.view(B, T, H, K).transpose(1, 2)
        k = k.view(B, T, H, K).transpose(1, 2)
        v = v.view(B, T, H, K).transpose(1, 2)
        w = -torch.exp(w.view(B, T, H, K).transpose(1, 2))

        # WKV computation
        x_out, wkv_state_out = rwkv6_wkv_torch(r, k, v, w, u, wkv_state)

        # GroupNorm + gate + output projection
        x_out = x_out.transpose(1, 2).reshape(B * T, D)
        x_out = self.ln_x(x_out).view(B, T, D)
        x_out = self.output(x_out * g)

        return x_out, shift_state_out, wkv_state_out


class RWKV6ChannelMix(nn.Module):
    """RWKV-6 Channel Mixing (FFN): sigmoid(r) * value(relu^2(key(x))).

    FFN dim = int((hidden_size * 3.5) // 32 * 32).
    Token shift is causal.

    Reference: RWKV-block/rwkv_block/v6_finch/block/rwkv6_channel_mix.py
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dtype: torch.dtype = torch.float32,
    ):
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

    def forward(self, x: torch.Tensor, shift_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            shift_state: (B, D) last token from previous chunk

        Returns:
            output: (B, T, D)
            new_shift_state: (B, D)
        """
        shift_state_out = x[:, -1]
        dxprev = torch.cat([shift_state.unsqueeze(1), x[:, :-1]], dim=1) - x

        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv, shift_state_out


class RWKV6Block(nn.Module):
    """Single RWKV-6 layer: ln0 (layer 0) -> ln1 -> TimeMix -> drop -> ln2 -> ChannelMix -> drop.

    Residual connections around both TimeMix and ChannelMix.
    """

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dropout: float,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.layer_id = layer_id

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
        else:
            self.ln0 = nn.Identity()

        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

        self.att = RWKV6TimeMix(
            hidden_size, n_head, head_size, num_hidden_layers, layer_id, dtype
        )
        self.ffn = RWKV6ChannelMix(
            hidden_size, num_hidden_layers, layer_id, dtype
        )

        self.drop0 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        tmix_shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        cmix_shift_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (output, new_tmix_shift, new_wkv_state, new_cmix_shift)
        """
        x = self.ln0(x)
        att_out, tmix_shift_out, wkv_state_out = self.att(self.ln1(x), tmix_shift_state, wkv_state)
        x = self.drop0(x + att_out)
        ffn_out, cmix_shift_out = self.ffn(self.ln2(x), cmix_shift_state)
        x = self.drop1(x + ffn_out)
        return x, tmix_shift_out, wkv_state_out, cmix_shift_out


class RWKV6Encoder(nn.Module):
    """RWKV-6 encoder — causal, carry-state capable.

    State per layer: (tmix_shift (B,D), wkv_state (B,H,K,K), cmix_shift (B,D))
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
        assert d_model % head_size == 0
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_head = d_model // head_size
        self.head_size = head_size

        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        self.layers = nn.ModuleList([
            RWKV6Block(d_model, self.n_head, head_size, n_layers, i, dropout, dtype)
            for i in range(n_layers)
        ])

    def init_state(self, batch_size: int, device: torch.device) -> List[Tuple]:
        """Initialize zero state for all layers."""
        D = self.d_model
        H = self.n_head
        K = self.head_size
        return [
            (
                torch.zeros(batch_size, D, device=device),              # tmix_shift
                torch.zeros(batch_size, H, K, K, dtype=torch.float32, device=device),  # wkv_state
                torch.zeros(batch_size, D, device=device),              # cmix_shift
            )
            for _ in range(self.n_layers)
        ]

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        x = self.pos_enc(x)
        B, T, D = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        if state is None:
            state = self.init_state(B, x.device)

        new_state = []
        for i, layer in enumerate(self.layers):
            tmix_shift, wkv_st, cmix_shift = state[i]
            x, tmix_shift, wkv_st, cmix_shift = layer(x, tmix_shift, wkv_st, cmix_shift)
            x = x * mask_f
            new_state.append((tmix_shift, wkv_st, cmix_shift))

        return x, new_state


# ── RWKV-6 WKV kernel (pure PyTorch, SmerkyG chunked algorithm) ──────────────

def rwkv6_wkv_torch(
    r: torch.Tensor,   # (B, H, T, K)
    k: torch.Tensor,   # (B, H, T, K)
    v: torch.Tensor,   # (B, H, T, K)
    w: torch.Tensor,   # (B, H, T, K) — log-decay (negative values, already -exp(raw))
    u: torch.Tensor,   # (1, H, 1, K) — bonus term
    wkv_state: torch.Tensor,  # (B, H, K, K)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked parallel WKV computation with carry-state.

    Processes T tokens using chunk_size sub-chunks for the parallel intra-chunk
    computation, with sequential inter-chunk state updates.

    Based on RWKVx060_subchunk_torch_inner from RWKV-block.

    Returns: (output (B,H,T,K), new_wkv_state (B,H,K,K))
    """
    B, H, T, K = r.shape

    # Process in chunks of decreasing size to handle any T
    processed = 0
    remaining = T
    out_parts = []
    state = wkv_state

    for chunk_len in [128, 16, 2, 1]:
        while remaining >= chunk_len:
            mul = remaining // chunk_len
            seg_len = chunk_len * mul

            out, state = _rwkv6_wkv_subchunk(
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


def _rwkv6_wkv_subchunk(
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
        kv = k.mT @ v                                    # (B, H, K, V)
        out = r @ (wkv_state + u.mT * kv)                # (B, H, 1, V)
        wkv_state = torch.exp(w).mT * wkv_state + kv     # (B, H, K, V)
        return out, wkv_state

    assert L % T == 0
    N = L // T

    # Numerical stability: clamp decay factor for precision
    precision_min_val = 0.005
    precision_dtype = torch.float64 if T > 24 else torch.float32
    # w is log-decay (negative): decay = exp(w) ∈ (0, 1), clamped for stability
    w_decay = torch.exp(w).clamp(min=precision_min_val)

    # Cumulative log-decay
    w_log = w_decay.float().log()  # (B, H, L, K)
    wc_log = w_log.view(w.size(0), H, N, T, K)
    wc_log_cum = wc_log.cumsum(dim=-2)
    shifted_wc_log_cum = F.pad(wc_log_cum, (0, 0, 1, -1))

    # Pre-compute decay weights
    ws = wc_log.sum(dim=-2, keepdim=True)          # total decay per chunk
    w_inter = ws - wc_log_cum                      # decay from position to end of chunk
    w_intra = wc_log_cum - wc_log                  # decay from start of chunk to position

    ws_list = list(ws.mT.exp().to(r.dtype).unbind(dim=-3))  # N x (B,H,K,1)
    w_inter = w_inter.exp().to(r.dtype)
    w_intra = w_intra.exp().to(r.dtype)

    # Reshape to chunks
    r = r.view(B, H, N, T, K)
    k = k.view(B, H, N, T, K)
    v = v.view(B, H, N, T, V)
    u_c = u.unsqueeze(2).to(r.dtype)  # (1, H, 1, 1, K)

    # Parallel intra-chunk attention
    wc_log_offset = shifted_wc_log_cum[..., T // 2:T // 2 + 1, :]
    r_decay = (shifted_wc_log_cum - wc_log_offset).to(precision_dtype).exp()
    k_inv_decay = (wc_log_offset - wc_log_cum).to(precision_dtype).exp()
    a = ((r * r_decay) @ (k * k_inv_decay).mT).to(r.dtype).tril(-1)  # (B,H,N,T,T)
    # Add bonus term on diagonal
    a = a + torch.einsum('bhntk,bhntk->bhnt', r, u_c * k).diag_embed()
    out = a @ v  # (B, H, N, T, V)

    # Pre-compute chunked kv products for state update
    wkv = (k * w_inter).mT @ v  # (B, H, N, K, V)
    wkv_list = list(wkv.unbind(dim=-3))

    # Recurrent inter-chunk state
    states = []
    for i in range(N):
        states.append(wkv_state)
        wkv_state = wkv_state * ws_list[i] + wkv_list[i]
    states = torch.stack(states, dim=2)  # (B, H, N, K, V)

    # Apply state to output
    out = out + (r * w_intra) @ states  # (B, H, N, T, V)
    out = out.view(B, H, L, V)

    return out, wkv_state


# =============================================================================
# 4. MAMBA (Gu & Dao 2023)
# =============================================================================
# Pure PyTorch Selective Scan (S6). No mamba-ssm dependency.
# Causal (unidirectional), supports carry-state.

class MambaBlock(nn.Module):
    """Mamba Selective Scan (S6) block.

    Architecture:
        in_proj -> (x, z) split -> depthwise conv1d -> x_proj(dt, B, C)
        -> dt_proj -> SSM scan -> gate(silu(z)) + D*skip -> out_proj

    Parameters:
        - in_proj:  d_model -> 2*d_inner (x and z branches)
        - conv1d:   depthwise conv (d_inner channels, kernel=d_conv)
        - x_proj:   d_inner -> dt_rank + 2*d_state
        - dt_proj:  dt_rank -> d_inner (with bias, softplus init)
        - A_log:    (d_inner, d_state) S4D real init
        - D:        (d_inner,) skip connection, init ones
        - out_proj: d_inner -> d_model

    Reference: state-spaces/mamba/mamba_ssm/modules/mamba_simple.py
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
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: x branch + z gate branch
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, dtype=dtype)

        # Depthwise causal conv1d on x branch
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            dtype=dtype,
        )

        # Projects x -> (dt, B, C) for selective scan
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False, dtype=dtype
        )

        # dt projection with special bias init
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, dtype=dtype)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(
            self.d_inner, -1
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, dtype=dtype)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D). Stateless training forward."""
        B, T, D = x.shape

        xz = self.in_proj(x)                     # (B, T, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)

        # Causal conv1d
        x_conv = x_inner.transpose(1, 2)         # (B, d_inner, T)
        x_conv = self.act(self.conv1d(x_conv)[..., :T])
        x_conv = x_conv.transpose(1, 2)          # (B, T, d_inner)

        # Project to dt, B_ssm, C_ssm
        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))         # (B, T, d_inner)

        A = -torch.exp(self.A_log.float())        # (d_inner, d_state)

        # SSM scan (sequential)
        y = self._ssm_scan(x_conv, dt, A, B_ssm, C_ssm)

        # Skip + gate + output (D skip must be gated, matching reference selective_scan_fn)
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * self.act(z)
        return self.out_proj(y)

    def _ssm_scan(
        self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor,
        B: torch.Tensor, C: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan: sequential over T. Returns (B, T, d_inner)."""
        batch, T, d_inner = x.shape
        d_state = A.shape[1]

        ssm_state = torch.zeros(batch, d_inner, d_state, dtype=torch.float32, device=x.device)
        x_f, dt_f, B_f, C_f = x.float(), dt.float(), B.float(), C.float()

        outputs = []
        for t in range(T):
            dA = torch.exp(dt_f[:, t, :].unsqueeze(-1) * A.unsqueeze(0))    # (B, d_inner, d_state)
            dB = dt_f[:, t, :].unsqueeze(-1) * B_f[:, t, :].unsqueeze(1)    # (B, d_inner, d_state)
            ssm_state = ssm_state * dA + x_f[:, t, :].unsqueeze(-1) * dB
            y_t = (ssm_state * C_f[:, t, :].unsqueeze(1)).sum(-1)           # (B, d_inner)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1).to(x.dtype)

    def step(
        self, x: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-token step for carry-state inference.

        Args:
            x: (B, D) single token
            conv_state: (B, d_inner, d_conv)
            ssm_state: (B, d_inner, d_state)
        Returns: (output (B, D), new_conv_state, new_ssm_state)
        """
        dtype = x.dtype
        xz = self.in_proj(x)                      # (B, 2*d_inner)
        x_inner, z = xz.chunk(2, dim=-1)

        # Conv step
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_inner
        x_conv = (conv_state * self.conv1d.weight.squeeze(1)).sum(dim=-1)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        x_conv = self.act(x_conv).to(dtype)

        # Project
        x_dbl = self.x_proj(x_conv)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())

        # Discretize and update state
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(1).float()
        ssm_state = ssm_state * dA + x_conv.unsqueeze(-1).float() * dB

        # Output
        y = (ssm_state.to(dtype) * C_ssm.unsqueeze(1)).sum(-1)
        y = y + self.D * x_conv
        y = y * self.act(z)

        return self.out_proj(y), conv_state, ssm_state


class MambaEncoder(nn.Module):
    """Mamba SSM encoder — causal, carry-state capable.

    Each layer: ln0 (layer 0) -> ln1 -> MambaBlock -> drop -> ln2 -> FFN -> drop
    FFN matched to RWKV-6: int((d_model * 3.5) // 32 * 32)
    """

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        ffn_dim: int = 896,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        self.pos_enc = SinusoidalPE(d_model, dropout=dropout)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                "ln0": nn.LayerNorm(d_model, dtype=dtype) if i == 0 else nn.Identity(),
                "ln1": nn.LayerNorm(d_model, dtype=dtype),
                "mamba": MambaBlock(d_model, d_state, d_conv, expand, dtype=dtype),
                "ln2": nn.LayerNorm(d_model, dtype=dtype),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, ffn_dim, dtype=dtype),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, d_model, dtype=dtype),
                    nn.Dropout(dropout),
                ),
                "drop": nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            })
            self.layers.append(layer)

    def init_state(self, batch_size: int, device: torch.device) -> List[Tuple]:
        """Initialize zero state for carry-state inference."""
        return [
            (
                torch.zeros(batch_size, self.d_inner, self.d_conv, device=device),   # conv_state
                torch.zeros(batch_size, self.d_inner, self.d_state, dtype=torch.float32, device=device),  # ssm_state
            )
            for _ in range(self.n_layers)
        ]

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor, state: Optional[List] = None,
    ) -> Tuple[torch.Tensor, Optional[List]]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        if state is not None:
            return self._forward_carry(x, mask_f, state)

        # Stateless training forward
        for layer in self.layers:
            x = layer["ln0"](x)
            residual = x
            x = residual + layer["drop"](layer["mamba"](layer["ln1"](x)))
            x = x + layer["ffn"](layer["ln2"](x))
            x = x * mask_f

        return x, None

    def _forward_carry(
        self, x: torch.Tensor, mask_f: torch.Tensor, state: List,
    ) -> Tuple[torch.Tensor, List]:
        """Token-by-token carry-state forward."""
        B, T, D = x.shape
        new_state = []

        for layer_idx, layer in enumerate(self.layers):
            conv_st, ssm_st = state[layer_idx]
            x = layer["ln0"](x)
            residual = x
            x_norm = layer["ln1"](x)

            outputs = []
            for t in range(T):
                out_t, conv_st, ssm_st = layer["mamba"].step(
                    x_norm[:, t, :], conv_st, ssm_st
                )
                outputs.append(out_t)

            x = residual + layer["drop"](torch.stack(outputs, dim=1))
            x = x + layer["ffn"](layer["ln2"](x))
            x = x * mask_f
            new_state.append((conv_st, ssm_st))

        return x, new_state
