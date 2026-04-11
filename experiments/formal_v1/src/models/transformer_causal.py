"""Causal (autoregressive) Transformer encoder with KV cache for streaming.

Mirrors `TransformerEncoder` (pre-norm, matched FFN dim, ln0, sinusoidal PE)
but applies a causal attention mask so each frame can only attend to past
frames. Also supports a streaming KV cache: on each forward pass with a
non-None `state`, new keys and values are appended to the per-layer caches
and attention is computed over the full history.

Why we have both causal and bidirectional Transformers:
  * The bidirectional variant is the Group B baseline for offline ASR.
  * This causal variant is the Group A baseline and the honest data point
    for the streaming-memory-vs-audio-duration demonstration. Its state
    grows linearly with audio length, unlike Mamba/RWKV-6 which are
    constant. See `scripts/measure_streaming_memory.py`.

Design notes:
  * Pure PyTorch multi-head attention (no `nn.MultiheadAttention`) so we
    can intercept K and V for the cache.
  * KV cache state schema:
        state = {
            "kv": [  # one per layer
                {"k": (B, H, T_seen, D_h), "v": (B, H, T_seen, D_h)}
                or None if no history yet
            ],
            "offset": int,  # absolute frame index for sinusoidal PE
        }
  * During offline training (state=None), the attention is a standard
    causal softmax over the current input's T frames.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components import SinusoidalPE


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional KV cache."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # (B, T, D)
        kv_cache: Optional[dict] = None,       # {"k", "v"} or None
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_total), True = pad
    ) -> Tuple[torch.Tensor, dict]:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        qkv = self.qkv(x).view(B, T, 3, H, Dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, T, Dh)

        # Append to KV cache if present
        if kv_cache is not None and kv_cache.get("k") is not None:
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        new_cache = {"k": k.detach(), "v": v.detach()}

        T_total = k.shape[2]
        T_new = q.shape[2]

        # Causal mask: q row i can attend to k columns 0..(T_total - T_new + i)
        # That is: past cached entries (columns 0..T_total-T_new-1) are all
        # visible, plus within-chunk causal visibility for the new entries.
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        # (B, H, T_new, T_total)

        causal = torch.ones(T_new, T_total, device=x.device, dtype=torch.bool).tril(
            diagonal=T_total - T_new
        )
        attn = attn.masked_fill(~causal[None, None, :, :], float("-inf"))

        if key_padding_mask is not None:
            # key_padding_mask: (B, T_total), True → pad, mask out
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T_new, Dh)
        out = out.transpose(1, 2).contiguous().view(B, T_new, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out, new_cache


class CausalTransformerLayer(nn.Module):
    """Pre-norm causal Transformer layer with matched FFN dim."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[dict] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        attn_out, new_cache = self.attn(
            self.ln1(x), kv_cache=kv_cache, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


class CausalTransformerEncoder(nn.Module):
    """Causal Transformer encoder with per-layer KV cache for streaming.

    Matches the shape and signature of the other encoders in this project.
    Used as:
      (a) the Group A Transformer baseline for causal / streaming ASR;
      (b) the honest measurement point in the streaming-memory vs duration
          demonstration — its KV cache grows linearly with audio length.
    """

    supports_carry_state = True

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.ln0 = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList([
            CausalTransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def init_state(self, batch_size: int, device: torch.device) -> dict:
        return {
            "kv": [{"k": None, "v": None} for _ in range(self.n_layers)],
            "offset": 0,
        }

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        B, T, _ = x.shape

        offset = state["offset"] if state is not None else 0
        kv_states: List[Optional[dict]] = (
            state["kv"] if state is not None else [None] * self.n_layers
        )

        x = self.ln0(x)
        x = self.pos_enc(x, offset=offset)

        # Padding mask: True where padded (absent) both in cache and current.
        # Cache entries are always real (we only append valid frames), so the
        # pad mask only marks the new chunk's tail. We extend it to full
        # cache length by prepending False for cache positions.
        new_pad = ~(torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
        if state is not None and kv_states[0] is not None and kv_states[0].get("k") is not None:
            cache_len = kv_states[0]["k"].shape[2]
            cache_pad = torch.zeros(B, cache_len, dtype=torch.bool, device=x.device)
            key_padding_mask = torch.cat([cache_pad, new_pad], dim=1)
        else:
            key_padding_mask = new_pad

        new_kv: List[dict] = []
        for i, layer in enumerate(self.layers):
            x, nc = layer(x, kv_cache=kv_states[i], key_padding_mask=key_padding_mask)
            new_kv.append(nc)

        new_state: Optional[dict] = None
        if state is not None:
            new_state = {"kv": new_kv, "offset": offset + T}

        return x, new_state
