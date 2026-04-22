"""RWKV-6 ChannelMix (FFN): sigmoid(r) * value(relu²(key(x))).

Supports both causal and bidirectional token shift via mode parameter.
FFN dim = int((hidden_size * 3.5) // 32 * 32) = 896 for hidden_size=256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RWKV6ChannelMix(nn.Module):
    """RWKV-6 channel mixing (FFN block).

    Architecture: sigmoid(r) * value(relu²(key(x)))
    Token shift is mode-dependent (causal or bidirectional).
    """

    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        layer_id: int,
        mode: str = "lion",
        use_chanmix_bypass: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.mode = mode
        self.use_chanmix_bypass = use_chanmix_bypass
        hidden_size_ffn = int((hidden_size * 3.5) // 32 * 32)
        self.hidden_size_ffn = hidden_size_ffn

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

        # Stage 10.4 — Avey partial-embedding bypass.
        # Split W_k output ∈ R^{FFN} at ρ=0.5 → [z_h, z_t]; tail path applies
        # ReLU², head path is linear; mix with α (scalar per layer, zero-init).
        # α=0 ⇒ vanilla ChannelMix bit-exact.
        if self.use_chanmix_bypass:
            assert hidden_size_ffn % 2 == 0, "ChannelMix bypass requires even FFN dim"
            self.bypass_alpha = nn.Parameter(torch.zeros(1, dtype=dtype))

    def _token_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ("lion", "bidir_serial"):
            left = F.pad(x[:, :-1, :], (0, 0, 1, 0))
            right = F.pad(x[:, 1:, :], (0, 0, 0, 1))
            return (left + right) * 0.5
        return F.pad(x[:, :-1, :], (0, 0, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dxprev = self._token_shift(x) - x

        xk = x + dxprev * self.time_maa_k
        xr = x + dxprev * self.time_maa_r

        k_pre = self.key(xk)                              # (B, T, FFN)
        k_relu2 = torch.relu(k_pre) ** 2                  # vanilla tail branch

        if self.use_chanmix_bypass:
            # Linear head / non-linear tail split at ρ=0.5, blended in FFN
            # space before a single W_v projection:
            #   z_blend = (1-α)·ReLU²(k) + α·[ k_head_linear || ReLU²(k_tail) ]
            # Equivalent to (1-α)·W_v·ReLU²(k) + α·W_v·z_mixed since W_v is
            # linear.  Saves one W_v call vs the two-projection form.
            half = self.hidden_size_ffn // 2
            z_head_linear = k_pre[..., :half]
            z_tail_relu2 = k_relu2[..., half:]
            z_mixed = torch.cat([z_head_linear, z_tail_relu2], dim=-1)  # (B, T, FFN)
            z_blend = (1.0 - self.bypass_alpha) * k_relu2 + self.bypass_alpha * z_mixed
            kv = self.value(z_blend)
        else:
            kv = self.value(k_relu2)

        return torch.sigmoid(self.receptance(xr)) * kv
