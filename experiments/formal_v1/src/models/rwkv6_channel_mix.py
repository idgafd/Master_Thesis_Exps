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
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.mode = mode
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

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
