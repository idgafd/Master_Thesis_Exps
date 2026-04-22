"""DWConvShift: learned depthwise Conv1d replacing fixed token shift.

Initialized to symmetric [0.5, 0, 0.5] to match the bidirectional
(x[t-1] + x[t+1]) / 2 token shift at initialization.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWConvShift(nn.Module):
    """Learned depthwise Conv1d token shift."""

    def __init__(self, d_model: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
            bias=False,
        )
        # Initialize to [0.5, 0, 0.5] — matches bidirectional average
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, 0] = 0.5   # left neighbor
            self.conv.weight[:, 0, -1] = 0.5   # right neighbor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class MultiDilationDWConvShift(nn.Module):
    """Stage 10.3 — Parallel dilated DWConv branches with learnable α_d.

    Parallel branches d ∈ dilations with kernel-3 DWConv1d, output mixed as
        x_mixed = Σ_d α_d · DWConv_d(x)
    with α_1 = 1 and α_{d>1} = 0 at init. At init the layer reduces to a
    single-dilation ConvShift.

    Padding mode:
      "causal"     — left-padded by (kernel_size - 1) * d (respects streaming).
      "symmetric"  — symmetric padding, matches existing DWConvShift.

    Alpha mode:
      content_conditional=False (default): α_d is a per-layer scalar
        vector (Stage 10.3 / 10.3-sym).
      content_conditional=True (CB-3): α_d(x_t) = softmax(W_α · x_t + b_d)
        per token; each frame selects its own dilation mix. b_d init
        = large log-one-hot on d=1, W_α=0 ⇒ softmax ≈ one-hot(d=1) at init,
        so the module reduces to single-dilation ConvShift like the
        scalar variant. Adds ~d_model × n_dilations params/layer.

    Zero-regression at init: reduces to a plain single-dilation DWConv1d
    with symmetric [0.5, 0, 0.5] init at dilation 1.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2, 4, 8),
        padding_mode: str = "causal",
        content_conditional: bool = False,
    ):
        super().__init__()
        assert padding_mode in ("causal", "symmetric")
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilations = tuple(int(d) for d in dilations)
        self.padding_mode = padding_mode
        self.content_conditional = content_conditional

        self.branches = nn.ModuleList()
        for _ in self.dilations:
            # Padding is handled manually in forward() to support causal mode;
            # the conv itself is unpadded here.
            branch = nn.Conv1d(
                d_model, d_model,
                kernel_size=kernel_size,
                padding=0,
                groups=d_model,
                bias=False,
            )
            self.branches.append(branch)

        if content_conditional:
            # Per-token content-conditional α via softmax. Logits for the
            # d=1 dilation are biased high at init so softmax collapses
            # to one-hot(d=1) regardless of x_t, matching the scalar variant's
            # initial behaviour. SGD then grows W_α and redistributes b_d.
            self.alpha_proj = nn.Linear(d_model, len(self.dilations), bias=True)
            with torch.no_grad():
                self.alpha_proj.weight.zero_()
                self.alpha_proj.bias.zero_()
                d1_idx = self.dilations.index(1) if 1 in self.dilations else 0
                # A bias of 10 gives softmax(10, 0, 0, 0) ≈ (0.9999, ~0, ~0, ~0)
                self.alpha_proj.bias[d1_idx] = 10.0
            self.alpha = None
        else:
            # Per-layer α_d: init α_1 = 1 and others 0.
            alphas = torch.zeros(len(self.dilations))
            if 1 in self.dilations:
                alphas[self.dilations.index(1)] = 1.0
            else:
                alphas[0] = 1.0
            self.alpha = nn.Parameter(alphas)
            self.alpha_proj = None

        # Init branch 0 (dilation=1) to reduce to single-dilation ConvShift.
        # Other branches are zero-init (inert at init).
        #
        # PyTorch conv1d with left-only pad = (k-1)*d produces, for kernel=3
        # and dilation=d: output[t] = w[0]·x[t-2d] + w[1]·x[t-d] + w[2]·x[t].
        # Hence the "taps in time" are: w[0] ↔ x[t-2d], w[1] ↔ x[t-d], w[2] ↔ x[t].
        # Symmetric pad = (k-1)*d//2 on each side gives:
        # output[t] = w[0]·x[t-d] + w[1]·x[t] + w[2]·x[t+d].
        with torch.no_grad():
            for branch in self.branches:
                branch.weight.zero_()
            b0 = self.branches[self.dilations.index(1)] if 1 in self.dilations else self.branches[0]
            if kernel_size == 3:
                if padding_mode == "symmetric":
                    # Bidirectional neighbor average: 0.5·x[t-1] + 0.5·x[t+1]
                    # (matches existing DWConvShift convention.)
                    b0.weight[:, 0, 0] = 0.5   # x[t-1]
                    b0.weight[:, 0, -1] = 0.5  # x[t+1]
                else:  # causal — past-only, x[t-1]/x[t] average
                    # Causal analog of the bidirectional average: skip the
                    # x[t-2] tap (index 0) and weight the two most recent
                    # causal taps equally: 0.5·x[t-1] + 0.5·x[t].
                    b0.weight[:, 0, 1] = 0.5   # x[t-1]
                    b0.weight[:, 0, 2] = 0.5   # x[t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        x_ch = x.transpose(1, 2)  # (B, D, T)

        if self.content_conditional:
            # Per-token α: shape (B, T, n_dilations), softmax over dilations.
            alpha_bt_n = torch.softmax(self.alpha_proj(x), dim=-1)    # (B, T, n_dil)

        out = None
        for idx, (d, branch) in enumerate(zip(self.dilations, self.branches)):
            pad = (self.kernel_size - 1) * d
            if self.padding_mode == "causal":
                x_p = F.pad(x_ch, (pad, 0))
            else:  # symmetric
                left = pad // 2
                right = pad - left
                x_p = F.pad(x_ch, (left, right))
            y_d = F.conv1d(
                x_p, branch.weight, bias=None, stride=1, padding=0,
                dilation=d, groups=self.d_model,
            )  # (B, D, T)

            if self.content_conditional:
                # alpha_bt_n[:, :, idx] is (B, T); broadcast over channels.
                alpha_d_bt = alpha_bt_n[:, :, idx].unsqueeze(1)       # (B, 1, T)
                contrib = alpha_d_bt * y_d
            else:
                contrib = self.alpha[idx] * y_d

            out = contrib if out is None else out + contrib
        return out.transpose(1, 2)  # (B, T, D)
