"""Shared model components: ConvSubsampling, CTCHead, SinusoidalPE."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 8000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

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

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, n_mels, T)
        lengths: (B,) original mel frame counts
        Returns: (B, T', d_model), new_lengths
        """
        x = x.unsqueeze(1)              # (B, 1, n_mels, T)
        x = self.conv(x)                # (B, C, n_mels/4, T/4)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        x = self.proj(x)

        new_lengths = ((lengths - 1) // 2 + 1)
        new_lengths = ((new_lengths - 1) // 2 + 1)
        new_lengths = torch.clamp(new_lengths, min=1)
        return x, new_lengths


class ConvSubsamplingV2(nn.Module):
    """CB-5 frontend_v2 — modern Conv1d + pre-norm LN + SiLU stack.

    Replaces the 2-stage Conv2d + Linear ConvSubsampling with a
    streamlined Conv1d stack treating n_mels as the channel dimension.
    Tests the hypothesis that the frontend is the binding gap at
    7 M / 30 ep / clean-100 (STAGE10_ANALYSIS §4.2).

    Two variants (per PI review follow-up — removes the
    structural-change / param-count confound):

    * ``variant="lean"`` (CB-5a, ~413 K params):
        mels → LN → Conv1d(n_mels → d_model/2, k=5, s=2) → SiLU
            → LN → Conv1d(d_model/2 → d_model, k=5, s=2) → SiLU
            → LN → Conv1d(d_model → d_model, k=3, s=1) → SiLU
        Deliberately leaner than v1 (−1.5 M params vs ConvSubsampling's
        1.9 M) — tests "streamlined structure beats v1" with a param
        reduction confound.

    * ``variant="matched"`` (CB-5b, ~1.94 M params, param-neutral vs v1):
        mels → LN → Conv1d(n_mels → d_model,   k=5, s=2) → SiLU
            → LN → Conv1d(d_model → 2·d_model, k=5, s=2) → SiLU
            → LN → Conv1d(2·d_model → 2·d_model, k=3, s=1) → SiLU
            → LN → Conv1d(2·d_model → d_model, k=3, s=1) → SiLU
        Matches v1's param budget to isolate the STRUCTURAL effect of
        the redesign from the parameter-count effect.  4-stage to fit
        the param budget while keeping d_model at output.

    Outcome interpretation matrix (PI review §2.3 CB-5 follow-up):

    | lean beats v1? | matched beats v1? | Interpretation |
    |---|---|---|
    | Yes | Yes | Structural change is the win; params secondary |
    | No  | Yes | Capacity was binding — thesis pivot: wider frontend |
    | Yes | No  | v1 overparameterised; leaner is Pareto-better |
    | No  | No  | Frontend not the binding gap |

    Zero-regression: NOT bit-exact vs the current ConvSubsampling — by
    design, frontend redesign without a zero-reduction contract.
    """

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        channels: int | None = None,
        variant: str = "lean",
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for the halved stage"
        assert variant in ("lean", "matched")
        self.variant = variant

        # Note: no pre-LN on raw mel input.  LayerNorm over n_mels at init
        # (γ=1, β=0) zeros the mean and normalizes variance per timestep
        # across mel bins, which obliterates the spectral envelope that
        # encodes phonetic content.  Observed in first CB-5b attempt:
        # CER stuck at 0.85+ through ep 3 (dev), compared to typical ep 1
        # values ~0.51.  Fix: conv1 reads raw mels; LN applies only to
        # post-conv features where normalisation is benign.
        if variant == "lean":
            mid = d_model // 2
            # 3-stage design, ~413 K params.
            self.conv1 = nn.Conv1d(n_mels, mid, kernel_size=5, stride=2, padding=2)
            self.ln2 = nn.LayerNorm(mid)
            self.conv2 = nn.Conv1d(mid, d_model, kernel_size=5, stride=2, padding=2)
            self.ln3 = nn.LayerNorm(d_model)
            self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1)
            self.ln4 = None
            self.conv4 = None
        else:  # matched — param-neutral vs v1
            wide = 2 * d_model                               # 512 @ d_model=256
            self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=5, stride=2, padding=2)
            self.ln2 = nn.LayerNorm(d_model)
            self.conv2 = nn.Conv1d(d_model, wide, kernel_size=5, stride=2, padding=2)
            self.ln3 = nn.LayerNorm(wide)
            self.conv3 = nn.Conv1d(wide, wide, kernel_size=3, stride=1, padding=1)
            self.ln4 = nn.LayerNorm(wide)
            self.conv4 = nn.Conv1d(wide, d_model, kernel_size=3, stride=1, padding=1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, n_mels, T)  →  (B, T/4, d_model). lengths → halved twice."""

        # Stage 1 (downsample 2×) — raw mels go directly into conv1.
        # No LN before conv1: at default init (γ=1, β=0), LN over the
        # n_mels channel dim destroys the spectral envelope and stalls
        # training at near-random CER.
        x = F.silu(self.conv1(x))                            # (B, C1, T/2)

        # Stage 2 (downsample 2×)
        x_bt_c = x.transpose(1, 2)
        x_bt_c = self.ln2(x_bt_c)
        x = x_bt_c.transpose(1, 2)
        x = F.silu(self.conv2(x))                            # (B, C2, T/4)

        # Stage 3 (no downsample; refinement / wide).
        # Activation choice: SiLU on conv3 ONLY for the lean variant (since
        # lean has no stage 4); for matched, conv3 uses SiLU (it's an
        # intermediate) and the final projection in conv4 is LINEAR — this
        # mirrors vanilla's Linear(5120, 256) final projection, which
        # produces symmetric ± features the encoder expects.  First CB-5
        # attempts had SiLU on the final conv and the output distribution
        # clipped at ~-0.3, 3× smaller std than vanilla — CTC collapsed
        # to blanks.  Linear final stage restores distribution symmetry.
        x_bt_c = x.transpose(1, 2)
        x_bt_c = self.ln3(x_bt_c)
        x = x_bt_c.transpose(1, 2)
        if self.conv4 is not None:
            # Matched variant: keep SiLU on conv3 (intermediate)
            x = F.silu(self.conv3(x))                        # (B, C3, T/4)
        else:
            # Lean variant: conv3 is the final stage → no activation
            x = self.conv3(x)                                # (B, d_model, T/4)

        # Stage 4 (matched variant only): LINEAR projection back to d_model.
        if self.conv4 is not None:
            x_bt_c = x.transpose(1, 2)
            x_bt_c = self.ln4(x_bt_c)
            x = x_bt_c.transpose(1, 2)
            x = self.conv4(x)                                # (B, d_model, T/4)  — no SiLU on final

        x = x.transpose(1, 2)                                # (B, T/4, d_model)

        # Length accounting: two stride-2 convs each halve (with rounding).
        new_lengths = ((lengths - 1) // 2 + 1)
        new_lengths = ((new_lengths - 1) // 2 + 1)
        new_lengths = torch.clamp(new_lengths, min=1)
        return x, new_lengths


class CTCHead(nn.Module):
    """LayerNorm + linear projection to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))
