"""RWKV-6 encoder — wraps N RWKV6Blocks with positional encoding and masking."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.components import SinusoidalPE
from src.models.rwkv6_block import RWKV6Block


class RWKV6Encoder(nn.Module):
    """RWKV-6 encoder supporting recurrent, LION, and bidir_serial modes.

    All configuration is passed through to RWKV6Block/TimeMix.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        dropout: float,
        head_size: int = 64,
        mode: str = "lion",
        conv_shift: bool = False,
        headscale: bool = False,
        delta_rule: bool = False,
        delta_warmstart: bool = False,
        lucid: bool = False,
        lucid_chunk_size: Optional[int] = None,
        lucid_self_reg: bool = False,
        temperature: bool = False,
        discretization: str = "zoh",
        discretization_init: str = "zoh",
        drop_u: bool = False,
        rse: bool = False,
        rse_n_scales: int = 1,
        rse_per_layer_overrides: list = None,
        p2rse: bool = False,
        p2rse_mixer: str = "linear",
        rse_viscosity: bool = False,
        p2rse_indep_lambda: bool = False,
        p2rse_indep_kv: bool = False,
        p2rse_kv_lora_dim: int = 32,
        use_rmsnorm: bool = False,
        use_hadamard_n2: bool = False,
        # When set, the Kronecker n=2 tail activates only on the top
        # `qtail_top_k` layers (indices [n_layers - qtail_top_k, n_layers)).
        # Aligned with the depth hierarchy observed in Stage-2 gen2 (α₁ grows
        # monotonically with depth) and Stage-4 `rse_depth`.
        use_qtail: bool = False,
        use_qtail_gamma: bool = False,
        use_qtail_dbeta: bool = False,
        use_qtail_lowrank: bool = False,
        qtail_lr_rank: int = 16,
        qtail_top_k: int = 2,
        use_qtail_beta_per_pair: bool = False,
        qtail_gamma_init: float = 1.0,
        # Stage 7A (A1′) — data-dependent readout phase
        use_data_dep_readphase: bool = False,
        # Soft clip on |φ| via tanh. None → TimeMix default (π).
        # Exposed so a regression-band single-seed result can be
        # retried once at a smaller clip (per STAGE7_DIAGNOSTICS
        # decision rule), without touching time-mix defaults.
        readphase_clip: Optional[float] = None,
        # Stage 8 T2 — non-normal RSE in polar parameterisation
        use_nonnormal_rse: bool = False,
        # κ in |ρ| ≤ κ · softplus(λ̃).  None → TimeMix default 0.6.
        nonnormal_rho_kappa: Optional[float] = None,
        # Stage 9 — sparse edge-layer specialist transition
        use_sparse_nonnormal_rse: bool = False,
        sparse_nn_edge_only: bool = False,
        nonnormal_psi_static: bool = False,
        use_loglinear: bool = False,
        loglinear_levels: int = 10,
        use_m2rnn: bool = False,
        m2rnn_layer: int = 5,
        use_conv_shift_multidilation: bool = False,
        conv_shift_multidil_padding_mode: str = "auto",
        conv_shift_multidil_content_conditional: bool = False,
        use_chanmix_bypass: bool = False,
        use_cayley_orthogonal: bool = False,
        cayley_rank: int = 1,
        use_pom_vlift: bool = False,
        pom_order: int = 2,
        pom_expansion: int = 64,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert d_model % 32 == 0
        assert d_model % head_size == 0

        self.d_model = d_model
        self.n_layers = n_layers
        self.mode = mode
        # Stage 10.1 / 10.2 — the new recurrent branches do not yet carry
        # per-bucket (loglinear) or per-layer-5 (M²RNN) state across chunks.
        # Disable carry-state advertising so evaluate.py skips *_carry metrics
        # rather than producing silently-wrong numbers.
        # Stage 10.5 Cayley and 10.6 PoM reuse the vanilla (B,H,K,K) carry
        # so they do support carry-state.
        self._supports_carry_state_override = not (use_loglinear or use_m2rnn)
        n_head = d_model // head_size

        self.pos_enc = SinusoidalPE(d_model, max_len=8000, dropout=dropout)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            per_layer = (rse_per_layer_overrides or [{}] * n_layers)[i]
            # Depth gating for the Kronecker n=2 tail: activate only on the
            # top `qtail_top_k` layers of the stack.
            qtail_active_here = use_qtail and (i >= n_layers - qtail_top_k)
            # Stage 9 Fix 2 — structural edge_only dispatch.  When sparse
            # edge_only is active, middle layers run plain RSE + viscosity
            # (no non-normal scan, no gate).  Only L0 and L_{n-1} carry the
            # non-normal extension.  Saves ~4/6 of the heavy 2×2 scan cost.
            is_edge = (i == 0) or (i == n_layers - 1)
            nonnormal_here = use_nonnormal_rse
            sparse_here = use_sparse_nonnormal_rse
            if sparse_nn_edge_only and use_sparse_nonnormal_rse and not is_edge:
                nonnormal_here = False
                sparse_here = False
            self.layers.append(
                RWKV6Block(
                    hidden_size=d_model,
                    n_head=n_head,
                    head_size=head_size,
                    num_hidden_layers=n_layers,
                    layer_id=i,
                    dropout=dropout,
                    mode=mode,
                    conv_shift=conv_shift,
                    headscale=headscale,
                    delta_rule=delta_rule,
                    delta_warmstart=delta_warmstart,
                    lucid=lucid,
                    lucid_chunk_size=lucid_chunk_size,
                    lucid_self_reg=lucid_self_reg,
                    temperature=temperature,
                    discretization=discretization,
                    discretization_init=discretization_init,
                    drop_u=drop_u,
                    rse=rse,
                    rse_n_scales=rse_n_scales,
                    rse_theta_init_scale=per_layer.get("theta_init_scale"),
                    rse_theta_clip=per_layer.get("theta_clip"),
                    rse_theta_lora_dim=per_layer.get("theta_lora_dim"),
                    p2rse=p2rse,
                    p2rse_mixer=p2rse_mixer,
                    rse_viscosity=rse_viscosity,
                    p2rse_indep_lambda=p2rse_indep_lambda,
                    p2rse_indep_kv=p2rse_indep_kv,
                    p2rse_kv_lora_dim=p2rse_kv_lora_dim,
                    use_rmsnorm=use_rmsnorm,
                    use_hadamard_n2=use_hadamard_n2,
                    use_qtail=qtail_active_here,
                    use_qtail_gamma=use_qtail_gamma and qtail_active_here,
                    use_qtail_dbeta=use_qtail_dbeta and qtail_active_here,
                    use_qtail_lowrank=use_qtail_lowrank and qtail_active_here,
                    qtail_lr_rank=qtail_lr_rank,
                    use_qtail_beta_per_pair=use_qtail_beta_per_pair and qtail_active_here,
                    qtail_gamma_init=qtail_gamma_init,
                    use_data_dep_readphase=use_data_dep_readphase,
                    readphase_clip=readphase_clip,
                    use_nonnormal_rse=nonnormal_here,
                    nonnormal_rho_kappa=nonnormal_rho_kappa,
                    use_sparse_nonnormal_rse=sparse_here,
                    sparse_nn_edge_only=sparse_nn_edge_only,
                    nonnormal_psi_static=nonnormal_psi_static,
                    use_loglinear=use_loglinear,
                    loglinear_levels=loglinear_levels,
                    use_m2rnn=use_m2rnn,
                    m2rnn_layer=m2rnn_layer,
                    use_conv_shift_multidilation=use_conv_shift_multidilation,
                    conv_shift_multidil_padding_mode=conv_shift_multidil_padding_mode,
                    conv_shift_multidil_content_conditional=conv_shift_multidil_content_conditional,
                    use_chanmix_bypass=use_chanmix_bypass,
                    use_cayley_orthogonal=use_cayley_orthogonal,
                    cayley_rank=cayley_rank,
                    use_pom_vlift=use_pom_vlift,
                    pom_order=pom_order,
                    pom_expansion=pom_expansion,
                    dtype=dtype,
                )
            )

    @property
    def supports_carry_state(self) -> bool:
        # Recurrent mode + no unsupported new-mechanism carry-state gap.
        return self.mode == "recurrent" and self._supports_carry_state_override

    def init_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize per-layer WKV states for carry-state inference.

        Single-pole RSE / vanilla RWKV-6: (B, H, K, K) per layer.
        Multi-rate RSE (M scales):         (M, B, H, K, K) per layer.
        Stage-5 P²-RSE (2 poles):          (2, B, H, K, K) per layer.
        """
        H = self.layers[0].att.n_head
        K = self.layers[0].att.head_size
        att0 = self.layers[0].att
        use_p2rse = getattr(att0, "use_p2rse", False)
        rse_M = getattr(att0, "rse_n_scales", 1)

        if use_p2rse:
            shape = (2, batch_size, H, K, K)
        elif rse_M > 1:
            shape = (rse_M, batch_size, H, K, K)
        else:
            shape = (batch_size, H, K, K)

        return [
            torch.zeros(*shape, dtype=torch.float32, device=device)
            for _ in range(self.n_layers)
        ]

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        x = self.pos_enc(x)
        B, T, _ = x.shape
        mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask_f = mask.unsqueeze(-1).float()

        new_states = [] if state is not None else None

        for i, layer in enumerate(self.layers):
            layer_state = state[i] if state is not None else None
            x, ns = layer(x, state=layer_state)
            x = x * mask_f
            if new_states is not None:
                new_states.append(ns)

        return x, new_states
