"""RWKV-6 layer block: ln0 (layer 0 only) -> ln1 -> TimeMix -> drop -> ln2 -> ChannelMix -> drop."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.rwkv6_time_mix import RWKV6TimeMix
from src.models.rwkv6_channel_mix import RWKV6ChannelMix


class RWKV6Block(nn.Module):
    """Single RWKV-6 encoder layer block."""

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_size: int,
        num_hidden_layers: int,
        layer_id: int,
        dropout: float,
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
        rse_theta_init_scale: float = None,
        rse_theta_clip: float = None,
        rse_theta_lora_dim: int = None,
        rse_split_real_frac: float = 0.0,
        rse_split_complex_init_scale: float = None,
        use_trwg: bool = False,
        trwg_threshold_init: float = 0.5,
        p2rse: bool = False,
        p2rse_mixer: str = "linear",
        rse_viscosity: bool = False,
        p2rse_indep_lambda: bool = False,
        p2rse_indep_kv: bool = False,
        p2rse_kv_lora_dim: int = 32,
        use_rmsnorm: bool = False,
        use_hadamard_n2: bool = False,
        use_qtail: bool = False,
        use_qtail_gamma: bool = False,
        use_qtail_dbeta: bool = False,
        use_qtail_lowrank: bool = False,
        qtail_lr_rank: int = 16,
        use_qtail_beta_per_pair: bool = False,
        qtail_gamma_init: float = 1.0,
        use_data_dep_readphase: bool = False,
        readphase_clip: float = None,
        use_nonnormal_rse: bool = False,
        nonnormal_rho_kappa: float = None,
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
        self.ln1 = nn.LayerNorm(hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(hidden_size, dtype=dtype)

        if layer_id == 0:
            self.ln0 = nn.LayerNorm(hidden_size, dtype=dtype)
        else:
            self.ln0 = nn.Identity()

        rse_kwargs = {}
        if rse_theta_init_scale is not None:
            rse_kwargs["rse_theta_init_scale"] = rse_theta_init_scale
        if rse_theta_clip is not None:
            rse_kwargs["rse_theta_clip"] = rse_theta_clip
        if rse_theta_lora_dim is not None:
            rse_kwargs["rse_theta_lora_dim"] = rse_theta_lora_dim
        if readphase_clip is not None:
            rse_kwargs["readphase_clip"] = readphase_clip
        if nonnormal_rho_kappa is not None:
            rse_kwargs["nonnormal_rho_kappa"] = nonnormal_rho_kappa
        if rse_split_real_frac > 0:
            rse_kwargs["rse_split_real_frac"] = rse_split_real_frac
        if rse_split_complex_init_scale is not None:
            rse_kwargs["rse_split_complex_init_scale"] = rse_split_complex_init_scale
        if use_trwg:
            rse_kwargs["use_trwg"] = True
            rse_kwargs["trwg_threshold_init"] = trwg_threshold_init
        self.att = RWKV6TimeMix(
            hidden_size=hidden_size,
            n_head=n_head,
            head_size=head_size,
            num_hidden_layers=num_hidden_layers,
            layer_id=layer_id,
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
            **rse_kwargs,
            p2rse=p2rse,
            p2rse_mixer=p2rse_mixer,
            rse_viscosity=rse_viscosity,
            p2rse_indep_lambda=p2rse_indep_lambda,
            p2rse_indep_kv=p2rse_indep_kv,
            p2rse_kv_lora_dim=p2rse_kv_lora_dim,
            use_rmsnorm=use_rmsnorm,
            use_hadamard_n2=use_hadamard_n2,
            use_qtail=use_qtail,
            use_qtail_gamma=use_qtail_gamma,
            use_qtail_dbeta=use_qtail_dbeta,
            use_qtail_lowrank=use_qtail_lowrank,
            qtail_lr_rank=qtail_lr_rank,
            use_qtail_beta_per_pair=use_qtail_beta_per_pair,
            qtail_gamma_init=qtail_gamma_init,
            use_data_dep_readphase=use_data_dep_readphase,
            use_nonnormal_rse=use_nonnormal_rse,
            use_sparse_nonnormal_rse=use_sparse_nonnormal_rse,
            sparse_nn_edge_only=sparse_nn_edge_only,
            nonnormal_psi_static=nonnormal_psi_static,
            use_loglinear=use_loglinear,
            loglinear_levels=loglinear_levels,
            use_m2rnn=use_m2rnn,
            m2rnn_layer=m2rnn_layer,
            use_conv_shift_multidilation=use_conv_shift_multidilation,
            conv_shift_multidil_padding_mode=conv_shift_multidil_padding_mode,
            conv_shift_multidil_content_conditional=conv_shift_multidil_content_conditional,
            use_cayley_orthogonal=use_cayley_orthogonal,
            cayley_rank=cayley_rank,
            use_pom_vlift=use_pom_vlift,
            pom_order=pom_order,
            pom_expansion=pom_expansion,
            dtype=dtype,
        )
        self.ffn = RWKV6ChannelMix(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            layer_id=layer_id,
            mode=mode,
            use_chanmix_bypass=use_chanmix_bypass,
            dtype=dtype,
        )

        if dropout > 0.0:
            self.drop0 = nn.Dropout(p=dropout)
            self.drop1 = nn.Dropout(p=dropout)
        else:
            self.drop0 = nn.Identity()
            self.drop1 = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.ln0(x)
        att_out, new_state = self.att(self.ln1(x), state=state)
        x = self.drop0(x + att_out)
        x = self.drop1(x + self.ffn(self.ln2(x)))
        return x, new_state
