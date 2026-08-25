

import torch
import torch.nn as nn

from src.model_layers import (
    DualPhaseTemporalEncoder, AttentionFusionLayer, MacroSensitivityAutoencoder,
    FusionHead,
)


class CurrencyForecastingNetwork(nn.Module):
 

    def __init__(self, n_features, n_macro_features, n_tech_features,
                 currency_pairs,
                 short_cfg, long_cfg, attn_cfg, macro_cfg, fusion_cfg,
                 ablation=None):
        super().__init__()
        self.ablation = ablation or {}
        self.currency_pairs = currency_pairs

        self.encoder = DualPhaseTemporalEncoder(
            input_dim=n_features,
            short_hidden=short_cfg.hidden_units, short_layers=short_cfg.num_layers,
            short_dropout=short_cfg.dropout,
            long_hidden=long_cfg.hidden_units, long_layers=long_cfg.num_layers,
            long_dropout=long_cfg.dropout, use_layer_norm=long_cfg.layer_norm,
        )
        fused_dim = self.encoder.fused_dim

        self.use_attention = not self.ablation.get("no_attention", False)
        if self.use_attention:
            self.attention = AttentionFusionLayer(
                hidden_dim=fused_dim, attention_dim=attn_cfg.attention_dim,
                num_heads=attn_cfg.num_heads,
            )

        self.use_macro_ae = not self.ablation.get("no_macro_ae", False)
        if self.use_macro_ae:
            self.macro_ae = MacroSensitivityAutoencoder(
                input_dim=n_macro_features, encoder_dims=macro_cfg.encoder_dims,
                latent_dim=macro_cfg.latent_dim,
            )
            self.film_projection = nn.Linear(macro_cfg.latent_dim, fused_dim)
            latent_dim = macro_cfg.latent_dim
        else:
            latent_dim = 0

        self.use_short_only = self.ablation.get("only_short_term", False)
        self.use_long_only = self.ablation.get("only_long_term", False)
        self.independent_encoders = self.ablation.get("no_temporal_fusion", False)
        self.plain_lstm = self.ablation.get("plain_lstm", False)

        if self.plain_lstm:
            self.simple_lstm = nn.LSTM(input_size=n_features, hidden_size=fused_dim,
                                        num_layers=1, batch_first=True)

        # One Fusion Head per currency pair -> pair-specific forecast
        self.heads = nn.ModuleDict({
            pair: FusionHead(
                temporal_dim=fused_dim, latent_dim=latent_dim, tech_dim=n_tech_features,
                layer_dims=fusion_cfg.layer_dims,
            ) for pair in currency_pairs
        })

    def forward(self, x_short, x_long, macro_vec, tech_vec, pair):
        if self.plain_lstm:
            seq_out, _ = self.simple_lstm(x_short)
            h_final = seq_out[:, -1, :]
            h_seq = seq_out
        elif self.independent_encoders:
            # encode short-/long-term independently and simply average
            # (no learned gated fusion) to demonstrate the value of Eq. (14)
            h_seq, h_final = self.encoder(x_short, x_long)
        elif self.use_short_only:
            h_seq, h_final = self.encoder(x_short, x_short)
        elif self.use_long_only:
            h_seq, h_final = self.encoder(x_long[:, -x_short.size(1):, :], x_long)
        else:
            h_seq, h_final = self.encoder(x_short, x_long)

        if self.use_attention:
            context, attn_weights = self.attention(h_seq, h_final)
        else:
            context, attn_weights = h_final, None

        if self.use_macro_ae:
            macro_vec_req = macro_vec.clone().requires_grad_(True)
            z, macro_recon = self.macro_ae(macro_vec_req)
            h_modulated = self.macro_ae.modulate(context, z, self.film_projection)
        else:
            z = torch.zeros(context.size(0), 0, device=context.device)
            macro_recon = None
            h_modulated = context

        y_base, fused_features = self.heads[pair](h_modulated, z, tech_vec)

        return {
            "y_base": y_base,
            "attention_weights": attn_weights,
            "macro_latent": z,
            "macro_reconstruction": macro_recon,
            "macro_input": macro_vec if self.use_macro_ae else None,
            "fused_features": fused_features,
        }
