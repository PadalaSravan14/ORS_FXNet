

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPhaseTemporalEncoder(nn.Module):


    def __init__(self, input_dim, short_hidden=128, short_layers=2, short_dropout=0.2,
                 long_hidden=256, long_layers=3, long_dropout=0.3, use_layer_norm=True):
        super().__init__()
        self.short_term_lstm = nn.LSTM(
            input_size=input_dim, hidden_size=short_hidden, num_layers=short_layers,
            batch_first=True, dropout=short_dropout if short_layers > 1 else 0.0,
        )
        self.long_term_lstm = nn.LSTM(
            input_size=input_dim, hidden_size=long_hidden, num_layers=long_layers,
            batch_first=True, dropout=long_dropout if long_layers > 1 else 0.0,
            bidirectional=True,
        )
        long_out_dim = long_hidden * 2  # bidirectional
        self.long_projection = nn.Linear(long_out_dim, short_hidden)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.short_norm = nn.LayerNorm(short_hidden)
            self.long_norm = nn.LayerNorm(short_hidden)

        self.gate = nn.Linear(short_hidden * 2, short_hidden)
        self.fused_dim = short_hidden

    def forward(self, x_short, x_long):
 
        h_s_seq, _ = self.short_term_lstm(x_short)          # (B, w_s, short_hidden)
        h_l_seq, _ = self.long_term_lstm(x_long)             # (B, w_l, 2*long_hidden)
        h_l_seq = self.long_projection(h_l_seq)               # (B, w_l, short_hidden)

        if self.use_layer_norm:
            h_s_seq = self.short_norm(h_s_seq)
            h_l_seq = self.long_norm(h_l_seq)

        # Align long-term representation to the short-term time axis by
        # broadcasting its final aggregated state across every short-term step.
        h_l_summary = h_l_seq[:, -1:, :].expand(-1, h_s_seq.size(1), -1)

        gate_input = torch.cat([h_s_seq, h_l_summary], dim=-1)
        alpha = torch.sigmoid(self.gate(gate_input))          # Eq. (13)
        h_fused_seq = alpha * h_s_seq + (1 - alpha) * h_l_summary  # Eq. (14)

        h_fused_final = h_fused_seq[:, -1, :]
        return h_fused_seq, h_fused_final


class AttentionFusionLayer(nn.Module):


    def __init__(self, hidden_dim, attention_dim=128, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.query_proj = nn.Linear(hidden_dim, attention_dim)
        self.key_proj = nn.Linear(hidden_dim, attention_dim)
        self.context_vector = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.01)
        self.output_proj = nn.Linear(attention_dim, hidden_dim)

    def forward(self, sequence, query_vector):
  
        B, T, _ = sequence.shape
        keys = self.key_proj(sequence).view(B, T, self.num_heads, self.head_dim)
        query = self.query_proj(query_vector).view(B, self.num_heads, self.head_dim)

        # e_i = v^T tanh(W1 h_i + W2 h_f)   -- Eq. (16), applied per head
        scores = torch.tanh(keys + query.unsqueeze(1))  # (B, T, heads, head_dim)
        scores = torch.einsum("bthd,hd->bth", scores, self.context_vector)  # (B, T, heads)
        weights = F.softmax(scores, dim=1)  # Eq. (17), softmax over time

        context = torch.einsum("bth,bthd->bhd", weights, keys)  # Eq. (18)
        context = context.reshape(B, -1)
        context = self.output_proj(context)
        attn_weights_avg = weights.mean(dim=-1)  # (B, T) averaged across heads for interpretability
        return context, attn_weights_avg



class MacroSensitivityAutoencoder(nn.Module):


    def __init__(self, input_dim, encoder_dims=(64, 32, 16), latent_dim=16):
        super().__init__()
        dims = [input_dim] + list(encoder_dims)
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)
        self.latent_head = nn.Linear(dims[-1], latent_dim)

        dec_dims = [latent_dim] + list(reversed(encoder_dims)) + [input_dim]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

        # modulation function g(.) mapping latent macro code -> per-feature
        # scaling factors (FiLM-style), applied to the temporal embedding
        self.modulation = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, latent_dim)
        )
        self.latent_dim = latent_dim

    def encode(self, macro_vec):
        h = self.encoder(macro_vec)
        z = self.latent_head(h)  # Eq. (19)
        return z

    def decode(self, z):
        return self.decoder(z)  # Eq. (20)

    def modulate(self, temporal_embedding, z, film_projection):

        scale = torch.sigmoid(film_projection(self.modulation(z))) * 2.0  # centred around 1
        return temporal_embedding * scale

    def forward(self, macro_vec):
        z = self.encode(macro_vec)
        recon = self.decode(z)
        return z, recon


def reconstruction_loss(macro_vec, recon):
    return F.mse_loss(recon, macro_vec)  # Eq. (21)


def sensitivity_loss(y_hat, z):

    grads = torch.autograd.grad(
        outputs=y_hat.sum(), inputs=z, create_graph=True, retain_graph=True,
    )[0]
    return grads.norm(p=2, dim=-1).mean()

class FusionHead(nn.Module):


    def __init__(self, temporal_dim, latent_dim, tech_dim, layer_dims=(256, 128, 1)):
        super().__init__()
        in_dim = temporal_dim + latent_dim + tech_dim
        layers = []
        prev = in_dim
        for i, dim in enumerate(layer_dims):
            layers.append(nn.Linear(prev, dim))
            if i < len(layer_dims) - 1:
                layers.append(nn.LayerNorm(dim))
                layers.append(nn.ReLU())
            prev = dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, h_modulated, z, tech_indicators):
        u = torch.cat([h_modulated, z, tech_indicators], dim=-1)  # Eq. (25)
        y_base = self.mlp(u)  # Eq. (26)
        return y_base.squeeze(-1), u
