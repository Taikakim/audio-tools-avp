"""Synth-JEPA: Joint Embedding Prediction Architecture for Synthesizer Parameter Search.

Reference: Hayes et al. (arXiv:2609.31024, September 2026).
Components:
1. AudioEncoder (E_a): 1D Conv projecting log-mel frames -> 150 tokens + [CLS] token ->
   8 Transformer Encoder blocks (d=512, 8 heads) -> z_a in R^512.
2. ParameterEncoder (E_p): Parameter projection + learned bias -> Perceiver bottleneck
   (32 learned queries) cross-attending to parameter tokens + 8 self-attention blocks -> z_p in R^512.
3. Predictors (f_a->p, f_p->a): 3-block Residual MLPs with hidden width 1024 -> R^512.
4. SIGRegLoss: Sliced Isotropic Gaussian Regularization via analytical Epps-Pulley test statistic
   on random 1D projections (LeJEPA / LeVLJEPA), preventing representation collapse.
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from surge_spec import CAT_INDICES, CONT_INDICES, NUM_PARAMS


# ==============================================================================
# 1. Audio Encoder (E_a)
# ==============================================================================
class AudioTransformerEncoder(nn.Module):
    """Transformer operating on log-mel spectrograms (128 bands, T frames).

    Projects via 1D convolution to 150 sequence tokens. A learned summary token
    [CLS] is prepended to the sequence and its output is taken as z_a in R^512.
    """
    def __init__(
        self,
        in_mels: int = 128,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 2048,
        target_tokens: int = 150,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1d = nn.Conv1d(in_mels, embed_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(target_tokens)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, target_tokens + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 1, 128, T] or [B, 128, T]
        if mel.ndim == 4:
            mel = mel.squeeze(1)
        B = mel.shape[0]

        # 1D conv over time dimension
        x = self.conv1d(mel)              # [B, embed_dim, T]
        x = self.pool(x).transpose(1, 2)  # [B, 150, embed_dim]

        # Prepend learned [CLS] token and add positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # [B, 151, embed_dim]

        h = self.transformer(x)
        h = self.norm(h)
        za = h[:, 0]  # [B, 512]
        return za


# ==============================================================================
# 2. Parameter Encoder (E_p) with Perceiver Bottleneck
# ==============================================================================
class PerceiverBlock(nn.Module):
    """One Perceiver block: Latents cross-attend to context, then self-attend, then MLP."""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.norm_lat1 = nn.LayerNorm(embed_dim)
        self.norm_ctx = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_lat2 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_lat3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # Cross-attention: latents query the context tokens
        q = self.norm_lat1(latents)
        kv = self.norm_ctx(context)
        out, _ = self.cross_attn(q, kv, kv)
        latents = latents + out

        # Self-attention among latents
        q2 = self.norm_lat2(latents)
        out2, _ = self.self_attn(q2, q2, q2)
        latents = latents + out2

        # Feedforward MLP
        latents = latents + self.mlp(self.norm_lat3(latents))
        return latents


class PerceiverParameterEncoder(nn.Module):
    """Represents each parameter with a linear projection plus a parameter-specific bias.

    A Perceiver bottleneck with 32 learned latents cross-attends to these tokens,
    followed by 8 self-attention blocks. Averaging the final outputs gives z_p in R^512.
    """
    def __init__(
        self,
        num_continuous: int = len(CONT_INDICES),
        cat_sizes: List[int] = [k for _, k in CAT_INDICES],
        embed_dim: int = 512,
        num_latents: int = 32,
        num_blocks: int = 8,
        num_heads: int = 8,
        ff_dim: int = 2048,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_continuous = num_continuous
        self.cat_sizes = cat_sizes
        total_tokens = num_continuous + len(cat_sizes)

        # Projections for continuous params (mapped to [-1, 1])
        self.cont_projs = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_continuous)])
        # Projections for discrete/categorical params (one-hot vectors)
        self.cat_projs = nn.ModuleList([nn.Linear(k, embed_dim) for k in cat_sizes])

        # Parameter-specific learned biases
        self.param_biases = nn.Parameter(torch.randn(total_tokens, embed_dim) * 0.02)

        # 32 learned Perceiver latent queries
        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)

        # 8 Perceiver blocks
        self.blocks = nn.ModuleList([
            PerceiverBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, cont_params: torch.Tensor, cat_onehots: List[torch.Tensor]) -> torch.Tensor:
        # cont_params: [B, num_continuous] in [-1, 1]
        # cat_onehots: list of [B, k] one-hot tensors
        B = cont_params.shape[0]
        tokens = []

        for i, proj in enumerate(self.cont_projs):
            t = proj(cont_params[:, i:i+1]) + self.param_biases[i]
            tokens.append(t)

        offset = self.num_continuous
        for j, (proj, onehot) in enumerate(zip(self.cat_projs, cat_onehots)):
            t = proj(onehot) + self.param_biases[offset + j]
            tokens.append(t)

        context = torch.stack(tokens, dim=1)  # [B, total_tokens, embed_dim]
        latents = self.latents.expand(B, -1, -1)

        for block in self.blocks:
            latents = block(latents, context)

        latents = self.norm(latents)
        zp = latents.mean(dim=1)  # [B, embed_dim]
        return zp


# ==============================================================================
# 3. Cross-Domain Predictors (f_a->p, f_p->a)
# ==============================================================================
class ResBlock1024(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class CrossDomainPredictor(nn.Module):
    """Three-block residual MLP with hidden width 1024 and linear output heads."""
    def __init__(self, in_dim: int = 512, hidden_dim: int = 1024, out_dim: int = 512, num_blocks: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock1024(hidden_dim) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z)
        for b in self.blocks:
            h = b(h)
        return self.out_proj(self.norm(h))


# ==============================================================================
# 4. SIGReg Loss (Provable Isotropic Gaussian Anti-Collapse)
# ==============================================================================
class SIGRegLoss(nn.Module):
    """Sliced Isotropic Gaussian Regularization (LeJEPA / LeVLJEPA).

    Projects batch embeddings onto M random unit directions in S^{D-1} and computes
    the analytical Epps-Pulley test statistic against N(0, 1).
    """
    def __init__(self, num_slices: int = 64, gamma: float = 1.0):
        super().__init__()
        self.num_slices = num_slices
        self.gamma = gamma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, D]
        B, D = z.shape
        u = torch.randn(D, self.num_slices, device=z.device, dtype=z.dtype)
        u = F.normalize(u, p=2, dim=0)

        # 1D projections: [B, M]
        s = torch.matmul(z, u)

        # Pairwise differences along each slice: [B, B, M]
        diff_sq = (s.unsqueeze(1) - s.unsqueeze(0)).pow(2)
        term1 = torch.exp(-diff_sq / (2.0 * self.gamma**2)).mean(dim=(0, 1))

        scale2 = self.gamma / math.sqrt(1.0 + self.gamma**2)
        term2 = -2.0 * scale2 * torch.exp(-s.pow(2) / (2.0 * (1.0 + self.gamma**2))).mean(dim=0)

        term3 = self.gamma / math.sqrt(2.0 + self.gamma**2)

        ep_stat = term1 + term2 + term3
        return ep_stat.mean()


# ==============================================================================
# 5. Full Synth-JEPA Wrapper
# ==============================================================================
class SynthJEPA(nn.Module):
    """Complete Synth-JEPA framework: E_a, E_p, f_a->p, f_p->a."""
    def __init__(
        self,
        embed_dim: int = 512,
        predictor_hidden: int = 1024,
        num_audio_layers: int = 8,
        num_param_layers: int = 8,
        num_slices_sigreg: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.audio_encoder = AudioTransformerEncoder(embed_dim=embed_dim, num_layers=num_audio_layers)
        self.param_encoder = PerceiverParameterEncoder(embed_dim=embed_dim, num_blocks=num_param_layers)
        self.f_a2p = CrossDomainPredictor(in_dim=embed_dim, hidden_dim=predictor_hidden, out_dim=embed_dim)
        self.f_p2a = CrossDomainPredictor(in_dim=embed_dim, hidden_dim=predictor_hidden, out_dim=embed_dim)
        self.sigreg = SIGRegLoss(num_slices=num_slices_sigreg)

    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder(mel)

    def encode_params(self, cont_params: torch.Tensor, cat_onehots: List[torch.Tensor]) -> torch.Tensor:
        return self.param_encoder(cont_params, cat_onehots)

    def predict_audio_latent(self, zp: torch.Tensor) -> torch.Tensor:
        return self.f_p2a(zp)

    def predict_param_latent(self, za: torch.Tensor) -> torch.Tensor:
        return self.f_a2p(za)

    def forward(
        self,
        mel: torch.Tensor,
        cont_params: torch.Tensor,
        cat_onehots: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoders
        za = self.audio_encoder(mel)
        zp = self.param_encoder(cont_params, cat_onehots)

        # Cross-Domain Predictions
        z_hat_p = self.f_a2p(za)
        z_hat_a = self.f_p2a(zp)

        # Stop-gradient targets
        loss_pred_p = F.mse_loss(z_hat_p, zp.detach())
        loss_pred_a = F.mse_loss(z_hat_a, za.detach())

        # SIGReg anti-collapse on both encoders independently
        loss_sig_a = self.sigreg(za)
        loss_sig_p = self.sigreg(zp)

        return loss_pred_p, loss_pred_a, loss_sig_a, loss_sig_p, za, zp
