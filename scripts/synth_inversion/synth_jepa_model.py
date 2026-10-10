"""Synth-JEPA: Joint Embedding Prediction Architecture for Synthesizer Parameter Search.

Reference: Hayes, Tian, Lattner, "Synth-JEPA: Joint Embedding Prediction for Renderer-Free
Synthesizer Parameter Search", arXiv:2609.31024 (Sept 2026). Section numbers below refer to it.
Components (Sec. 3.2):
1. AudioEncoder (E_a): transformer on log-mel frames, "projected via 1D convolution to 150
   tokens" + a learned summary token whose output is z_a in R^512. The paper's 150 tokens come
   from 3.0 s at a 10 ms hop (300 frames) with a stride-2 conv; our clips are 0.8 s (81 frames),
   so the same stride-2 conv gives 41 tokens. (v1 of this file instead average-pooled 81 frames
   UP to 150 tokens, i.e. duplicated frames.)
2. ParameterEncoder (E_p): each parameter -> a linear projection plus a parameter-specific bias;
   "a Perceiver bottleneck with 32 learned latents cross-attends to these tokens, followed by
   eight self-attention blocks. Averaging the final outputs gives z_p." ONE cross-attention,
   then 8 self-attention blocks (v1 had 8 blocks that each re-ran cross-attention).
3. Predictors (f_a->p, f_p->a): three-block residual MLPs, hidden width 1024, linear output heads.
4. SIGReg (LeJEPA, Sec. 2.1), applied to each encoder branch independently (LeVLJEPA).
Not specified by the paper: encoder depth and FFN width (only d=512, 8 heads, 53M total). With
8 layers per encoder, ff_dim=1024 gives 50.7M (the default, as the nearest to 53M) and
ff_dim=2048 gives 68.6M. This is a guess at the paper's config, not a reading of it. The EMA-teacher
variant (the paper's ablation, which SIGReg beats) is not implemented.
"""
import math
from inharmonicity_aux_head import InharmonicityHead
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

    A stride-2 1D convolution turns the frames into tokens (81 frames -> 41 tokens). A learned
    summary token [CLS] is prepended and its output is taken as z_a in R^512.
    """
    def __init__(
        self,
        in_mels: int = 128,
        embed_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_dim: int = 2048,
        in_frames: int = 81,
        token_stride: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv1d = nn.Conv1d(in_mels, embed_dim, kernel_size=3, stride=token_stride, padding=1)
        n_tokens = (in_frames + 2 - 3) // token_stride + 1  # conv output length
        self.n_tokens = n_tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens + 1, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 1, 128, T] or [B, 128, T]
        if mel.ndim == 4:
            mel = mel.squeeze(1)
        B = mel.shape[0]

        # 1D conv over time: [B, embed_dim, n_tokens] -> [B, n_tokens, embed_dim]
        x = self.conv1d(mel).transpose(1, 2)
        if x.shape[1] != self.n_tokens:
            raise ValueError(f"expected {self.n_tokens} tokens from the mel, got {x.shape[1]} (wrong frame count?)")

        # Prepend learned [CLS] token and add positional embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # [B, n_tokens + 1, embed_dim]

        h = self.transformer(x)
        h = self.norm(h)
        za = h[:, 0]  # [B, 512]
        return za


# ==============================================================================
# 2. Parameter Encoder (E_p) with Perceiver Bottleneck
# ==============================================================================
class CrossAttentionBlock(nn.Module):
    """Perceiver read-in: latents cross-attend to the parameter tokens, then an MLP."""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.norm_lat = nn.LayerNorm(embed_dim)
        self.norm_ctx = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        kv = self.norm_ctx(context)
        out, _ = self.cross_attn(self.norm_lat(latents), kv, kv)
        latents = latents + out
        return latents + self.mlp(self.norm_mlp(latents))


class PerceiverParameterEncoder(nn.Module):
    """Each parameter -> linear projection + parameter-specific bias. 32 learned latents
    cross-attend to these tokens once, followed by 8 self-attention blocks; averaging the
    final latents gives z_p in R^512 (Sec. 3.2)."""
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

        # Continuous params (in [-1, 1]): one linear projection per parameter
        self.cont_projs = nn.ModuleList([nn.Linear(1, embed_dim, bias=False) for _ in range(num_continuous)])
        # Categorical params (one-hot): one linear projection per parameter
        self.cat_projs = nn.ModuleList([nn.Linear(k, embed_dim, bias=False) for k in cat_sizes])
        # Parameter-specific learned biases (the only bias term per token)
        self.param_biases = nn.Parameter(torch.randn(total_tokens, embed_dim) * 0.02)

        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        self.read_in = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
                                           dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.self_blocks = nn.TransformerEncoder(layer, num_layers=num_blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, cont_params: torch.Tensor, cat_onehots: List[torch.Tensor]) -> torch.Tensor:
        # cont_params: [B, num_continuous] in [-1, 1]; cat_onehots: list of [B, k]
        B = cont_params.shape[0]
        tokens = [proj(cont_params[:, i:i + 1]) + self.param_biases[i] for i, proj in enumerate(self.cont_projs)]
        offset = self.num_continuous
        tokens += [proj(oh) + self.param_biases[offset + j] for j, (proj, oh) in enumerate(zip(self.cat_projs, cat_onehots))]
        context = torch.stack(tokens, dim=1)  # [B, total_tokens, embed_dim]

        latents = self.read_in(self.latents.expand(B, -1, -1), context)
        latents = self.norm(self.self_blocks(latents))
        return latents.mean(dim=1)  # [B, embed_dim]


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

    def forward(self, z: torch.Tensor, generator: torch.Generator = None) -> torch.Tensor:
        # z: [B, D]. Closed-form Epps-Pulley (BHEP) statistic with bandwidth beta = 1/gamma:
        #   T = mean_ij exp(-(si-sj)^2 / 2g^2) - 2 g/sqrt(1+g^2) mean_i exp(-si^2 / 2(1+g^2)) + g/sqrt(2+g^2)
        # which is 0 in expectation-limit iff the slice is N(0, 1).
        B, D = z.shape
        u = torch.randn(D, self.num_slices, device=z.device, dtype=z.dtype, generator=generator)
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
        ff_dim: int = 1024,
        in_frames: int = 81,
        inharmonicity_head: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.audio_encoder = AudioTransformerEncoder(embed_dim=embed_dim, num_layers=num_audio_layers,
                                                     ff_dim=ff_dim, in_frames=in_frames)
        self.param_encoder = PerceiverParameterEncoder(embed_dim=embed_dim, num_blocks=num_param_layers, ff_dim=ff_dim)
        self.f_a2p = CrossDomainPredictor(in_dim=embed_dim, hidden_dim=predictor_hidden, out_dim=embed_dim)
        self.f_p2a = CrossDomainPredictor(in_dim=embed_dim, hidden_dim=predictor_hidden, out_dim=embed_dim)
        self.sigreg = SIGRegLoss(num_slices=num_slices_sigreg)
        if inharmonicity_head:
            self.inharmonicity_head = InharmonicityHead(embed_dim=embed_dim)

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
        sigreg_generator: torch.Generator = None,
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
        loss_sig_a = self.sigreg(za, sigreg_generator)
        loss_sig_p = self.sigreg(zp, sigreg_generator)

        return loss_pred_p, loss_pred_a, loss_sig_a, loss_sig_p, za, zp
