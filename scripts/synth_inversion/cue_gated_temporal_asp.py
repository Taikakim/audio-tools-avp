# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Cue-Gated Temporal Audio Encoder (Step 1 Cheap Variant).

Implements CONTINUITY's suggestion:
  Adds an independent sigmoid cue gate directly to the existing TemporalAudioEncoderV4
  (which already pools mean + std via AttentiveStatsPooling), aligning directly to
  the existing Synth-JEPA embedding space.

Preserves the existing V6 C flow generator without needing to retrain Stage 2.
Fully compatible with cross-attention (xattn=True) and pack_tokens.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class CueGatedTemporalAudioEncoder(nn.Module):
    """Wraps TemporalAudioEncoderV4 with a cue-conditioned sigmoid gate.
    
    Stages:
      1. Tokens: base_encoder.tokens(mel) -> [B, 41, d_model]
      2. Cue gating: a_i = sigmoid((W_h h_i)^T W_q q / sqrt(d)) in [0, 1]
      3. Gated tokens: gated_tok = a * tok
      4. Weighted stats: mu, sigma across time
      5. Zero-init residual: z = norm(q) + R_theta([mu; sigma])
      6. Output: packed with gated_tok if base_encoder.pack_tokens is True
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        cue_dim: int = 512,
        proj_dim: int = 128,
        hidden_dim: int = 512,
        freeze_base: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.base_encoder = base_encoder
        self.cue_dim = cue_dim
        self.d_model = getattr(base_encoder, "d_model", 256)
        self.n_tokens = getattr(base_encoder, "n_tokens", 41)
        self.embed_dim = getattr(base_encoder, "embed_dim", 512)
        self.pack_tokens = getattr(base_encoder, "pack_tokens", False)
        self.proj_dim = proj_dim
        self.scale = 1.0 / math.sqrt(proj_dim)
        self.eps = eps

        if freeze_base:
            for p in self.base_encoder.parameters():
                p.requires_grad = False

        # Projections to shared d-dimensional space for gating
        self.W_h = nn.Linear(self.d_model, proj_dim, bias=False)
        self.W_q = nn.Linear(cue_dim, proj_dim, bias=False)

        # Residual mapping R_theta from [mu; sigma] (dim: 2 * d_model) to cue_dim
        self.R_theta = nn.Sequential(
            nn.Linear(2 * self.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cue_dim),
        )

        # Zero-initialize the final layer of R_theta so z initially equals norm(q)
        nn.init.zeros_(self.R_theta[-1].weight)
        nn.init.zeros_(self.R_theta[-1].bias)

    def unpack(self, packed: torch.Tensor):
        return self.base_encoder.unpack(packed)

    def forward(
        self,
        mel: torch.Tensor,
        cue: torch.Tensor,
        return_gates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            mel: Log-mel spectrogram of mixture [B, (1,) 128, 81]
            cue: Anchor cue embedding [B, cue_dim]
            return_gates: If True, returns (out, gates)
            
        Returns:
            out: Extracted embedding in Synth-JEPA space [B, embed_dim (+ tokens if packed)]
        """
        tok = self.base_encoder.tokens(mel)  # [B, 41, d_model]
        q_norm = F.normalize(cue, p=2, dim=-1)  # [B, cue_dim]

        # Stage 1: Cue-token gating
        h_proj = self.W_h(tok)             # [B, 41, proj_dim]
        q_proj = self.W_q(q_norm)          # [B, proj_dim]

        logits = (h_proj * q_proj.unsqueeze(1)).sum(dim=-1, keepdim=True) * self.scale  # [B, 41, 1]
        a = torch.sigmoid(logits)          # [B, 41, 1]

        # Stage 2: Weighted statistics aggregation
        a_sum = a.sum(dim=1, keepdim=True) + self.eps  # [B, 1, 1]
        mu = (a * tok).sum(dim=1, keepdim=True) / a_sum  # [B, 1, d_model]

        var = (a * ((tok - mu) ** 2)).sum(dim=1, keepdim=True) / a_sum + self.eps
        sigma = torch.sqrt(var)  # [B, 1, d_model]

        # Stage 3: Residual fusion
        stats = torch.cat([mu.squeeze(1), sigma.squeeze(1)], dim=-1)  # [B, 2 * d_model]
        res = self.R_theta(stats)                                      # [B, cue_dim]

        z = cue + res

        # Gated tokens for cross-attention
        gated_tok = a * tok

        out = torch.cat([z, gated_tok.flatten(1)], dim=1) if self.pack_tokens else z

        if return_gates:
            return out, a.squeeze(-1)
        return out
