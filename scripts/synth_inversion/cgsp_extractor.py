# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Cue-Gated Statistics Pooling (CGSP) and Timbre Extractor module.

Adapted from 'Point-and-Play: Instrument Cloning from Music Mixtures'
(Tian et al., Univ. of Rochester, arXiv:2610.19445v1, Oct 14, 2026).

Enables extracting target instrument/synth timbre directly from polyphonic
mixtures without intermediate source separation artifacts.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import laion_clap
except ImportError:
    laion_clap = None


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) layer wrapping a frozen Linear layer."""

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        for p in self.original_linear.parameters():
            p.requires_grad = False

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        # Zero-init B, Kaiming-init A
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original_linear(x)
        lora = self.lora_B(self.lora_A(x)) * self.scaling
        return base + lora


def apply_lora_to_htsat(module: nn.Module, rank: int = 8, alpha: float = 16.0, target_subnames=("qkv", "fc1", "fc2")) -> int:
    """Recursively replaces targeted Linear layers in HTSAT layers with LoRALinear.
    
    Returns the count of adapted layers.
    """
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and any(sub in name for sub in target_subnames):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += apply_lora_to_htsat(child, rank=rank, alpha=alpha, target_subnames=target_subnames)
    return count


class CueGatedStatisticsPooling(nn.Module):
    """Cue-Gated Statistics Pooling (CGSP) module (Eq. 1 in Tian et al., 2026).
    
    Stages:
      1. Cue-token gating: elementwise sigmoid attention a_i = sigmoid((W_h h_i)^T W_q q / sqrt(d))
      2. Weighted statistics aggregation: mean mu and standard deviation sigma_h
      3. Residual fusion: z = norm(norm(q) + R_theta([mu; sigma_h]))
         where the final layer of R_theta is zero-initialized.
    """

    def __init__(
        self,
        token_dim: int = 1024,
        cue_dim: int = 512,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.cue_dim = cue_dim
        self.proj_dim = proj_dim
        self.eps = eps

        # Projections to shared d-dimensional space for gating
        self.W_h = nn.Linear(token_dim, proj_dim, bias=False)
        self.W_q = nn.Linear(cue_dim, proj_dim, bias=False)
        self.scale = 1.0 / math.sqrt(proj_dim)

        # Residual mapping R_theta from [mu; sigma_h] (dim: 2 * token_dim) to cue_dim
        self.R_theta = nn.Sequential(
            nn.Linear(2 * token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cue_dim),
        )

        # CRITICAL: Zero-initialize the final layer of R_theta so z initially equals norm(q)
        nn.init.zeros_(self.R_theta[-1].weight)
        nn.init.zeros_(self.R_theta[-1].bias)

    def forward(
        self,
        H: torch.Tensor,
        q: torch.Tensor,
        return_gates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Args:
            H: Mixture token sequence of shape [B, T, D_h] (e.g. [B, 1024, 1024])
            q: Cue embedding of shape [B, D_q] (e.g. [B, 512])
            return_gates: If True, returns (z, a) where a is [B, T]
            
        Returns:
            z: Extracted timbre code of shape [B, D_q], L2-normalized.
        """
        B, T, Dh = H.shape
        q_norm = F.normalize(q, p=2, dim=-1)  # [B, D_q]

        # Stage 1: Cue-token gating
        h_proj = self.W_h(H)             # [B, T, proj_dim]
        q_proj = self.W_q(q_norm)        # [B, proj_dim]

        # Compute dot product (W_h h_i)^T (W_q q)
        logits = (h_proj * q_proj.unsqueeze(1)).sum(dim=-1, keepdim=True) * self.scale  # [B, T, 1]
        a = torch.sigmoid(logits)  # [B, T, 1]

        # Stage 2: Weighted statistics aggregation
        a_sum = a.sum(dim=1, keepdim=True) + self.eps  # [B, 1, 1]
        mu = (a * H).sum(dim=1, keepdim=True) / a_sum  # [B, 1, Dh]

        var = (a * ((H - mu) ** 2)).sum(dim=1, keepdim=True) / a_sum + self.eps
        sigma_h = torch.sqrt(var)  # [B, 1, Dh]

        # Stage 3: Residual fusion
        stats = torch.cat([mu.squeeze(1), sigma_h.squeeze(1)], dim=-1)  # [B, 2 * Dh]
        residual = self.R_theta(stats)                                   # [B, D_q]

        z = F.normalize(q_norm + residual, p=2, dim=-1)

        if return_gates:
            return z, a.squeeze(-1)
        return z


class PointAndPlayExtractor(nn.Module):
    """End-to-end extractor wrapping CLAP audio/text branches with CGSP.
    
    Supports:
      - Mixture encoding with fine-grained tokens H and global code g
      - Text cue encoding q_text
      - Audio cue encoding q_audio
      - CGSP pooling to extracted timbre z
    """

    def __init__(
        self,
        clap_module: Optional[nn.Module] = None,
        token_dim: int = 1024,
        cue_dim: int = 512,
        proj_dim: int = 256,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.cue_dim = cue_dim
        self.token_dim = token_dim

        if clap_module is None:
            if laion_clap is None:
                raise ImportError("laion_clap is required to instantiate PointAndPlayExtractor")
            self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        else:
            self.clap = clap_module

        # Freeze entire base CLAP model first
        for p in self.clap.parameters():
            p.requires_grad = False

        # Apply LoRA to mixture audio branch
        self.adapted_count = apply_lora_to_htsat(
            self.clap.model.audio_branch,
            rank=lora_rank,
            alpha=lora_alpha,
            target_subnames=("qkv", "fc1", "fc2"),
        )

        # CGSP pooling module
        self.cgsp = CueGatedStatisticsPooling(
            token_dim=token_dim,
            cue_dim=cue_dim,
            proj_dim=proj_dim,
            hidden_dim=cue_dim,
        )

    def encode_mixture(self, mix_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes mixture waveform to frame tokens H and global normalized code g.
        
        Args:
            mix_wav: Tensor of shape [B, samples] (48kHz audio)
            
        Returns:
            H: Frame tokens [B, T, D_h] (fine_grained_embedding)
            g: Global code [B, cue_dim] (L2-normalized)
        """
        out = self.clap.model.audio_branch({"waveform": mix_wav})
        H = out["fine_grained_embedding"]  # [B, 1024, 1024]
        raw_g = out["embedding"]           # [B, 1024]
        g = self.clap.model.audio_projection(raw_g)
        g = F.normalize(g, p=2, dim=-1)
        return H, g

    @torch.no_grad()
    def encode_text_cue(self, texts: List[str]) -> torch.Tensor:
        """Encodes text prompts to normalized cue embedding q_text."""
        q = self.clap.get_text_embedding(texts, use_tensor=True)
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=torch.float32)
        return F.normalize(q, p=2, dim=-1)

    @torch.no_grad()
    def encode_audio_cue(self, cue_wav: torch.Tensor) -> torch.Tensor:
        """Encodes clean audio cue snippet to normalized cue embedding q_audio."""
        out = self.clap.model.audio_branch({"waveform": cue_wav})
        raw_g = out["embedding"]
        q = self.clap.model.audio_projection(raw_g)
        return F.normalize(q, p=2, dim=-1)

    def forward(
        self,
        mix_wav: torch.Tensor,
        cue: torch.Tensor,
        return_gates: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Extracts timbre code z from mixture given cue vector q.
        
        Args:
            mix_wav: [B, samples] audio waveform
            cue: [B, 512] cue embedding (from text or audio)
            return_gates: If True, also returns attention weights [B, T]
            
        Returns:
            z: [B, 512] extracted timbre code
            g: [B, 512] global mixture embedding
            (gates): [B, T] optional attention gates
        """
        H, g = self.encode_mixture(mix_wav)
        if return_gates:
            z, gates = self.cgsp(H, cue, return_gates=True)
            return z, g, gates
        z = self.cgsp(H, cue, return_gates=False)
        return z, g
