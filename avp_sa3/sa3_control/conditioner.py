"""Control conditioners: raw control signals -> control tokens for the adapters.

MVP = the audio-reference (riffer) branch: encode a reference SAME latent into a
compact set of style/content tokens. (Attribute branches — dynamics/rhythm/melody
from the .TIMESERIES.npz — slot in the same way later.)
"""

from __future__ import annotations

import torch
from torch import nn


class ScalarAttributeEncoder(nn.Module):
    """A single scalar control (e.g. normalized onset_density) -> control tokens.

    The FIRST explicit attribute branch (vs the opaque audio-reference riffer). A bank of
    `n_tokens` learned base tokens is FiLM-modulated by the scalar, giving cross-attention a
    small, expressive token set parametrised by one interpretable number. Same output contract
    as AudioRefEncoder -> (B, n_tokens, control_dim), so it drops into the same adapters.

    Input scalar is expected pre-normalised (~standardised); 0 == the dataset mean == the
    natural cfg-dropout null.
    """

    def __init__(self, control_dim: int = 768, n_tokens: int = 16, hidden: int = 256):
        super().__init__()
        self.n_tokens = n_tokens
        self.control_dim = control_dim
        self.tokens = nn.Parameter(torch.randn(n_tokens, control_dim) * 0.02)
        self.film = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, n_tokens * control_dim * 2),     # per (token,dim) scale + shift
        )
        nn.init.zeros_(self.film[-1].weight)                   # start as identity (scale=0, shift=0)
        nn.init.zeros_(self.film[-1].bias)

    def forward(self, scalar):                                 # (B,) or (B,1)
        x = scalar.reshape(-1, 1).to(self.tokens.dtype)
        gb = self.film(x).view(-1, self.n_tokens, self.control_dim, 2)
        scale, shift = gb[..., 0], gb[..., 1]
        return self.tokens[None] * (1.0 + scale) + shift       # (B, n_tokens, control_dim)


class AudioRefEncoder(nn.Module):
    """Reference SAME latent (B, latent_dim, T) -> control tokens (B, n_tokens, control_dim).

    Downsamples the 4096-frame latent to a small token set (the reference is a
    style/content summary, not time-aligned to the target), so cross-attention stays
    cheap. Strided convs (/16) then adaptive pool to exactly `n_tokens`.
    """

    def __init__(self, latent_dim: int = 256, control_dim: int = 768,
                 n_tokens: int = 256, hidden: int = 512):
        super().__init__()
        self.n_tokens = n_tokens
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 4, stride=4, padding=0), nn.SiLU(),   # T/4
            nn.Conv1d(hidden, hidden, 4, stride=4, padding=0), nn.SiLU(),   # T/16
            nn.Conv1d(hidden, control_dim, 1),
        )
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)

    def forward(self, ref_latent):                 # (B, latent_dim, T)
        h = self.net(ref_latent.to(self.net[0].weight.dtype))   # (B, control_dim, ~T/16)
        h = self.pool(h)                           # (B, control_dim, n_tokens)
        return h.transpose(1, 2).contiguous()      # (B, n_tokens, control_dim)
