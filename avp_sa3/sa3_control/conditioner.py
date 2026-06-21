"""Control conditioners: raw control signals -> control tokens for the adapters.

MVP = the audio-reference (riffer) branch: encode a reference SAME latent into a
compact set of style/content tokens. (Attribute branches — dynamics/rhythm/melody
from the .TIMESERIES.npz — slot in the same way later.)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class AttributeEncoder(nn.Module):
    """Time-varying control feature (B, C_in, T) -> TIME-ORDERED control tokens (B, T', control_dim).

    The time-aligned attribute branch — the chroma/curve analog of ScalarAttributeEncoder (one number)
    and AudioRefEncoder (global pool). Strided convs reduce T -> T' = T / 2**ceil(log2(downsample)) while
    STRICTLY preserving time order (no pooling), so the adapter's `add_fractional_positions` aligns each
    control token to its own time region — every output latent frame can attend to the control *at its
    moment*, approximating local conditioning.

    Use for SAME chroma (C_in = 384 = 3 octave-bands x 128), or any stacked per-frame feature
    (dynamics 4 / rhythm 3 / melody 12, …). The adapter's zero-init output gives the no-op training
    start, so this encoder is normally initialised (it should carry the time-varying signal from step 0).
    """

    def __init__(self, in_channels: int, control_dim: int = 768, hidden: int = 512, downsample: int = 8):
        super().__init__()
        self.in_channels = int(in_channels)
        self.control_dim = int(control_dim)
        n_down = max(0, round(math.log2(max(1, downsample))))     # number of stride-2 halvings
        self.downsample = 2 ** n_down
        layers = [nn.Conv1d(self.in_channels, hidden, 3, padding=1), nn.SiLU()]
        for _ in range(n_down):
            layers += [nn.Conv1d(hidden, hidden, 4, stride=2, padding=1), nn.SiLU()]   # exact T -> T/2
        layers += [nn.Conv1d(hidden, control_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat):                                      # (B, C_in, T)
        h = self.net(feat.to(self.net[0].weight.dtype))          # (B, control_dim, T')
        return h.transpose(1, 2).contiguous()                    # (B, T', control_dim)


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
