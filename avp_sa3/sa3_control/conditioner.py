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


class ChromaAttributeEncoder(nn.Module):
    """SAME chroma (B, 384 = 3 bands x 128 bins, T) -> time-aligned control tokens (B, T', control_dim).

    Chroma-AWARE (vs the generic AttributeEncoder's flat 1-D conv): reshapes the 384 channels back
    into the 3 octave-bands x 128 pitch-class bins SAME regresses (paper §3.3.2), and convolves the
    pitch axis CIRCULARLY — pitch class wraps at the octave, so a chord spanning bin 127->0 is seen as
    adjacent (a plain conv would treat it as a hard edge). Bands enter as conv input channels (register
    is preserved); strided convs downsample TIME only (time-ordered tokens for the adapter's fractional
    positions); pitch is pooled after circular mixing. Output contract identical to AttributeEncoder.
    """

    def __init__(self, n_bands: int = 3, n_bins: int = 128, control_dim: int = 768,
                 hidden: int = 128, downsample: int = 8, pitch_pool: int = 8, pitch_k: int = 3):
        super().__init__()
        self.n_bands, self.n_bins, self.pitch_k = int(n_bands), int(n_bins), int(pitch_k)
        self.control_dim = int(control_dim)
        self.in_channels = self.n_bands * self.n_bins
        self.conv_in = nn.Conv2d(self.n_bands, hidden, (self.pitch_k, 3), padding=(0, 1))   # pitch pad=circular(manual)
        n_down = max(0, round(math.log2(max(1, downsample))))
        self.downsample = 2 ** n_down
        self.time_convs = nn.ModuleList([
            nn.Conv2d(hidden, hidden, (self.pitch_k, 4), stride=(1, 2), padding=(0, 1))      # time /2, pitch kept
            for _ in range(n_down)])
        self.pitch_pool = nn.AdaptiveAvgPool2d((pitch_pool, None))                           # pitch -> pitch_pool
        self.proj = nn.Linear(hidden * pitch_pool, control_dim)

    def _cpad(self, x):                                # circular pad the pitch axis (dim=2)
        p = (self.pitch_k - 1) // 2
        return torch.cat([x[:, :, -p:, :], x, x[:, :, :p, :]], dim=2) if p else x

    def forward(self, feat):                           # (B, 384, T) or (B, 3, 128, T)
        w = self.conv_in.weight
        if feat.dim() == 3:                            # (B, 384, T) -> (B, 3, 128, T)
            B, C, T = feat.shape
            feat = feat.view(B, self.n_bands, self.n_bins, T)
        x = torch.relu(self.conv_in(self._cpad(feat.to(w.dtype))))     # (B, hidden, 128, T)
        for cv in self.time_convs:
            x = torch.relu(cv(self._cpad(x)))                          # (B, hidden, 128, T/2^k)
        x = self.pitch_pool(x)                                         # (B, hidden, pitch_pool, T')
        B, h, pp, Tp = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, Tp, h * pp)               # (B, T', hidden*pitch_pool)
        return self.proj(x)                                            # (B, T', control_dim)


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
