"""Control conditioners: raw control signals -> control tokens for the adapters.

MVP = the audio-reference (riffer) branch: encode a reference SAME latent into a
compact set of style/content tokens. (Attribute branches — dynamics/rhythm/melody
from the .TIMESERIES.npz — slot in the same way later.)
"""

from __future__ import annotations

from torch import nn


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
        h = self.net(ref_latent.float())           # (B, control_dim, ~T/16)
        h = self.pool(h)                           # (B, control_dim, n_tokens)
        return h.transpose(1, 2).contiguous()      # (B, n_tokens, control_dim)
