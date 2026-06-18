"""sa3-control — our own MuseControlLite-style control adapters for SA3 medium-base.

Trains decoupled cross-attention adapters on pre-encoded SAME-L latents
(`latents_sa3`) + grid-aligned mir control features (`.TIMESERIES.npz`), against
our ROCm SA3 fork. Audio-reference branch = the similarity riffer.
"""

__version__ = "0.0.1"
