# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Mixture-Conditioned Flow Matching Inverter for V7 Layout.

Connects the Stage 1 CGSP Timbre Extractor directly to the flow-matching
Surge XT parameter generation network, eliminating the need for intermediate
stem separation or log-mel spectrogram pre-processing.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import param_codec_v6 as codec
from models_v6 import NoteConditionedFlow
from cgsp_extractor import PointAndPlayExtractor


class MixtureConditionedFlowV7(NoteConditionedFlow):
    """Flow inverter conditioned on 512-dim CGSP timbre embedding z and MIDI note."""

    def __init__(
        self,
        param_dim: int = codec.ENCODED_DIM,
        hidden_dim: int = 512,
        num_layers: int = 8,
        timbre_dim: int = 512,
        note_features: int = 8,
        time_scale: float = 1000.0,
        cond_noise: bool = False,
        **kwargs,
    ):
        # Initialize NoteConditionedFlow with dummy encoder (we pass z directly as audio_emb)
        super().__init__(
            param_dim=param_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            time_scale=time_scale,
            cond_noise=cond_noise,
            note_features=note_features,
            **kwargs,
        )
        self.timbre_dim = timbre_dim
        # audio_proj maps the 512-dim timbre embedding z to hidden_dim
        self.audio_proj = nn.Linear(timbre_dim, hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        tau: Optional[torch.Tensor] = None,
        note: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass predicting flow velocity v."""
        if z is not None:
            emb = self.audio_proj(z)
        elif audio_emb is not None:
            if audio_emb.shape[-1] == self.timbre_dim:
                emb = self.audio_proj(audio_emb)
            else:
                emb = audio_emb
        else:
            raise ValueError("MixtureConditionedFlowV7 requires timbre embedding z or audio_emb")
        return super().forward(x_t, t, audio_emb=emb, tau=tau, note=note)

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        note: torch.Tensor,
        num_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        tau: Optional[float] = None,
    ) -> torch.Tensor:
        """Generates parameter vector x_1 clamped to [0, 1] conditioned on z and note."""
        B = z.shape[0]
        x = torch.randn(B, self.param_dim, device=z.device, generator=generator)
        tau_t = None
        if self.cond_noise and tau is not None:
            tau_t = torch.full((B,), float(tau), device=z.device, dtype=torch.float32)

        dt = 1.0 / float(num_steps)
        for i in range(num_steps):
            t = torch.full((B,), i / float(num_steps), device=z.device, dtype=torch.float32)
            v = self(x, t, z=z, tau=tau_t, note=note)
            x = x + v * dt

        return torch.clamp(x, 0.0, 1.0)

    @torch.no_grad()
    def sample_from_mixture(
        self,
        mix_wav: torch.Tensor,
        cue: torch.Tensor,
        note: torch.Tensor,
        extractor: PointAndPlayExtractor,
        num_steps: int = 25,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """End-to-end extraction and parameter sampling directly from raw mixture."""
        z, _ = extractor(mix_wav, cue)
        return self.sample(z=z, note=note, num_steps=num_steps, generator=generator)


def build_model_v7(
    param_dim: int = codec.ENCODED_DIM,
    hidden_dim: int = 512,
    num_layers: int = 8,
    timbre_dim: int = 512,
    note_features: int = 8,
    cond_noise: bool = False,
) -> MixtureConditionedFlowV7:
    """Builds a fresh MixtureConditionedFlowV7 instance."""
    return MixtureConditionedFlowV7(
        param_dim=param_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        timbre_dim=timbre_dim,
        note_features=note_features,
        cond_noise=cond_noise,
    )
