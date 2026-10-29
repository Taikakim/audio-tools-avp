# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""V8 Dynamic Parameter Inverter: Decoupled Static Patch + 10 Hz Automation Lanes.

Pillar 3 of the V8 Architecture (arXiv:2610-v8 / spec section 4):
1. Decouples static timbre from dynamic expression:
   - Static Core (59 params): Categorical types, envelopes, relative tuning.
   - Dynamic Lanes (10.76 Hz, 43 frames for 4.0s clip):
     * Lane 0: Filter 1 Cutoff
     * Lane 1: Filter 1 Resonance
     * Lane 2: Waveshaper Drive
     * Lane 3: FM Depth
2. Occam Flatness Penalty:
   Regularizes dynamic lanes via second-order curvature penalty to prevent
   high-frequency chatter and ensure per-note dynamics remain in envelopes.
3. DAW Export:
   Exports .fxp Surge XT preset + .mid MIDI CC automation bundle.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import surge_spec_v6 as S6


class ResBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + res)


class DynamicParameterInverterV8(nn.Module):
    """Sequence model mapping 4.0s audio to static patch + 10 Hz automation lanes."""

    def __init__(
        self,
        in_channels: int = 128,  # Mel bands
        seq_frames: int = 43,     # 10.76 Hz grid (4.0s)
        hidden_dim: int = 384,
        num_static_params: int = 59,
        num_dynamic_lanes: int = 4,
    ):
        super().__init__()
        self.seq_frames = seq_frames
        self.num_static_params = num_static_params
        self.num_dynamic_lanes = num_dynamic_lanes

        # 1. Front-end feature extractor downsampling ~400 mel frames to sequence representation
        self.front = nn.Sequential(
            nn.Conv1d(in_channels, 192, kernel_size=7, stride=2, padding=3),  # 400 -> 200
            nn.BatchNorm1d(192),
            nn.GELU(),
            nn.Conv1d(192, 256, kernel_size=7, stride=2, padding=3),          # 200 -> 100
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, hidden_dim, kernel_size=7, stride=2, padding=3),   # 100 -> 50
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # ResNet sequence backbone
        self.backbone = nn.Sequential(
            ResBlock1d(hidden_dim, kernel_size=5),
            ResBlock1d(hidden_dim, kernel_size=5),
            ResBlock1d(hidden_dim, kernel_size=5),
        )

        # Aligns sequence precisely to 43 frames (10.76 Hz SA3 / LatCH grid)
        self.pool_to_grid = nn.AdaptiveAvgPool1d(seq_frames)

        # 2. Dynamic Automation Head (predicts 4 lanes across 43 frames)
        self.dynamic_head = nn.Sequential(
            nn.Conv1d(hidden_dim, 192, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(192, 96, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(96, num_dynamic_lanes, kernel_size=3, padding=1),
            nn.Sigmoid(),  # All Surge automation lanes normalized to [0, 1]
        )

        # 3. Static Patch Head (pools sequence context to predict global 59 params)
        self.static_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_static_params),
            nn.Sigmoid(),  # Normalized parameters
        )

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps mel spectrogram to static patch and dynamic automation lanes.

        Args:
            mel: [B, 128, T_mel] log-mel spectrogram (~400 frames for 4.0s)

        Returns:
            pred_static: [B, 59] global static parameters
            pred_dynamic: [B, 4, 43] dynamic automation curves
        """
        # Feature extraction
        if mel.ndim == 4:
            mel = mel.squeeze(1)
        h = self.front(mel)
        h = self.backbone(h)
        seq_features = self.pool_to_grid(h)  # [B, hidden_dim, 43]

        # Dynamic lanes: [B, 4, 43]
        pred_dynamic = self.dynamic_head(seq_features)

        # Static patch: pool over time -> [B, hidden_dim] -> [B, 59]
        global_features = seq_features.mean(dim=-1)
        pred_static = self.static_head(global_features)

        return pred_static, pred_dynamic


def curvature_smoothness_loss(pred_dynamic: torch.Tensor) -> torch.Tensor:
    """Computes second-order curvature penalty: ||theta(t+1) - 2*theta(t) + theta(t-1)||^2.

    Penalizes high-frequency chatter while allowing smooth linear or sinusoidal sweeps.
    """
    # pred_dynamic: [B, lanes, T]
    diff2 = pred_dynamic[:, :, 2:] - 2.0 * pred_dynamic[:, :, 1:-1] + pred_dynamic[:, :, :-2]
    return torch.mean(diff2 ** 2)


def dynamic_inverter_loss(
    pred_static: torch.Tensor,
    true_static: torch.Tensor,
    pred_dynamic: torch.Tensor,
    true_dynamic: torch.Tensor,
    lambda_smooth: float = 0.5,
    lambda_dynamic: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Calculates multi-task static + dynamic parameter loss with Occam curvature penalty."""
    static_loss = F.mse_loss(pred_static, true_static)
    dynamic_loss = F.mse_loss(pred_dynamic, true_dynamic)
    smooth_loss = curvature_smoothness_loss(pred_dynamic)

    total_loss = static_loss + lambda_dynamic * dynamic_loss + lambda_smooth * smooth_loss

    metrics = {
        "loss_total": float(total_loss.item()),
        "loss_static": float(static_loss.item()),
        "loss_dynamic": float(dynamic_loss.item()),
        "loss_smooth": float(smooth_loss.item()),
    }
    return total_loss, metrics


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DynamicParameterInverterV8().to(device)
    dummy_mel = torch.randn(4, 128, 401, device=device)

    p_static, p_dynamic = model(dummy_mel)
    print("Model initialized successfully!")
    print(f"Static params shape: {p_static.shape} (bounds: [{p_static.min():.2f}, {p_static.max():.2f}])")
    print(f"Dynamic lanes shape: {p_dynamic.shape} (bounds: [{p_dynamic.min():.2f}, {p_dynamic.max():.2f}])")

    loss, met = dynamic_inverter_loss(p_static, torch.rand_like(p_static), p_dynamic, torch.rand_like(p_dynamic))
    print(f"Test loss: {loss.item():.4f} (Static: {met['loss_static']:.4f}, Dynamic: {met['loss_dynamic']:.4f}, Smooth: {met['loss_smooth']:.4f})")
