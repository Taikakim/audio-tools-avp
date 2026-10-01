import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. Conv2D Spectrogram Audio Encoder
# ==============================================================================
class AudioEncoder(nn.Module):
    """Encodes a (B, 1, 128, 81) mel-spectrogram into a compact 1D latent embedding."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # 64 x 41
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 x 21
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 16 x 11
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# 8 x 6
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1) # [B, 1, 128, 81]
        feat = self.conv(x).flatten(1)
        return self.proj(feat)

# ==============================================================================
# 2. Residual MLP Block
# ==============================================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.fc2(self.act(self.fc1(self.norm(x))))
        return x + h

# ==============================================================================
# 3. Direct Feedforward ResMLP Inverter
# ==============================================================================
class ResMLPInverter(nn.Module):
    def __init__(self, param_dim=18, hidden_dim=256, num_layers=4):
        super().__init__()
        self.encoder = AudioEncoder(embed_dim=hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, param_dim)

    def forward(self, mel):
        emb = self.encoder(mel)
        h = emb
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        # Parameters are strictly normalized in [0, 1]
        pred_params = torch.sigmoid(self.head(h))
        return pred_params

# ==============================================================================
# 4. Conditional Flow Matching DiT Inverter
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class FlowMatchingResMLP(nn.Module):
    """Conditional Flow Matching velocity predictor v_t(x_t, t | audio)."""
    def __init__(self, param_dim=18, hidden_dim=256, num_layers=4):
        super().__init__()
        self.param_dim = param_dim
        self.encoder = AudioEncoder(embed_dim=hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.param_proj = nn.Linear(param_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, param_dim)

    def forward(self, x_t, t, mel):
        # x_t: [B, param_dim], t: [B], mel: [B, 1, 128, 81]
        audio_emb = self.encoder(mel)
        time_emb = self.time_embed(t)
        param_emb = self.param_proj(x_t)
        
        h = audio_emb + time_emb + param_emb
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h) # Velocity field v_t

    @torch.no_grad()
    def sample(self, mel, num_steps=20):
        device = mel.device
        B = mel.shape[0]
        # Sample standard Gaussian noise
        x = torch.randn(B, self.param_dim, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=device, dtype=torch.float32)
            v = self(x, t, mel)
            x = x + v * dt
        return torch.clamp(x, 0.0, 1.0)
