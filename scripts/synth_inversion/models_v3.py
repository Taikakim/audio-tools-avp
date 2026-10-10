import math

import torch
import torch.nn as nn

import param_codec_v3 as codec
from surge_spec_v3 import NUM_PARAMS


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
    """Point-estimate inverter.

    encoding="onehot_v2": sigmoid head for continuous params + logits per categorical.
    encoding="ordinal_v1": legacy single sigmoid head over all 23 (loads old checkpoints).
    """
    def __init__(self, param_dim=NUM_PARAMS, hidden_dim=256, num_layers=4, encoding=codec.ENCODING_V2):
        super().__init__()
        self.encoding = encoding
        self.encoder = AudioEncoder(embed_dim=hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        if encoding == codec.ENCODING_V1:
            self.head = nn.Linear(hidden_dim, param_dim)
        else:
            self.head_cont = nn.Linear(hidden_dim, codec.N_CONT)
            self.head_cat = nn.Linear(hidden_dim, sum(codec.CAT_SIZES))

    def forward(self, mel):
        """v2: returns [B, ENCODED_DIM] = [sigmoid continuous | raw logits per categorical].
        v1: returns [B, 23] sigmoid ordinal params."""
        h = self.encoder(mel)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        if self.encoding == codec.ENCODING_V1:
            return torch.sigmoid(self.head(h))
        return torch.cat([torch.sigmoid(self.head_cont(h)), self.head_cat(h)], dim=1)

    @torch.no_grad()
    def predict_params(self, mel, **_):
        """[B, 23] ordinal-layout parameters (categoricals as class / (n-1))."""
        out = self(mel)
        if self.encoding == codec.ENCODING_V1:
            return out
        return codec.decode(out)

# ==============================================================================
# 4. Conditional Flow Matching ResMLP Inverter
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding of t in [0, 1].

    `scale` multiplies t first. With scale=1 (the v1 models) the highest frequency is
    1 rad per unit t, so over t in [0, 1] every channel moves by < 1 rad and most barely
    move at all. scale=1000 is the usual DDPM-style choice and spreads t across channels.
    """
    def __init__(self, dim, scale=1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = (t * self.scale).unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class FlowMatchingResMLP(nn.Module):
    """Conditional flow matching velocity predictor v_t(x_t, t | audio).

    Path (OT-CFM, Lipman et al.): x_t = (1 - (1 - sigma_min) t) x0 + t x1, x0 ~ N(0, I),
    target u_t = x1 - (1 - sigma_min) x0; t = 0 is noise, t = 1 is data.
    encoding="onehot_v2": x lives in the codec's [continuous | one-hot] space.
    encoding="ordinal_v1": legacy 23-d ordinal space (loads old checkpoints; use time_scale=1).
    """
    def __init__(self, param_dim=None, hidden_dim=256, num_layers=4,
                 encoding=codec.ENCODING_V2, time_scale=1000.0, sigma_min=1e-5, cond_noise=False, n_mels=128):
        super().__init__()
        self.encoding = encoding
        # cond_noise (EXPERIMENTS H6, after Synth-JDF 2609.29320): the audio condition may be partly noised,
        # mel_tau = (1 - tau) * eps + tau * mel in standardised space, and tau is an input. tau = 1 is the
        # clean condition every older model saw. The standardisation stats live in the model (buffers), so a
        # checkpoint can noise its own input at inference.
        self.cond_noise = cond_noise
        if cond_noise:
            self.tau_embed = nn.Sequential(
                SinusoidalTimeEmbedding(hidden_dim, scale=time_scale),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.register_buffer("mel_mean", torch.zeros(n_mels))
            self.register_buffer("mel_std", torch.ones(n_mels))
        if param_dim is None:
            param_dim = NUM_PARAMS if encoding == codec.ENCODING_V1 else codec.ENCODED_DIM
        self.param_dim = param_dim
        self.sigma_min = sigma_min
        self.encoder = AudioEncoder(embed_dim=hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim, scale=time_scale),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.param_proj = nn.Linear(param_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, param_dim)

    def noisy(self, mel, tau, generator=None):
        """Partly noise a mel condition: (1 - tau) * eps + tau * mel, in standardised space. tau: float or [B]."""
        B = mel.shape[0]
        tau = torch.as_tensor(tau, dtype=mel.dtype, device=mel.device).expand(B) if not torch.is_tensor(tau) or tau.dim() == 0 \
            else tau.to(mel.dtype)
        shape = (B,) + (1,) * (mel.dim() - 1)
        mean, std = self.mel_mean[:, None].to(mel.dtype), self.mel_std[:, None].to(mel.dtype)  # broadcast over time
        z = (mel - mean) / std
        eps = torch.randn(mel.shape, device=mel.device, dtype=mel.dtype, generator=generator)
        tv = tau.view(shape)
        return mean + std * ((1.0 - tv) * eps + tv * z)

    def forward(self, x_t, t, mel=None, audio_emb=None, tau=None):
        # x_t: [B, param_dim], t: [B], mel: [B, 1, 128, 81]; tau: [B] conditioning level (cond_noise models only)
        if audio_emb is None:
            audio_emb = self.encoder(mel)
        h = audio_emb + self.time_embed(t) + self.param_proj(x_t)
        if self.cond_noise:
            if tau is None:
                tau = torch.ones(x_t.shape[0], device=x_t.device)
            h = h + self.tau_embed(tau)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return self.head(h) # Velocity field v_t

    def path(self, x0, x1, t):
        """Returns (x_t, target velocity) for training."""
        te = t.unsqueeze(-1)
        xt = (1.0 - (1.0 - self.sigma_min) * te) * x0 + te * x1
        return xt, x1 - (1.0 - self.sigma_min) * x0

    @torch.no_grad()
    def sample(self, mel, num_steps=20, generator=None, tau=None):
        """Euler ODE solve from noise; returns the raw x_1 (clamped to [0, 1]).
        tau < 1 (cond_noise models only) conditions on a partly noised reference (Synth-JDF's smoothed posterior).
        tau=None uses self.default_tau (1.0 unless a caller set it, e.g. invert_stem_collection.py --cond_tau)."""
        if tau is None:
            tau = getattr(self, "default_tau", 1.0)
        B = mel.shape[0]
        tau_t = None
        if self.cond_noise:
            tau_t = torch.full((B,), float(tau), device=mel.device, dtype=torch.float32)
            if float(tau) < 1.0:
                mel = self.noisy(mel, tau_t, generator=generator)
        elif float(tau) != 1.0:
            raise ValueError("tau < 1 needs a model trained with cond_noise")
        audio_emb = self.encoder(mel)  # encode once, not once per step
        x = torch.randn(B, self.param_dim, device=mel.device, generator=generator)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=mel.device, dtype=torch.float32)
            x = x + self(x, t, audio_emb=audio_emb, tau=tau_t) * dt
        return torch.clamp(x, 0.0, 1.0)

    @torch.no_grad()
    def predict_params(self, mel, num_steps=20, generator=None):
        """One posterior draw per input, as [B, 23] ordinal-layout parameters."""
        x = self.sample(mel, num_steps=num_steps, generator=generator)
        if self.encoding == codec.ENCODING_V1:
            return x
        return codec.decode(x)


# ==============================================================================
# 5. Checkpoint loading (v1 and v2)
# ==============================================================================
def build_model(model_type, hidden_dim, num_layers, encoding=codec.ENCODING_V2, time_scale=1000.0, cond_noise=False):
    if model_type == "resmlp":
        return ResMLPInverter(hidden_dim=hidden_dim, num_layers=num_layers, encoding=encoding)
    if model_type == "flow":
        return FlowMatchingResMLP(hidden_dim=hidden_dim, num_layers=num_layers,
                                  encoding=encoding, time_scale=time_scale, cond_noise=cond_noise)
    raise ValueError(f"Unknown model type: {model_type}")


def load_inverter(ckpt_path, device="cpu", model_type=None):
    """Rebuilds the model a checkpoint was trained as. Checkpoints written before the
    v2 fixes carry no 'param_encoding' and load as ordinal_v1 with time_scale=1."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {}) or {}
    state = ckpt["model_state"]
    # Architecture from the state dict itself, so checkpoints from older scripts with
    # incomplete args still load.
    model_type = model_type or args.get("model_type") or ("flow" if "time_embed.1.weight" in state else "resmlp")
    hidden_dim = state["encoder.proj.weight"].shape[0]
    num_layers = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    is_v2 = "head_cont.weight" in state or state["head.weight"].shape[0] == codec.ENCODED_DIM
    encoding = args.get("param_encoding", codec.ENCODING_V2 if is_v2 else codec.ENCODING_V1)
    time_scale = args.get("time_scale", 1.0 if encoding == codec.ENCODING_V1 else 1000.0)
    cond_noise = "tau_embed.1.weight" in state
    model = build_model(model_type, hidden_dim, num_layers, encoding=encoding, time_scale=time_scale,
                        **({"cond_noise": True} if cond_noise else {}))
    model.load_state_dict(state)
    return model.to(device).eval()
