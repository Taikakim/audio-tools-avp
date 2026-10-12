# Owner: KUANG (created 2026-10-11)
"""Temporal conditioning for the flow inverter (research Q5; review B5, B6).

The v3 flow condition is ONE vector from `models_v3.AudioEncoder`: four stride-2 convolutions (time 81 -> 41 -> 21 ->
11 -> 6) and a global average pool. This module keeps the time axis and pools it with something that knows order:

  TemporalAudioEncoderV4(pool=...)   drop-in for `FlowMatchingResMLP.encoder`: mel [B, (1,) 128, 81] -> [B, embed_dim]
      conv stack with stride 2 on frequency only for the first three layers (time stays 81), stride 2 in time at the
      last layer (41 tokens of 20 ms, the same token grid as the Synth-JEPA audio encoder); a linear token projection,
      a learned positional embedding, an optional small transformer, then
        pool="asp"       attentive statistics pooling (weighted mean and std; Okabe et al. 2018) -> 2d
        pool="segments"  attack tokens (first att_ms) + sustain tokens (from sus_start_frac) + ASP -> 4d. The brief's
                         attack/sustain idea with the indices computed from the real hop (10 ms) and token stride
        pool="queries"   n_queries learned queries cross-attend to the tokens -> n_queries * d
  FlowMatchingResMLPV4               `FlowMatchingResMLP` with that encoder and optional per-block FiLM
                                     (zero-initialised, so it starts equal to the un-modulated flow)

Everything else (cond_noise, OT coupling, `sample`, `path`, the loss) is inherited unchanged, so `flow_loss` and the
inference code work as they are. A state dict is NOT compatible with a v3 flow (different encoder keys): start a new
out_dir. `load_inverter_v4` rebuilds either kind from a flow_best.pt / flow_latest.pt.

Status: unit-tested on CPU (shapes, gradients, FiLM identity at init, cond_noise, sampling, save/load); NOT trained on
Surge data. The toy probe (probe_encoders_toy_v4.py) is the only learning evidence, on a stand-in synth.
"""
import math

import torch
import torch.nn as nn

import param_codec_v3 as codec
from models_v3 import AudioEncoder, FlowMatchingResMLP


class AttentiveStatsPooling(nn.Module):
    """Weighted mean and standard deviation over time with learned frame weights. x [B, T, C] -> [B, 2C]."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        w = torch.softmax(self.score(x), dim=1)
        mu = (w * x).sum(dim=1)
        var = (w * (x - mu[:, None, :]) ** 2).sum(dim=1)
        return torch.cat([mu, var.clamp_min(1e-5).sqrt()], dim=-1)


class TemporalAudioEncoderV4(nn.Module):
    def __init__(self, embed_dim: int = 512, in_mels: int = 128, in_frames: int = 81,
                 channels=(32, 64, 128, 256), time_strides=(1, 1, 1, 2), d_model: int = 256, n_layers: int = 1,
                 n_heads: int = 4, pool: str = "asp", n_queries: int = 4, hop_ms: float = 10.0, att_ms: float = 40.0,
                 sus_start_frac: float = 0.25):
        super().__init__()
        if pool not in ("asp", "segments", "queries"):
            raise ValueError("pool must be 'asp', 'segments' or 'queries'")
        if len(channels) != len(time_strides):
            raise ValueError("channels and time_strides must have the same length")
        self.pool_kind = pool
        layers, c_in = [], 1
        for c_out, ts in zip(channels, time_strides):
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=(2, ts), padding=1), nn.BatchNorm2d(c_out), nn.GELU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        f_out, t_out = in_mels, in_frames
        for ts in time_strides:
            f_out, t_out = (f_out + 2 - 3) // 2 + 1, (t_out + 2 - 3) // ts + 1
        self.n_tokens, self.f_out = t_out, f_out
        self.token_ms = hop_ms * in_frames / t_out if t_out else hop_ms   # spacing of one token
        self.tok_proj = nn.Linear(channels[-1] * f_out, d_model)
        self.pos = nn.Parameter(torch.randn(t_out, d_model) * 0.02)      # 2-D on purpose: no unusual parameter rank for the optimizer
        if n_layers > 0:
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        else:
            self.transformer = nn.Identity()
        self.norm = nn.LayerNorm(d_model)
        self.n_att = max(1, int(round(att_ms / self.token_ms)))
        self.sus_start = min(int(round(sus_start_frac * t_out)), t_out - 1)
        if pool == "queries":
            self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            out_dim = n_queries * d_model
        else:
            self.asp = AttentiveStatsPooling(d_model)
            out_dim = 2 * d_model if pool == "asp" else 4 * d_model
        self.d_model = d_model
        self.proj = nn.Linear(out_dim, embed_dim)

    def tokens(self, mel):
        if mel.ndim == 3:
            mel = mel.unsqueeze(1)
        h = self.conv(mel)                                      # [B, C, F', T']
        h = h.permute(0, 3, 1, 2).flatten(2)                    # [B, T', C * F']
        return self.norm(self.transformer(self.tok_proj(h) + self.pos.unsqueeze(0)))

    def forward(self, mel):
        tok = self.tokens(mel)
        if self.pool_kind == "asp":
            pooled = self.asp(tok)
        elif self.pool_kind == "segments":
            pooled = torch.cat([tok[:, :self.n_att].mean(dim=1), tok[:, self.sus_start:].mean(dim=1), self.asp(tok)], dim=-1)
        else:
            q = self.queries.unsqueeze(0).expand(tok.shape[0], -1, -1)
            out, _ = self.attn(q, tok, tok)
            pooled = out.flatten(1)
        return self.proj(pooled)


class FiLMBlock(nn.Module):
    """gamma, beta from the audio embedding through a low-rank bottleneck; zero-initialised (identity at start)."""

    def __init__(self, dim: int, rank: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, rank), nn.GELU(), nn.Linear(rank, 2 * dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h, cond):
        g, b = self.net(cond).chunk(2, dim=-1)
        return h * (1.0 + g) + b


ENCODERS = ("conv", "temporal_asp", "temporal_segments", "temporal_queries")


def make_encoder_v4(kind: str, embed_dim: int, n_mels: int = 128, **kw) -> nn.Module:
    if kind == "conv":
        return AudioEncoder(embed_dim=embed_dim)
    if kind in ("temporal_asp", "temporal_segments", "temporal_queries"):
        return TemporalAudioEncoderV4(embed_dim=embed_dim, in_mels=n_mels, pool=kind.split("_", 1)[1], **kw)
    raise ValueError(f"encoder must be one of {ENCODERS}, got {kind!r}")


class FlowMatchingResMLPV4(FlowMatchingResMLP):
    def __init__(self, param_dim=None, hidden_dim=256, num_layers=4, encoding=codec.ENCODING_V2, time_scale=1000.0,
                 sigma_min=1e-5, cond_noise=False, n_mels=128, encoder: str = "temporal_asp", film: bool = False,
                 film_rank: int = 128, enc_kwargs=None):
        super().__init__(param_dim=param_dim, hidden_dim=hidden_dim, num_layers=num_layers, encoding=encoding,
                         time_scale=time_scale, sigma_min=sigma_min, cond_noise=cond_noise, n_mels=n_mels)
        self.v4_config = dict(encoder=encoder, film=bool(film), film_rank=int(film_rank), enc_kwargs=dict(enc_kwargs or {}))
        self.encoder = make_encoder_v4(encoder, hidden_dim, n_mels, **(enc_kwargs or {}))
        self.film = nn.ModuleList([FiLMBlock(hidden_dim, film_rank) for _ in range(num_layers)]) if film else None

    def forward(self, x_t, t, mel=None, audio_emb=None, tau=None):
        if audio_emb is None:
            audio_emb = self.encoder(mel)
        h = audio_emb + self.time_embed(t) + self.param_proj(x_t)
        if self.cond_noise:
            if tau is None:
                tau = torch.ones(x_t.shape[0], device=x_t.device)
            h = h + self.tau_embed(tau)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.film is not None:
                h = self.film[i](h, audio_emb)
        h = self.norm(h)
        return self.head(h)


def build_model_v4(model_type="flow", hidden_dim=512, num_layers=8, encoding=codec.ENCODING_V2, time_scale=1000.0,
                   cond_noise=False, encoder="temporal_asp", film=False, film_rank=128, enc_kwargs=None):
    """Same signature as `models_v3.build_model` plus the encoder choice. `encoder="conv"` with film=False returns the
    v3 flow unchanged."""
    if model_type != "flow":
        raise ValueError("build_model_v4 builds the flow model only")
    if encoder == "conv" and not film:
        return FlowMatchingResMLP(hidden_dim=hidden_dim, num_layers=num_layers, encoding=encoding,
                                  time_scale=time_scale, cond_noise=cond_noise)
    return FlowMatchingResMLPV4(hidden_dim=hidden_dim, num_layers=num_layers, encoding=encoding, time_scale=time_scale,
                                cond_noise=cond_noise, encoder=encoder, film=film, film_rank=film_rank, enc_kwargs=enc_kwargs)


def load_inverter_v4(ckpt_path, device="cpu", v4_config=None):
    """Rebuild a v3 or v4 flow from a flow_*.pt written by the trainer ({'model_state', 'args', ...}).

    The encoder kind is inferred from the state-dict keys; pass `v4_config` (the dict `model.v4_config` holds) to
    override, which is needed only for non-default encoder kwargs (heads, channels, strides)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if "encoder.tok_proj.weight" not in state:
        from models_v3 import load_inverter
        return load_inverter(ckpt_path, device=device)
    hidden = state["head.weight"].shape[1]
    num_layers = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    cfg = dict(v4_config or {})
    if not cfg:
        d_model = state["encoder.tok_proj.weight"].shape[0]
        n_in = state["encoder.proj.weight"].shape[1]
        if "encoder.queries" in state:
            kind = "temporal_queries"
        elif n_in == 4 * d_model:
            kind = "temporal_segments"
        else:
            kind = "temporal_asp"
        n_layers = len({k.split(".")[3] for k in state if k.startswith("encoder.transformer.layers.")})
        cfg = dict(encoder=kind, film=any(k.startswith("film.") for k in state), film_rank=128,
                   enc_kwargs=dict(d_model=d_model, n_layers=n_layers))
    model = FlowMatchingResMLPV4(hidden_dim=hidden, num_layers=num_layers, cond_noise="tau_embed.1.weight" in state, **cfg)
    model.load_state_dict(state)
    return model.to(device).eval()


def param_counts(model: nn.Module) -> dict:
    enc = sum(p.numel() for p in model.encoder.parameters())
    tot = sum(p.numel() for p in model.parameters())
    return {"encoder": enc, "total": tot}
