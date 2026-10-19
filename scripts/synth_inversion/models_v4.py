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
      xattn=True (v5, 2026-10-13)    per-block CROSS-ATTENTION from the flow state to the encoder's time tokens: after
                                     every ResBlock the hidden state (which already carries x_t, t and tau) forms one
                                     query per head over the 41 tokens, so different blocks and different ODE times can
                                     read different parts of the note (attack for envelope amounts, tail for sustain).
                                     The output projection is zero-initialised: at step 0 the model equals the same
                                     flow without cross-attention. Needs a temporal_* encoder.

Everything else (cond_noise, OT coupling, `sample`, `path`, the loss) is inherited unchanged, so `flow_loss` and the
inference code work as they are. A state dict is NOT compatible with a v3 flow (different encoder keys): start a new
out_dir. `load_inverter_v4` rebuilds either kind from a flow_best.pt / flow_latest.pt.

How xattn stays compatible with callers that do `audio_emb = model.encoder(mel)` and pass it back as
`model(x, t, audio_emb=audio_emb)` (models_v3.sample, inference_v3.predict_vectors_pinned): with xattn the encoder
returns ONE packed tensor [B, hidden + n_tokens * d_model] (pooled embedding, then the flattened tokens), and the
forward unpacks it. A batch-1 condition broadcasts over a batch of draws, as before.

Status: unit-tested on CPU (shapes, gradients, FiLM and cross-attention identity at init, cond_noise, sampling,
save/load); NOT trained on Surge data. The toy probe (probe_encoders_toy_v4.py) is the only learning evidence for the
temporal encoders, on a stand-in synth; there is none yet for cross-attention.
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
                 sus_start_frac: float = 0.25, pack_tokens: bool = False):
        super().__init__()
        self.pack_tokens = bool(pack_tokens)
        self.embed_dim = embed_dim
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
        emb = self.proj(pooled)
        return torch.cat([emb, tok.flatten(1)], dim=1) if self.pack_tokens else emb

    def unpack(self, packed):
        """[B, embed_dim + n_tokens * d_model] -> (emb [B, embed_dim], tokens [B, n_tokens, d_model])."""
        emb = packed[:, :self.embed_dim]
        return emb, packed[:, self.embed_dim:].reshape(packed.shape[0], self.n_tokens, self.d_model)


class CrossAttnBlock(nn.Module):
    """h [B, dim] reads tokens [B, T, ctx_dim]: one query per head from LayerNorm(h), keys and values from the tokens,
    residual output through a zero-initialised projection (identity at init). attn_dim is the inner width."""

    def __init__(self, dim: int, ctx_dim: int, attn_dim: int = 256, n_heads: int = 4):
        super().__init__()
        if attn_dim % n_heads:
            raise ValueError(f"attn_dim {attn_dim} is not divisible by n_heads {n_heads}")
        self.n_heads, self.attn_dim = n_heads, attn_dim
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, attn_dim)
        self.kv = nn.Linear(ctx_dim, 2 * attn_dim)
        self.out = nn.Linear(attn_dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def attend(self, h, tok):
        B, H = h.shape[0], self.n_heads
        if tok.shape[0] != B:
            if tok.shape[0] != 1:
                raise ValueError(f"token batch {tok.shape[0]} does not match {B}")
            tok = tok.expand(B, -1, -1)
        T, dh = tok.shape[1], self.attn_dim // H
        q = self.q(self.norm(h)).view(B, H, 1, dh)
        k, v = self.kv(tok).chunk(2, dim=-1)
        k = k.reshape(B, T, H, dh).transpose(1, 2)
        v = v.reshape(B, T, H, dh).transpose(1, 2)
        w = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(dh), dim=-1)    # [B, H, 1, T]
        return (w @ v).reshape(B, self.attn_dim), w.squeeze(2)

    def forward(self, h, tok):
        o, _ = self.attend(h, tok)
        return h + self.out(o)


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
                 film_rank: int = 128, enc_kwargs=None, xattn: bool = False, xattn_dim=None, xattn_heads: int = 4):
        super().__init__(param_dim=param_dim, hidden_dim=hidden_dim, num_layers=num_layers, encoding=encoding,
                         time_scale=time_scale, sigma_min=sigma_min, cond_noise=cond_noise, n_mels=n_mels)
        if xattn and not encoder.startswith("temporal_"):
            raise ValueError("xattn needs a temporal_* encoder (the conv encoder has no time tokens)")
        enc_kwargs = dict(enc_kwargs or {})
        self.encoder = make_encoder_v4(encoder, hidden_dim, n_mels, **enc_kwargs)
        self.film = nn.ModuleList([FiLMBlock(hidden_dim, film_rank) for _ in range(num_layers)]) if film else None
        self.xattn = None
        if xattn:
            self.encoder.pack_tokens = True
            ctx = self.encoder.d_model
            xattn_dim = int(xattn_dim or ctx)
            self.xattn = nn.ModuleList([CrossAttnBlock(hidden_dim, ctx, xattn_dim, xattn_heads) for _ in range(num_layers)])
        self.v4_config = dict(encoder=encoder, film=bool(film), film_rank=int(film_rank), enc_kwargs=enc_kwargs,
                              xattn=bool(xattn), xattn_dim=int(xattn_dim) if xattn else None, xattn_heads=int(xattn_heads))

    def _split(self, audio_emb):
        if self.xattn is None:
            return audio_emb, None
        return self.encoder.unpack(audio_emb)

    def forward(self, x_t, t, mel=None, audio_emb=None, tau=None):
        if audio_emb is None:
            audio_emb = self.encoder(mel)
        emb, tok = self._split(audio_emb)
        h = emb + self.time_embed(t) + self.param_proj(x_t)
        if self.cond_noise:
            if tau is None:
                tau = torch.ones(x_t.shape[0], device=x_t.device)
            h = h + self.tau_embed(tau)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.xattn is not None:
                h = self.xattn[i](h, tok)
            if self.film is not None:
                h = self.film[i](h, emb)
        h = self.norm(h)
        return self.head(h)

    @torch.no_grad()
    def attention_maps(self, x_t, t, mel, tau=None):
        """Diagnostics: per block, the cross-attention weights over time tokens [B, heads, T] for one (x_t, t)."""
        if self.xattn is None:
            raise ValueError("model has no cross-attention")
        emb, tok = self._split(self.encoder(mel))
        h = emb + self.time_embed(t) + self.param_proj(x_t)
        if self.cond_noise:
            h = h + self.tau_embed(tau if tau is not None else torch.ones(x_t.shape[0], device=x_t.device))
        maps = []
        for i, block in enumerate(self.blocks):
            h = block(h)
            o, w = self.xattn[i].attend(h, tok)
            maps.append(w)
            h = h + self.xattn[i].out(o)
            if self.film is not None:
                h = self.film[i](h, emb)
        return maps


def build_model_v4(model_type="flow", param_dim=None, hidden_dim=512, num_layers=8, encoding=codec.ENCODING_V2, time_scale=1000.0,
                   cond_noise=False, encoder="temporal_asp", film=False, film_rank=128, enc_kwargs=None,
                   xattn=False, xattn_dim=None, xattn_heads=4):
    """Same signature as `models_v3.build_model` plus the encoder choice. `encoder="conv"` with film=False returns the
    v3 flow unchanged."""
    if model_type != "flow":
        raise ValueError("build_model_v4 builds the flow model only")
    if encoder == "conv" and not film and not xattn:
        return FlowMatchingResMLP(param_dim=param_dim, hidden_dim=hidden_dim, num_layers=num_layers, encoding=encoding,
                                  time_scale=time_scale, cond_noise=cond_noise)
    return FlowMatchingResMLPV4(param_dim=param_dim, hidden_dim=hidden_dim, num_layers=num_layers, encoding=encoding, time_scale=time_scale,
                                cond_noise=cond_noise, encoder=encoder, film=film, film_rank=film_rank, enc_kwargs=enc_kwargs,
                                xattn=xattn, xattn_dim=xattn_dim, xattn_heads=xattn_heads)


def load_inverter_v4(ckpt_path, device="cpu", v4_config=None):
    """Rebuild a v3 or v4 flow from a flow_*.pt written by the trainer ({'model_state', 'args', ...}).

    The config comes from `v4_config` if given, else from the checkpoint's args['v4_config'] (written by the trainer
    since v5), else it is inferred from the state-dict keys (encoder kind, FiLM, cross-attention width; the number of
    cross-attention heads cannot be inferred and defaults to 4)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if "encoder.tok_proj.weight" not in state:
        from models_v3 import load_inverter
        return load_inverter(ckpt_path, device=device)
    param_dim = state["head.weight"].shape[0]
    hidden = state["head.weight"].shape[1]
    num_layers = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    cfg = dict(v4_config or (ckpt.get("args") or {}).get("v4_config") or {})
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
        has_x = "xattn.0.q.weight" in state
        cfg = dict(encoder=kind, film=any(k.startswith("film.") for k in state), film_rank=128,
                   enc_kwargs=dict(d_model=d_model, n_layers=n_layers), xattn=has_x,
                   xattn_dim=state["xattn.0.q.weight"].shape[0] if has_x else None, xattn_heads=4)
    model = FlowMatchingResMLPV4(param_dim=param_dim, hidden_dim=hidden, num_layers=num_layers, cond_noise="tau_embed.1.weight" in state, **cfg)
    model.load_state_dict(state)
    return model.to(device).eval()


def param_counts(model: nn.Module) -> dict:
    enc = sum(p.numel() for p in model.encoder.parameters())
    tot = sum(p.numel() for p in model.parameters())
    return {"encoder": enc, "total": tot}
