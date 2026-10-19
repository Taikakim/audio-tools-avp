# Owner: KUANG (created 2026-10-19)
"""Note-conditioned flow for the v6 layout (V6_SPECIFICATION.md section 3.1): the normalised MIDI note is an
input, not a predicted parameter.

The note enters through the existing parameter projection: param_proj sees [x_t | sin(2^j pi n) | cos(2^j pi n)]
(j = 0..note_features-1), so the network, the audio encoders, cross-attention and FiLM are those of models_v4.
Every call that runs the network passes note=[B] (normalised (midi - 28) / 22); calling without it raises.

  flow = build_model_v6(param_dim=param_codec_v6.ENCODED_DIM, hidden_dim=512, num_layers=8, encoder="temporal_asp")
  v = flow(x_t, t, mel, note=note)            # velocity
  x1 = flow.sample(mel, note, num_steps=25)   # encoded draw, clamped to [0, 1]
  load_inverter_v4 hands checkpoints with a note projection to load_inverter_v6, so every tool that loads a flow
  through models_v4 also loads v6 flows.
"""
import math

import torch
import torch.nn as nn

from models_v4 import FlowMatchingResMLPV4
import param_codec_v6 as codec


class NoteConditionedFlow(FlowMatchingResMLPV4):
    note_conditioned = True

    def __init__(self, *args, note_features=8, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = self.param_proj.out_features
        self.note_features = int(note_features)
        self.param_proj = nn.Linear(self.param_dim + 2 * self.note_features, hidden)
        self.register_buffer("note_freqs", math.pi * 2.0 ** torch.arange(self.note_features, dtype=torch.float32))
        self.v4_config = dict(self.v4_config, note_features=self.note_features)

    def _with_note(self, x_t, note):
        if note is None:
            raise ValueError("NoteConditionedFlow needs note=[B] (normalised MIDI note)")
        n = torch.as_tensor(note, dtype=x_t.dtype, device=x_t.device).reshape(-1)
        if n.numel() == 1:
            n = n.expand(x_t.shape[0])
        f = n[:, None] * self.note_freqs.to(x_t.dtype)
        return torch.cat([x_t, torch.sin(f), torch.cos(f)], dim=1)

    def forward(self, x_t, t, mel=None, audio_emb=None, tau=None, note=None):
        return super().forward(self._with_note(x_t, note), t, mel=mel, audio_emb=audio_emb, tau=tau)

    @torch.no_grad()
    def attention_maps(self, x_t, t, mel, tau=None, note=None):
        return super().attention_maps(self._with_note(x_t, note), t, mel, tau=tau)

    @torch.no_grad()
    def sample(self, mel, note, num_steps=20, generator=None, tau=None):
        """Euler ODE solve from noise -> encoded x_1 clamped to [0, 1] (decode with param_codec_v6.decode(x, note))."""
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
        audio_emb = self.encoder(mel)
        x = torch.randn(B, self.param_dim, device=mel.device, generator=generator)
        for i in range(num_steps):
            t = torch.full((B,), i / num_steps, device=mel.device, dtype=torch.float32)
            x = x + self(x, t, audio_emb=audio_emb, tau=tau_t, note=note) * (1.0 / num_steps)
        return torch.clamp(x, 0.0, 1.0)


def build_model_v6(model_type="flow", param_dim=None, hidden_dim=512, num_layers=8, encoding=codec.ENCODING_V2,
                   time_scale=1000.0, cond_noise=False, encoder="temporal_asp", film=False, film_rank=128,
                   enc_kwargs=None, xattn=False, xattn_dim=None, xattn_heads=4, note_features=8):
    if model_type != "flow":
        raise ValueError("build_model_v6 builds the flow model only")
    return NoteConditionedFlow(param_dim=param_dim or codec.ENCODED_DIM, hidden_dim=hidden_dim, num_layers=num_layers,
                               encoding=encoding, time_scale=time_scale, cond_noise=cond_noise, encoder=encoder,
                               film=film, film_rank=film_rank, enc_kwargs=enc_kwargs, xattn=xattn, xattn_dim=xattn_dim,
                               xattn_heads=xattn_heads, note_features=note_features)


def is_v6_state(state):
    return "note_freqs" in state


def load_inverter_v6(ckpt_path, device="cpu", v4_config=None):
    """Rebuild a NoteConditionedFlow from a flow_*.pt (the trainer writes args['v4_config'], note_features included)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    cfg = dict(v4_config or (ckpt.get("args") or {}).get("v4_config") or {})
    if not cfg:
        raise ValueError(f"{ckpt_path}: a v6 flow checkpoint needs args['v4_config'] (written by train_realistic_bass_v6)")
    cfg.setdefault("note_features", int(state["note_freqs"].numel()))
    param_dim = state["head.weight"].shape[0]
    hidden = state["head.weight"].shape[1]
    num_layers = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    model = NoteConditionedFlow(param_dim=param_dim, hidden_dim=hidden, num_layers=num_layers,
                                cond_noise="tau_embed.1.weight" in state, **cfg)
    model.load_state_dict(state)
    return model.to(device).eval()


def flow_call(flow, x, t, audio_emb=None, mel=None, tau=None, note=None):
    """Run any flow: note-conditioned ones get the note, older ones do not take it."""
    if getattr(flow, "note_conditioned", False):
        return flow(x, t, mel=mel, audio_emb=audio_emb, tau=tau, note=note)
    return flow(x, t, mel=mel, audio_emb=audio_emb, tau=tau)
