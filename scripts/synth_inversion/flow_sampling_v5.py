# Owner: KUANG (created 2026-10-13)
"""K independent flow candidates per target, with the MIDI note pinned along the ODE path (as
inference_v3.predict_vectors_pinned), batched over targets.

Difference from predict_vectors_pinned, on purpose: with tau < 1 (cond_noise models) every candidate gets ITS OWN
noise draw of the condition. predict_vectors_pinned noises the mel once and shares it across all candidates, so at
tau < 1 its candidates are not independent draws of the smoothed posterior (EVAL_PARADIGM_V4.md, finding P3).

  vecs = sample_candidates(flow, mel [B, 128, 81], midi [B], k=8, steps=25, seed=0, tau=1.0)   # -> [B, k, P]
Candidate j of target b uses the generator stream of one seeded torch.Generator, so a (seed, B, k) triple is
reproducible on one device.

The codec and note range come from the flow's own layout (layout_v6.layout_for_flow: 92 -> v3, 108 -> v4,
126 -> v5); the returned vectors are in that layout's ordinal storage, to be rendered with the same spec.
"""
import numpy as np
import torch

import param_codec_v3 as codec_v3
from layout_v6 import layout_for_dim


@torch.no_grad()
def sample_candidates(flow, mel, midi, k=8, steps=25, seed=0, tau=1.0, chunk=512):
    device = next(flow.parameters()).device
    was = flow.training
    flow.eval()
    mel = mel.to(device)
    B = mel.shape[0]
    midi = np.asarray(midi, dtype=np.float64).reshape(B)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    ordinal = getattr(flow, "encoding", codec_v3.ENCODING_V2) == codec_v3.ENCODING_V1
    lay = None if ordinal else layout_for_dim(flow.param_dim)
    lo, hi = lay.note_range if lay is not None else (28, 50)        # ordinal (v1) flows are v3-era: 28..50
    note = torch.from_numpy(np.clip((midi - lo) / (hi - lo), 0.0, 1.0)).float().to(device)
    mel_k = mel.repeat_interleave(k, dim=0)                      # [B * k, ...]: target b's candidates are rows b*k..
    note_k = note.repeat_interleave(k)
    out = []
    for s in range(0, B * k, chunk):
        m = mel_k[s:s + chunk]
        n = note_k[s:s + chunk]
        R = m.shape[0]
        tau_t = None
        if getattr(flow, "cond_noise", False):
            tau_t = torch.full((R,), float(tau), device=device)
            if float(tau) < 1.0:
                m = flow.noisy(m, tau_t, generator=gen)          # an independent noise draw per candidate
        elif float(tau) != 1.0:
            raise ValueError("tau < 1 needs a model trained with cond_noise")
        emb = flow.encoder(m)
        x = torch.randn(R, flow.param_dim, device=device, generator=gen)
        for i in range(steps):
            tv = i / steps
            t = torch.full((R,), tv, device=device)
            x[:, 0] = (1.0 - tv) * x[:, 0] + tv * n
            x = x + flow(x, t, audio_emb=emb, tau=tau_t) * (1.0 / steps)
        x[:, 0] = n
        x = torch.clamp(x, 0.0, 1.0)
        out.append(x if ordinal else lay.codec.decode(x))
    flow.train(was)
    return torch.cat(out).cpu().numpy().reshape(B, k, -1).astype(np.float32)


def mel_of(audio, device, chunk=128):
    """Model mel [n, 128, 81] (ExactGpuMel, as the trainer) of float audio [n, 35280]."""
    from audio_utils import ExactGpuMel
    fn = ExactGpuMel(device=device)
    a = torch.as_tensor(np.asarray(audio, dtype=np.float32))
    with torch.no_grad():
        return torch.cat([fn(a[s:s + chunk].to(device)).float().cpu() for s in range(0, len(a), chunk)])


def file_md5(path, block=1 << 20):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()
