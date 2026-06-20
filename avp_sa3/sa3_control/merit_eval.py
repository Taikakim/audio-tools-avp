"""MERIT disentangled music similarity for control-adapter eval.

github.com/AMAAI-Lab/MERIT — frozen MERT-330M + 3 pre-trained projection heads give
three *independent* cosine similarities per audio pair:
    S_mel (melody) · S_rhy (rhythm) · S_tim (timbre)

This is the metric the cross-ref-diff / chroma checks couldn't give us: it says *which*
factor matched. Load once (`MeritScorer`), score many. CPU is fine (scoring is not on the
hot path); pass device="cuda" if the GPU is free.

Heads cache under MERIT/models/ (downloaded on first use). MERT-330M downloads ~1.3 GB once.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MODEL_ID = "m-a-p/MERT-v1-330M"
EXTRACT_LAYERS = (3, 4, 5, 6, 23)        # MERIT's fixed layer set → 5×1024 = 5120-dim backbone
SR = 24_000
FACTORS = ("mel", "rhy", "tim")
_MODELS_DIR = "/home/kim/Projects/MERIT/models"


class _Head(nn.Module):
    """MERIT projection head: Linear→ReLU→Linear→L2-norm (matches the published checkpoints)."""
    def __init__(self, in_dim=5120, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class MeritScorer:
    """Frozen MERT-330M + the 3 MERIT heads. `score(a, sr_a, b, sr_b)` → {mel,rhy,tim} cosines."""

    def __init__(self, device: str = "cpu", models_dir: str = _MODELS_DIR):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.device = device
        self.proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.mert = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(device).eval()
        self.heads = {}
        for f in FACTORS:
            p = hf_hub_download("amaai-lab/merit", f"head_{f}/best_head.pt", local_dir=models_dir)
            ck = torch.load(p, map_location=device, weights_only=True)
            h = _Head(ck["in_dim"], ck["hidden_dim"], ck["out_dim"])
            h.load_state_dict(ck["state_dict"])
            self.heads[f] = h.to(device).eval()

    @torch.no_grad()
    def embed(self, wav, sr: int) -> dict:
        """wav: np.ndarray or torch.Tensor, mono (T,) or (C,T). Returns {factor: (1,128) unit vec}."""
        import torchaudio
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        wav = wav.float()
        if wav.ndim == 2:
            wav = wav.mean(0)
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        wav = wav[: SR * 30]                                   # MERIT truncates to 30 s
        inp = self.proc(wav.cpu().numpy(), sampling_rate=SR, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.mert(**inp, output_hidden_states=True)
        bb = torch.cat([out.hidden_states[l].mean(1) for l in EXTRACT_LAYERS], dim=-1)  # (1,5120)
        return {f: self.heads[f](bb) for f in FACTORS}

    def score(self, wav_a, sr_a: int, wav_b, sr_b: int) -> dict:
        """Per-factor cosine similarity in [-1,1]: {'S_mel':_, 'S_rhy':_, 'S_tim':_}."""
        ea, eb = self.embed(wav_a, sr_a), self.embed(wav_b, sr_b)
        return {f"S_{f}": float((ea[f] * eb[f]).sum().item()) for f in FACTORS}


if __name__ == "__main__":   # tiny self-check on noise (cosine of identical clips ≈ 1.0)
    s = MeritScorer(device="cpu")
    w = np.random.randn(SR * 5).astype(np.float32) * 0.1
    print("self-sim (≈1.0):", s.score(w, SR, w, SR))
    print("noise-vs-noise:", s.score(w, SR, np.random.randn(SR * 5).astype(np.float32) * 0.1, SR))
