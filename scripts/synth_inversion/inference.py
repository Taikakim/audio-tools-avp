"""Model -> Surge patch(es), shared by the evaluation scripts."""
import time

import numpy as np
import torch

from models import FlowMatchingResMLP, load_inverter  # noqa: F401  (load_inverter re-exported)
from surge_spec import vector_to_patch


def parse_ckpt_specs(specs):
    """['name=/path/a.pt', ...] -> [(name, path), ...]"""
    out = []
    for s in specs:
        if "=" not in s:
            raise ValueError(f"--ckpt expects NAME=PATH, got {s!r}")
        name, path = s.split("=", 1)
        out.append((name, path))
    return out


def is_flow(model) -> bool:
    return isinstance(model, FlowMatchingResMLP)


@torch.no_grad()
def predict_vectors(model, mel, n_draws=1, steps=20, seed=0):
    """mel [B, 128, 81] tensor -> np.array [n_draws, B, 23] (point models: n_draws = 1).
    Returns (vectors, latency_ms per input for one draw)."""
    device = next(model.parameters()).device
    mel = mel.to(device)
    t0 = time.time()
    if not is_flow(model):
        vecs = model.predict_params(mel).unsqueeze(0)
    else:
        gen = torch.Generator(device=device).manual_seed(seed)
        vecs = torch.stack([model.predict_params(mel, num_steps=steps, generator=gen) for _ in range(n_draws)])
    lat_ms = (time.time() - t0) * 1000.0 / max(1, vecs.shape[0]) / mel.shape[0]
    return vecs.cpu().numpy(), lat_ms


def vectors_to_patches(vecs_one_input):
    """[n_draws, 23] -> list of patch dicts."""
    return [vector_to_patch(v) for v in np.atleast_2d(vecs_one_input)]
