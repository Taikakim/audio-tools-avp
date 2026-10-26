# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Unit tests for CueGatedTemporalAudioEncoder."""

import torch
import torch.nn.functional as F

from models_v4 import load_inverter_v4
from cue_gated_temporal_asp import CueGatedTemporalAudioEncoder
import param_codec_v6 as codec


def test_cue_gated_asp_shapes_and_compat():
    flow = load_inverter_v4('/home/kim/sao_runs/v6_C/flow_best.pt', device='cpu')
    base_encoder = flow.encoder

    gated_encoder = CueGatedTemporalAudioEncoder(base_encoder, cue_dim=512, proj_dim=128)

    B = 2
    mel = torch.randn(B, 128, 81)
    cue = torch.randn(B, 512)

    # 1. Forward shapes & gates
    out, gates = gated_encoder(mel, cue, return_gates=True)
    expected_dim = 512 + (41 * 256 if gated_encoder.pack_tokens else 0)
    assert out.shape == (B, expected_dim), f"Expected shape ({B}, {expected_dim}), got {out.shape}"
    assert gates.shape == (B, 41), f"Expected gate shape ({B}, 41), got {gates.shape}"
    assert (gates >= 0.0).all() and (gates <= 1.0).all(), "Gates must be in [0, 1]"

    # 2. Zero-init check: initially z (the first 512 elements) must equal cue
    z = out[:, :512]
    diff = (z - cue).abs().max().item()
    assert diff < 1e-6, f"At step 0, z must equal cue, diff was {diff}"

    # 3. Direct compatibility with V6 C flow model!
    note = torch.tensor([0.45, 0.70])
    x_t = torch.randn(B, codec.ENCODED_DIM)
    t = torch.tensor([0.3, 0.8])
    v = flow(x_t, t, audio_emb=out, note=note)
    assert v.shape == (B, codec.ENCODED_DIM), f"Expected velocity shape ({B}, {codec.ENCODED_DIM}), got {v.shape}"

    # 4. Test sampling with V6 C flow model conditioned on cue-gated output
    gen = torch.Generator(device='cpu').manual_seed(42)
    # Replicate sample with audio_emb
    x = torch.randn(B, flow.param_dim, generator=gen)
    num_steps = 5
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_step = torch.full((B,), i / num_steps, dtype=torch.float32)
        vel = flow(x, t_step, audio_emb=out, note=note)
        x = x + vel * dt
    sampled_patch = torch.clamp(x, 0.0, 1.0)
    assert sampled_patch.shape == (B, codec.ENCODED_DIM)

    decoded = codec.decode(sampled_patch, note)
    assert decoded.shape == (B, 59)

    print("test_cue_gated_asp_shapes_and_compat: PASS")


if __name__ == "__main__":
    test_cue_gated_asp_shapes_and_compat()
    print("All Cue-Gated ASP unit tests PASSED successfully!")
