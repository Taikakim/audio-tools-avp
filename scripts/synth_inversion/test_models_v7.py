# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Unit tests for models_v7.py (MixtureConditionedFlowV7)."""

import torch
import param_codec_v6 as codec
from surge_spec_v6 import NUM_PARAMS, PARAM_INDEX
from models_v7 import build_model_v7


def test_v7_flow_shapes():
    B = 2
    param_dim = codec.ENCODED_DIM  # 114
    timbre_dim = 512

    model = build_model_v7(
        param_dim=param_dim,
        hidden_dim=256,
        num_layers=4,
        timbre_dim=timbre_dim,
        note_features=8,
    )

    x_t = torch.randn(B, param_dim)
    t = torch.tensor([0.2, 0.7])
    z = torch.randn(B, timbre_dim)
    note = torch.tensor([0.35, 0.80])

    # 1. Forward velocity prediction
    v = model(x_t, t, z=z, note=note)
    assert v.shape == (B, param_dim), f"Expected shape ({B}, {param_dim}), got {v.shape}"

    # 2. ODE Sampling pass (5 steps)
    x1 = model.sample(z=z, note=note, num_steps=5)
    assert x1.shape == (B, param_dim), f"Expected shape ({B}, {param_dim}), got {x1.shape}"
    assert (x1 >= 0.0).all() and (x1 <= 1.0).all(), "Sampled parameters must be clamped to [0, 1]"

    # 3. Check parameter decoding to storage format [B, 59]
    decoded_storage = codec.decode(x1, note)
    assert decoded_storage.shape == (B, NUM_PARAMS), f"Expected shape ({B}, {NUM_PARAMS}), got {decoded_storage.shape}"
    assert (decoded_storage >= 0.0).all() and (decoded_storage <= 1.0).all(), "Decoded storage parameters must be in [0, 1]"

    # Check that note is correctly placed in storage
    assert torch.allclose(decoded_storage[:, PARAM_INDEX["midi_note"]], note), "Note condition was not preserved in storage"

    print("test_v7_flow_shapes: PASS")


if __name__ == "__main__":
    test_v7_flow_shapes()
    print("All V7 Flow unit tests PASSED successfully!")
