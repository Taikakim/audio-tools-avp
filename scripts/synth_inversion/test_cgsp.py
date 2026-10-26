# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Unit tests for CueGatedStatisticsPooling (CGSP) and LoRALinear."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cgsp_extractor import CueGatedStatisticsPooling, LoRALinear


def test_lora_linear():
    base_linear = nn.Linear(128, 64)
    lora = LoRALinear(base_linear, rank=8, alpha=16.0)

    x = torch.randn(4, 128)
    base_out = base_linear(x)
    lora_out = lora(x)

    # 1. Zero-init check: initial output must match frozen base linear exactly
    diff = (lora_out - base_out).abs().max().item()
    assert diff < 1e-6, f"LoRA initial output differs from base: {diff}"

    # 2. Gradient check: gradients must flow to LoRA parameters, not base linear
    loss = lora_out.sum()
    loss.backward()

    assert base_linear.weight.grad is None, "Base linear should not have gradients"
    assert lora.lora_A.weight.grad is not None, "LoRA A must have gradients"
    assert lora.lora_B.weight.grad is not None, "LoRA B must have gradients"
    print("test_lora_linear: PASS")


def test_cgsp_shapes_and_zero_init():
    B, T, Dh, Dq = 4, 128, 1024, 512
    cgsp = CueGatedStatisticsPooling(token_dim=Dh, cue_dim=Dq, proj_dim=256, hidden_dim=512)

    H = torch.randn(B, T, Dh)
    q = torch.randn(B, Dq)

    z, gates = cgsp(H, q, return_gates=True)

    # 1. Shape check
    assert z.shape == (B, Dq), f"Expected shape ({B}, {Dq}), got {z.shape}"
    assert gates.shape == (B, T), f"Expected gate shape ({B}, {T}), got {gates.shape}"

    # 2. Gate bounds check (sigmoid output in (0, 1))
    assert (gates >= 0.0).all() and (gates <= 1.0).all(), "Gates must be bounded in [0, 1]"

    # 3. L2 Normalization check
    norms = torch.norm(z, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"Norms not 1.0: {norms}"

    # 4. Zero-initialization check: at step 0, z must equal normalized q exactly
    q_norm = F.normalize(q, p=2, dim=-1)
    init_diff = (z - q_norm).abs().max().item()
    assert init_diff < 1e-6, f"At step 0, z must equal norm(q), diff was {init_diff}"

    # 5. Gradient check: ensure gradients flow backwards to H and q
    H.requires_grad_(True)
    q.requires_grad_(True)
    z = cgsp(H, q)
    loss = z.sum()
    loss.backward()

    assert H.grad is not None, "Gradients must flow to mixture tokens H"
    assert q.grad is not None, "Gradients must flow to cue embedding q"
    assert cgsp.W_h.weight.grad is not None, "Gradients must flow to W_h"
    assert cgsp.W_q.weight.grad is not None, "Gradients must flow to W_q"
    assert cgsp.R_theta[0].weight.grad is not None, "Gradients must flow to R_theta"
    print("test_cgsp_shapes_and_zero_init: PASS")


if __name__ == "__main__":
    test_lora_linear()
    test_cgsp_shapes_and_zero_init()
    print("All CGSP unit tests PASSED successfully!")
