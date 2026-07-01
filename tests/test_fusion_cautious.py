"""Unit tests for FusionOpt cautious masking (C-Muon / C-AdamW).

Runnable directly: `sat-venv/bin/python tests/test_fusion_cautious.py`
(pytest is not installed in sat-venv; the __main__ block runs every test_*).

Cautious masking (Liang et al., "Cautious Optimizers", 2024): after the update
direction U is computed, zero the coordinates whose update disagrees with the
gradient (U*g <= 0 — moving along -U would *increase* loss there), then rescale
the survivors so their mean magnitude is preserved. One extra elementwise op,
no extra optimizer state. Targets the constant-magnitude wandering of LMO
updates (Muon/NS5) in the under-determined control-head landscape.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_audio_tools.training.fusion_opt import FusionOpt, apply_cautious


def test_cautious_is_a_valid_component():
    """FusionOpt must accept 'cautious' in its components set (was rejected before)."""
    w = torch.nn.Parameter(torch.randn(8, 8))
    groups = [{"params": [w], "group_type": "spectral"}]
    # Should not raise "unknown components".
    opt = FusionOpt(groups, lr=3e-4,
                    components={"ns5", "normuon", "sf", "cautious"},
                    hot_dtype="fp32")
    assert "cautious" in opt.components


def test_apply_cautious_all_agree_is_identity():
    """When every coord agrees with the gradient, the update is unchanged."""
    U = torch.tensor([1.0, 2.0, 3.0, 4.0])
    g = torch.tensor([0.5, 9.0, 0.1, 2.0])           # all same sign as U
    out = apply_cautious(U, g)
    assert torch.allclose(out, U, atol=1e-6), out


def test_apply_cautious_masks_disagreeing_and_preserves_mean():
    """Disagreeing coords are zeroed; survivors rescaled to preserve mean magnitude."""
    U = torch.tensor([1.0, -2.0, 3.0, 4.0])
    g = torch.tensor([1.0,  1.0, 1.0, 1.0])          # coord 1 (-2) disagrees
    out = apply_cautious(U, g)
    # disagreeing coord zeroed
    assert out[1].abs() < 1e-6, out
    # survivors scaled by 1/keep_frac = 1/0.75
    assert torch.allclose(out[[0, 2, 3]], U[[0, 2, 3]] / 0.75, atol=1e-5), out
    # mean magnitude preserved: sum(out) == sum(U over kept) / keep_frac == sum(U_kept)*4/3
    assert abs(float(out.sum()) - float(U[[0, 2, 3]].sum()) / 0.75) < 1e-4


def test_apply_cautious_all_disagree_is_zero():
    """If every coord disagrees, the update collapses to ~0 (no blow-up)."""
    U = torch.tensor([1.0, 2.0, 3.0])
    g = torch.tensor([-1.0, -1.0, -1.0])
    out = apply_cautious(U, g)
    assert torch.all(out.abs() < 1e-6), out


def test_cautious_alters_the_spectral_update():
    """End-to-end: with a gradient that disagrees with the orthogonalised update on
    some coords, a step WITH cautious must move the weight differently than without.
    Proves the mask is wired into the spectral path (not just a free function)."""
    torch.manual_seed(0)
    w0 = torch.randn(16, 16)
    g = torch.randn(16, 16)

    def one_step(components):
        w = torch.nn.Parameter(w0.clone())
        groups = [{"params": [w], "group_type": "spectral"}]
        opt = FusionOpt(groups, lr=1e-2, components=components, hot_dtype="fp32")
        opt.train()
        w.grad = g.clone()
        opt.set_loss(torch.tensor(1.0))
        opt.step()
        return opt.state[w]["z"].clone()

    base = one_step({"ns5", "normuon", "sf"})
    caut = one_step({"ns5", "normuon", "sf", "cautious"})
    assert not torch.allclose(base, caut, atol=1e-6), \
        "cautious masking did not change the spectral update — not wired in"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
