"""Numeric tests for the full-FT regularization A/B levers (CONTINUITY spec 2026-08-10,
docs/superpowers/specs/2026-08-10-fullft-regularization-ab.md).

Two new SA3-training levers bound the FusionOpt full-FT latent-scale runaway more gently than
broad weight decay:
  (1) AGC — scale-relative per-output-unit gradient clipping (NFNets/Brock), in
      stable_audio_3/training/diffusion.py::agc_clip_grads_.
  (2) output-std penalty — penalize the reconstructed clean latent z0_hat's per-channel std
      drifting off the real data's, inlined in training_step (uses the RF z0_hat reconstruction,
      stable_audio_3/training/stereo_loss.py::rf_z0_hat).

This pins the MATH, no GPU, in seconds — a bug here wastes a multi-hour LUMI run. We import the
REAL agc_clip_grads_ and rf_z0_hat; the output-std penalty is inlined in training_step so its
std-MSE core is replicated here (marked) byte-for-byte against that block.

Runnable directly (pytest is not installed in sat-venv):
    sat-venv/bin/python tests/test_agc_and_outputstd.py

Imports the SA3 package (sibling repo). FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE is set before
torch import per MASTER §5 (the import prints a harmless "flash_attn not installed" note under
sat-venv, which lacks the CK ext — irrelevant to this CPU math test).
"""
import os
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
import sys
from pathlib import Path

# SA3 lives in the sibling repo: SAO/stable-audio-3 (parents[2] == SAO).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stable-audio-3"))

import torch

from stable_audio_3.training.diffusion import agc_clip_grads_
from stable_audio_3.training.stereo_loss import rf_z0_hat


def _unit_norm(x):
    """Per-output-unit (dim 0) L2 norm, keepdim — mirrors agc_clip_grads_'s reduction."""
    dims = tuple(range(1, x.ndim))
    return x.float().pow(2).sum(dim=dims, keepdim=True).sqrt()


# ---------------------------------------------------------------------------
# (a) AGC: large-weight/small-grad ~unchanged; small-weight/large-grad scaled by the right coef;
#     1D params skipped.
# ---------------------------------------------------------------------------
def test_agc_leaves_small_grad_large_weight_unchanged():
    w = torch.nn.Parameter(torch.full((4, 8), 10.0))     # large weight norm
    w.grad = torch.full((4, 8), 1e-4)                     # tiny gradient
    g0 = w.grad.clone()
    agc_clip_grads_([w], agc_lambda=0.01)
    # coef = min(1, 0.01 * ||w|| / (||g||+1e-6)); here ||w||/||g|| ~ 1e5 => coef == 1 => untouched.
    assert torch.allclose(w.grad, g0, atol=0, rtol=0), \
        f"expected unchanged grad, max delta {(w.grad - g0).abs().max():.3e}"


def test_agc_scales_large_grad_small_weight_by_right_coef():
    w = torch.nn.Parameter(torch.full((4, 8), 0.01))     # small weight norm
    w.grad = torch.full((4, 8), 1.0)                     # large gradient
    g0 = w.grad.clone()
    lam = 0.01
    pn = _unit_norm(w.detach()).clamp_min(1e-3)          # (4,1)
    gn = _unit_norm(g0)                                  # (4,1)
    expected_coef = (lam * pn / (gn + 1e-6)).clamp_max(1.0)   # per-row
    expected = g0 * expected_coef
    agc_clip_grads_([w], agc_lambda=lam)
    # coef ~ lam * (pn/gn) = 0.01 * 0.01 = 1e-4 (scaled DOWN, well below 1).
    assert float(expected_coef.max()) < 1.0, "test setup: coef should be < 1 (grad reined in)"
    assert torch.allclose(w.grad, expected, atol=1e-8), \
        f"AGC scaling mismatch, max delta {(w.grad - expected).abs().max():.3e}"


def test_agc_skips_1d_params():
    b = torch.nn.Parameter(torch.full((8,), 0.001))      # 1D (bias/norm) — must be skipped
    b.grad = torch.full((8,), 5.0)
    g0 = b.grad.clone()
    agc_clip_grads_([b], agc_lambda=0.01)
    assert torch.equal(b.grad, g0), "1D param grad must be left untouched by AGC"


# ---------------------------------------------------------------------------
# (b) RF z0 reconstruction round-trips: with the TRUE velocity, z0_hat = z_t - t*v == z0.
#     RF convention (diffusion.py training_step): z_t = z0*(1-t) + eps*t, v = eps - z0.
# ---------------------------------------------------------------------------
def test_z0_reconstruction_roundtrips():
    torch.manual_seed(0)
    z0 = torch.randn(3, 16, 32)
    eps = torch.randn(3, 16, 32)
    worst = 0.0
    for tval in (0.02, 0.1, 0.3, 0.5, 0.8, 0.99):
        t = torch.full((3,), tval)
        tb = t.view(-1, 1, 1)
        z_t = z0 * (1 - tb) + eps * tb          # forward interpolant
        v = eps - z0                            # TRUE velocity target
        z0_hat = rf_z0_hat(z_t, v, t)           # real helper: z_t - t*v
        z0_hat_inline = z_t - tb * v            # mirrors training_step's inlined z0_hat
        err = (z0_hat - z0).abs().max().item()
        worst = max(worst, err)
        assert err < 1e-4, f"z0 round-trip err {err:.3e} at t={tval} exceeds 1e-4"
        assert torch.allclose(z0_hat, z0_hat_inline, atol=0), "rf_z0_hat != inlined formula"
    print(f"  [z0 round-trip] worst |z0_hat - z0| over t = {worst:.2e}")


# ---------------------------------------------------------------------------
# (c) Output-std penalty ~0 when z0_hat matches the data's per-channel std, grows when it drifts.
#     Replicates training_step's inlined std-MSE (per-channel std over (batch,time), t-gated).
# ---------------------------------------------------------------------------
def _out_std_loss(z0_hat, z0_data, t, t_gate=0.5):
    """Byte-for-byte mirror of the training_step output-std-penalty block."""
    gate = t < t_gate
    if not bool(gate.any()):
        return z0_hat.new_zeros(())
    zc = z0_hat[gate].float()
    zr = z0_data[gate].detach().float()
    std_hat = zc.std(dim=(0, 2))                 # (C,) per-channel std over batch+time
    std_real = zr.std(dim=(0, 2))
    return (std_hat - std_real).pow(2).mean()


def _data_with_channel_stds(B=8, C=16, T=64, seed=0):
    torch.manual_seed(seed)
    per_chan = torch.linspace(0.5, 2.0, C).view(1, C, 1)   # deliberately varied per-channel scale
    return torch.randn(B, C, T) * per_chan


def test_output_std_penalty_zero_when_matched():
    z0 = _data_with_channel_stds()
    t = torch.zeros(z0.shape[0])                  # all low-noise => all gated IN
    loss = _out_std_loss(z0.clone(), z0, t)       # z0_hat == data => identical per-channel std
    assert float(loss) < 1e-10, f"expected ~0 penalty when stds match, got {float(loss):.3e}"


def test_output_std_penalty_grows_with_runaway():
    z0 = _data_with_channel_stds()
    t = torch.zeros(z0.shape[0])
    losses = []
    for scale in (1.0, 1.5, 3.0):                 # 1.0 = matched; >1 = latent-scale runaway
        losses.append(float(_out_std_loss(z0 * scale, z0, t)))
    assert losses[0] < 1e-10, f"scale 1.0 should be ~0, got {losses[0]:.3e}"
    assert losses[0] < losses[1] < losses[2], f"penalty must grow with runaway, got {losses}"
    print(f"  [out-std penalty] scale 1.0/1.5/3.0 -> {losses[0]:.2e} / {losses[1]:.3f} / {losses[2]:.3f}")


def test_output_std_penalty_gate_skips_high_noise():
    z0 = _data_with_channel_stds()
    t = torch.full((z0.shape[0],), 0.9)           # all high-noise => gated OUT
    loss = _out_std_loss(z0 * 5.0, z0, t, t_gate=0.5)
    assert float(loss) == 0.0, f"expected 0 (all t >= t_gate gated out), got {float(loss)}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    sys.exit(1 if failed else 0)
