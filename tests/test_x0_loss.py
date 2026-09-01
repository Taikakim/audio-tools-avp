"""Numeric tests for the x0-reconstruction ADD-a-term RF loss (2026-08-11).

A new SA3-training lever adds an EXPLICIT clean-latent reconstruction term to the
rectified-flow loss:

    z0_hat = noised - t*v_pred            (rf_z0_hat, stable_audio_3/training/stereo_loss.py)
    loss  += lambda_x0 * MSE(z0_hat, z0)  (training_step, stable_audio_3/training/diffusion.py)

This is the ADD form (loss += term). It is DISTINCT from the pre-existing
`x0_equiv_loss` flag, which REPLACES the v-loss with its sigma^2-weighted form
(mathematically an x0-space MSE for a v-output head). Predicting toward the clean
latent removes the ambient isotropic-noise component that pressures output-scale
growth (arXiv 2605.27102 JLT; the x0equiv thread).

This pins the MATH, no GPU, in seconds — a bug here wastes a multi-hour LUMI run.
We import the REAL rf_z0_hat + torch's F.mse_loss; the add-a-term is inlined in
training_step so its core is replicated here (marked) against that block.

Runnable directly (pytest is not installed in sat-venv):
    sat-venv/bin/python tests/test_x0_loss.py

Imports the SA3 package (sibling repo). FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE is set
before torch import per MASTER §5 (the import prints a harmless "flash_attn not installed"
note under sat-venv, which lacks the CK ext — irrelevant to this CPU math test).
"""
import os
os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
import sys
from pathlib import Path

# SA3 lives in the sibling repo: SAO/stable-audio-3 (parents[2] == SAO).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stable-audio-3"))

import torch
import torch.nn.functional as F

from stable_audio_3.training.stereo_loss import rf_z0_hat


def _x0_recon_term(noised, v_pred, t, z0_data, weight):
    """Byte-for-byte mirror of the training_step x0-reconstruction ADD-a-term block.

    Returns the ADDITIVE contribution (weight * MSE(z0_hat, z0)); weight<=0 => 0.0.
    """
    if not (weight > 0):
        return torch.zeros(())
    z0_hat = rf_z0_hat(noised, v_pred.to(noised.dtype), t)
    x0_recon_loss = F.mse_loss(z0_hat, z0_data.detach().to(z0_hat.dtype))
    return weight * x0_recon_loss


def _make_rf_batch(B=3, C=16, T=32, tval=0.4, seed=0):
    """Build a consistent RF batch: z0, eps, noised = z0*(1-t)+eps*t, true v = eps - z0."""
    torch.manual_seed(seed)
    z0 = torch.randn(B, C, T)
    eps = torch.randn(B, C, T)
    t = torch.full((B,), tval)
    tb = t.view(-1, 1, 1)
    noised = z0 * (1 - tb) + eps * tb
    v_true = eps - z0
    return z0, eps, t, noised, v_true


# ---------------------------------------------------------------------------
# (a) The x0 term is ~0 when z0_hat == z0 (i.e. the model predicts the TRUE velocity),
#     across the timestep range — z0_hat = noised - t*v_true == z0 exactly.
# ---------------------------------------------------------------------------
def test_x0_term_zero_when_reconstruction_exact():
    worst = 0.0
    for tval in (0.02, 0.1, 0.3, 0.5, 0.8, 0.99):
        z0, eps, t, noised, v_true = _make_rf_batch(tval=tval)
        term = float(_x0_recon_term(noised, v_true, t, z0, weight=1.0))
        worst = max(worst, term)
        assert term < 1e-8, f"expected ~0 x0 term with exact v at t={tval}, got {term:.3e}"
    print(f"  [x0 exact] worst term over t = {worst:.2e}")


# ---------------------------------------------------------------------------
# (b) The term grows monotonically with the velocity-prediction error (mismatch).
# ---------------------------------------------------------------------------
def test_x0_term_grows_with_mismatch():
    z0, eps, t, noised, v_true = _make_rf_batch(tval=0.4)
    torch.manual_seed(1)
    err_dir = torch.randn_like(v_true)
    terms = []
    for scale in (0.0, 0.25, 1.0, 4.0):        # 0 = exact => 0 term; larger = more mismatch
        v_pred = v_true + scale * err_dir
        terms.append(float(_x0_recon_term(noised, v_pred, t, z0, weight=1.0)))
    assert terms[0] < 1e-8, f"scale 0.0 (exact v) should be ~0, got {terms[0]:.3e}"
    assert terms[0] < terms[1] < terms[2] < terms[3], f"term must grow with mismatch, got {terms}"
    print(f"  [x0 mismatch] scale 0/0.25/1/4 -> "
          f"{terms[0]:.2e} / {terms[1]:.3f} / {terms[2]:.3f} / {terms[3]:.3f}")


# ---------------------------------------------------------------------------
# (c) weight == 0 is a byte-identical no-op: the additive contribution is exactly 0.0,
#     independent of how large the reconstruction error is.
# ---------------------------------------------------------------------------
def test_x0_term_weight_zero_is_noop():
    z0, eps, t, noised, v_true = _make_rf_batch(tval=0.4)
    torch.manual_seed(2)
    v_pred = v_true + 5.0 * torch.randn_like(v_true)     # large mismatch
    # Sanity: the term WOULD be large at weight 1.0 ...
    assert float(_x0_recon_term(noised, v_pred, t, z0, weight=1.0)) > 0.1, \
        "test setup: mismatch should give a sizable term at weight 1.0"
    # ... but weight 0.0 contributes exactly nothing.
    term0 = _x0_recon_term(noised, v_pred, t, z0, weight=0.0)
    assert float(term0) == 0.0, f"weight==0 must be a no-op (0.0), got {float(term0)}"


# ---------------------------------------------------------------------------
# (d) The linear-in-v structure: MSE(z0_hat, z0) == t^2 * MSE(v_pred, v_true), since
#     z0_hat - z0 = -t*(v_pred - v_true). Confirms the ADD term is a t^2-weighted
#     v-MSE (the same sigma^2 geometry the x0_equiv REPLACE form encodes), reconstructed
#     rather than reweighted. Uses a scalar t so the closed form holds elementwise.
# ---------------------------------------------------------------------------
def test_x0_term_equals_t2_weighted_vmse():
    tval = 0.4
    z0, eps, t, noised, v_true = _make_rf_batch(tval=tval)
    torch.manual_seed(3)
    v_pred = v_true + torch.randn_like(v_true)
    x0_mse = float(F.mse_loss(rf_z0_hat(noised, v_pred, t), z0))
    v_mse = float(F.mse_loss(v_pred, v_true))
    expected = (tval ** 2) * v_mse
    assert abs(x0_mse - expected) < 1e-5, \
        f"x0 MSE {x0_mse:.5f} != t^2 * v_MSE {expected:.5f} (t={tval})"
    print(f"  [x0 == t^2*vMSE] {x0_mse:.5f} vs {expected:.5f}")


# ---------------------------------------------------------------------------
# (e) The wrapper stores x0_loss_weight and guards weight<=0 (attribute smoke test):
#     confirm the __init__ arg exists and defaults to 0.0 (OFF => byte-identical).
# ---------------------------------------------------------------------------
def test_wrapper_signature_has_x0_loss_weight():
    import inspect
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper
    sig = inspect.signature(DiffusionCondTrainingWrapper.__init__)
    assert "x0_loss_weight" in sig.parameters, "wrapper __init__ missing x0_loss_weight param"
    assert sig.parameters["x0_loss_weight"].default == 0.0, \
        f"x0_loss_weight default must be 0.0 (OFF), got {sig.parameters['x0_loss_weight'].default}"


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
