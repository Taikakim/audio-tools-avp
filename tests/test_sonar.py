"""Unit tests for the sonar probe (radial loss probes -> trust-scaled step).
Runnable directly: `sat-venv/bin/python tests/test_sonar.py` (or pytest).

Design refs (2026-07 research pass): SALSA (2407.20650) EMA-smoothed criterion +
conservative growth; critical sharpness (2601.16979) forward-only probes along the
APPLIED update; Distance-Aware Muon (2605.18999) trust radius. Probes displace live
params, must restore EXACTLY; per-event scale change clamped to [0.5, 2].
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from stable_audio_tools.training.sonar import SonarProbe
from stable_audio_tools.training.fusion_opt import FusionOpt


def _quad_problem(lr, dim=32, seed=0):
    """Quadratic bowl: L(w) = ||w - w*||^2 / dim. Plain SGD so the step scale is exact."""
    torch.manual_seed(seed)
    target = torch.randn(dim)
    w = torch.nn.Parameter(torch.randn(dim))
    opt = torch.optim.SGD([w], lr=lr)
    loss_fn = lambda: ((w - target) ** 2).mean()
    return w, target, opt, loss_fn


def test_probe_restores_params_exactly():
    w, _, opt, loss_fn = _quad_problem(lr=0.05)
    sonar = SonarProbe(every=1)
    sonar.before_step([w])
    loss_fn().backward(); opt.step()
    before = w.detach().clone()
    sonar.after_step([w], loss_fn)
    assert torch.equal(w.detach(), before), "probe must restore params bit-exactly"


def test_too_small_step_raises_scale():
    """lr far below optimal on a quadratic -> parabola says go further -> scale > 1."""
    w, _, opt, loss_fn = _quad_problem(lr=0.01)   # optimal lr for this loss is 0.5*dim... tiny here
    sonar = SonarProbe(every=1)
    for _ in range(5):
        sonar.before_step([w])
        opt.zero_grad(); loss_fn().backward(); opt.step()
        sonar.after_step([w], loss_fn)
    assert sonar.scale > 1.05, sonar.scale


def test_overshoot_lowers_scale():
    """lr above optimal (loss increases along the applied step) -> scale < 1."""
    w, _, opt, loss_fn = _quad_problem(lr=30.0)   # near the stability edge (opt lr = 16)
    sonar = SonarProbe(every=1)
    for _ in range(6):
        sonar.before_step([w])
        opt.zero_grad(); loss_fn().backward(); opt.step()
        sonar.after_step([w], loss_fn)
    assert sonar.scale < 0.9, sonar.scale


def test_per_event_clamp():
    """A single probe can move the scale by at most 2x/0.5x (SALSA-conservative)."""
    w, _, opt, loss_fn = _quad_problem(lr=0.001)
    sonar = SonarProbe(every=1)
    sonar.before_step([w])
    opt.zero_grad(); loss_fn().backward(); opt.step()
    sonar.after_step([w], loss_fn)
    assert 0.5 - 1e-6 <= sonar.scale <= 2.0 + 1e-6


def test_cadence_skips():
    w, _, opt, loss_fn = _quad_problem(lr=0.01)
    sonar = SonarProbe(every=10)
    n_probes = 0
    for _ in range(10):
        sonar.before_step([w])
        opt.zero_grad(); loss_fn().backward(); opt.step()
        n_probes += int(sonar.after_step([w], loss_fn))
    assert n_probes == 1, n_probes


def test_fusionopt_gamma_scale_multiplies():
    """FusionOpt must honor an external gamma scale (sonar writes it)."""
    torch.manual_seed(0)
    w1 = torch.nn.Parameter(torch.randn(16, 16))
    w2 = torch.nn.Parameter(torch.randn(16, 16))
    g = torch.randn(16, 16)

    def displacement(param, scale):
        w0 = param.detach().clone()          # pre-step reference (train mode rewrites p.data)
        opt = FusionOpt([{"params": [param], "group_type": "spectral"}], lr=1e-2,
                        components={"ns5", "normuon", "sf"}, hot_dtype="fp32")
        opt.gamma_scale = scale
        opt.train(); param.grad = g.clone(); opt.set_loss(torch.tensor(1.0)); opt.step()
        return (opt.state[param]["z"] - w0).norm()

    d1 = displacement(w1, 1.0)
    d2 = displacement(w2, 0.5)
    assert abs(d2 / d1 - 0.5) < 0.05, (float(d1), float(d2))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except Exception as e:
            bad += 1; print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed"); sys.exit(1 if bad else 0)
