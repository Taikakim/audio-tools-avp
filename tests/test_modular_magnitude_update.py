"""ModularOptimizer magnitude_update: multiplicative DoRA-magnitude steps (C, 2026-09-24).

Regression for goa3_avp_r256_2026-09-23: small DoRA magnitudes (~0.13) walked through zero under
fixed-size sign steps. Run: .venv/bin/python -m pytest stable-audio-tools/tests/test_modular_magnitude_update.py -q
"""
import math

import pytest
import torch
import torch.nn as nn

from stable_audio_tools.training.modular_opt import ModularOptimizer, build_modular_param_groups


def _model(mag_value):
    torch.manual_seed(0)
    m = nn.Module()
    m.blk = nn.Module()
    m.blk.lora_A = nn.Parameter(torch.randn(16, 64) * 0.05)
    m.blk.lora_B = nn.Parameter(torch.zeros(64, 16))
    # named like DoRA's parametrization so routing sends it to the sign group
    m.blk.magnitude = nn.Parameter(torch.full((64,), float(mag_value)))
    m.blk.bias = nn.Parameter(torch.zeros(64))
    return m


def _run(mode, sf, steps, mag_value=0.13, lr=6e-4, grad_sign=+1.0):
    m = _model(mag_value)
    opt = ModularOptimizer(build_modular_param_groups(m), lr=lr, schedule_free=sf,
                           normuon=True, magnitude_update=mode)
    for _ in range(steps):
        for p in m.parameters():
            p.grad = torch.randn_like(p) * 1e-3
        # a CONSISTENT push toward zero on the magnitude -- the failure's worst case
        m.blk.magnitude.grad = torch.full_like(m.blk.magnitude, grad_sign)
        opt.step()
    if sf and hasattr(opt, "eval"):
        opt.eval()
    return m


@pytest.mark.parametrize("sf", [False, True])
def test_additive_small_magnitude_crosses_zero(sf):
    # the bug as it was: 0.13 - 6e-4 * 400 = -0.11
    m = _run("additive", sf, steps=400)
    assert (m.blk.magnitude <= 0).all()


@pytest.mark.parametrize("sf", [False, True])
def test_multiplicative_stays_positive_and_relative(sf):
    m = _run("multiplicative", sf, steps=400)
    mag = m.blk.magnitude.detach()
    assert (mag > 0).all()
    # z shrinks by exp(-lr) per step -> 0.13*exp(-0.24); the SF average sits between that and 0.13
    lo = 0.13 * math.exp(-6e-4 * 400)
    assert (mag >= lo * 0.999).all() and (mag <= 0.13).all()


def test_multiplicative_scale_invariance():
    small = _run("multiplicative", False, steps=200, mag_value=0.13).blk.magnitude.detach()
    large = _run("multiplicative", False, steps=200, mag_value=2.4).blk.magnitude.detach()
    assert torch.allclose(small / 0.13, large / 2.4, rtol=1e-5)


def test_non_magnitude_params_unchanged_by_mode():
    a = _run("additive", True, steps=50)
    b = _run("multiplicative", True, steps=50)
    for name in ("lora_A", "lora_B", "bias"):
        assert torch.equal(getattr(a.blk, name), getattr(b.blk, name)), name


def test_default_is_additive():
    m = _model(0.13)
    opt = ModularOptimizer(build_modular_param_groups(m), lr=1e-3)
    assert all(g.get("magnitude_update") == "additive" for g in opt.param_groups)
