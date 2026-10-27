"""ModularOptimizer strict_group_lrs (default True): a spectral group and a 1-D group on one lr is refused (CONTINUITY 2026-10-27).

Kim's rule (Muon guide review): old scripts should fail with parameters that anticipate problems, unless explicitly overdriven.
Run: .venv/bin/python -m pytest stable-audio-tools/tests/test_modular_group_lrs.py -q
"""
import warnings

import pytest
import torch
import torch.nn as nn

from stable_audio_tools.training.modular_opt import ModularOptimizer, build_modular_param_groups


def _model():
    torch.manual_seed(0)
    m = nn.Module()
    m.blk = nn.Module()
    m.blk.weight = nn.Parameter(torch.randn(256, 256) * 0.05)   # hidden matrix: spectral
    m.blk.bias = nn.Parameter(torch.zeros(256))                 # 1-D list: sign / colnorm / adam
    return m


def _groups(m, lr_scalar=None):
    groups = build_modular_param_groups(m)
    kinds = {g["group_type"] for g in groups}
    assert "spectral" in kinds and len(kinds) >= 2, kinds       # the test needs both kinds present
    if lr_scalar is not None:
        for g in groups:
            if g["group_type"] != "spectral":
                g["lr"] = lr_scalar
    return groups


def test_default_refuses_a_uniform_lr():
    with pytest.raises(ValueError, match="share one learning rate"):
        ModularOptimizer(_groups(_model()), lr=3e-3)


def test_explicit_override_accepts_it_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModularOptimizer(_groups(_model()), lr=3e-3, strict_group_lrs=False)


def test_separate_lrs_pass_the_default():
    ModularOptimizer(_groups(_model(), lr_scalar=3e-4), lr=3e-3)


def test_step_unchanged_by_the_check():
    torch.manual_seed(1)
    ms = [_model(), _model()]
    opts = [ModularOptimizer(_groups(ms[0]), lr=1e-3, schedule_free=False, warmup_steps=0, strict_group_lrs=False),
            ModularOptimizer(_groups(ms[1]), lr=1e-3, schedule_free=False, warmup_steps=0, strict_group_lrs=False)]
    for _ in range(3):
        g = [torch.randn_like(p) for p in ms[0].parameters()]
        for m, o in zip(ms, opts):
            for p, gi in zip(m.parameters(), g):
                p.grad = gi.clone()
            o.step()
    for a, b in zip(ms[0].parameters(), ms[1].parameters()):
        assert torch.equal(a, b)
