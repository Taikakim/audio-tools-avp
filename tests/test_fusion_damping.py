"""Unit tests for FusionOpt's two damping mechanisms (C, 2026-08-19, Kim: "write the code for those
muon damping tests"):
  1. an LR DECAY SCHEDULE inside the optimizer (cosine / linear / wsd over total_steps, after warmup)
     — the thing a magnitude-blind spectral step (NS5 -> NorMuon = unit-RMS per neuron whether the
     gradient is signal or noise) otherwise never has: a way to slow down;
  2. an SNR GATE ('snr' component): scale each row's (or element's) step by |EMA(U)| / RMS(U) — a
     data-driven brake that engages where the update direction stops being consistent. On iid noise
     with beta=0.9 the gate settles at sqrt((1-b)/(1+b)) ~= 0.23 (Adam's built-in brake); on a
     constant direction it is 1.
Runnable directly: `sat-venv/bin/python tests/test_fusion_damping.py`.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from stable_audio_tools.training.fusion_opt import FusionOpt, decay_factor, snr_gate


# ---- 1. decay schedule -------------------------------------------------------------------------

def test_decay_none_is_one_everywhere():
    for t in (0, 10, 999):
        assert decay_factor("none", t, total_steps=1000, warmup=0, decay_min=0.0) == 1.0


def test_cosine_endpoints_and_midpoint():
    T = 1000
    assert abs(decay_factor("cosine", 0, T, 0, 0.0) - 1.0) < 1e-9
    assert abs(decay_factor("cosine", T, T, 0, 0.0) - 0.0) < 1e-9
    assert abs(decay_factor("cosine", T // 2, T, 0, 0.0) - 0.5) < 1e-9
    # decay_min floors it
    assert abs(decay_factor("cosine", T, T, 0, 0.1) - 0.1) < 1e-9
    # past the end stays at the floor (a run that overshoots total_steps must not re-warm)
    assert abs(decay_factor("cosine", 2 * T, T, 0, 0.1) - 0.1) < 1e-9


def test_cosine_starts_after_warmup():
    T, W = 1000, 100
    assert abs(decay_factor("cosine", 50, T, W, 0.0) - 1.0) < 1e-9        # inside warmup: no decay
    assert abs(decay_factor("cosine", W, T, W, 0.0) - 1.0) < 1e-9
    assert abs(decay_factor("cosine", (T + W) // 2, T, W, 0.0) - 0.5) < 1e-9


def test_linear_and_wsd():
    T = 1000
    assert abs(decay_factor("linear", 500, T, 0, 0.0) - 0.5) < 1e-9
    # wsd: flat until start_frac, then linear to decay_min at T
    assert abs(decay_factor("wsd", 700, T, 0, 0.0, decay_start_frac=0.8) - 1.0) < 1e-9
    assert abs(decay_factor("wsd", 900, T, 0, 0.0, decay_start_frac=0.8) - 0.5) < 1e-9
    assert abs(decay_factor("wsd", 1000, T, 0, 0.0, decay_start_frac=0.8) - 0.0) < 1e-9


def test_decay_requires_total_steps():
    try:
        decay_factor("cosine", 5, total_steps=0, warmup=0, decay_min=0.0)
    except ValueError:
        return
    raise AssertionError("cosine with total_steps=0 must raise, not silently not decay")


# ---- 2. SNR gate ---------------------------------------------------------------------------------

def _run_gate(make_U, steps, mode, beta=0.9):
    st = {}
    g = None
    for t in range(1, steps + 1):
        U = make_U(t)
        _, g = snr_gate(U, st, mode=mode, beta=beta, floor=0.0, power=1.0, step=t)
    return g


def test_snr_gate_on_iid_noise_settles_near_adam_brake():
    torch.manual_seed(0)
    g = _run_gate(lambda t: torch.randn(64, 256), steps=300, mode="row")
    expect = math.sqrt((1 - 0.9) / (1 + 0.9))          # 0.229
    assert abs(float(g.mean()) - expect) < 0.05, (float(g.mean()), expect)
    ge = _run_gate(lambda t: torch.randn(64, 256), steps=300, mode="elem")
    assert abs(float(ge.mean()) - expect) < 0.08, float(ge.mean())


def test_snr_gate_on_constant_direction_is_one():
    D = torch.randn(64, 256)
    g = _run_gate(lambda t: D, steps=100, mode="row")
    assert float(g.min()) > 0.99
    ge = _run_gate(lambda t: D, steps=100, mode="elem")
    assert float(ge.min()) > 0.99


def test_snr_gate_shapes_floor_and_power():
    torch.manual_seed(1)
    st = {}
    U = torch.randn(8, 16)
    out, g = snr_gate(U, st, mode="row", beta=0.9, floor=0.5, power=1.0, step=1)
    assert out.shape == U.shape and g.shape == (8,)
    assert float(g.min()) >= 0.5                          # floor holds
    st = {}
    out2, g2 = snr_gate(U, st, mode="elem", beta=0.9, floor=0.0, power=2.0, step=1)
    assert g2.shape == U.shape and float(g2.max()) <= 1.0 + 1e-6


def test_snr_gate_measures_on_ref_when_given():
    """Gate statistics come from `ref` (the raw gradient), not from U: a perfectly consistent U
    with a pure-noise ref must brake to ~0.23; a noise U with a constant ref must pass at ~1."""
    torch.manual_seed(3)
    st = {}; U = torch.randn(32, 64); g = None
    for t in range(1, 200):
        _, g = snr_gate(U, st, mode="row", beta=0.9, floor=0.0, power=1.0, step=t, ref=torch.randn(32, 64))
    assert abs(float(g.mean()) - math.sqrt(0.1 / 1.9)) < 0.06, float(g.mean())
    st = {}; C = torch.randn(32, 64)
    for t in range(1, 50):
        _, g = snr_gate(torch.randn(32, 64), st, mode="row", beta=0.9, floor=0.0, power=1.0, step=t, ref=C)
    assert float(g.min()) > 0.99


def test_snr_is_a_valid_component_and_optimizer_steps():
    w = torch.nn.Parameter(torch.randn(8, 8))
    groups = [{"params": [w], "group_type": "spectral"}]
    opt = FusionOpt(groups, lr=1e-2, components={"ns5", "normuon", "sf", "snr"}, hot_dtype="fp32",
                    snr_mode="row", decay_schedule="cosine", total_steps=20)
    assert "snr" in opt.components
    w0 = w.detach().clone()
    for _ in range(5):
        (w * torch.randn_like(w)).sum().backward()
        opt.step(); opt.zero_grad()
    assert not torch.allclose(w.detach(), w0)


def test_default_construction_unchanged():
    """No decay, no snr => the constructor and a step run exactly as before (no new required args)."""
    w = torch.nn.Parameter(torch.randn(8, 8))
    opt = FusionOpt([{"params": [w], "group_type": "spectral"}], lr=1e-2,
                    components={"ns5", "normuon", "sf"}, hot_dtype="fp32")
    (w * 2).sum().backward(); opt.step()


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f(); print("PASS", f.__name__)
    print(f"{len(fns)} passed")
