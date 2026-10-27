"""ModularOptimizer C-series options and T0 checks (KUANG, 2026-10-17).

docs/modular-opt-test-plan-2026-10-17.md (avp-audio-craft) sections T0, C2-C7, N. CPU only, small tensors.
Run: .venv/bin/python -m pytest stable-audio-tools/tests/test_modular_c_series.py -q
"""
import math

import pytest
import torch
import torch.nn as nn

from stable_audio_tools.training.modular_opt import (
    GradNoiseMeter, ModularOptimizer, build_modular_param_groups, newton_schulz_quintic,
)


def _MO(*a, **k):
    """These tests use ONE lr for every group on purpose; strict_group_lrs (default True) would refuse it."""
    k.setdefault("strict_group_lrs", False)
    return ModularOptimizer(*a, **k)


def _single(shape, radius=1.0, seed=0, dtype=torch.float32, **kw):
    torch.manual_seed(seed)
    w = nn.Parameter((torch.randn(*shape) * 0.02).to(dtype))
    g = {"params": [w], "group_type": "spectral", "param_names": ["w"], "radii": [radius], "blocks": [1],
         "whitening": kw.pop("whitening", "none"), "weight_decay": kw.pop("weight_decay", 0.0)}
    opt = _MO([g], **kw)
    opt._telem_on = False
    return w, opt


def _drive(w, opt, steps, seed=1, signal=0.3):
    gen = torch.Generator().manual_seed(seed)
    deltas = []
    for _ in range(steps):
        before = w.detach().float().clone()
        w.grad = (torch.randn(w.shape, generator=gen) + signal).to(w.dtype)
        opt.step()
        deltas.append(w.detach().float() - before)
    return deltas


# T0.1 ---------------------------------------------------------------------------------------------
def test_radius_noop_with_normuon():
    a = _drive(*_single((128, 32), radius=1.0, lr=1e-2, schedule_free=False), 10)
    b = _drive(*_single((128, 32), radius=4.0, lr=1e-2, schedule_free=False), 10)
    assert max(float((x - y).abs().max()) for x, y in zip(a, b)) == 0.0


def test_radius_acts_without_normuon():
    a = _drive(*_single((128, 32), radius=1.0, lr=1e-2, schedule_free=False, normuon=False), 3)
    b = _drive(*_single((128, 32), radius=4.0, lr=1e-2, schedule_free=False, normuon=False), 3)
    assert b[-1].norm() / a[-1].norm() == pytest.approx(4.0, rel=1e-4)


# T0.2 ---------------------------------------------------------------------------------------------
def _moved(dtype, lr, steps=300):
    w, opt = _single((256, 256), lr=lr, dtype=dtype, weight_decay=0.01)
    w0 = w.detach().float().clone()
    target = torch.randn(256, 256, generator=torch.Generator().manual_seed(7)) * 0.02
    gen = torch.Generator().manual_seed(3)
    for _ in range(steps):
        w.grad = ((w.detach().float() - target) + 0.5 * 0.02 * torch.randn(256, 256, generator=gen)).to(dtype)
        opt.step()
    opt.eval()
    return float((w.detach().float() - w0).pow(2).mean().sqrt())


def test_bf16_storage_loses_small_steps():
    """F21: with bf16 parameters (and therefore bf16 momentum / Schedule-Free z, x) a small-lr run moves
    much less than the same run in fp32; at a large lr the loss is small. Regression guard for C1."""
    lo = _moved(torch.bfloat16, 3e-4) / _moved(torch.float32, 3e-4)
    hi = _moved(torch.bfloat16, 3e-3) / _moved(torch.float32, 3e-3)
    assert lo < 0.7, lo
    assert hi > 0.8, hi
    assert hi > lo + 0.15


# T0.3 / C2 ----------------------------------------------------------------------------------------
def _dion_reference(grads, shape, lr, beta1, beta2):
    """microsoft/dion NorMuon (normuon.py::_normuon_normalization_core + adjust_lr_spectral_norm), with
    our momentum convention and NS polynomial (both are scale-free after Newton-Schulz)."""
    m, n = shape
    mom = torch.zeros(shape)
    v = torch.zeros(m)
    out = []
    for t, g in enumerate(grads):
        mom.mul_(beta1).add_(g, alpha=1 - beta1)
        U = newton_schulz_quintic(mom / (1 - beta1 ** (t + 1)))
        nU = U.norm()
        v = torch.lerp(v, (U * U).mean(-1), 1 - beta2)
        U = U / (v.sqrt() + 1e-8).unsqueeze(-1)
        U = U * (nU / U.norm().clamp_min(1e-8))
        out.append(-lr * math.sqrt(m / n) * U)
    return out


@pytest.mark.parametrize("shape", [(64, 64), (128, 32), (32, 128)])
def test_dion_mode_matches_reference(shape):
    w, opt = _single(shape, radius=3.0, lr=1e-2, schedule_free=False, normuon_mode="dion", beta1=0.9)
    gen = torch.Generator().manual_seed(5)
    grads = [torch.randn(*shape, generator=gen) * (1 + torch.arange(shape[0]).float().unsqueeze(-1) / shape[0])
             for _ in range(12)]
    mine = []
    for g in grads:
        before = w.detach().clone()
        w.grad = g.clone()
        opt.step()
        mine.append(w.detach() - before)
    ref = _dion_reference(grads, shape, 1e-2, 0.9, 0.95)
    for a, b in zip(mine, ref):
        assert torch.allclose(a, b, rtol=1e-4, atol=1e-8)


def test_row_unit_vs_dion_scale_on_wide_matrix():
    """F23: on a wide matrix the historical NorMuon steps at least sqrt(d_in/d_out) larger than dion's.
    On top of that, ours rescales rows to exactly unit length while dion keeps the Newton-Schulz output
    size (singular values ~0.7-1.0 after 5 quintic steps), so the measured ratio is a little larger
    (GHOST-NOTE's desktop check 2026-10-16: 4-8% on square/tall; 4.76 vs 4.0 here, 32x512)."""
    rms = lambda ds: sum(float(d.pow(2).mean().sqrt()) for d in ds[5:]) / len(ds[5:])
    ours = rms(_drive(*_single((32, 512), lr=1e-2, schedule_free=False), 20))
    dion = rms(_drive(*_single((32, 512), lr=1e-2, schedule_free=False, normuon_mode="dion"), 20))
    assert 4.0 <= ours / dion <= 5.0


# T0.4 ---------------------------------------------------------------------------------------------
def test_vadd_damping_is_applied_twice():
    """Documents current behaviour (optimizer.py Stage 5): when VADD fires, var_damp multiplies the step
    twice, once through ev_mult and once directly, so the damping is var_damp ** 2. Kim decides whether
    that is intended; change this test with the code, not silently."""
    def first_step(observed):
        w, opt = _single((64, 64), lr=1e-2, schedule_free=False, var_dampening_threshold=1.0)
        if observed is not None:
            opt.set_observed_variance(observed)
        return _drive(w, opt, 1)[0].norm()
    plain, damped = first_step(None), first_step(2.0)      # var_damp = 1.0 / 2.0 = 0.5
    assert float(damped / plain) == pytest.approx(0.25, rel=1e-5)


# T0.5 / C3 ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(64, 64), (256, 32), (16, 512)])
def test_rms02_update_scale(shape):
    w, opt = _single(shape, lr=1e-2, schedule_free=False, update_scale="rms0.2")
    d = _drive(w, opt, 3)
    for x in d:
        assert float(x.pow(2).mean().sqrt()) == pytest.approx(0.2 * 1e-2, rel=1e-4)


# C7 -----------------------------------------------------------------------------------------------
def test_bf16_newton_schulz_direction_close():
    a = _drive(*_single((128, 64), lr=1e-2, schedule_free=False), 8)
    b = _drive(*_single((128, 64), lr=1e-2, schedule_free=False, ns_dtype="bf16"), 8)
    for x, y in zip(a, b):
        cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0)
        assert float(cos) > 0.99
        assert float(y.norm() / x.norm()) == pytest.approx(1.0, rel=0.05)


def test_ns_dtype_validated():
    with pytest.raises(ValueError):
        _single((8, 8), ns_dtype="int8")


# C5 -----------------------------------------------------------------------------------------------
def test_normuon_after_unwhiten_equalises_rows():
    """Square matrix -> left-side Shampoo, whose pullback re-mixes rows. With C5, NorMuon runs last, so on
    step 1 (bias-corrected EMA = this step's rows) every row of the update has the same length."""
    def row_spread(after):
        w, opt = _single((64, 64), lr=1e-2, schedule_free=False, whitening="shampoo",
                         normuon_after_unwhiten=after)
        d = _drive(w, opt, 1, signal=0.0)[0]
        r = d.norm(dim=1)
        return float(r.std() / r.mean())
    assert row_spread(True) < 1e-4
    assert row_spread(False) > 1e-3


# C3 adam group ------------------------------------------------------------------------------------
def _scalar_model():
    torch.manual_seed(0)
    m = nn.Module()
    m.lin = nn.Linear(64, 64)
    m.norm = nn.LayerNorm(64)
    return m


def test_scalar_opt_adam_routing():
    groups = build_modular_param_groups(_scalar_model(), scalar_opt="adam", sign_lr=3e-3, scalar_sf_beta=0.95)
    by_type = {g["group_type"]: g for g in groups}
    assert set(by_type) == {"spectral", "adam"}
    assert by_type["adam"]["lr"] == 3e-3 and by_type["adam"]["sf_beta"] == 0.95
    assert "norm.weight" in by_type["adam"]["param_names"] and "lin.bias" in by_type["adam"]["param_names"]
    with pytest.raises(ValueError):
        build_modular_param_groups(_scalar_model(), scalar_opt="lion")


def test_adam_group_matches_manual_adam_without_sf():
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(32))
    opt = _MO([{"params": [w], "group_type": "adam", "param_names": ["b"], "radii": [1.0],
                             "blocks": [1], "whitening": "none", "weight_decay": 0.0}],
                           lr=1e-2, schedule_free=False, beta1=0.9, adam_beta2=0.99)
    opt._telem_on = False
    ref = w.detach().clone()
    m = torch.zeros(32)
    v = torch.zeros(32)
    gen = torch.Generator().manual_seed(2)
    for t in range(1, 6):
        g = torch.randn(32, generator=gen)
        w.grad = g.clone()
        opt.step()
        m = 0.9 * m + 0.1 * g
        v = 0.99 * v + 0.01 * g * g
        ref -= 1e-2 * (m / (1 - 0.9 ** t)) / ((v / (1 - 0.99 ** t)).sqrt() + 1e-8)
    assert torch.allclose(w.detach(), ref, atol=1e-7)


def test_adam_group_under_schedule_free_has_no_momentum():
    """Under Schedule-Free the adam group follows SF-AdamW (no momentum buffer): z moves by
    lr * g / sqrt(v_hat), which on step 1 is lr * sign(g)."""
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(16))
    opt = _MO([{"params": [w], "group_type": "adam", "param_names": ["b"], "radii": [1.0],
                             "blocks": [1], "whitening": "none", "weight_decay": 0.0}], lr=1e-2)
    opt._telem_on = False
    z0 = w.detach().clone()
    g = torch.randn(16)
    w.grad = g.clone()
    opt.step()
    assert torch.allclose(opt.state[w]["z"] - z0, -1e-2 * torch.sign(g), atol=1e-7)
    assert "adam_m" not in opt.state[w]


# C4 -----------------------------------------------------------------------------------------------
def test_per_group_telemetry():
    m = _scalar_model()
    opt = _MO(build_modular_param_groups(m), lr=1e-2)
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    t = opt.component_telemetry
    keys = [k for k in t if k.endswith("/rel_update")]
    assert {k.split("/")[1] for k in keys} == {"g0_spectral", "g1_sign"}
    assert all(t[k] > 0 for k in keys)
    assert t["comp/eigh_failures"] == 0.0


# C6 -----------------------------------------------------------------------------------------------
def test_grad_noise_meter_recovers_noise_scale(tmp_path):
    """Synthetic: true gradient G, per-sample noise sigma, halves of b samples. Expected
    B_simple = tr(Sigma) / |G|^2 = sigma^2 * numel / |G|^2."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.zeros(64, 32))
    meter = GradNoiseMeter([("w", p)], every=1, out_path=str(tmp_path / "gn.jsonl"), ema=0.98)
    G = torch.randn(64, 32) * 0.1
    sigma, b = 0.5, 8
    expected = sigma ** 2 * G.numel() / float((G * G).sum())
    gen = torch.Generator().manual_seed(1)
    for step in range(400):
        h1 = G + sigma / math.sqrt(b) * torch.randn(64, 32, generator=gen)
        h2 = G + sigma / math.sqrt(b) * torch.randn(64, 32, generator=gen)
        out = meter.measure(step, [h1], [h2], b_half=b)
    assert out["grad_noise/b_simple_ema"] == pytest.approx(expected, rel=0.25)
    # converted cosine: two independent batches of 2b samples
    exp_cos = 1.0 / (1.0 + expected / (2 * b))
    assert out["grad_noise/cos_fullbatch_est"] == pytest.approx(exp_cos, abs=0.08)
    assert (tmp_path / "gn.jsonl").read_text().count("\n") == 400


# C1 state_dtype="fp32" ----------------------------------------------------------------------------
def _moved_state(dtype, lr, state_dtype, sf=True, whitening="none", steps=300):
    w, opt = _single((256, 256), lr=lr, dtype=dtype, weight_decay=0.01, state_dtype=state_dtype,
                     schedule_free=sf, whitening=whitening)
    w0 = w.detach().float().clone()
    target = torch.randn(256, 256, generator=torch.Generator().manual_seed(7)) * 0.02
    gen = torch.Generator().manual_seed(3)
    for _ in range(steps):
        w.grad = ((w.detach().float() - target) + 0.5 * 0.02 * torch.randn(256, 256, generator=gen)).to(dtype)
        opt.step()
    opt.eval()
    return float((w.detach().float() - w0).pow(2).mean().sqrt()), opt, w


@pytest.mark.parametrize("sf", [True, False])
def test_fp32_state_recovers_bf16_motion(sf):
    ref = _moved_state(torch.float32, 3e-4, "param", sf)[0]
    bf16_plain = _moved_state(torch.bfloat16, 3e-4, "param", sf)[0]
    bf16_master = _moved_state(torch.bfloat16, 3e-4, "fp32", sf)[0]
    assert bf16_master / ref == pytest.approx(1.0, abs=0.05)
    assert bf16_master > bf16_plain * 1.3


def test_fp32_state_with_shampoo_runs_and_stays_fp32():
    _, opt, w = _moved_state(torch.bfloat16, 1e-3, "fp32", whitening="shampoo", steps=25)
    st = opt.state[w]
    assert w.dtype == torch.bfloat16
    assert st["momentum"].dtype == st["z"].dtype == st["x"].dtype == torch.float32


def test_fp32_state_survives_resume():
    _, opt, w = _moved_state(torch.bfloat16, 1e-3, "fp32", steps=5)
    sd = opt.state_dict()
    w2 = nn.Parameter(w.detach().clone())
    opt2 = _MO([{"params": [w2], "group_type": "spectral", "param_names": ["w"], "radii": [1.0],
                              "blocks": [1], "whitening": "none", "weight_decay": 0.01}],
                            lr=1e-3, state_dtype="fp32")
    opt2.load_state_dict(sd)
    for k in ("momentum", "z", "x"):
        assert opt2.state[w2][k].dtype == torch.float32
        assert torch.equal(opt2.state[w2][k], opt.state[w][k])


# ns_batch -----------------------------------------------------------------------------------------
def _batch_model():
    torch.manual_seed(0)
    m = nn.Module()
    m.a1, m.a2, m.a3 = nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 64)  # one shape, batched together
    m.up, m.down = nn.Linear(64, 256), nn.Linear(256, 64)                       # tall, wide
    m.blk = nn.Module()
    m.blk.to_qkv = nn.Linear(64, 192)                                           # 3 QKV blocks of 64x64
    m.blk.to_out = nn.Linear(64, 64)
    m.norm = nn.LayerNorm(64)
    return m


@pytest.mark.parametrize("kw", [dict(), dict(default_whitening="shampoo"), dict(normuon_mode="dion"),
                                dict(update_scale="rms0.2"), dict(ns_poly="cubic5"), dict(schedule_free=False),
                                dict(normuon=False)])
def test_ns_batch_matches_per_matrix(kw):
    kw = dict(kw)
    wh = kw.pop("default_whitening", "none")
    ma, mb = _batch_model(), _batch_model()
    oa = _MO(build_modular_param_groups(ma, default_whitening=wh), lr=1e-2, **kw)
    ob = _MO(build_modular_param_groups(mb, default_whitening=wh), lr=1e-2, ns_batch=True, **kw)
    gen = torch.Generator().manual_seed(4)
    for _ in range(15):
        for pa, pb in zip(ma.parameters(), mb.parameters()):
            g = torch.randn(pa.shape, generator=gen)
            pa.grad, pb.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert torch.allclose(pa, pb, rtol=1e-5, atol=1e-6)


def test_ns_batch_with_bf16_params_and_fp32_state():
    ma, mb = _batch_model().to(torch.bfloat16), _batch_model().to(torch.bfloat16)
    oa = _MO(build_modular_param_groups(ma), lr=1e-3, state_dtype="fp32", ns_dtype="bf16")
    ob = _MO(build_modular_param_groups(mb), lr=1e-3, state_dtype="fp32", ns_dtype="bf16", ns_batch=True)
    gen = torch.Generator().manual_seed(5)
    for _ in range(10):
        for pa, pb in zip(ma.parameters(), mb.parameters()):
            g = torch.randn(pa.shape, generator=gen).to(torch.bfloat16)
            pa.grad, pb.grad = g.clone(), g.clone()
        oa.step()
        ob.step()
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        za, zb = oa.state[pa].get("z"), ob.state[pb].get("z")
        if za is not None:
            assert torch.allclose(za, zb, rtol=1e-3, atol=1e-5)


# direction agreement / momentum spectra ----------------------------------------------------------
def test_direction_agreement_signal_vs_noise():
    """A shared signal in the top direction plus independent noise: rank 1 agrees strongly between two
    independent gradients, the residual bulk (pure noise) does not."""
    from stable_audio_tools.training.modular_opt import direction_agreement, momentum_basis
    torch.manual_seed(0)
    u, v = torch.randn(64, 1), torch.randn(32, 1)
    signal = 3.0 * (u @ v.T) / (u.norm() * v.norm())
    M = signal + 0.05 * torch.randn(64, 32)                 # momentum from a third sample group
    basis = momentum_basis(M, k=8)
    gA = signal + 0.02 * torch.randn(64, 32)
    gB = signal + 0.02 * torch.randn(64, 32)
    out = direction_agreement(gA, gB, basis, bands=((0, 1), (1, 8)))
    assert out["r1-1"]["cos"] > 0.99 and out["r1-1"]["agree_momentum"] == 1.0
    assert abs(out["residual"]["cos"]) < 0.2
    assert out["residual"]["beyond_rank"] == 8


def test_direction_agreement_band_is_rotation_invariant():
    """Two (near-)equal singular values form one band: the band cosine must not depend on how the
    degenerate pair is rotated."""
    from stable_audio_tools.training.modular_opt import direction_agreement
    torch.manual_seed(1)
    U, _ = torch.linalg.qr(torch.randn(16, 4))
    V, _ = torch.linalg.qr(torch.randn(12, 4))
    S = torch.tensor([2.0, 1.0, 1.0, 0.5])
    gA, gB = torch.randn(16, 12), torch.randn(16, 12)
    c = torch.tensor(0.6)
    R = torch.eye(4)
    R[1:3, 1:3] = torch.tensor([[c, -(1 - c * c).sqrt()], [(1 - c * c).sqrt(), c]])
    a = direction_agreement(gA, gB, {"U": U, "V": V, "S": S, "k": 4}, bands=((0, 1), (1, 3), (3, 4)))
    b = direction_agreement(gA, gB, {"U": U @ R, "V": V @ R, "S": S, "k": 4}, bands=((0, 1), (1, 3), (3, 4)))
    assert a["r2-3"]["cos"] == pytest.approx(b["r2-3"]["cos"], abs=1e-6)
    assert a["r2-3"]["edge_unstable"] is False and a["r4-4"]["edge_ratio"] == pytest.approx(2.0)


def test_momentum_spectra_reads_without_mutating():
    from stable_audio_tools.training.modular_opt import momentum_spectra
    m = _scalar_model()
    opt = _MO(build_modular_param_groups(m), lr=1e-2)
    for p in m.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    before = {k: v.clone() for k, v in opt.state[m.lin.weight].items() if torch.is_tensor(v)}
    rows = momentum_spectra(opt, m.named_parameters(), k=4)
    assert [r["name"] for r in rows] == ["lin.weight"]
    assert rows[0]["s_top"][0] == pytest.approx(1.0) and 1.0 <= rows[0]["stable_rank"] <= 64
    for k, v in before.items():
        assert torch.equal(v, opt.state[m.lin.weight][k])


def test_momentum_spectra_skips_zero_and_nonfinite_momentum():
    # P0 (2026-10-19) crashed at step 0: svdvals on the all-zero initial momentum raised LinAlgError on the GPU
    from types import SimpleNamespace
    from stable_audio_tools.training.modular_opt import momentum_spectra
    pz, pn, pr = (torch.nn.Parameter(torch.zeros(16, 8)) for _ in range(3))
    bad = torch.randn(16, 8); bad[0, 0] = float("nan")
    opt = SimpleNamespace(state={pz: {"momentum": torch.zeros(16, 8)}, pn: {"momentum": bad},
                                 pr: {"momentum": torch.randn(16, 8)}})
    rows = momentum_spectra(opt, [("zero", pz), ("nan", pn), ("rand", pr)])
    assert [r["name"] for r in rows] == ["rand"]
