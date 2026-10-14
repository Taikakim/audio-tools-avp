# Owner: KUANG (created 2026-10-13)
"""CPU tests for the v5 tools (audio metrics, ladder gate, cross-attention flow, model size flags, render bank, Flow-DPO,
listening test). No Surge, no GPU. Run from scripts/synth_inversion:

    python -m pytest test_v5.py -q

What these tests are: unit tests on synthetic signals, the stand-in synth (standin_synth_v4.py) and tiny random models.
They show each piece does what its docstring says in a controlled setting. They do NOT show that a metric agrees with
listeners, that cross-attention or DPO helps on Surge audio, or that the bank matches live rendering on the desktop;
those are the desktop smoke tests listed in V5_PROPOSAL_REVIEW.md.
"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import audio_metrics_v5 as am  # noqa: E402
from standin_synth_v4 import StandinPatch, render  # noqa: E402

SR = 44100
# The trainer (and the bank, which reuses its random-patch sampler) imports Essentia at module level; off the training
# machine the essentia/mido fakes under $TEMP/inharm-review/fakes are used when present (as in test_eval_v4.py).
FAKES = os.path.join(os.environ.get("TEMP", os.environ.get("TMPDIR", "/tmp")), "inharm-review", "fakes")
if os.path.isdir(FAKES) and FAKES not in sys.path:
    sys.path.insert(0, FAKES)


def need_trainer():
    try:
        import train_realistic_bass_v3 as T
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"trainer not importable here: {e}")
    return T


def note(**kw):
    p = dict(f0=55.0, fc_hz=800.0, q=0.9, amp_decay_s=0.25, amp_sustain=0.5, release_s=0.05, phase_seed=1)
    p.update(kw)
    nd = p.pop("note_dur", 0.5)
    return render(StandinPatch(**p), dur=0.8, note_dur=nd)


def two_voice_saw(f0=110.0, cents=10.0, seed=0, dur=0.8, n_harm=40):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for f in (f0 * 2 ** (-cents / 2400), f0 * 2 ** (cents / 2400)):
        ph = rng.uniform(0, 2 * np.pi, n_harm)
        for k in range(1, n_harm + 1):
            out += np.sin(2 * np.pi * k * f * t + ph[k - 1]) / k
    env = np.minimum(t / 0.005, 1.0) * np.exp(-t / 0.6)
    out *= env
    return (out / np.max(np.abs(out))).astype(np.float32)


# ------------------------------------------------------------------------------------------------ audio metrics
def test_ssim_matches_slow_reference():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (20, 25))
    y = np.clip(x + rng.normal(0, 0.1, x.shape), 0, 1)
    win, pad = 7, 3
    fast = am._crop(am.ssim_map(x, y, win=win), win)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    slow = np.zeros_like(fast)
    for i in range(pad, x.shape[0] - pad):
        for j in range(pad, x.shape[1] - pad):
            a = x[i - pad:i + pad + 1, j - pad:j + pad + 1].ravel()
            b = y[i - pad:i + pad + 1, j - pad:j + pad + 1].ravel()
            va, vb = a.var(ddof=1), b.var(ddof=1)
            cov = np.cov(a, b, ddof=1)[0, 1]
            slow[i - pad, j - pad] = ((2 * a.mean() * b.mean() + c1) * (2 * cov + c2)) / \
                                     ((a.mean() ** 2 + b.mean() ** 2 + c1) * (va + vb + c2))
    np.testing.assert_allclose(fast, slow, atol=1e-9)


def test_identity_is_zero_for_every_metric():
    x = note()
    for name, (fn, _, _) in am.METRICS.items():
        d = fn(x, x.copy())
        assert abs(d) < 1e-5, (name, d)


def test_logmel_matches_model_input():
    from audio_utils import make_mel_spec
    x = note()
    np.testing.assert_allclose(am.logmel(x), make_mel_spec(x), atol=1e-5)


@pytest.mark.parametrize("name", ["logmel_l1", "mss", "mel_ssim_dist", "mel_ssim_gated_dist", "octave_db_l1"])
def test_cutoff_sweep_is_monotone(name):
    fcs = np.geomspace(150.0, 1500.0, 8)
    clips = [note(fc_hz=f) for f in fcs]
    d = [am.METRICS[name][0](clips[0], c) for c in clips[1:]]
    assert spearmanr(np.arange(len(d)), d).correlation >= 0.9, (name, np.round(d, 4))


def _beat_ratio(metric, f0, c1, c2):
    """distance to a DIFFERENT detune / distance to the SAME detune with new random phases (mean of 3 seeds)."""
    same = np.mean([metric(two_voice_saw(f0, c1, s), two_voice_saw(f0, c1, s + 10)) for s in range(3)])
    diff = np.mean([metric(two_voice_saw(f0, c1, s), two_voice_saw(f0, c2, s + 10)) for s in range(3)])
    return diff / same


def test_modspec_sees_beating_logmel_does_not():
    # Measured 2026-10-13 on this synthetic two-voice saw: log-mel L1 cannot tell 40 from 80 cents (ratio ~1.0:
    # free-running phase noise is as large as the detune change), the modulation spectrum can (ratio ~3.4).
    # At 55 Hz with 10 vs 30 cents the modulation spectrum only reaches ~1.3: beats slower than ~2 Hz do not
    # complete a cycle in 0.8 s, so they stay phase-dependent.
    assert _beat_ratio(am.modspec_dist, 110.0, 40, 80) > 2.0
    assert _beat_ratio(am.logmel_l1, 110.0, 40, 80) < 1.3


def test_flatness_gate_ignores_silent_tail():
    short = note(note_dur=0.15, release_s=0.01)     # long near-silent tail
    long_ = note(note_dur=0.70, release_s=0.01)     # same timbre, no tail
    ungated = am.sm_flatness_err(short, long_)
    gated = am.flatness_gated_err(short, long_)
    assert ungated > 3.0 * gated, (ungated, gated)


def test_ssim_gate_ignores_silent_tail():
    a = note(note_dur=0.15, release_s=0.01, fc_hz=300.0)
    b = note(note_dur=0.15, release_s=0.01, fc_hz=1200.0)
    # the same spectral difference, but plain SSIM averages it with the identical silent tail
    assert am.mel_ssim_gated_dist(a, b) > am.mel_ssim_dist(a, b)


def test_score_many_shapes_and_errors():
    xs = [note(fc_hz=f) for f in (300.0, 600.0, 900.0)]
    ys = [note(fc_hz=f * 1.1) for f in (300.0, 600.0, 900.0)]
    out = am.score_many(xs, ys, ("logmel_l1", "mel_ssim_gated_dist"))
    assert set(out) == {"logmel_l1", "mel_ssim_gated_dist"} and all(v.shape == (3,) for v in out.values())
    with pytest.raises(KeyError):
        am.score_many(xs, ys, ("nope",))
    with pytest.raises(ValueError):
        am.score_many(xs, ys[:2])


# ------------------------------------------------------------------------------------------------- ladder gate
def _vec(seed):
    from surge_spec_v3 import draw_patch, patch_to_vector
    return patch_to_vector(draw_patch(seed)).astype(np.float32)


class FakePrior:
    N = 4

    def sample_patch_and_midi(self, rng):
        preset = int(rng.randint(0, self.N))                 # like RealisticBassPrior: the archetype index first
        vec = _vec(int(rng.randint(0, 10_000)) + preset)
        return None, vec, int(rng.randint(28, 51)), 0.3


def fake_render(vec, midi):
    from surge_spec_v3 import PARAM_INDEX as I
    from standin_synth_v4 import midi_to_hz
    return render(StandinPatch(f0=midi_to_hz(midi), fc_hz=60.0 * 2 ** (7.0 * float(vec[I["cutoff"]])),
                               q=0.7 + 8.0 * float(vec[I["resonance"]]), amp_sustain=float(vec[I["aeg_sustain"]]),
                               amp_decay_s=0.05 + 0.6 * float(vec[I["aeg_decay"]]), phase_seed=1), note_dur=0.5)


def test_new_ladder_axes_sweep_one_parameter():
    import ladder_metric_check_v5 as lg
    from surge_spec_v3 import PARAM_INDEX as I
    rng = np.random.RandomState(0)
    base = _vec(3)
    for axis in lg.NEW_AXES:
        lad = lg.make_ladder_v5(base, axis, 6, rng)
        k = I[lg.AXIS_PARAM_V5[axis]]
        assert np.all(np.diff(lad[:, k]) > 0), axis               # canonicalisation must not zero the swept parameter
        other = [j for j in range(lad.shape[1]) if j != k]
        assert np.all(lad[:, other] == lad[:1, other]), axis      # everything else fixed along the ladder
    assert lg.make_ladder_v5(base, "unison_detune", 4, rng)[0, I["unison"]] == 1.0
    assert lg.make_ladder_v5(base, "fm_depth", 4, rng)[0, I["fm_routing"]] == pytest.approx(1 / 3)


def test_ladder_gate_pipeline_on_fake_renders():
    import ladder_metric_check_v5 as lg
    from render_pool_v5 import RenderPool
    axes = ("cutoff_low", "resonance")
    lad = lg.build_ladders(FakePrior(), axes, n_anchors=3, n_steps=6, seed=1)
    assert lad["vecs"].shape[:2] == (6, 6)
    with RenderPool(render_fn=fake_render) as pool:
        audio, rep = lg.render_ladders(lad, pool)
    dists = lg.ladder_distances(audio, rep, ("logmel_l1", "mel_ssim_gated_dist"))
    s = lg.summarize(dists, lad["axis"], list(axes), n_boot=200)
    assert s["logmel_l1"]["axes"]["cutoff_low"]["rho"] > 0.8
    assert s["logmel_l1"]["axes"]["cutoff_low"]["n"] == 3
    md = lg.report_md(s, list(axes))
    assert "logmel_l1" in md and "cutoff_low" in md


# ----------------------------------------------------------------------------- cross-attention flow and model size
torch = pytest.importorskip("torch")


def _mel(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, 128, 81, generator=g) * 2 - 1


def _flows(**kw):
    import models_v4 as m4
    base = dict(hidden_dim=64, num_layers=2, encoder="temporal_asp", enc_kwargs={"d_model": 32})
    torch.manual_seed(0)
    a = m4.build_model_v4(**base, **kw)
    torch.manual_seed(0)
    b = m4.build_model_v4(**base, xattn=True, xattn_heads=4, **kw)
    b.load_state_dict(a.state_dict(), strict=False)          # shared weights equal; cross-attention keeps its init
    return a.eval(), b.eval()


def _xt(B=3):
    import param_codec_v3 as codec
    g = torch.Generator().manual_seed(1)
    return torch.randn(B, codec.ENCODED_DIM, generator=g), torch.rand(B, generator=g)


@pytest.mark.parametrize("cond_noise", [False, True])
def test_xattn_is_identity_at_init(cond_noise):
    a, b = _flows(cond_noise=cond_noise)
    x, t = _xt()
    mel = _mel()
    assert torch.allclose(a(x, t, mel), b(x, t, mel), atol=1e-6)


def test_xattn_gradients_reach_every_part():
    _, b = _flows()
    b.train()
    for blk in b.xattn:                                     # leave the zero init so q / kv receive gradient too
        torch.nn.init.normal_(blk.out.weight, std=0.02)
    x, t = _xt()
    b(x, t, _mel()).pow(2).mean().backward()
    for name, p in b.named_parameters():
        if name.startswith("xattn.") or name.startswith("encoder.tok_proj"):
            assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_xattn_sampling_paths_and_broadcast():
    from inference_v3 import predict_vectors_pinned
    _, b = _flows(cond_noise=True)
    for blk in b.xattn:
        torch.nn.init.normal_(blk.out.weight, std=0.05)
    out = b.sample(_mel(2), num_steps=4, generator=torch.Generator().manual_seed(0))
    assert out.shape[0] == 2 and torch.isfinite(out).all()
    vecs = predict_vectors_pinned(b, _mel(1), midi_note=36, n_draws=5, steps=4, seed=0, tau=0.7)
    assert vecs.shape[0] == 5 and np.isfinite(vecs).all()   # a batch-1 condition broadcasts over 5 draws
    maps = b.attention_maps(*_xt(2), _mel(2))
    assert len(maps) == 2 and maps[0].shape == (2, 4, b.encoder.n_tokens)
    assert torch.allclose(maps[0].sum(-1), torch.ones(2, 4), atol=1e-5)


def test_xattn_save_load_roundtrip(tmp_path):
    import models_v4 as m4
    torch.manual_seed(3)
    b = m4.build_model_v4(hidden_dim=64, num_layers=2, encoder="temporal_asp", enc_kwargs={"d_model": 32},
                          xattn=True, xattn_heads=2, xattn_dim=16).eval()
    for blk in b.xattn:
        torch.nn.init.normal_(blk.out.weight, std=0.05)
    x, t = _xt()
    mel = _mel()
    p = tmp_path / "flow.pt"
    torch.save({"model_state": b.state_dict(), "args": {"v4_config": b.v4_config}}, p)
    c = m4.load_inverter_v4(str(p))
    assert c.v4_config["xattn_heads"] == 2 and c.v4_config["xattn_dim"] == 16
    assert torch.allclose(b(x, t, mel), c(x, t, mel), atol=1e-6)
    torch.save({"model_state": b.state_dict(), "args": {}}, p)    # older export: config inferred from the keys
    d = m4.load_inverter_v4(str(p))
    assert d.xattn is not None and d.v4_config["xattn_dim"] == 16 and d.v4_config["xattn_heads"] == 4


def test_xattn_needs_temporal_encoder():
    import models_v4 as m4
    with pytest.raises(ValueError):
        m4.build_model_v4(hidden_dim=64, num_layers=2, encoder="conv", xattn=True)


# ------------------------------------------------------------------------------------------------------- bank
class BankPrior(FakePrior):
    calibrated = True

    def support_bounds(self):
        return {"midi_note": (0.0, 1.0)}


def _bank_args(out, **kw):
    import argparse
    d = dict(manifold="fake.npz", plugin=None, out=str(out), seed=5, random_patch_frac=0.2, n_items=10, shard_size=4,
             n_ladders=6, ladder_steps=4, ladder_shard=2, inharm_config=None, workers=0)
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture
def fake_md5(monkeypatch):
    import realistic_bass_prior_v3 as rp
    monkeypatch.setattr(rp, "manifold_md5", lambda path: "fakemd5")


def _build(out, **kw):
    need_trainer()
    import render_bank_v5 as rb
    from render_pool_v5 import RenderPool
    rb.build(_bank_args(out, **kw), prior=BankPrior(), pool=RenderPool(render_fn=fake_render), log=lambda *_: None)


def test_bank_reproducible_and_resumable(tmp_path, fake_md5):
    import render_bank_v5 as rb
    _build(tmp_path)
    ds = rb.BankMainDataset(str(tmp_path))
    assert len(ds) == 10
    a, v, inh, val = ds[5]
    vec, midi, _ = rb.draw_main_item(BankPrior(), 5, 5, 0.2)
    assert np.array_equal(v.numpy(), vec)
    np.testing.assert_allclose(a.numpy(), fake_render(vec, midi), atol=1.0 / 32767 + 1e-7)
    shard1 = np.load(tmp_path / "main_00001_audio.npy")
    del ds                                                   # drop the memory map (Windows locks mapped files)
    import gc
    gc.collect()
    os.remove(tmp_path / "main_00001_audio.npy")
    _build(tmp_path)                                         # resume re-creates only the missing shard, identically
    assert np.array_equal(np.load(tmp_path / "main_00001_audio.npy"), shard1)
    with pytest.raises(SystemExit):
        _build(tmp_path, seed=6)                             # a different configuration is refused


def test_bank_ladder_items_and_compatibility(tmp_path, fake_md5):
    import render_bank_v5 as rb
    _build(tmp_path)
    lad = rb.BankLadderDataset(str(tmp_path), n_anchors=2)
    assert len(lad) == 3
    audio, vecs, axis = lad[1]
    assert audio.shape == (8, 35280) and vecs.shape[0] == 8 and axis.shape == (8,)
    m = rb.read_manifest(str(tmp_path))
    assert rb.check_compatible(m, "fakemd5", 0.0, None, 4, 4, 0.2) == []
    probs = rb.check_compatible(m, "other", 1.0, None, 16, 4, 0.05)
    assert len(probs) == 4


def test_bank_refuses_other_code(tmp_path, fake_md5):
    import json as _json
    import render_bank_v5 as rb
    _build(tmp_path)
    m = rb.read_manifest(str(tmp_path))
    fp = rb.code_fingerprint()
    assert m["code_fingerprint"] == fp and rb.check_compatible(m, "fakemd5", 0.0, None, 4, 4, 0.2, fingerprint=fp) == []
    assert len(rb.check_compatible(m, "fakemd5", 0.0, None, 4, 4, 0.2, fingerprint="0" * 32)) == 1
    del m["code_fingerprint"]                                  # a bank from before fingerprints
    with open(tmp_path / "manifest.json", "w") as f:
        _json.dump(m, f)
    with pytest.raises(SystemExit):
        _build(tmp_path)                                       # resume refused ...
    _build(tmp_path, trust_existing=True)                     # ... unless explicitly trusted
    assert rb.read_manifest(str(tmp_path))["code_fingerprint"] == fp


def test_bank_order_does_not_depend_on_model_rng(tmp_path, fake_md5):
    import render_bank_v5 as rb
    from torch.utils.data import DataLoader, RandomSampler
    _build(tmp_path)
    ds = rb.BankMainDataset(str(tmp_path))

    def first_vecs(model_seed):
        torch.manual_seed(model_seed)
        torch.randn(1000)                                     # an arm that consumes RNG differently
        it = rb.cycle(DataLoader(ds, batch_size=4, drop_last=True,
                                 sampler=RandomSampler(ds, generator=torch.Generator().manual_seed(0))))
        return torch.cat([next(it)[1] for _ in range(5)])     # 5 batches > one pass of 2 batches: cycles
    assert torch.equal(first_vecs(1), first_vecs(2))


# ------------------------------------------------------------------------------------- trainer, end to end on CPU
def _fake_fixed_set(synth, prior, n, seed, extractor=None):
    rng = np.random.RandomState(seed)
    vecs, audio = [], []
    for _ in range(n):
        _, v, m, _ = prior.sample_patch_and_midi(rng)
        vecs.append(v)
        audio.append(fake_render(v, m))
    return torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(vecs))


def _fake_ladder_set(synth, prior, n_anchors_per_axis, n_steps, seed):
    from training_controls_v3 import AXES, make_ladder
    rng = np.random.RandomState(seed)
    audio, vecs, ax_ids = [], [], []
    for ax, axis in enumerate(AXES):
        for _ in range(n_anchors_per_axis):
            _, base, m, _ = prior.sample_patch_and_midi(rng)
            for v in make_ladder(base, axis, n_steps, rng):
                audio.append(fake_render(v, m))
                vecs.append(v)
                ax_ids.append(ax)
    return torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(vecs).astype(np.float32)), np.array(ax_ids)


def _train_args(out, bank, cache, steps):
    return ["--out_dir", str(out), "--manifold", "fake.npz", "--bank", str(bank), "--val_cache", str(cache),
            "--max_steps", str(steps), "--batch_size", "4", "--optimizer", "adamw", "--device", "cpu", "--seed", "0",
            "--embed_dim", "32", "--ff_dim", "64", "--jepa_layers", "1", "--predictor_hidden", "32",
            "--flow_dim", "32", "--flow_layers", "2", "--flow_encoder", "temporal_asp", "--flow_enc_dim", "16",
            "--flow_xattn", "--flow_xattn_heads", "2", "--cond_noise", "--ot_coupling", "--flow_loss", "mae",
            "--ladder_every", "4", "--ladder_anchors", "2", "--ladder_steps", "4", "--ladder_val_anchors", "1",
            "--ema_halflife_steps", "4", "--val_size", "8", "--val_interval_steps", "4",
            "--checkpoint_interval_steps", "4", "--log_interval_steps", "2", "--norm_batches", "2",
            "--num_workers", "0", "--ladder_workers", "0", "--random_patch_frac", "0.2"]


def test_trainer_end_to_end_with_bank_and_xattn(tmp_path, fake_md5, monkeypatch):
    import models_v4 as m4
    tr = need_trainer()
    from synth_jepa_search_v3 import load_synth_jepa
    bank = tmp_path / "bank"
    _build(bank, n_items=48, shard_size=16, n_ladders=8, ladder_shard=4)
    monkeypatch.setattr(tr, "RealisticBassPrior", lambda path, split="train": BankPrior())
    monkeypatch.setattr(tr, "manifold_md5", lambda path: "fakemd5")
    monkeypatch.setattr(tr, "init_synth", lambda *a, **k: object())
    monkeypatch.setattr(tr, "render_fixed_set", _fake_fixed_set)
    monkeypatch.setattr(tr, "render_ladder_set", _fake_ladder_set)
    out = tmp_path / "run"
    tr.train(tr.build_parser().parse_args(_train_args(out, bank, tmp_path / "vc", 8)))
    for f in ("flow_latest.pt", "jepa_latest.pt", "flow_ema_latest.pt", "jepa_online_latest.pt", "val_kinds.jsonl"):
        assert (out / f).exists(), f
    log = (out / "training.log").read_text()
    assert "samples/s, data wait" in log and "Cached validation sets" in log and "passes" in log
    flow = m4.load_inverter_v4(str(out / "flow_ema_latest.pt"))
    assert flow.xattn is not None and flow.v4_config["xattn_heads"] == 2 and len(flow.blocks) == 2
    jepa, _ = load_synth_jepa(str(out / "jepa_latest.pt"))
    assert sum(1 for k in jepa.state_dict() if k.startswith("audio_encoder.") and k.endswith("linear1.weight")) == 1
    out2 = tmp_path / "run2"                                 # a second run reuses the cached validation renders
    tr.train(tr.build_parser().parse_args(_train_args(out2, bank, tmp_path / "vc", 2)))
    assert "Loaded cached validation sets" in (out2 / "training.log").read_text()


# --------------------------------------------------------------------------------------- sampling and Flow-DPO
def _save_flow(path, cond_noise=True, xattn=False, seed=0):
    import models_v4 as m4
    torch.manual_seed(seed)
    f = m4.build_model_v4(hidden_dim=32, num_layers=2, encoder="temporal_asp", enc_kwargs={"d_model": 16},
                          cond_noise=cond_noise, xattn=xattn, xattn_heads=2)
    torch.save({"model_state": f.state_dict(), "args": {"v4_config": f.v4_config}, "step": 7, "prior_bounds": None},
               path)
    return f


def test_sample_candidates_pins_midi_and_draws_independent_noise(tmp_path):
    from flow_sampling_v5 import sample_candidates
    from surge_spec_v3 import NOTE_HIGH, NOTE_LOW
    f = _save_flow(tmp_path / "f.pt").eval()
    mel = _mel(2)
    v = sample_candidates(f, mel, [30, 45], k=4, steps=3, seed=0, tau=0.5)
    assert v.shape[:2] == (2, 4)
    assert np.allclose(v[0, :, 0], (30 - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW))
    assert np.allclose(v[1, :, 0], (45 - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW))
    assert not np.allclose(v[0, 0], v[0, 1])                 # candidates differ
    v2 = sample_candidates(f, mel, [30, 45], k=4, steps=3, seed=0, tau=0.5)
    assert np.array_equal(v, v2)                             # reproducible


def _write_pairs(d, flow_path, n=48, k=4, seed=0):
    """Synthetic preference data: the 'good' candidates have low cutoff, the bad ones high (condition-independent)."""
    from flow_sampling_v5 import file_md5
    from render_pool_v5 import to_int16
    from surge_spec_v3 import PARAM_INDEX as I
    rng = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    vecs = np.stack([_vec(int(s)) for s in rng.integers(0, 10_000, n * k)]).reshape(n, k, -1)
    vecs[:, :, I["cutoff"]] = rng.uniform(0.1, 0.9, (n, k))
    score = vecs[:, :, I["cutoff"]].copy()                    # lower cutoff = better
    audio = np.stack([note(fc_hz=float(f)) for f in rng.uniform(200, 1200, n)])
    np.savez(os.path.join(d, "shard_00000.npz"), cand_vecs=vecs.astype(np.float32), score__logmel_l1=score,
             noise__logmel_l1=np.full(n, 0.01), target_audio=to_int16(audio), midi=np.full(n, 36),
             true_vecs=vecs[:, 0].astype(np.float32), cond_index=np.arange(n))
    with open(os.path.join(d, "manifest.json"), "w") as fh:
        json.dump({"flow": str(flow_path), "flow_md5": file_md5(flow_path), "metrics": ["logmel_l1"], "tau": 1.0}, fh)


import json  # noqa: E402


def test_load_pairs_margins(tmp_path):
    import train_flow_dpo_v5 as dpo
    _save_flow(tmp_path / "f.pt")
    _write_pairs(tmp_path / "p", tmp_path / "f.pt", n=10, k=4)
    bw = dpo.load_pairs([str(tmp_path / "p")])
    assert len(bw["cond"]) == 10 and np.all(bw["gap"] > 0.01)
    assert np.all(bw["w"][:, 9] < bw["l"][:, 9])              # winner has the lower score (cutoff)
    am_ = dpo.load_pairs([str(tmp_path / "p")], mode="all_margin", max_per_cond=3)
    assert len(am_["cond"]) <= 30 and len(am_["cond"]) > 10
    with pytest.raises(SystemExit):
        dpo.load_pairs([str(tmp_path / "p")], margin=5.0)


def test_dpo_is_neutral_at_init_and_learns_a_preference(tmp_path):
    import models_v4 as m4
    import train_flow_dpo_v5 as dpo
    flow_path = tmp_path / "f.pt"
    _save_flow(flow_path)
    _write_pairs(tmp_path / "p", flow_path, n=64, k=4)
    # neutral at init: policy == reference -> delta 0, loss log 2
    f = m4.load_inverter_v4(str(flow_path)).eval()
    import copy as _copy
    import param_codec_v3 as codec
    from surge_spec_v3 import domain_weights
    w_enc = codec.encoded_weights(torch.from_numpy(domain_weights()))
    data = dpo.load_pairs([str(tmp_path / "p")])
    from flow_sampling_v5 import mel_of
    mel = mel_of(data["cond_audio"][:8].astype(np.float32) / 32767.0, "cpu")
    loss, s = dpo.dpo_terms(f, _copy.deepcopy(f), mel, torch.from_numpy(data["w"][:8]), torch.from_numpy(data["l"][:8]),
                            w_enc, 100.0, torch.Generator().manual_seed(0), tau=0.6)
    assert abs(s["delta"]) < 1e-6 and abs(float(loss) - np.log(2)) < 1e-5
    # learns: 150 steps at a high learning rate on a condition-independent preference
    a = dpo.build_parser().parse_args(["--flow", str(flow_path), "--pairs", str(tmp_path / "p"), "--out",
                                       str(tmp_path / "o"), "--beta", "50", "--lr", "1e-3", "--steps", "150",
                                       "--batch", "32", "--warmup", "10", "--log_every", "50", "--val_every", "150",
                                       "--val_frac", "0.25", "--device", "cpu", "--sft_weight", "0.1",
                                       "--sft_target", "true"])
    ref_before = {k: v.clone() for k, v in m4.load_inverter_v4(str(flow_path)).state_dict().items()}
    policy, ref = dpo.train(a, log=lambda *_: None)
    recs = [json.loads(x) for x in open(tmp_path / "o" / "dpo_log.jsonl")]
    assert recs[-1]["loss"] < np.log(2) - 0.05 and recs[-1]["acc"] > 0.6, recs[-1]
    assert recs[-1]["val_acc"] > 0.6, recs[-1]               # held-out conditions too
    for k, v in ref.state_dict().items():                    # the reference never moves
        assert torch.equal(v, ref_before[k]), k
    g = m4.load_inverter_v4(str(tmp_path / "o" / "flow_dpo_latest.pt"))
    assert any(not torch.equal(g.state_dict()[k], ref_before[k]) for k in ref_before)


def test_dpo_refuses_pairs_from_another_flow(tmp_path):
    import train_flow_dpo_v5 as dpo
    _save_flow(tmp_path / "f.pt", seed=0)
    _save_flow(tmp_path / "g.pt", seed=1)
    _write_pairs(tmp_path / "p", tmp_path / "g.pt", n=8)
    a = dpo.build_parser().parse_args(["--flow", str(tmp_path / "f.pt"), "--pairs", str(tmp_path / "p"), "--out",
                                       str(tmp_path / "o"), "--steps", "1", "--device", "cpu"])
    with pytest.raises(SystemExit):
        dpo.train(a, log=lambda *_: None)


def test_dpo_pairs_shard_and_headroom_with_fake_renders(tmp_path):
    import dpo_pairs_v5 as dp
    from render_pool_v5 import RenderPool
    f = _save_flow(tmp_path / "f.pt").eval()
    prior = BankPrior()
    true_vecs, midi = dp.synth_conditions(prior, 0, 4, seed=3)
    with RenderPool(render_fn=fake_render) as pool:
        targets = pool.render(true_vecs, midi)
        sh = dp.build_shard(f, pool, "cpu", targets, midi, k=3, steps=2, tau=1.0, seed=0,
                            metrics=("logmel_l1", "mss"), true_vecs=true_vecs)
    assert sh["cand_vecs"].shape[:2] == (4, 3) and sh["score__logmel_l1"].shape == (4, 3)
    assert np.allclose(sh["noise__logmel_l1"], 0.0, atol=1e-6)   # the fake renderer is deterministic
    h = dp.headroom([sh], "logmel_l1")
    assert h["best"] <= h["draw0"] <= h["worst"]


def test_audio_eval_end_to_end_with_fake_renders(tmp_path):
    import audio_eval_v5 as ae
    from render_pool_v5 import RenderPool
    _save_flow(tmp_path / "f.pt", cond_noise=False)
    _save_flow(tmp_path / "g.pt", cond_noise=False, xattn=True, seed=1)
    out = tmp_path / "ev"
    s = ae.main(["--manifold", "fake.npz", "--flow", f"A={tmp_path / 'f.pt'}", "--flow", f"B={tmp_path / 'g.pt'}",
                 "--n", "6", "--k", "3", "--steps", "2", "--metrics", "logmel_l1", "mel_ssim_gated_dist",
                 "--workers", "0", "--listen", "2", "--device", "cpu", "--out", str(out)],
                pool=RenderPool(render_fn=fake_render), prior=BankPrior())
    assert set(s["systems"]) == {"A/draw0", "A/best3", "B/draw0", "B/best3"}
    sa = s["systems"]
    assert sa["A/best3"]["logmel_l1"]["estimate"] <= sa["A/draw0"]["logmel_l1"]["estimate"] + 1e-9
    assert (out / "report.md").exists() and (out / "listen" / "index.html").exists()
    assert (out / "listen" / "refs" / "syn000.wav").exists() and (out / "listen" / "syn001__B_best3.wav").exists()


# ------------------------------------------------------------------------------------------------ listening test
def test_bs1770_meter_on_a_sine():
    import listening_test_v5 as lt
    t = np.arange(3 * SR) / SR
    x = 0.1 * np.sin(2 * np.pi * 997.0 * t)
    assert lt.integrated_loudness(x, SR) == pytest.approx(-23.01, abs=0.15)   # 0 dBFS sine = -3.01 LUFS by design
    y, g, lim = lt.normalize_loudness(0.5 * x, SR, -23.0)
    assert not lim and lt.integrated_loudness(y, SR) == pytest.approx(-23.0, abs=0.05)
    spike = np.zeros(SR)
    spike[100] = 0.5                                          # quiet overall, high peak: cannot reach -23 unclipped
    _, _, lim = lt.normalize_loudness(spike, SR, -23.0)
    assert lim


def _lt_fixture(tmp_path):
    import soundfile as sf
    clips, refs = tmp_path / "clips", tmp_path / "refs"
    clips.mkdir()
    refs.mkdir()
    for s, fc in (("syn000", 300), ("syn001", 500), ("stem_a", 700), ("stem_b", 900)):
        sf.write(refs / f"{s}.wav", 0.5 * note(fc_hz=fc), SR, subtype="PCM_16")
        for sysname, mult in (("X", 1.05), ("Y", 1.6), ("Z", 2.2)):
            sf.write(clips / f"{s}__{sysname}.wav", 0.5 * note(fc_hz=fc * mult), SR, subtype="PCM_16")
    return clips, refs


def test_listening_test_make_and_analyze(tmp_path):
    import listening_test_v5 as lt
    clips, refs = _lt_fixture(tmp_path)
    out = tmp_path / "lt"
    key = lt.make(lt.main.__globals__["argparse"].Namespace(
        clips=str(clips), refs=str(refs), pair=[["X", "Y"], ["X", "Z"]], stems=None,
        domain=["syn*=synthetic", "stem_*=real"], listeners=3, checks=2, check_system=None, lufs=-23.0, seed=0,
        out=str(out)))
    assert all((out / f"listener_{i:02d}" / "index.html").exists() for i in (1, 2, 3))
    assert all(len(key["listeners"][f"L{i:02d}"]) == 4 * 2 + 2 for i in (1, 2, 3))
    orders = [[(t["stem"], t["A"], t["B"]) for _, t in sorted(key["listeners"][l].items())] for l in ("L01", "L02")]
    assert orders[0] != orders[1]                             # each listener has their own order and sides
    names = os.listdir(out / "audio")
    assert not any(s in n for n in names for s in ("X", "Y", "Z", "syn", "stem"))
    import soundfile as sf
    for n in names[:6]:
        y, sr = sf.read(out / "audio" / n)
        assert lt.integrated_loudness(y, sr) == pytest.approx(-23.0, abs=0.3)
    answers = []
    for lid, prefer_x, fail in (("L01", 1.0, False), ("L02", 0.5, False), ("L03", 1.0, True)):
        rng = np.random.default_rng(1)
        for tid, t in key["listeners"][lid].items():
            if t["check"]:
                want = t["B"] if fail else ("A" if t["A"] == "__truth__" else "B")
                choice = want if not fail else ("A" if t["A"] != "__truth__" else "B")
            else:
                pick_x = rng.random() < prefer_x
                choice = ("A" if t["A"] == "X" else "B") if pick_x else ("B" if t["A"] == "X" else "A")
            answers.append({"trial": tid, "choice": choice, "listener": lid})
    res = lt.analyze_answers(key, answers, focus="X", n_boot=2000)
    assert res["excluded"] == ["L03"] and res["n_listeners"] == 2
    xy = res["pairs"][0]["all"]
    assert 0.5 <= xy["pref"] <= 1.0 and xy["n_trials"] == 8
    lo, hi = xy["ci_participants"]
    assert lo <= xy["pref"] <= hi
    assert res["pairs"][0]["synthetic"]["n_trials"] == 4 and res["pairs"][0]["real"]["n_trials"] == 4
    assert res["focus"]["all"]["n_trials"] == 16
