# Owner: KUANG (created 2026-10-19)
"""CPU tests for the V6 review fixes and tools (V6_REVIEW.md). No Surge, no GPU. Run from scripts/synth_inversion:

    python -m pytest test_v6.py -q

What these show: the layout registry, the real-stem eval's note handling, the live-mode trainer data path, the bank
checks, the acoustic-loss options, layout-aware Flow-DPO and the ablation tools behave as documented, on tiny random
models and a fake renderer. They do NOT show that any of it improves Surge audio; that is the desktop plan in
V6_REVIEW.md.
"""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _install_fakes():
    """essentia (inharmonicity_target imports it at module level) and mido are not on this laptop; minimal stand-ins
    so the trainers and the real-stem eval import. Real modules win when present."""
    def fake(name):
        m = types.ModuleType(name)

        def ga(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **k: None
        m.__getattr__ = ga
        sys.modules[name] = m
        return m
    try:
        import essentia.standard  # noqa: F401
    except Exception:
        fake("essentia").standard = fake("essentia.standard")
    try:
        import mido  # noqa: F401
    except Exception:
        mido = fake("mido")

        class Message:
            def __init__(self, type, **kw):
                self.type = type
                self.__dict__.update(kw)
        mido.Message = Message


_install_fakes()

import layout_v6 as L  # noqa: E402
import surge_spec_v5 as S5  # noqa: E402
import param_codec_v5 as C5  # noqa: E402


# ------------------------------------------------------------------------------------------------ fixtures
def _manifolds(tmp_path, n=12, seed=0):
    """Two small soundset manifolds (60-d prior draws), with a val split."""
    from realistic_prior_v4 import sample_unconstrained_random_patch
    rng = np.random.RandomState(seed)
    paths = []
    for name in ("bass", "all"):
        params = np.stack([sample_unconstrained_random_patch(rng)[0] for _ in range(n)]).astype(np.float64)
        is_val = np.arange(n) % 4 == 0
        p = tmp_path / f"{name}.npz"
        np.savez(p, params=params, is_val=is_val)
        paths.append(str(p))
    return paths


def _prior(tmp_path, split="train"):
    from realistic_prior_v4 import UnifiedSoundsetPrior
    b, a = _manifolds(tmp_path)
    return UnifiedSoundsetPrior(b, a, split=split)


def _v5_flow(path=None, seed=0, cond_noise=False):
    import models_v4 as m4
    torch.manual_seed(seed)
    f = m4.build_model_v4(param_dim=C5.ENCODED_DIM, hidden_dim=32, num_layers=2, encoder="temporal_asp",
                          enc_kwargs={"d_model": 16}, cond_noise=cond_noise)
    if path is not None:
        torch.save({"model_state": f.state_dict(), "args": {"v4_config": f.v4_config}, "step": 3, "prior_bounds": None},
                   path)
    return f.eval()


def _mel(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, 128, 81, generator=g) * 2 - 1


SR = 44100
N_AUDIO = 35280


def fake_render_v5(vec, midi):
    """Deterministic v5 stand-in: a saw-like tone whose harmonics depend on cutoff; the waveshaper adds harmonics
    in proportion to drive; oscillator 2 adds an octave-up tone in proportion to its volume (inaudible below ~0.02)."""
    I = S5.PARAM_INDEX
    patch = S5.vector_to_patch(np.asarray(vec))
    f0 = 440.0 * 2 ** ((int(midi) + 12 * patch["osc_1_octave_offset"] - 69) / 12)
    t = np.arange(N_AUDIO) / SR
    cutoff = 200.0 * 2 ** (6.0 * float(vec[I["cutoff"]]))
    y = sum(np.sin(2 * np.pi * k * f0 * t) / k / (1 + (k * f0 / cutoff) ** 2) for k in range(1, 30))
    y = y + patch["osc_2_volume"] * np.sin(2 * np.pi * 2 * f0 * t)
    if patch["ws_idx"] > 0:
        g = 1.0 + 30.0 * (patch["drive_raw"] - 0.5)
        y = np.tanh(g * y) / max(np.tanh(g), 1e-6)
    y = y * np.exp(-t / 0.4)
    return (y / (np.max(np.abs(y)) + 1e-7)).astype(np.float32)


class FakePool:
    """RenderPool stand-in: render(vecs, midi) -> [n, N]."""
    def __init__(self, fn=fake_render_v5):
        self.fn = fn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def render(self, vecs, midi):
        return np.stack([self.fn(v, int(m)) for v, m in zip(np.asarray(vecs), np.asarray(midi))])


def _score(metric="logmel_l1"):
    import audio_metrics_v5 as am
    return lambda t, p: am.score_many(list(t), list(p), [metric])[metric]


def _base_vec(seed=0):
    from realistic_prior_v4 import sample_unconstrained_random_patch
    v, m = sample_unconstrained_random_patch(np.random.RandomState(seed))
    I = S5.PARAM_INDEX
    v = v.copy()
    for n in ("osc_2_volume", "osc_3_volume", "ring_1x2_volume", "ring_2x3_volume", "noise_volume", "sub_mix",
              "fm_depth", "sync", "drift"):
        v[I[n]] = 0.0
    v[I["unison"]], v[I["unison_detune"]] = 0.0, 0.0
    v[I["waveshaper_type"]], v[I["drive"]] = 0.0, 0.0
    v[I["filter_feedback"]] = 0.5
    v[I["osc_2_pitch"]] = v[I["osc_3_pitch"]] = 0.5
    v[I["osc_1_octave"]] = 0.5
    v[I["cutoff"]] = 0.5
    return v, int(m)


# ------------------------------------------------------------------------------------------------ layout
def test_layout_widths_come_from_the_codecs():
    assert L.dims() == {92: "v3", 108: "v4", 126: "v5", 114: "v6"}
    assert L.layout_for_dim(126).name == "v5" and L.layout_named("v4").num_params == 60
    with pytest.raises(ValueError):
        L.layout_for_dim(99)


def test_fold_note_and_octave_shift():
    v5 = L.layout_named("v5")
    assert L.fold_note(40, v5) == (40, 0)
    assert L.fold_note(62, v5) == (50, 1)          # one octave down into 28..50
    assert L.fold_note(20, v5) == (32, -1)
    assert L.fold_note(76, v5) == (40, 3)
    v = np.zeros(60, np.float32)
    I = S5.PARAM_INDEX
    v[I["osc_1_octave"]], v[I["osc_2_octave"]], v[I["osc_3_octave"]] = 3 / 6, 0.0, 6 / 6
    w, clamped = L.shift_octaves(v, -1, v5)
    cls = [c.item() for c in C5.target_classes(torch.tensor(w[None]))]
    oct_pos = {i: k for k, (i, _) in enumerate(S5.CAT_INDICES)}
    assert cls[oct_pos[I["osc_1_octave"]]] == 2 and cls[oct_pos[I["osc_3_octave"]]] == 5
    assert cls[oct_pos[I["osc_2_octave"]]] == 0 and clamped == 1       # osc 2 was already at -3


def test_v5_render_octave_matches_codec_label():
    rng = np.random.default_rng(0)
    I = S5.PARAM_INDEX
    oct_pos = [k for k, (i, _) in enumerate(S5.CAT_INDICES) if i == I["osc_1_octave"]][0]
    for raw in np.concatenate([rng.uniform(0, 1, 400), np.array(S5.OCTAVE_THRESHOLDS) + 1e-4,
                               np.array(S5.OCTAVE_THRESHOLDS) - 1e-4, np.arange(7) / 6]):
        v = np.full(60, 0.5, np.float32)
        v[I["osc_1_octave"]] = raw
        rendered = S5.vector_to_patch(v)["osc_1_octave_idx"]
        label = int(C5.target_classes(torch.tensor(v[None]))[oct_pos])
        assert rendered == label, raw


# ------------------------------------------------------------------------------------------------ sampling / DPO
def test_sample_candidates_v5_layout_pins_the_note():
    from flow_sampling_v5 import sample_candidates
    f = _v5_flow()
    v = sample_candidates(f, _mel(2), [30, 45], k=3, steps=2, seed=0)
    assert v.shape == (2, 3, 60)
    assert np.allclose(v[0, :, 0], (30 - 28) / 22) and np.allclose(v[1, :, 0], (45 - 28) / 22)


def test_dpo_v5_neutral_at_init_and_weighted_metric(tmp_path):
    import copy
    import train_flow_dpo_v5 as dpo
    from flow_sampling_v5 import file_md5, mel_of
    from render_pool_v5 import to_int16
    fp = tmp_path / "f.pt"
    f = _v5_flow(fp)
    prior = _prior(tmp_path)
    rng = np.random.RandomState(0)
    n, k = 12, 3
    vecs = np.stack([prior.sample_vector(rng)[0] for _ in range(n * k)]).reshape(n, k, 60).astype(np.float32)
    s1 = rng.uniform(0, 1, (n, k))
    s2 = rng.uniform(0, 1, (n, k))
    audio = np.stack([fake_render_v5(vecs[i, 0], 36) for i in range(n)])
    d = tmp_path / "p"
    os.makedirs(d)
    np.savez(d / "shard_00000.npz", cand_vecs=vecs, score__logmel_l1=s1, score__mss=s2,
             noise__logmel_l1=np.full(n, 0.01), noise__mss=np.full(n, 0.02), target_audio=to_int16(audio),
             midi=np.full(n, 36), cond_index=np.arange(n))
    with open(d / "manifest.json", "w") as fh:
        json.dump({"flow": str(fp), "flow_md5": file_md5(fp), "metrics": ["logmel_l1", "mss"], "tau": 1.0}, fh)
    data = dpo.load_pairs([str(d)], metric="logmel_l1=1,mss=0.5")
    comb = s1 + 0.5 * s2
    for c, gap in zip(data["cond"], data["gap"]):
        assert gap == pytest.approx(comb[c].max() - comb[c].min())          # best vs worst of the weighted sum
    assert np.all(data["gap"] > 0.01 + 0.5 * 0.02)                          # weighted noise floor as the margin
    with pytest.raises(SystemExit):
        dpo.load_pairs([str(d)], metric="modspec_dist")
    mel = mel_of(data["cond_audio"][:4].astype(np.float32) / 32767.0, "cpu")
    w_enc = C5.encoded_weights(torch.from_numpy(S5.domain_weights()))
    loss, st = dpo.dpo_terms(f, copy.deepcopy(f), mel, torch.from_numpy(data["w"][:4]), torch.from_numpy(data["l"][:4]),
                             w_enc, 100.0, torch.Generator().manual_seed(0))
    assert abs(st["delta"]) < 1e-6 and abs(float(loss) - np.log(2)) < 1e-5


# ------------------------------------------------------------------------------------------------ bank / trainer
def test_bank_check_compatible(tmp_path):
    import render_bank_v4 as rb
    b, a = _manifolds(tmp_path)
    good = {"format": 4, "num_params": 60, "bass_manifold_md5": rb.file_md5(b), "all_manifold_md5": rb.file_md5(a),
            "inharm_labels": True, "inharm_config_md5": None}
    assert rb.check_compatible(good, b, a) == ([], [])
    old = {"format": 4, "num_params": 60, "bass_manifold": b, "all_manifold": a}      # pre-fix manifest
    hard, soft = rb.check_compatible(old, b, a, inharm_weight=100.0)
    assert any("inharmonicity" in h for h in hard) and soft == []
    hard, _ = rb.check_compatible(good, b, a, ladder_every=8)
    assert any("ladders" in h for h in hard)
    other = tmp_path / "other.npz"
    np.savez(other, params=np.zeros((4, 60)), is_val=np.zeros(4, bool))
    _, soft = rb.check_compatible(good, str(other), a)
    assert len(soft) == 1 and "bass_manifold" in soft[0]
    hard, _ = rb.check_compatible({**good, "format": 5}, b, a)
    assert hard


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_live_dataset_starts_with_the_soundset_prior(tmp_path, monkeypatch, version):
    T = __import__(f"train_realistic_bass_{version}")
    b, a = _manifolds(tmp_path)
    monkeypatch.setattr(T, "init_synth", lambda *x, **k: object())
    monkeypatch.setattr(T, "render_patch", lambda synth, patch, midi, dur: np.zeros(N_AUDIO, np.float32))
    np.random.seed(0)
    ds = T.RealisticOnlineDataset(b, random_patch_frac=0.5, all_manifold_path=a)
    for i in range(12):                                   # both branches: random patches and prior draws
        audio, vec, _, _ = ds[i]
        assert vec.shape == (60,) and audio.shape == (N_AUDIO,)


def test_acoustic_consistency_options():
    import train_realistic_bass_v5 as T
    from synth_jepa_model import SynthJEPA
    torch.manual_seed(0)
    jepa = SynthJEPA(embed_dim=16, predictor_hidden=32, num_audio_layers=1, num_param_layers=1, num_slices_sigreg=4,
                     ff_dim=32, num_continuous=len(S5.CONT_INDICES), cat_sizes=[k for _, k in S5.CAT_INDICES])
    B = 6
    x = torch.rand(B, C5.ENCODED_DIM, requires_grad=True)
    za = torch.randn(B, 16)
    t = torch.tensor([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    seen = {}
    orig = jepa.encode_params

    def spy(cont, cats):
        seen["cats"] = cats
        return orig(cont, cats)
    jepa.encode_params = spy
    loss_all, used_all = T.acoustic_consistency(jepa, x, za, t)
    assert used_all == 1.0 and all((c.sum(-1) - 1).abs().max() < 1e-5 for c in seen["cats"])
    assert all(c.max() < 1.0 for c in seen["cats"])                          # soft: never a one-hot
    loss_st, used = T.acoustic_consistency(jepa, x, za, t, cat_mode="st", t_min=0.5)
    assert used == pytest.approx(0.5)
    assert all(torch.allclose(c.detach(), torch.nn.functional.one_hot(c.argmax(-1), c.shape[-1]).float(), atol=1e-6)
               for c in seen["cats"])                                         # straight-through: one-hot forward
    loss_st.backward()
    assert x.grad is not None and x.grad[:3].abs().sum() == 0 and x.grad[3:].abs().sum() > 0   # t < 0.5 masked
    assert all(p.grad is None for p in jepa.parameters()) and all(p.requires_grad for p in jepa.parameters())


# ------------------------------------------------------------------------------------------------ real-stem eval
def test_eval_folds_notes_and_prefers_the_real_octave(monkeypatch):
    import eval_real_stems_consensus_v5 as E
    notes = [{"note": 40, "start": 0.0, "dur": 0.2}, {"note": 52, "start": 0.3, "dur": 0.1},
             {"note": 52, "start": 0.5, "dur": 0.2}]
    assert E.octave_partner(notes[0], notes) is notes[2] and E.octave_partner(notes[2], notes) is None
    monkeypatch.setattr(E.librosa.effects, "pitch_shift", lambda *a, **k: (_ for _ in ()).throw(AssertionError("used")))
    chunk = np.sin(np.arange(4000) / 10.0).astype(np.float32)
    ph, _ = E.build_conditioning_targets(chunk, 0.1, chunk_oct=0.5 * chunk)
    assert np.abs(ph[int(0.25 * SR): int(0.25 * SR) + 4000]).max() > 0.9          # the real partner, renormalised
    f = _v5_flow()
    mel = _mel(1)
    direct, k0 = E.sample_candidates_adaptive(f, mel, 50, k=3, steps=2, seed=7)
    folded, k1 = E.sample_candidates_adaptive(f, mel, 62, k=3, steps=2, seed=7)
    assert k0 == 0 and k1 == 1
    want, _ = L.shift_octaves(direct, -1, L.layout_named("v5"))
    assert np.allclose(folded, want)                     # same draw, played an octave up at the real note
    assert np.allclose(direct[:, 0], 1.0)                # note 50 = top of 28..50 (was 0.45 with the 12..96 range)


def test_audio_eval_soundset_draws_follow_the_prior_stream(tmp_path):
    import audio_eval_v5 as ae
    prior = _prior(tmp_path, split="val")
    items = ae.draw_items_soundset(prior, 40, seed=3)
    rng = np.random.RandomState(3)
    for v in items["vecs"]:
        assert np.array_equal(np.asarray(prior.sample_vector(rng)[0], np.float32), v)
    for src, c in zip(items["source"], items["preset"]):
        assert c.startswith("r") if src == "random" else c.split(":")[0] in ("bass", "all")


# ------------------------------------------------------------------------------------------------ ablation
def test_simplify_and_active():
    import ablation_v6 as ab
    I = S5.PARAM_INDEX
    v, _ = _base_vec()
    assert not any(ab.active(v, n, S5) for n in ab.SIMPLIFICATIONS)          # a plain one-oscillator patch
    v2 = v.copy()
    v2[I["unison_detune"]] = 0.3                                              # detune with 1 voice: no effect
    assert not ab.active(v2, "unison_1", S5)
    v2[I["waveshaper_type"]], v2[I["drive"]] = 2 / 7, 0.5
    assert ab.active(v2, "waveshaper_off", S5)
    s = ab.simplify(v2, "waveshaper_off", S5)
    assert S5.vector_to_patch(s)["ws_idx"] == 0 and not ab.active(s, "waveshaper_off", S5)


def test_counterfactual_labels_and_occam():
    import ablation_v6 as ab
    I = S5.PARAM_INDEX
    base, midi = _base_vec(1)
    loud = base.copy()
    loud[I["waveshaper_type"]], loud[I["drive"]] = 2 / 7, 0.9            # audible distortion
    faint = base.copy()
    faint[I["osc_2_volume"]] = 0.002                                      # an oscillator nobody can hear
    vecs = np.stack([loud, faint, base])
    pool = FakePool()
    score = _score()
    audio, noise = ab.counterfactual(vecs, [midi] * 3, pool.render, score)
    assert np.allclose(noise, 0.0, atol=1e-9)                             # the fake renderer is deterministic
    d = ab.counterfactual_deltas(vecs, [midi] * 3, audio, pool.render, score, ["waveshaper_off", "osc2_off"], S5)
    assert np.isfinite(d["waveshaper_off"][0]) and np.isnan(d["waveshaper_off"][1:]).all()
    aud_ws = ab.audible(d["waveshaper_off"], noise, floor=0.01)
    aud_o2 = ab.audible(d["osc2_off"], noise, floor=0.01)
    assert aud_ws.tolist() == [True, False, False] and aud_o2.tolist() == [False, False, False]
    # Occam toward each item's own render: the inaudible oscillator goes, the audible waveshaper stays
    new, start, final, log = ab.occam(vecs, audio, [midi] * 3, pool.render, score, S5, tol_rel=0.02, tol_abs=0.005)
    assert S5.vector_to_patch(new[0])["ws_idx"] > 0
    assert new[1][I["osc_2_volume"]] == 0.0 and [e[1] for e in log[1]] == ["osc2_off"]
    assert log[2] == [] and np.all(final <= start + 0.005 + 1e-9)


def test_auroc_and_probe():
    import ablation_v6 as ab
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=60), rng.random(60) < 0.4
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert ab.auroc(s, y) == pytest.approx(brute)
    assert ab.auroc([1, 1, 2, 2], [0, 1, 0, 1]) == pytest.approx(0.5)
    X = rng.normal(size=(200, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=200) > 0
    assert ab.auroc(ab.probe_cv(X, y), y) > 0.95
    assert abs(ab.auroc(ab.probe_cv(X, rng.random(200) < 0.5), rng.random(200) < 0.5) - 0.5) < 0.2


def test_labels_and_occam_cli_with_fake_renders(tmp_path):
    import ablation_v6 as ab
    import audio_eval_v5  # noqa: F401  (items.npz layout below follows audio_eval_v5)
    from render_pool_v5 import to_int16
    prior = _prior(tmp_path)
    out = tmp_path / "lab"
    s = ab.main(["labels", "--n", "16", "--seed", "1", "--enrich", "waveshaper_off=0.5", "--metric", "logmel_l1",
                 "--floor", "0.01", "--features", "waveshaper_off", "osc2_off", "--workers", "0", "--out", str(out)],
                pool=FakePool(), prior=prior)
    assert s["features"]["waveshaper_off"]["on_share"] >= 0.5
    z = np.load(out / "labels.npz")
    assert z["audio"].shape == (16, N_AUDIO) and np.isfinite(z["delta__waveshaper_off"]).sum() >= 8
    # occam on an audio_eval-style items file whose 'system' proposes the target plus a faint extra oscillator
    I = S5.PARAM_INDEX
    vecs = z["vecs"][:4].copy()
    cand = np.repeat(vecs[:, None], 2, axis=1)
    cand[:, 0, I["osc_2_volume"]] = np.maximum(cand[:, 0, I["osc_2_volume"]], 0.002)
    tgt = np.stack([fake_render_v5(v, m) for v, m in zip(vecs, z["midi"][:4])])
    np.savez(tmp_path / "items.npz", target_audio=to_int16(tgt), midi=z["midi"][:4], S__vecs=cand,
             S__score__logmel_l1=np.array([[0.0, 1.0]] * 4))
    r = ab.main(["occam", "--items", str(tmp_path / "items.npz"), "--system", "S", "--metrics", "logmel_l1",
                 "--workers", "0", "--tol_abs", "0.005", "--out", str(tmp_path / "oc")], pool=FakePool())
    assert r["n"] == 4 and r["metric_after"]["logmel_l1"] <= r["metric_before"]["logmel_l1"] + 0.005


def test_legacy_jepa_loads_as_trained_and_new_jepas_refuse_truncation(tmp_path):
    """v3-trainer JEPAs before 2026-10-19 used only the first 20 continuous / 3 categorical parameters; they load
    with that behaviour (legacy_param_truncation), and a correctly sized model refuses mismatched input."""
    import surge_spec_v3 as S3
    from synth_jepa_model import SynthJEPA
    from synth_jepa_search_v3 import WelfordNormalizer, load_synth_jepa
    torch.manual_seed(0)
    kw = dict(embed_dim=16, predictor_hidden=16, num_audio_layers=1, num_param_layers=1, num_slices_sigreg=4, ff_dim=16)
    old = SynthJEPA(**kw).eval()                                   # surge_spec (v1) sizes, as the v3 trainer built it
    norm = WelfordNormalizer(n_mels=128)
    norm.update(np.random.default_rng(0).normal(size=(4, 128, 81)))
    norm.freeze()
    torch.save({"model_state_dict": old.state_dict(), "model_kwargs": kw, "normalizer": norm.state_dict()},
               tmp_path / "jepa_old.pt")
    with pytest.warns(UserWarning, match="legacy JEPA"):
        m, _ = load_synth_jepa(str(tmp_path / "jepa_old.pt"))
    B = 3
    cont = torch.rand(B, len(S3.CONT_INDICES)) * 2 - 1
    cats = [torch.nn.functional.one_hot(torch.randint(0, k, (B,)), k).float() for _, k in S3.CAT_INDICES]
    with torch.no_grad():
        got = m.encode_params(cont, cats)
        want = old.param_encoder.__class__.forward(old.param_encoder, cont[:, :20], cats[:3])   # what it was trained on
    assert torch.allclose(got, want, atol=1e-6)
    full = SynthJEPA(**kw, num_continuous=len(S3.CONT_INDICES), cat_sizes=[k for _, k in S3.CAT_INDICES]).eval()
    with torch.no_grad():
        assert full.encode_params(cont, cats).shape == (B, 16)
    with pytest.raises(ValueError):
        old.encode_params(cont, cats)                              # built without the legacy flag: refuses
