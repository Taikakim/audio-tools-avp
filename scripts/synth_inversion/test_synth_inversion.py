"""Tests for the synth-inversion toolchain that run WITHOUT Surge XT or a GPU.

A FakeSurge object stands in for the pedalboard plugin (same .parameters / .process /
.reset / .raw_state surface), so the evaluation scripts run end to end on CPU.

    cd scripts/synth_inversion && python -m pytest -q test_synth_inversion.py
"""
import os
import sys

import h5py
import numpy as np
import pytest
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_utils  # noqa: E402
import param_codec as codec  # noqa: E402
import surge_spec  # noqa: E402
from models import FlowMatchingResMLP, ResMLPInverter, load_inverter  # noqa: E402
from optimizer import build_optimizers, partition_params  # noqa: E402


# ---------------------------------------------------------------------------
# Fake Surge
# ---------------------------------------------------------------------------
class _Param:
    def __init__(self, name, strings):
        self.name, self._strings, self.raw_value = name, strings, 0.0

    @property
    def string_value(self):
        if self._strings is None:
            return f"{self.raw_value:.3f}"
        return min(self._strings.items(), key=lambda kv: abs(kv[0] - self.raw_value))[1]


class FakeSurge:
    """Deterministic toy synth: the output depends on the patch so losses vary."""

    def __init__(self, wrong_filter_strings=False):
        names = [n for n, _ in surge_spec.LP_FILTERS]
        if wrong_filter_strings:  # simulate a Surge update that shifted the enum by one
            names = names[1:] + names[:1]
        strings = {
            "a_filter_1_type": {raw: n for n, (_, raw) in zip(names, surge_spec.LP_FILTERS)},
            "a_waveshaper_type": {raw: n for n, raw in surge_spec.WAVESHAPER_TYPES},
            "fx_a1_fx_type": {0.3103: "Chorus", 0.0: "Off"},
            "fx_a2_fx_type": {0.0345: "Delay", 0.0: "Off"},
            "a_osc_1_unison_voices": {0.0: "1 voice", 0.05: "2 voices"},
        }
        self.parameters = _ParamDict(strings)

    def reset(self):
        pass

    def process(self, events, duration, sample_rate, num_channels):
        n = int(round(duration * sample_rate))
        t = np.arange(n) / sample_rate
        P = {k: p.raw_value for k, p in self.parameters.items()}
        note = events[0].note if events else 36
        f0 = 440.0 * 2 ** ((note - 69) / 12)
        env = np.exp(-t / (0.02 + P.get("a_amp_eg_decay", 0.3)))
        saw = 2 * ((t * f0) % 1.0) - 1
        sq = np.sign(np.sin(2 * np.pi * f0 * t))
        shape = P.get("a_osc_1_shape", 0.5)
        sig = ((1 - shape) * saw + shape * sq) * env * (0.2 + P.get("a_filter_1_cutoff", 0.5))
        return np.stack([sig] * num_channels).astype(np.float32)

    @property
    def raw_state(self):
        return repr(sorted((k, p.raw_value) for k, p in self.parameters.items())).encode()


class _ParamDict(dict):
    def __init__(self, strings):
        super().__init__()
        self._strings = strings

    def __getitem__(self, name):
        if name not in self:
            dict.__setitem__(self, name, _Param(name, self._strings.get(name)))
        return dict.__getitem__(self, name)


def _v1_reference_vector(seed):
    """VERBATIM copy of the draw sequence in generate_bass_dataset.py @ 686ec81 (the code
    that wrote surge_bass_200k.h5). Frozen: if this test fails, draw_patch has changed the
    RNG stream and v1 datasets can no longer be replayed."""
    np.random.seed(seed)
    filter_choice = int(np.random.randint(0, 10))
    midi_note = int(np.random.randint(28, 51))
    shape = float(np.random.uniform(0.0, 1.0))
    width = float(np.random.uniform(0.0, 1.0))
    sub_mix = float(np.random.uniform(0.0, 0.85))
    sync = float(np.random.uniform(0.0, 0.40)) if np.random.rand() < 0.3 else 0.0
    fm_depth = float(np.random.uniform(0.0, 0.45)) if np.random.rand() < 0.4 else 0.0
    use_unison = 1.0 if np.random.rand() < 0.25 else 0.0
    unison_detune = float(np.random.uniform(0.05, 0.35)) if use_unison > 0.0 else 0.0
    cutoff = float(np.random.uniform(0.08, 0.92))
    resonance = float(np.random.uniform(0.0, 0.85))
    keytrack_raw = float(np.random.uniform(0.5, 1.0))
    feg_amount = float(np.random.uniform(0.2, 0.95))
    feg_decay = float(np.random.uniform(0.03, 0.65))
    feg_sustain = float(np.random.uniform(0.0, 0.60))
    aeg_decay = float(np.random.uniform(0.05, 0.65))
    aeg_sustain = float(np.random.uniform(0.0, 0.80))
    aeg_release = float(np.random.uniform(0.01, 0.40))
    if np.random.rand() < 0.45:
        ws_idx = int(np.random.randint(1, 6))
        drive_raw = float(np.random.uniform(0.50, 0.82))
    else:
        ws_idx = 0
        drive_raw = 0.50
    chorus_mix = float(np.random.uniform(0.1, 0.6)) if np.random.rand() < 0.35 else 0.0
    delay_mix = float(np.random.uniform(0.1, 0.45)) if np.random.rand() < 0.25 else 0.0
    delay_fb = float(np.random.uniform(0.1, 0.5)) if delay_mix > 0.0 else 0.0
    note_dur = float(np.random.uniform(0.18, 0.45))
    vec = np.array([
        (midi_note - 28.0) / 22.0, filter_choice / 9, shape, width, sub_mix, sync, fm_depth, use_unison,
        unison_detune / 0.35, cutoff, resonance, (keytrack_raw - 0.5) / 0.5, feg_amount, feg_decay, feg_sustain,
        aeg_decay, aeg_sustain, aeg_release, ws_idx / 5, (drive_raw - 0.50) / 0.32, chorus_mix, delay_mix, delay_fb,
    ], dtype=np.float32)
    return vec, note_dur


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------
def test_draw_patch_reproduces_v1_rng_stream():
    for seed in range(5000, 7000):
        ref_vec, ref_dur = _v1_reference_vector(seed)
        patch = surge_spec.draw_patch(seed)
        np.testing.assert_array_equal(surge_spec.patch_to_vector(patch), ref_vec)
        assert patch["note_dur"] == ref_dur


def test_note_dur_range_keeps_stream():
    a, b = surge_spec.draw_patch(123), surge_spec.draw_patch(123, (0.06, 0.45))
    np.testing.assert_array_equal(surge_spec.patch_to_vector(a), surge_spec.patch_to_vector(b))
    assert b["note_dur"] < a["note_dur"]


def test_vector_patch_roundtrip():
    for seed in range(200):
        patch = surge_spec.draw_patch(seed)
        back = surge_spec.vector_to_patch(surge_spec.patch_to_vector(patch))
        for k, v in back.items():
            assert v == pytest.approx(patch[k], abs=1e-6), k


def test_codec_roundtrip_and_weights():
    p = torch.from_numpy(np.stack([surge_spec.patch_to_vector(surge_spec.draw_patch(s)) for s in range(256)]))
    enc = codec.encode(p)
    assert enc.shape == (256, codec.ENCODED_DIM)
    assert torch.allclose(codec.decode(enc), p, atol=1e-6)
    w = codec.encoded_weights(torch.ones(surge_spec.NUM_PARAMS))
    assert w.shape == (codec.ENCODED_DIM,)
    err = codec.per_param_error(p, p)
    assert float(err.abs().max()) == 0.0


def test_categorical_error_is_misclassification():
    p = torch.from_numpy(surge_spec.patch_to_vector(surge_spec.draw_patch(1))).unsqueeze(0)
    q = p.clone()
    fi = surge_spec.PARAM_INDEX["filter_type"]
    q[0, fi] = (round(float(p[0, fi]) * 9) + 1) % 10 / 9
    assert float(codec.per_param_error(q, p)[0, fi]) == 1.0


def test_verify_mapping():
    surge_spec.verify_surge_mapping(FakeSurge())
    with pytest.raises(RuntimeError, match="a_filter_1_type"):
        surge_spec.verify_surge_mapping(FakeSurge(wrong_filter_strings=True))


def test_apply_patch_sets_everything_no_leak():
    synth = FakeSurge()
    a, b = surge_spec.draw_patch(1), surge_spec.draw_patch(2)
    surge_spec.apply_patch(synth, a)
    surge_spec.apply_patch(synth, b)
    after_b = synth.raw_state
    fresh = FakeSurge()
    surge_spec.apply_patch(fresh, b)
    assert fresh.raw_state == after_b


# ---------------------------------------------------------------------------
# Models / optimizer
# ---------------------------------------------------------------------------
def _batch(B=8):
    p = torch.from_numpy(np.stack([surge_spec.patch_to_vector(surge_spec.draw_patch(s)) for s in range(B)]))
    return torch.randn(B, 128, 81), p


def test_models_v2_shapes_and_decode():
    mel, p = _batch()
    r = ResMLPInverter(hidden_dim=32, num_layers=2)
    assert r(mel).shape == (8, codec.ENCODED_DIM)
    assert r.predict_params(mel).shape == (8, surge_spec.NUM_PARAMS)
    f = FlowMatchingResMLP(hidden_dim=32, num_layers=2)
    assert f.param_dim == codec.ENCODED_DIM
    out = f.predict_params(mel, num_steps=3, generator=torch.Generator().manual_seed(0))
    assert out.shape == (8, surge_spec.NUM_PARAMS)
    # categoricals decode to valid class positions
    for i, k in surge_spec.CAT_INDICES:
        cls = out[:, i] * (k - 1)
        assert torch.allclose(cls, cls.round(), atol=1e-5)


def test_time_embedding_scale_spreads_t():
    from models import SinusoidalTimeEmbedding
    t = torch.tensor([0.0, 1.0])
    old = SinusoidalTimeEmbedding(256, scale=1.0)(t)
    new = SinusoidalTimeEmbedding(256, scale=1000.0)(t)
    assert (old[1] - old[0]).abs().mean() < (new[1] - new[0]).abs().mean()


def test_optimizer_partition_keeps_io_layers_on_adam():
    f = FlowMatchingResMLP(hidden_dim=32, num_layers=2)
    muon, adam = partition_params(f)
    muon_ids = {id(p) for p in muon}
    named = dict(f.named_parameters())
    assert id(named["param_proj.weight"]) not in muon_ids
    assert id(named["head.weight"]) not in muon_ids
    assert id(named["blocks.0.fc1.weight"]) in muon_ids
    opt_m, opt_a = build_optimizers(f)
    opt_a.train()
    mel, p = _batch()
    from train_bracket import flow_loss
    loss = flow_loss(f, mel, p, codec.encoded_weights(torch.ones(23)))
    loss.backward()
    opt_m.step()
    opt_a.step()
    assert opt_m.last_update_norm > 0


def test_load_inverter_v1_checkpoints(tmp_path):
    # v1 ResMLP with full args, v1 flow with NO args (older scripts)
    r1 = ResMLPInverter(param_dim=23, hidden_dim=32, num_layers=2, encoding=codec.ENCODING_V1)
    torch.save({"model_state": r1.state_dict(), "args": {"model_type": "resmlp"}}, tmp_path / "r1.pt")
    f1 = FlowMatchingResMLP(hidden_dim=32, num_layers=3, encoding=codec.ENCODING_V1, time_scale=1.0)
    torch.save({"model_state": f1.state_dict()}, tmp_path / "f1.pt")
    mel, _ = _batch(2)
    lr = load_inverter(tmp_path / "r1.pt")
    assert lr.encoding == codec.ENCODING_V1 and torch.allclose(lr.predict_params(mel), r1.eval().predict_params(mel))
    lf = load_inverter(tmp_path / "f1.pt")
    assert isinstance(lf, FlowMatchingResMLP) and lf.encoding == codec.ENCODING_V1
    assert lf.time_embed[0].scale == 1.0 and len(lf.blocks) == 3


# ---------------------------------------------------------------------------
# Audio utils
# ---------------------------------------------------------------------------
def test_stft_loss_identity_and_order():
    loss = audio_utils.MultiScaleSTFTLoss()
    t = np.arange(surge_spec.AUDIO_LEN) / surge_spec.SAMPLE_RATE
    a = np.sin(2 * np.pi * 110 * t).astype(np.float32)
    b = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    assert loss(a, a) < 1e-3 < loss(a, b)


def test_prepare_target_resamples_trims_and_flags(tmp_path):
    sr = 48000
    clip = np.zeros(int(0.15 * sr), dtype=np.float32)
    onset = int(0.03 * sr)
    clip[onset:] = np.sin(2 * np.pi * 65.4 * np.arange(len(clip) - onset) / sr)
    path = tmp_path / "hit.wav"
    sf.write(path, clip, sr)
    with pytest.warns(UserWarning, match="OUT OF DISTRIBUTION"):
        y, info = audio_utils.prepare_target(str(path), note_dur=0.08)
    assert len(y) == surge_spec.AUDIO_LEN and info["orig_sr"] == sr
    assert 25 < info["trimmed_leading_ms"] < 30
    assert len(info["out_of_distribution"]) == 2
    assert info["score_len"] < surge_spec.AUDIO_LEN
    audio_utils.make_mel_spec(y)
    with pytest.raises(ValueError):
        audio_utils.make_mel_spec(np.zeros(1000, dtype=np.float32))


# ---------------------------------------------------------------------------
# End to end on a tiny synthetic h5
# ---------------------------------------------------------------------------
def _tiny_h5(path, n=96, store_note_dur=False):
    synth = FakeSurge()
    with h5py.File(path, "w") as h5:
        h5.attrs["seed_offset"] = surge_spec.SEED_OFFSET
        da = h5.create_dataset("audio", (n, surge_spec.AUDIO_LEN), dtype=np.float32)
        dm = h5.create_dataset("mel", (n, *audio_utils.MEL_SHAPE), dtype=np.float32)
        dp = h5.create_dataset("params", (n, surge_spec.NUM_PARAMS), dtype=np.float32)
        nd = h5.create_dataset("note_dur", (n,), dtype=np.float32) if store_note_dur else None
        for i in range(n):
            patch = surge_spec.draw_patch(i + surge_spec.SEED_OFFSET)
            mono = surge_spec.render_patch(synth, patch, patch["midi_note"], patch["note_dur"])
            da[i], dm[i], dp[i] = mono, audio_utils.make_mel_spec(mono), surge_spec.patch_to_vector(patch)
            if nd is not None:
                nd[i] = patch["note_dur"]


@pytest.mark.parametrize("model_type,opt", [("resmlp", "normuon_sf"), ("flow", "normuon_sf"), ("flow", "adamw")])
def test_train_end_to_end(tmp_path, model_type, opt):
    import train_bracket
    h5p = tmp_path / "tiny.h5"
    _tiny_h5(h5p)
    # Defaults come from the script's own parser, so new flags never break this test.
    args = train_bracket.build_parser().parse_args([
        "--h5_path", str(h5p), "--model_type", model_type, "--opt_family", opt, "--hidden_dim", "32",
        "--num_layers", "2", "--batch_size", "16", "--adam_warmup_steps", "5", "--epochs", "2",
        "--val_batches", "2", "--flow_val_draws", "3", "--flow_val_steps", "3",
        "--run_id", f"t_{model_type}_{opt}", "--output_dir", str(tmp_path / "out"), "--device", "cpu"])
    train_bracket.train(args)
    ckpt = tmp_path / "out" / f"{args.run_id}_best.pt"
    m = load_inverter(ckpt)
    assert m.encoding == codec.ENCODING_V2
    lines = (tmp_path / "out" / "leaderboard_v2.tsv").read_text().strip().splitlines()
    assert len(lines) == 2 and len(lines[1].split("\t")) == len(train_bracket.LEADERBOARD_COLS)


@pytest.mark.parametrize("store_note_dur", [False, True])
def test_evaluate_holdout_audio_end_to_end(tmp_path, monkeypatch, store_note_dur):
    import evaluate_holdout_audio as eh
    h5p = tmp_path / "tiny.h5"
    _tiny_h5(h5p, n=40, store_note_dur=store_note_dur)
    r = ResMLPInverter(hidden_dim=32, num_layers=2)
    f = FlowMatchingResMLP(hidden_dim=32, num_layers=2)
    torch.save({"model_state": r.state_dict(), "args": {"model_type": "resmlp", "param_encoding": codec.ENCODING_V2}}, tmp_path / "r.pt")
    torch.save({"model_state": f.state_dict(), "args": {"model_type": "flow", "param_encoding": codec.ENCODING_V2}}, tmp_path / "f.pt")
    monkeypatch.setattr(eh, "init_synth", lambda *a, **k: FakeSurge())
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["x", "--h5", str(h5p), "--ckpt", f"r={tmp_path/'r.pt'}", "--ckpt", f"f={tmp_path/'f.pt'}",
                                      "--n", "6", "--flow_draws", "2", "--flow_steps", "2", "--out_dir", str(out)])
    eh.main()
    import json
    s = json.loads((out / "holdout_audio_summary.json").read_text())
    # oracle = re-render of the float32-stored true params: a near-zero floor (log-magnitude
    # picks up round-off in near-silent tail bins), far below the trivial baseline
    assert s["methods"]["oracle"]["median"] < 1e-3
    assert s["methods"]["oracle"]["mean"] < 0.05 * s["methods"]["mean_patch"]["mean"]
    assert {"r/point", "f/draw0", "f/best_of_2", "mean_patch"} <= set(s["methods"])


def test_holdout_replay_detects_foreign_h5(tmp_path):
    import evaluate_holdout_audio as eh
    h5p = tmp_path / "tiny.h5"
    _tiny_h5(h5p, n=10)
    with h5py.File(h5p, "a") as h5:
        h5["params"][3] = 0.5
    with h5py.File(h5p, "r") as h5, pytest.raises(RuntimeError, match="RNG replay"):
        eh.note_durations(h5, list(range(10)))


def test_match_and_stem_scripts_end_to_end(tmp_path, monkeypatch):
    import evaluate_200k_inversion as ev
    import match_untitled_note as mu
    sr = 44100
    t = np.arange(int(0.3 * sr)) / sr
    sf.write(tmp_path / "note.wav", (np.sign(np.sin(2 * np.pi * 65.4 * t)) * np.exp(-t / 0.1)).astype(np.float32), sr)
    f = FlowMatchingResMLP(hidden_dim=32, num_layers=2)
    torch.save({"model_state": f.state_dict(), "args": {"model_type": "flow", "param_encoding": codec.ENCODING_V2}}, tmp_path / "f.pt")
    for mod in (mu, ev):
        monkeypatch.setattr(mod, "init_synth", lambda *a, **k: FakeSurge())
    monkeypatch.setattr(sys, "argv", ["x", "--target", str(tmp_path / "note.wav"), "--note_dur", "0.2",
                                      "--ckpt", f"f={tmp_path/'f.pt'}", "--flow_draws", "2", "--flow_steps", "2",
                                      "--filters", "LP 12 dB,LP K35", "--de_maxiter", "1", "--de_popsize", "2",
                                      "--out_dir", str(tmp_path / "m")])
    mu.main()
    assert (tmp_path / "m" / "untitled_de_reference.pedalboard_state").exists()
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", f"f={tmp_path/'f.pt'}", "--stem", f"n:{tmp_path/'note.wav'}:36:0.2",
                                      "--flow_draws", "2", "--flow_steps", "2", "--out_dir", str(tmp_path / "s")])
    ev.main()
    assert (tmp_path / "s" / "stem_inversion_results.json").exists()


# ---------------------------------------------------------------------------
# Synth-JEPA (arXiv:2609.31024)
# ---------------------------------------------------------------------------
def test_cont_bounds_cover_training_draws_and_canonical():
    vecs = np.stack([surge_spec.patch_to_vector(surge_spec.draw_patch(s)) for s in range(3000)])
    for name, (lo, hi) in surge_spec.CONT_BOUNDS.items():
        col = vecs[:, surge_spec.PARAM_INDEX[name]]
        assert col.min() >= lo - 1e-6 and col.max() <= hi + 1e-6, name
    assert set(surge_spec.CONT_BOUNDS) == {surge_spec.PARAM_NAMES[i] for i in surge_spec.CONT_INDICES}
    np.testing.assert_array_equal(surge_spec.canonicalize_vector(vecs), vecs)  # training data is canonical


def test_welford_is_channelwise_and_matches_numpy():
    from synth_jepa_search import WelfordNormalizer
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 128, 81)) * rng.uniform(0.5, 3, size=(1, 128, 1)) + rng.normal(size=(1, 128, 1))
    norm = WelfordNormalizer()
    for s in range(0, 50, 7):
        norm.update(x[s:s + 7])
    norm.freeze()
    frames = x.transpose(1, 0, 2).reshape(128, -1)
    np.testing.assert_allclose(norm.frozen_mean.numpy(), frames.mean(1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(norm.frozen_std.numpy(), frames.std(1, ddof=1), rtol=1e-4)
    with pytest.raises(RuntimeError):
        WelfordNormalizer().normalize(torch.zeros(1, 128, 81))


def _tiny_jepa():
    from synth_jepa_model import SynthJEPA
    return SynthJEPA(embed_dim=32, predictor_hidden=64, num_audio_layers=1, num_param_layers=2, ff_dim=64)


def test_jepa_model_structure():
    from synth_jepa_model import SynthJEPA
    m = _tiny_jepa()
    assert m.audio_encoder.n_tokens == 41  # stride-2 conv over 81 frames, no up-pooling
    cross = [n for n, _ in m.param_encoder.named_modules() if n.endswith("cross_attn")]
    assert cross == ["read_in.cross_attn"]  # ONE cross-attention, then self-attention blocks
    assert len(m.param_encoder.self_blocks.layers) == 2
    mel, p = _batch(4)
    from train_synth_jepa import prepare_inputs
    cont, cats = prepare_inputs(p)
    assert float(cont.min()) >= -1 and float(cont.max()) <= 1
    lp, la, sa, sp, za, zp = m(mel, cont, cats)
    assert za.shape == zp.shape == (4, 32)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert float(m.sigreg(za.detach(), g1)) == float(m.sigreg(za.detach(), g2))
    full = SynthJEPA()
    assert 45e6 < sum(q.numel() for q in full.parameters()) < 60e6  # paper: 53M


def test_jepa_search_contract():
    from synth_jepa_search import SynthJEPASearcher, WelfordNormalizer
    norm = WelfordNormalizer()
    norm.update(np.random.default_rng(0).normal(size=(8, 128, 81)))
    norm.freeze()
    s = SynthJEPASearcher(_tiny_jepa(), norm, device="cpu", seed=0)
    res = s.search(torch.randn(128, 81), midi_note=40, total_eval_budget=512)
    assert 0.9 * 512 <= res["n_evals"] <= 512
    assert res["patch"]["midi_note"] == 40
    v = res["vector"]
    for name, (lo, hi) in surge_spec.CONT_BOUNDS.items():
        if name != "midi_note":
            assert lo - 1e-5 <= v[surge_spec.PARAM_INDEX[name]] <= hi + 1e-5, name
    np.testing.assert_array_equal(surge_spec.canonicalize_vector(v), v)
    again = SynthJEPASearcher(s.model, norm, device="cpu", seed=0)  # same seed -> same evaluation count
    assert again.search(torch.randn(128, 81), midi_note=40, total_eval_budget=512)["n_evals"] == res["n_evals"]


def test_train_synth_jepa_then_eval_end_to_end(tmp_path, monkeypatch):
    import json
    import evaluate_holdout_audio as eh
    import train_synth_jepa
    from synth_jepa_search import load_synth_jepa
    h5p = tmp_path / "tiny.h5"
    _tiny_h5(h5p, n=80)
    monkeypatch.setattr(sys, "argv", ["x", "--h5_path", str(h5p), "--out_dir", str(tmp_path / "j"), "--run_id", "t",
                                      "--epochs", "2", "--batch_size", "8", "--embed_dim", "32", "--num_layers", "1",
                                      "--ff_dim", "64", "--norm_spectrograms", "40", "--val_batches", "2",
                                      "--num_workers", "0", "--device", "cpu"])
    train_synth_jepa.main()
    ckpt = tmp_path / "j" / "t" / "checkpoint_latest.pt"
    model, norm = load_synth_jepa(str(ckpt))
    assert norm.frozen and norm.n_spectrograms == 40
    lines = (tmp_path / "j" / "t" / "val.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and "r_a2p" in json.loads(lines[-1])["val"]
    monkeypatch.setattr(eh, "init_synth", lambda *a, **k: FakeSurge())
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["x", "--h5", str(h5p), "--jepa", f"j={ckpt}", "--jepa_budget", "128",
                                      "--n", "3", "--out_dir", str(out)])
    eh.main()
    s = json.loads((out / "holdout_audio_summary.json").read_text())
    assert "j/search" in s["methods"] and s["jepa_search"]["j"]["evals_per_target_max"] <= 128


def test_exact_gpu_mel_equivalence():
    from audio_utils import ExactGpuMel, make_mel_spec
    t = np.linspace(0, 0.8, 35280, endpoint=False)
    audio = (0.6 * np.sin(2 * np.pi * 130.81 * t) + 0.3 * np.sin(2 * np.pi * 261.63 * t)).astype(np.float32)
    cpu_mel = make_mel_spec(audio)

    gpu_mel = ExactGpuMel(device="cpu")
    t_audio = torch.from_numpy(audio).unsqueeze(0)
    mel_out = gpu_mel(t_audio).squeeze(0).numpy()

    max_diff = float(np.max(np.abs(cpu_mel - mel_out)))
    assert max_diff <= 1e-4, f"ExactGpuMel drifted from librosa make_mel_spec: max_diff={max_diff:.2e}"


def test_envelope_extractor_profiling_and_decoupling():
    from envelope_extractor import (butter_bandpass_filter, compute_envelope_loss_np,
                                    detect_gate_off_changepoint, profile_note_envelope,
                                    zolzer_envelope_follower)

    fs = 44100
    t = np.linspace(0, 0.4, int(0.4 * fs), endpoint=False)
    # Synthesize a note: 100ms gate, fast attack, 60ms decay, 15ms release after gate
    # Sub-band 65 Hz (C2 fundamental), harmonic band 1300 Hz
    gate_s = 0.080
    amp_env = np.where(t < gate_s, np.exp(-t / 0.050), np.exp(-gate_s / 0.050) * np.exp(-(t - gate_s) / 0.010))
    filt_env = np.exp(-t / 0.020)  # filter decays faster than amp

    sub = np.sin(2 * np.pi * 65.4 * t) * amp_env
    harm = np.sin(2 * np.pi * 1308.0 * t) * amp_env * filt_env
    audio = (sub + 0.5 * harm).astype(np.float32)

    profile = profile_note_envelope(audio, fs=fs, bpm=143.0)
    assert 0.050 <= profile.gate_off_s <= 0.095, f"Gate off detection failed: {profile.gate_off_s}"
    assert profile.feg_decay_half_ms < profile.aeg_decay_half_ms, "FEG should decay faster than AEG"
    assert "a_amp_eg_decay" in profile.suggested_params

    # Test loss function
    loss_same = compute_envelope_loss_np(audio, audio, fs=fs)
    assert loss_same["loss_env_total"] == 0.0

    audio_diff = (sub * 0.5).astype(np.float32)
    loss_diff = compute_envelope_loss_np(audio, audio_diff, fs=fs)
    assert loss_diff["loss_env_total"] > 0.0



# ---------------------------------------------------------------------------
# Real-preset prior (v2): extractor conversions, sampler, trainer, watchdog
# ---------------------------------------------------------------------------
def _display(kind, x):
    """What Surge would show for native value x (the units extract_real_bass_manifold parses)."""
    if kind == "pct":
        return f"{100 * x:.2f} %"
    if kind == "semitones":
        return f"{x:.2f} semitones"
    if kind == "cents":
        return f"{100 * x:.2f} cents"
    if kind == "db":
        return f"{x:.2f} dB"
    if kind == "hz":
        hz = 440.0 * 2 ** (x / 12)
        return f"{hz / 1000:.3f} kHz" if hz >= 1000 else f"{hz:.2f} Hz"
    if kind == "seconds":
        s = 2 ** x
        return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.3f} s"
    raise ValueError(kind)


class _DisplayParam:
    def __init__(self, lo, hi, kind):
        self.lo, self.hi, self.kind, self.raw_value = lo, hi, kind, 0.0

    @property
    def string_value(self):
        return _display(self.kind, self.lo + self.raw_value * (self.hi - self.lo))


class _DisplaySurge:
    def __init__(self, override=None):
        import extract_real_bass_manifold as ex
        self.parameters = {}
        for name, (_tag, pname, lo, hi, kind) in ex.CONTINUOUS.items():
            lo, hi = (override or {}).get(name, (lo, hi))
            self.parameters[pname] = _DisplayParam(lo, hi, kind)


def _xml_preset(values: dict) -> bytes:
    body = "".join(f'<{k} type="2" value="{v:.6f}" />' for k, v in values.items())
    return b"CcnK\x00\x00\x00\x00FPCh" + b"\x00" * 48 + \
        f'<?xml version="1.0" encoding="UTF-8"?><patch revision="21"><parameters>{body}</parameters></patch>'.encode()


def _preset_values(rng=None, **kw):
    import extract_real_bass_manifold as ex
    rng = rng or np.random.RandomState(0)
    v = {"a_osc1_type": 0, "a_osc1_param6": 1, "a_filter1_type": 2, "a_ws_type": 0, "a_fm_switch": 0}
    for name, (tag, _p, lo, hi, _k) in ex.CONTINUOUS.items():
        v[tag] = lo + (hi - lo) * rng.uniform(0.3, 0.7)
    v["a_osc1_param5"] = rng.uniform(0.05, 0.3)  # detune inside our 0..0.35 range (no clipping)
    v.update(kw)
    return v


def test_extractor_calibration_accepts_matching_and_rejects_wrong_ranges():
    import extract_real_bass_manifold as ex
    ranges = ex.calibrate_ranges(_DisplaySurge())
    assert ranges == {n: (c[2], c[3]) for n, c in ex.CONTINUOUS.items()}
    with pytest.raises(RuntimeError, match="fm_depth"):
        ex.calibrate_ranges(_DisplaySurge(override={"fm_depth": (-48.0, 24.0)}))
    measured = ex.calibrate_ranges(_DisplaySurge(override={"fm_depth": (-48.0, 24.0)}), use_measured=True)
    assert abs(measured["fm_depth"][1] - 24.0) < 0.1


def test_extractor_roundtrip_through_apply_patch():
    """XML native value -> vector -> vector_to_patch -> apply_patch must set exactly the raw value
    Surge would compute from the native value: raw = (x - min) / (max - min)."""
    import extract_real_bass_manifold as ex
    ranges = {n: (c[2], c[3]) for n, c in ex.CONTINUOUS.items()}
    xml = _preset_values(**{"a_osc1_param6": 2, "a_osc1_param5": 0.2, "a_filter1_type": 15, "a_ws_type": 23,
                            "a_ws_drive": 6.0, "a_filter1_keytrack": 0.5, "a_osc1_param0": -0.4,
                            "a_fm_switch": 1, "a_fm_depth": -10.0, "a_filter1_cutoff": -12.0})
    vec, clipped, reason = ex.preset_to_vector(ex.parse_fxp(_xml_preset(xml)), ranges)
    assert reason is None and clipped == []
    patch = surge_spec.vector_to_patch(vec)
    assert surge_spec.LP_FILTERS[patch["filter_idx"]][0] == "LP Diode Ladder"
    assert surge_spec.WAVESHAPER_TYPES[patch["ws_idx"]][0] == "Fuzz" and patch["unison"]
    synth = FakeSurge()
    surge_spec.apply_patch(synth, patch)
    for name, (tag, pname, lo, hi, _k) in ex.CONTINUOUS.items():
        expected = (xml[tag] - lo) / (hi - lo)
        assert abs(synth.parameters[pname].raw_value - expected) < 1e-5, name
    assert abs(synth.parameters["a_filter_1_type"].raw_value - 15 / 33) < 1e-3


def test_extractor_skips_and_zeroes_inactive_controls():
    import extract_real_bass_manifold as ex
    ranges = {n: (c[2], c[3]) for n, c in ex.CONTINUOUS.items()}
    run = lambda **kw: ex.preset_to_vector(ex.parse_fxp(_xml_preset(_preset_values(**kw))), ranges)  # noqa: E731
    assert run(a_osc1_type=1)[2] == "osc 1 not Classic"
    assert "lowpass" in run(a_filter1_type=4)[2]  # a highpass is skipped, not relabelled
    assert "waveshaper" in run(a_ws_type=7)[2]
    vec, _, _ = run(a_fm_switch=0, a_ws_type=0, a_osc1_param6=1)
    for n in ("fm_depth", "drive", "unison_detune"):
        assert vec[surge_spec.PARAM_INDEX[n]] == 0.0
    vec, clipped, _ = run(a_ws_type=1, a_ws_drive=20.0)  # +20 dB is past our +15.36 dB ceiling
    assert vec[surge_spec.PARAM_INDEX["drive"]] == 1.0 and clipped == ["drive"]
    assert ex.parse_fxp(b"garbage") is None


def _tiny_manifold(path, n=60, seed=0):
    import extract_real_bass_manifold as ex
    rng = np.random.RandomState(seed)
    presets = []
    for i in range(n):
        kw = {"a_filter1_type": [1, 2, 3, 10, 15][i % 5], "a_ws_type": [0, 1, 23][i % 3],
              "a_osc1_param6": 1 + (i % 2), "a_fm_switch": i % 2}
        if i % 3 == 0:
            kw["a_osc1_param4"] = 0.0  # sync off in a third of presets
        presets.append((f"p{i}.fxp", _xml_preset(_preset_values(rng, **kw))))
    presets.append(("hp.fxp", _xml_preset(_preset_values(rng, a_filter1_type=4))))
    presets.append(("dup.fxp", presets[0][1]))
    ranges = {k: (c[2], c[3]) for k, c in ex.CONTINUOUS.items()}
    params, names, is_val, skips, clip = ex.build_manifold(presets, ranges, val_frac=0.2)
    import json
    np.savez_compressed(path, params=params, preset_names=np.array(names), is_val=is_val,
                        param_names=np.array(surge_spec.PARAM_NAMES), clip_frac=clip.astype(np.float32),
                        skip_reasons=json.dumps(skips), ranges=json.dumps(ranges), calibrated=True,
                        format_version=ex.FORMAT_VERSION)
    return params, is_val, skips


def test_manifold_split_dedup_and_prior_sampling(tmp_path):
    from realistic_bass_prior import JITTER_IDX, RealisticBassPrior
    path = str(tmp_path / "m.npz")
    params, is_val, skips = _tiny_manifold(path)
    assert skips.get("duplicate vector") == 1 and any("lowpass" in k for k in skips)
    assert 0 < is_val.sum() < len(is_val)
    prior = RealisticBassPrior(path, split="train")
    train = params[~is_val]
    sync = surge_spec.PARAM_INDEX["sync"]
    rng = np.random.RandomState(0)
    bounds = prior.support_bounds()
    draws = np.stack([prior.sample_vector(rng) for _ in range(3000)])
    # sync is a spike dimension: off in ~1/3 of presets, and the draws keep roughly that rate
    assert abs(np.mean(draws[:, sync] == 0) - np.mean(train[:, sync] == 0)) < 0.05
    # every draw is inside the recorded support box, and nothing piles up at a box edge
    for i in JITTER_IDX:
        lo, hi = bounds[surge_spec.PARAM_NAMES[i]]
        col = draws[:, i]
        assert col.min() >= lo - 1e-6 and col.max() <= hi + 1e-6
        on = col[col > 0]
        if len(on) > 100 and hi > lo:
            assert np.mean(np.isclose(on, on.min())) < 0.02 and np.mean(np.isclose(on, hi)) < 0.02
    # categoricals are copied from a preset, never jittered into new classes
    cats = [i for i, _ in surge_spec.CAT_INDICES]
    train_cats = {tuple(r) for r in np.round(train[:, cats], 5)}
    assert all(tuple(r) in train_cats for r in np.round(draws[:, cats], 5))
    for n in ("chorus_mix", "delay_mix", "delay_fb"):
        assert np.all(draws[:, surge_spec.PARAM_INDEX[n]] == 0) and bounds[n] == (0.0, 0.0)
    v1 = tmp_path / "v1.npz"
    np.savez(v1, params=params)
    with pytest.raises(ValueError, match="v1 manifold"):
        RealisticBassPrior(str(v1))


def test_realistic_trainer_fresh_resume_and_exports(tmp_path, monkeypatch):
    import train_realistic_bass_overnight as tr
    from synth_jepa_search import SynthJEPASearcher, checkpoint_bounds, load_synth_jepa
    manifold = str(tmp_path / "m.npz")
    _tiny_manifold(manifold)
    monkeypatch.setattr(tr, "init_synth", lambda *a, **k: FakeSurge())
    out = tmp_path / "run"
    base = ["--out_dir", str(out), "--manifold", manifold, "--device", "cpu", "--num_workers", "0",
            "--batch_size", "4", "--val_size", "6", "--val_interval_steps", "2", "--checkpoint_interval_steps", "2",
            "--norm_batches", "2", "--warmup_steps", "2", "--log_interval_steps", "2", "--keep_last", "1"]
    tr.train(tr.build_parser().parse_args(base + ["--max_steps", "2"]))
    for f in ("checkpoint_latest.pt", "checkpoint_best.pt", "jepa_best.pt", "flow_best.pt", "run_meta.json", "val.jsonl"):
        assert (out / f).exists(), f
    tr.train(tr.build_parser().parse_args(base + ["--max_steps", "4"]))  # resumes, does not restart
    ck = torch.load(out / "checkpoint_latest.pt", weights_only=False)
    assert ck["step"] == 4 and "opt_jepa" in ck and ck["prior_bounds"]["chorus_mix"] == (0.0, 0.0)
    assert len(list(out.glob("checkpoint_step_*.pt"))) == 1
    log = (out / "training.log").read_text()
    assert "Resumed at step 2" in log and "VAL step 4" in log
    flow = load_inverter(str(out / "flow_latest.pt"))
    assert isinstance(flow, FlowMatchingResMLP)
    model, norm = load_synth_jepa(str(out / "jepa_latest.pt"))
    s = SynthJEPASearcher(model, norm, device="cpu", seed=0, bounds=checkpoint_bounds(str(out / "jepa_latest.pt")))
    res = s.search(torch.zeros(128, 81), midi_note=36, total_eval_budget=64)
    assert res["vector"][surge_spec.PARAM_INDEX["chorus_mix"]] == 0.0
    torch.save({"step": 1}, out / "checkpoint_latest.pt")  # a v1 checkpoint is refused
    with pytest.raises(SystemExit, match="v1 trainer"):
        tr.train(tr.build_parser().parse_args(base + ["--max_steps", "6"]))


def test_watchdog_restarts_on_crash_and_gives_up(tmp_path):
    import watchdog_supervisor as wd
    crash = [sys.executable, "-c", "import sys; sys.exit(1)"]
    rc = wd.main(["--out_dir", str(tmp_path), "--poll_s", "0.05", "--max_restarts", "2", "--"] + crash)
    log = (tmp_path / "watchdog.log").read_text()
    assert rc == 1 and log.count("Launching") == 3 and "Giving up" in log
    assert wd.main(["--out_dir", str(tmp_path / "ok"), "--poll_s", "0.05", "--",
                    sys.executable, "-c", "pass"]) == 0
