# Owner: KUANG (created 2026-10-19)
"""CPU tests for the V6 foundation (V6_SPECIFICATION.md phase 1): surge_spec_v6 / param_codec_v6 (note as input,
osc 1 at octave 0, relative osc 2 / 3 octaves, +-0.2 semitone detune), soundset_prior_v6 (simplicity prior),
models_v6 (note-conditioned flow), render_bank_v6 and train_realistic_bass_v6, plus the v6 paths of the sampling,
evaluation and DPO tools. No Surge, no GPU: fake renderers and tiny models.

    python -m pytest test_v6_foundation.py -q
"""
import argparse
import copy

import numpy as np
import pytest
import torch

from test_v6 import FakePool, N_AUDIO, _manifolds, _mel, fake_render_v5  # noqa: F401  (also installs the fakes)
import layout_v6 as L  # noqa: E402
import param_codec_v6 as C6  # noqa: E402
import surge_spec_v5 as S5  # noqa: E402
import surge_spec_v6 as S6  # noqa: E402


def fake_render_v6(vec, midi):
    return fake_render_v5(S6.to_v5(vec), midi)


def _v6_prior(tmp_path, split="train"):
    from soundset_prior_v6 import SoundsetPriorV6
    b, a = _manifolds(tmp_path)
    return SoundsetPriorV6(b, a, split=split)


def _v6_flow(seed=0, cond_noise=False):
    import models_v6 as m6
    torch.manual_seed(seed)
    return m6.build_model_v6(hidden_dim=32, num_layers=2, encoder="temporal_asp", enc_kwargs={"d_model": 16},
                             cond_noise=cond_noise).eval()


# ------------------------------------------------------------------------------------------------ layout
def test_layout_sizes_and_v5_round_trip(tmp_path):
    assert (S6.NUM_PARAMS, len(S6.CONT_INDICES), len(S6.FLOW_CONT_INDICES), C6.ENCODED_DIM) == (59, 44, 43, 114)
    assert L.layout_for_dim(114).name == "v6" and L.layout_named("v6").note_conditioned
    prior = _v6_prior(tmp_path)
    rng = np.random.RandomState(0)
    for _ in range(200):
        v, midi, _ = prior.sample_vector(rng)
        v5 = S6.to_v5(v)
        assert S5.vector_to_patch(v5)["osc_1_octave_idx"] == 3                  # osc 1 always at octave 0
        back, note = S6.from_v5(v5, midi)
        assert note == midi and np.allclose(back, v, atol=1e-6)
    I5, I6 = S5.PARAM_INDEX, S6.PARAM_INDEX
    v5 = np.full(60, 0.5, np.float32)
    v5[I5["midi_note"]] = (30 - 28) / 22
    v5[I5["osc_1_octave"]], v5[I5["osc_2_octave"]] = 4 / 6, 3 / 6                # osc 1 +1, osc 2 at 0
    v5[I5["osc_2_pitch"]] = 0.6                                                  # +0.2 semitone in v5 units
    v6, note = S6.from_v5(v5)
    assert note == 42                                                            # the pitch moved into the note
    assert S6.vector_to_patch(v6)["osc_2_octave_rel"] == -1                      # osc 2 now an octave below osc 1
    assert v6[I6["osc_2_pitch"]] == pytest.approx(1.0)                           # +0.2 st = top of the v6 range


def test_codec_round_trip_and_jepa_input(tmp_path):
    prior = _v6_prior(tmp_path)
    rng = np.random.RandomState(1)
    p = torch.tensor(np.stack([prior.sample_vector(rng)[0] for _ in range(32)]))
    enc = C6.encode(p)
    assert enc.shape == (32, 114)
    back = C6.decode(enc, C6.note_of(p))
    for i in range(59):
        assert torch.allclose(back[:, i], p[:, i], atol=1e-6), S6.PARAM_NAMES[i]
    cont, _ = C6.split_encoded(enc)
    assert torch.allclose(C6.jepa_cont(cont, C6.note_of(p)), p[:, S6.CONT_INDICES])
    assert C6.encoded_weights(torch.from_numpy(S6.domain_weights())).shape == (114,)


def test_simplicity_prior_shares(tmp_path):
    import ablation_v6 as ab
    prior = _v6_prior(tmp_path)
    rng = np.random.RandomState(2)
    draws = [prior.sample_vector(rng) for _ in range(3000)]
    src = np.array([d[2] for d in draws])
    for name, share in prior.mix.items():
        assert abs(np.mean(src == name) - share) < 0.03, name
    sparse = np.stack([d[0] for d in draws if d[2] == "sparse"])
    on = {n: np.mean([ab.active(v, n, S6) for v in sparse]) for n in ("osc2_off", "waveshaper_off", "unison_1")}
    assert abs(on["osc2_off"] - 0.20) < 0.04 and abs(on["waveshaper_off"] - 0.25) < 0.04, on
    assert abs(on["unison_1"] - 0.15) < 0.04, on
    assert all(28 <= d[1] <= 50 for d in draws)


# ------------------------------------------------------------------------------------------------ model / tools
def test_flow_needs_the_note_and_round_trips(tmp_path):
    import models_v4 as m4
    import models_v6 as m6
    f = _v6_flow(cond_noise=True)
    x, t, mel = torch.rand(2, 114), torch.rand(2), _mel(2)
    with pytest.raises(ValueError):
        f(x, t, mel)
    a = f(x, t, mel, note=torch.tensor([0.1, 0.1]))
    assert not torch.allclose(a, f(x, t, mel, note=torch.tensor([0.9, 0.9])))     # the note changes the velocity
    path = tmp_path / "f6.pt"
    torch.save({"model_state": f.state_dict(), "args": {"v4_config": f.v4_config}, "step": 1}, path)
    g = m4.load_inverter_v4(str(path))                                          # shared loader -> models_v6
    assert isinstance(g, m6.NoteConditionedFlow) and g.note_features == 8
    assert torch.allclose(g(x, t, mel, note=torch.tensor([0.1, 0.1])), a)
    x1 = g.sample(mel, torch.tensor([0.5, 0.2]), num_steps=2, generator=torch.Generator().manual_seed(0))
    v = L.layout_named("v6").decode(x1, torch.tensor([0.5, 0.2]))
    assert v.shape == (2, 59) and torch.allclose(v[:, 0], torch.tensor([0.5, 0.2]))


def test_sampling_eval_and_dpo_use_the_note():
    import eval_real_stems_consensus_v5 as E
    import train_flow_dpo_v5 as dpo
    from flow_sampling_v5 import sample_candidates
    f = _v6_flow()
    v = sample_candidates(f, _mel(2), [30, 45], k=3, steps=2, seed=0)
    assert v.shape == (2, 3, 59)
    assert np.allclose(v[0, :, 0], (30 - 28) / 22) and np.allclose(v[1, :, 0], (45 - 28) / 22)
    mel = _mel(1)
    direct, k0 = E.sample_candidates_adaptive(f, mel, 50, k=2, steps=2, seed=3)
    folded, k1 = E.sample_candidates_adaptive(f, mel, 62, k=2, steps=2, seed=3)
    assert (k0, k1) == (0, 1) and np.allclose(direct, folded)       # v6: nothing to shift back after folding
    w_enc = C6.encoded_weights(torch.from_numpy(S6.domain_weights()))
    loss, st = dpo.dpo_terms(f, copy.deepcopy(f), _mel(2), torch.from_numpy(v[:, 0]), torch.from_numpy(v[:, 1]),
                             w_enc, 100.0, torch.Generator().manual_seed(0))
    assert abs(st["delta"]) < 1e-6 and abs(float(loss) - np.log(2)) < 1e-5


def test_eval_helpers_handle_v6():
    import eval_real_stems_consensus_v5 as E
    rng = np.random.RandomState(0)
    vecs = np.clip(rng.uniform(0, 1, (5, 59)), 0, 1).astype(np.float32)
    cons = E.build_consensus_vector(vecs, param_dim=114)
    assert cons.shape == (59,)
    patch = E.vector_to_patch_adaptive(cons, param_dim=114)
    assert patch["osc_1_octave_idx"] == 3 and "osc_2_octave_rel" in patch
    assert "F1:" in E.format_patch_details(patch, param_dim=114)


# ------------------------------------------------------------------------------------------------ bank / trainer
def _bank(tmp_path, n=24, shard=8):
    import render_bank_v6 as rb6
    b, a = _manifolds(tmp_path)
    out = tmp_path / "bank6"
    args = argparse.Namespace(out=str(out), bass_manifold=b, all_manifold=a, seed=3, n_items=n, shard_size=shard,
                              plugin=None, workers=0, inharm_config=None)
    rb6.build(args, log=lambda *_: None, pool=FakePool(fake_render_v6))
    return out, args, b, a


def test_bank_build_and_checks(tmp_path):
    import render_bank_v6 as rb6
    out, args, b, a = _bank(tmp_path)
    m = rb6.read_manifest(str(out))
    assert m["format"] == 6 and m["num_params"] == 59 and m["inharm_labels"] is False
    audio, vec, inh, valid = rb6.BankMainDataset(str(out))[5]
    assert vec.shape == (59,) and audio.shape == (N_AUDIO,) and valid == 0.0
    hard, soft = rb6.check_compatible(m, b, a, inharm_weight=100.0)
    assert any("inharmonicity" in h for h in hard) and soft == []
    assert rb6.check_compatible(m, b, a) == ([], [])
    rb6.build(args, log=lambda *_: None, pool=FakePool(fake_render_v6))                  # resume: nothing to do
    assert rb6.verify(argparse.Namespace(bank=str(out), n=4), log=lambda *_: None,
                      pool=FakePool(fake_render_v6)) < 1e-3          # int16 storage rounding only


def test_bank_redraws_glitches(tmp_path):
    import render_bank_v6 as rb6
    prior = _v6_prior(tmp_path)
    rng = np.random.RandomState(5)
    draws = [prior.sample_vector(rng) for _ in range(6)]
    vecs = np.stack([d[0] for d in draws]).astype(np.float32)
    midi = np.array([d[1] for d in draws], np.int16)
    src = np.array([d[2] for d in draws], dtype="<U12")
    bad = {vecs[1].tobytes()}

    def render(v, m):                                         # the second item renders silent until redrawn
        return np.zeros(N_AUDIO, np.float32) if v.tobytes() in bad else fake_render_v6(v, m)
    audio, _, _, v2, m2, s2 = rb6.render_items(FakePool(render), prior, vecs, midi, src, labels=False)
    assert s2[1] == "redraw" and not np.array_equal(v2[1], vecs[1]) and np.abs(audio[1]).max() > 0
    assert np.array_equal(v2[[0, 2, 3, 4, 5]], vecs[[0, 2, 3, 4, 5]])


def test_trainer_end_to_end_on_a_bank(tmp_path, monkeypatch):
    import models_v4 as m4
    import models_v6 as m6
    import train_realistic_bass_v6 as tr
    bank, _, b, a = _bank(tmp_path, n=48, shard=16)

    def fixed_set(synth, prior, n, seed, extractor=None):
        rng = np.random.RandomState(seed)
        vecs, audio = [], []
        for _ in range(n):
            _, v, m, _ = prior.sample_patch_and_midi(rng)
            vecs.append(v)
            audio.append(fake_render_v6(v, m))
        return torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(vecs))
    monkeypatch.setattr(tr, "init_synth", lambda *x, **k: object())
    monkeypatch.setattr(tr, "render_fixed_set", fixed_set)
    out = tmp_path / "run6"
    argv = ["--out_dir", str(out), "--bass_manifold", b, "--all_manifold", a, "--bank", str(bank),
            "--val_cache", str(tmp_path / "vc"), "--max_steps", "6", "--batch_size", "4", "--optimizer", "adamw",
            "--device", "cpu", "--seed", "0", "--embed_dim", "32", "--ff_dim", "64", "--jepa_layers", "1",
            "--predictor_hidden", "32", "--flow_dim", "32", "--flow_layers", "2", "--flow_enc_dim", "16",
            "--cond_noise", "--flow_loss", "mae", "--lambda_acoustic", "0.1", "--acoustic_cat", "st",
            "--acoustic_t_min", "0.5", "--ema_halflife_steps", "4", "--val_size", "8", "--val_interval_steps", "3",
            "--checkpoint_interval_steps", "3", "--log_interval_steps", "2", "--norm_batches", "2",
            "--num_workers", "0", "--ladder_every", "0"]
    tr.train(tr.build_parser().parse_args(argv))
    flow = m4.load_inverter_v4(str(out / "flow_ema_latest.pt"))
    assert isinstance(flow, m6.NoteConditionedFlow) and flow.param_dim == 114
    log = (out / "training.log").read_text()
    assert "V6 joint training" in log and "(Ac " in log
    bad = argv[:-2] + ["--ladder_every", "4"]
    bad[bad.index("--out_dir") + 1] = str(tmp_path / "x")
    with pytest.raises(SystemExit):                                           # ladders are refused for v6
        tr.train(tr.build_parser().parse_args(bad))


def _saw(f, seconds=1.75, sr=44100):
    t = np.arange(int(seconds * sr)) / sr
    return sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 12))


def _spec_render(note, pitch_span_st=1.0):
    """A renderer that follows the v6 mapping, with the v5 fine-pitch span as a parameter (1.0 = the assumption)."""
    def render(vec):
        p = S6.vector_to_patch(vec)
        f0 = 440.0 * 2 ** ((note - 69) / 12)
        if p["osc_1_volume"] > 0:
            return _saw(f0)
        st = (p["osc_2_pitch"] - 0.5) * 2.0 * pitch_span_st                  # v5 value -> semitones
        return _saw(f0 * 2.0 ** (p["osc_2_octave_idx"] - 3) * 2.0 ** (st / 12.0))
    return render


def test_check_v6_spec_measures_detune_and_octaves():
    import check_v6_spec as chk
    ok = chk.run(_spec_render(45), 45, log=lambda *_: None)
    assert all(r[3] for r in ok), ok
    assert [round(r[2]) for r in ok] == [0, -20, 20, -2400, -1200, 1200, 2400]
    wrong = chk.run(_spec_render(45, pitch_span_st=1.6), 45, log=lambda *_: None)   # if v5's span were +-1.6 st
    assert not wrong[1][3] and not wrong[2][3] and abs(wrong[2][1] - 32.0) < 3.0


# ------------------------------------------------------------------------------------------------ real-stem tools
def test_transcription_filter_and_phrase_events():
    import eval_real_stems_consensus_v5 as E
    notes = [{"note": 36, "start": 0.0, "dur": 0.2, "vel": 100}, {"note": 43, "start": 0.1, "dur": 0.2, "vel": 90},
             {"note": 38, "start": 0.5, "dur": 0.2, "vel": 100}, {"note": 72, "start": 0.8, "dur": 0.1, "vel": 100}]
    assert [n["note"] for n in E.filter_transcription(notes, max_midi=60)] == [36, 43, 38]
    assert [n["note"] for n in E.filter_transcription(notes, max_midi=60, drop_polyphony=True)] == [38]
    ev = E.events_from_notes(E.filter_transcription(notes, 60, True), max_dur=1.0)
    assert [(m.type, m.note) for m in ev] == [("note_on", 38), ("note_off", 38)]
    assert ev[0].time == pytest.approx(0.5) and ev[1].time == pytest.approx(0.7)


def _summary(path, rows):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return str(path)


def test_stem_stats_paired_and_old_ssim_files(tmp_path):
    import stem_stats_v6 as ss
    rng = np.random.default_rng(0)
    base = rng.uniform(3, 9, 24)
    rows = [{"stem_id": f"s{i:02d}", "models": {
        "REF": {"phrase_mss": float(base[i]), "mel_ssim": 0.7},                         # old file: 1 - SSIM
        "BETTER": {"phrase_mss": float(base[i] - 0.5 - 0.05 * rng.random()), "mel_ssim": 0.6},
        "SAME": {"phrase_mss": float(base[i] + rng.normal(0, 0.1)), "mel_ssim": 0.7}}} for i in range(24)]
    res = ss.table(ss.load([_summary(tmp_path / "run" / "consensus_eval_summary.json", rows)]), ref="REF", n_boot=2000)
    by = {r["model"]: r for r in res["paired"]}
    assert by["BETTER"]["hi"] < 0 and by["BETTER"]["better"] == 24 and by["BETTER"]["sign_test_p"] < 1e-6
    assert by["SAME"]["lo"] < 0 < by["SAME"]["hi"]
    assert res["models"]["BETTER"]["wins_on_common"] == 24
    sim = ss.table(ss.load([str(tmp_path / "run" / "consensus_eval_summary.json")]), metric="mel_ssim", n_boot=200)
    assert sim["models"]["BETTER"]["mean"] == pytest.approx(0.4) and sim["models"]["REF"]["mean"] == pytest.approx(0.3)
    assert "lower is better" in ss.report(res)
