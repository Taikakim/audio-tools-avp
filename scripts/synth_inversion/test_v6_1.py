# Owner: KUANG (created 2026-10-20)
"""CPU tests for the V6.1 tools (V6_1_PLAN.md, V6_REVIEW.md section 9): harmonic parity per note / first phrase
note, the Welch spectral shoulder, the vector-space cutoff / FEG / keytrack polish (the plan's
test_polish_keytrack_feg), gate normalisation of note crops, the guided DPO candidate and the guided pair generator.
No Surge, no GPU.

    python -m pytest test_v6_1.py -q
"""
import json

import numpy as np
import pytest
import torch

from test_v6 import FakePool, N_AUDIO, _manifolds  # noqa: F401  (also installs the fakes)
from test_v6_foundation import fake_render_v6, _v6_flow
import audio_metrics_v6 as am6
import surge_spec_v6 as S6

SR = 44100


def tone(f0, odd_only, seconds, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    ks = range(1, 40, 2) if odd_only else range(1, 40)
    return sum(np.sin(2 * np.pi * k * f0 * t) / k for k in ks if k * f0 < sr / 2)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ------------------------------------------------------------------------------------------------ metrics
def test_parity_square_vs_saw_and_the_octave_note():
    f0 = midi_hz(33)
    assert am6.compute_harmonic_parity(tone(f0, True, 0.24), f0) < 0.02
    assert 0.2 < am6.compute_harmonic_parity(tone(f0, False, 0.24), f0) < 0.4
    phrase = np.zeros(N_AUDIO)                       # training phrase: square at 0 s, square an octave up at 0.25 s
    phrase[:int(0.125 * SR)] = tone(f0, True, 0.125)
    phrase[int(0.25 * SR):int(0.5 * SR)] = tone(2 * f0, True, 0.25)
    assert am6.compute_harmonic_parity(phrase, f0) > 1.0          # the octave note's odd harmonics count as even
    assert am6.phrase_parity(phrase, 33) < 0.02                   # first note only


def test_parity_reads_the_whole_signal_and_each_note():
    y = np.zeros(3 * SR)
    y[2 * SR:int(2.25 * SR)] = tone(midi_hz(40), True, 0.25)     # the only note starts at 2 s
    assert am6.compute_harmonic_parity(y, midi_hz(40), start_s=2.0, dur_s=0.25) < 0.02
    notes = [{"note": 40, "start": 2.0, "dur": 0.25}, {"note": 47, "start": 0.5, "dur": 0.2}]
    seg = tone(midi_hz(47), False, 0.2)
    y[int(0.5 * SR):int(0.5 * SR) + len(seg)] = seg
    par = am6.parity_over_notes(y, notes)
    assert 0.0 < par < 0.4 and np.isfinite(par)
    assert np.isnan(am6.parity_over_notes(y, [{"note": 40, "start": 0.0, "dur": 0.02}]))   # too short


def test_shoulder_uses_the_whole_clip():
    from scipy.signal import butter, lfilter
    t = np.arange(SR) / SR
    saw = sum(np.sin(2 * np.pi * k * 55.0 * t) / k for k in range(1, 300) if k * 55.0 < 20000)
    b, a = butter(2, 1000 / (SR / 2))
    late = np.concatenate([np.zeros(SR // 2), lfilter(b, a, saw)])    # silent first half: the old 93 ms saw nothing
    assert 150.0 < am6.compute_spectral_shoulder(late) < 14000.0


# ------------------------------------------------------------------------------------------------ polish
def _polish_case(span, start=None):
    import eval_real_stems_consensus_v6 as E
    I = S6.PARAM_INDEX
    v = np.full(S6.NUM_PARAMS, 0.5, np.float32) if start is None else start
    v[I["cutoff"]], v[I["feg_amount"]], v[I["keytrack"]] = 0.30, 0.50, 0.0

    def score(vec):                                   # best at cutoff 0.45, feg 0.65, keytrack 100 % (raw 0.75)
        s = abs(vec[I["cutoff"]] - 0.45) + abs(vec[I["feg_amount"]] - 0.65) + abs((0.5 + 0.5 * vec[I["keytrack"]]) - 0.75)
        return float(s), {"cutoff": float(vec[I["cutoff"]])}
    centroid = lambda out: 300.0 + 3000.0 * out["cutoff"]          # noqa: E731  brighter with the cutoff
    s0, o0 = score(v)
    return E.polish_vector(v, S6, score, centroid, real_centroid=300.0 + 3000.0 * 0.45, pitch_span=span,
                           start_score=s0, start_out=o0), s0


def test_polish_moves_toward_the_target_within_bounds():
    I = S6.PARAM_INDEX
    (v, s, out, log), s0 = _polish_case(span=12)
    assert s < s0 and [e[0] for e in log] == ["cutoff", "feg_amount", "keytrack_raw"]
    assert 0.30 < v[I["cutoff"]] <= 0.45 + 1e-6 and v[I["feg_amount"]] == pytest.approx(0.65)
    assert 0.5 + 0.5 * v[I["keytrack"]] == pytest.approx(0.75) and v[I["filter_2_keytrack"]] == pytest.approx(0.75)
    lo, hi = S6.CONT_BOUNDS["cutoff"]
    assert lo <= v[I["cutoff"]] <= hi


def test_polish_skips_keytrack_on_a_narrow_phrase_and_never_gets_worse():
    I = S6.PARAM_INDEX
    (v, s, out, log), s0 = _polish_case(span=5)
    assert "keytrack_raw" not in [e[0] for e in log] and v[I["keytrack"]] == 0.0 and s <= s0


def test_cut_note_stops_at_the_next_onset():
    import eval_real_stems_consensus_v6 as E
    y = np.ones(SR, np.float32)
    n = {"note": 40, "start": 0.1, "dur": 0.09}
    chunk, dur = E.cut_note(y, n, SR, next_start=0.2)
    assert dur == pytest.approx(0.125) and len(chunk) == int(round(0.125 * SR))
    assert np.all(chunk[int(0.1 * SR):] == 0.0) and chunk[100] > 0.9               # silence after the next onset
    assert E.next_onset(n, [n, {"note": 43, "start": 0.2, "dur": 0.1}]) == pytest.approx(0.2)


# ------------------------------------------------------------------------------------------------ guided DPO
def test_guided_candidate_rules():
    import render_pairs_dpo_v6 as rp
    I = S6.PARAM_INDEX
    v = np.full(S6.NUM_PARAMS, 0.5, np.float32)
    v[I["shape"]], v[I["cutoff"]], v[I["feg_amount"]], v[I["keytrack"]] = 0.0, 0.3, 0.3, 0.0
    v[I["resonance"]] = 0.6
    f0 = midi_hz(48)
    target = np.zeros(N_AUDIO)
    target[:int(0.125 * SR)] = tone(f0, True, 0.125)               # bright square
    cand = np.zeros(N_AUDIO)
    cand[:int(0.125 * SR)] = tone(f0, False, 0.125) * np.exp(-np.arange(int(0.125 * SR)) / 300.0)
    g, fired = rp.guide_candidate_vector(v, cand, target, 48, square_shape=0.5)
    assert "parity" in fired and g[I["shape"]] == pytest.approx(0.5) and g[I["ring_1x2_volume"]] == 0.0
    assert ("cutoff" in fired) == (abs(am6.get_centroid(target) - am6.get_centroid(cand)) > 120.0)
    assert "keytrack" in fired and g[I["keytrack"]] == pytest.approx(0.5)
    same, fired2 = rp.guide_candidate_vector(v, target, target, 40)          # candidate sounds like the target
    assert "cutoff" not in fired2 and "keytrack" not in fired2


def test_guided_pairs_end_to_end(tmp_path):
    import render_pairs_dpo_v6 as rp
    b, a = _manifolds(tmp_path)
    flow = _v6_flow()
    fp = tmp_path / "f6.pt"
    torch.save({"model_state": flow.state_dict(), "args": {"v4_config": flow.v4_config}, "step": 1}, fp)
    out = tmp_path / "pairs"
    rp.main(["--flow", str(fp), "--bass_manifold", b, "--all_manifold", a, "--n", "6", "--k_passive", "2",
             "--steps", "2", "--shard", "3", "--workers", "0", "--device", "cpu", "--out", str(out)],
            pool=FakePool(fake_render_v6), flow=flow)
    m = json.loads((out / "manifest.json").read_text())
    assert m["k"] == 3 and set(m["guided_best_share"]) == {"logmel_l1", "mss", "mel_ssim_dist"}
    assert 0.0 <= m["guided_best_share"]["mss"] <= 1.0 and "headroom" in m
    z = np.load(out / "shard_00000.npz")
    assert z["cand_vecs"].shape == (3, 3, 59) and z["score__mss"].shape == (3, 3) and len(z["guided_rules"]) == 3
    import train_flow_dpo_v5 as dpo                                   # the trainer reads the guided pairs
    data = dpo.load_pairs([str(out)], metric="mss", margin=0.0, noise_mult=0.0)
    assert data["w"].shape[1] == 59
