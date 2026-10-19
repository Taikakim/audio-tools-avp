# Owner: KUANG (created 2026-10-20)
"""CPU tests for the tools built from Kim's 24-stem listening test (HUMAN_EVALUATION_SYNTHESIS.md, a4076e3):
band limit, band-hierarchical score, spectral EMD, resonance, square vs triangle, MIDI tempo check, the agreement
report. No Surge, no GPU.

    python -m pytest test_v6_listening.py -q
"""
import numpy as np
import pytest
from scipy.signal import butter, iirpeak, lfilter

import test_v6  # noqa: F401  (installs the fakes: mido, pedalboard)
import audio_metrics_v6 as am6

SR = 44100


def wave(kind, f0=55.0, seconds=0.25, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    if kind == "saw":
        ks, amp = range(1, 400), (lambda k: 1.0 / k)
    elif kind == "square":
        ks, amp = range(1, 400, 2), (lambda k: 1.0 / k)
    elif kind == "triangle":
        ks, amp = range(1, 400, 2), (lambda k: (-1) ** ((k - 1) // 2) / k ** 2)
    else:
        return np.sin(2 * np.pi * f0 * t)
    return sum(amp(k) * np.sin(2 * np.pi * k * f0 * t) for k in ks if k * f0 < 20000)


def lp(y, hz, order=2):
    b, a = butter(order, hz / (SR / 2))
    return lfilter(b, a, y)


# ------------------------------------------------------------------------------------------------ waveform shape
def test_square_triangle_and_sine_are_told_apart():
    import check_v6_spec as chk
    f0 = 55.0
    sq, tri, sine = wave("square"), wave("triangle"), wave("sine")
    assert am6.compute_harmonic_parity(sq, f0) < 0.01 and am6.compute_harmonic_parity(tri, f0) < 0.01
    assert chk.classify_wave(am6.compute_harmonic_parity(sq, f0), am6.odd_harmonic_slope(sq, f0)) == "square"
    assert chk.classify_wave(am6.compute_harmonic_parity(tri, f0), am6.odd_harmonic_slope(tri, f0)) == "triangle"
    assert am6.harmonic_richness(sine, f0) < 0.01 < am6.MIN_RICHNESS < am6.harmonic_richness(sq, f0)
    y = np.zeros(int(0.5 * SR))
    y[:len(sine)] = sine                                           # a filtered-down bass is not a square
    assert np.isnan(am6.phrase_parity(y, 33))
    assert np.isnan(am6.parity_over_notes(y, [{"note": 33, "start": 0.0, "dur": 0.2}]))


def test_shape_scan_picks_the_square_not_the_triangle():
    import check_v6_spec as chk
    import surge_spec_v6 as S
    I = S.PARAM_INDEX

    def render(v):                                  # Surge's classic osc as we read it: tri (0) -> saw (.5) -> pulse (1)
        sh = float(v[I["shape"]])
        return wave("triangle" if sh < 0.25 else ("saw" if sh < 0.75 else "square"), f0=110.0, seconds=0.5)
    rows, square = chk.shape_scan(render, 45, log=lambda *_: None)
    assert square == pytest.approx(0.75) and rows[0][3] == "triangle" and rows[4][3] == "saw/mixed"
    _, none = chk.shape_scan(lambda v: wave("saw", 110.0, 0.5), 45, log=lambda *_: None)
    assert none is None


def test_guided_parity_rule_needs_a_measured_square_shape():
    import render_pairs_dpo_v6 as rp
    import surge_spec_v6 as S6
    I = S6.PARAM_INDEX
    v = np.full(S6.NUM_PARAMS, 0.5, np.float32)
    target = np.zeros(35280)
    target[:int(0.125 * SR)] = wave("square", 130.81, 0.125)
    assert "parity" not in rp.guide_candidate_vector(v, target, target, 48)[1]
    g, fired = rp.guide_candidate_vector(v, target, target, 48, square_shape=1.0)
    assert "parity" in fired and g[I["shape"]] == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------ resonance / EMD
def test_resonance_excess_separates_a_resonant_filter():
    saw = wave("saw")
    plain = lp(saw, 800.0)
    bp, ap = iirpeak(800.0, 4.0, SR)
    resonant = plain + 3.0 * lfilter(bp, ap, saw)
    e_plain, _ = am6.resonance_excess_db(plain, 55.0)
    e_res, f_res = am6.resonance_excess_db(resonant, 55.0)
    assert e_plain < 3.0 and e_res > 5.0 and 600.0 < f_res < 1000.0
    assert am6.resonance_excess_db(wave("square"), 55.0)[0] < 1.0       # odd-only: missing evens are not dips
    assert np.isnan(am6.resonance_excess_db(wave("sine"), 55.0)[0])
    notes = [{"note": 33, "start": 0.0, "dur": 0.2}]
    assert am6.resonance_over_notes(resonant, notes)[0] == pytest.approx(e_res, abs=1.5)


def test_spectral_emd_grows_with_the_filter_distance():
    saw = wave("saw", seconds=0.5)
    ref = lp(saw, 400.0)
    d_same = am6.spectral_emd_oct(ref, ref)
    d_near, d_far = am6.spectral_emd_oct(lp(saw, 600.0), ref), am6.spectral_emd_oct(lp(saw, 2400.0), ref)
    assert d_same == pytest.approx(0.0, abs=1e-9) and 0.0 < d_near < d_far


# ------------------------------------------------------------------------------------------------ bands / hierarchy
def test_lowpass_and_band_dists():
    t = np.arange(SR // 2) / SR
    y = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 8000 * t)
    z = am6.lowpass(y, SR, 4000.0)
    spec = np.abs(np.fft.rfft(z))
    f = np.fft.rfftfreq(len(z), 1 / SR)
    assert spec[np.argmin(abs(f - 8000))] < 1e-3 * spec[np.argmin(abs(f - 200))]
    saw = wave("saw", seconds=0.5)
    d = am6.band_dists(lp(saw, 800.0), lp(saw, 800.0))
    assert set(d) == {"sub", "body", "air"} and max(d.values()) < 1e-6
    d2 = am6.band_dists(lp(saw, 2000.0), lp(saw, 800.0))
    assert d2["body"] > 1.0 and d2["sub"] < d2["body"]


def test_lexi_score_order_tolerance_and_occam_check():
    L = am6.LexiScore
    a = L([1.0, 5.0, 9.0], total=3.0)                   # better sub
    b = L([2.0, 1.0, 1.0], total=0.5)                   # better everything else
    assert a < b and not b < a and min([b, a]) is a
    c = L([1.02, 4.0, 9.0], total=9.0)                  # sub within tolerance of a -> body decides
    assert c < a
    d = L([1.0, 5.0, 9.0], total=2.0)                   # all bands tied -> composite decides
    assert d < a and a < float("inf") and f"{a:.1f}" == "3.0" and round(a, 1) == 3.0
    assert am6.lexi_no_worse(L([1.04, 5.1, 9.0], 1.0), a) and not am6.lexi_no_worse(L([1.0, 6.0, 9.0], 1.0), a)


def test_polish_runs_on_a_lexi_score():
    import eval_real_stems_consensus_v6 as E
    import surge_spec_v6 as S6
    I = S6.PARAM_INDEX
    v = np.full(S6.NUM_PARAMS, 0.5, np.float32)
    v[I["cutoff"]], v[I["feg_amount"]], v[I["keytrack"]] = 0.30, 0.50, 0.0

    def score(vec):                                     # body band best at cutoff 0.45; total prefers feg 0.65
        return am6.LexiScore([0.5, abs(vec[I["cutoff"]] - 0.45) * 20, 1.0],
                             total=abs(vec[I["feg_amount"]] - 0.65)), {"cutoff": float(vec[I["cutoff"]])}
    s0, o0 = score(v)
    v2, s2, _, log = E.polish_vector(v, S6, score, lambda o: 300 + 3000 * o["cutoff"], 300 + 3000 * 0.45, 0, s0, o0)
    assert isinstance(s2, am6.LexiScore) and s2 < s0 and v2[I["cutoff"]] > 0.30
    assert [e[0] for e in log][:1] == ["cutoff"]


# ------------------------------------------------------------------------------------------------ MIDI tempo
def test_onset_scale_check_finds_a_wrong_export_tempo():
    import eval_real_stems_consensus_v6 as E
    starts = np.arange(16) * 60.0 / 141.0 / 2                # eighths at 141 BPM in the audio
    y = np.zeros(int((starts[-1] + 0.5) * SR), np.float32)
    rng = np.random.default_rng(0)
    for s in starts:
        a = int(s * SR)
        y[a:a + 2000] += rng.normal(0, 1, 2000).astype(np.float32) * np.exp(-np.arange(2000) / 300.0)
    notes = [{"note": 36, "start": s * 141.0 / 100.0, "dur": 0.1} for s in starts]   # MIDI written at 100 BPM
    best, sc_best, sc_one = E.onset_scale_check(notes, y, SR)
    assert best == pytest.approx(100.0 / 141.0, abs=0.02) and sc_best > 1.2 * sc_one
    fixed = E.scale_notes(notes, 100.0 / 141.0)
    assert fixed[3]["start"] == pytest.approx(starts[3]) and fixed[0]["dur"] == pytest.approx(0.1 * 100 / 141)
    assert E.onset_scale_check(fixed, y, SR)[0] == pytest.approx(1.0, abs=0.02)


# ------------------------------------------------------------------------------------------------ agreement report
def test_human_pref_agreement():
    import human_pref_agreement as hp
    rng = np.random.default_rng(1)
    models = ["A", "B", "C", "D"]
    prefs, summary = {}, []
    for i in range(12):
        ranks = rng.permutation(4) + 1
        prefs[f"s{i}"] = dict(zip(models, ranks.tolist()))
        summary.append({"stem_id": f"s{i}", "real_centroid_hz": 400.0, "models": {
            m: {"phrase_mss": float(rng.uniform(1, 5)),                      # noise
                "centroid_hz": 400.0 * 2 ** (0.2 * int(r))}                  # follows the human rank
            for m, r in zip(models, ranks)}})
    lines, res = hp.agreement(prefs, summary)
    c = res["metrics"]["abs_log2_centroid_ratio"]
    assert c["median_rho"] == pytest.approx(1.0) and c["top1_hits"] == 12 and c["wilcoxon_p"] < 0.01
    assert abs(res["metrics"]["phrase_mss"]["median_rho"]) < 0.8 and res["friedman"]["n_stems"] == 12


# ------------------------------------------------------------------------------------------------ eval smoke test
class _Param:
    def __init__(self):
        self.raw_value = 0.5


class _FakeSurge:
    """Enough of a pedalboard Surge for the real-stem eval: parameters spring into existence, process() renders a
    saw per note, low-passed by the a_filter_1_cutoff raw value."""

    def __init__(self):
        self.parameters = _AutoParams()

    def reset(self):
        pass

    def process(self, events, duration, sample_rate=SR, num_channels=2):
        n = int(round(duration * sample_rate))
        y = np.zeros(n)
        ons = [e for e in events if getattr(e, "type", "") == "note_on"]
        for e in ons:
            a = int(e.time * sample_rate)
            b = min(n, a + int(0.12 * sample_rate))
            if a >= n:
                continue
            t = np.arange(b - a) / sample_rate
            f0 = 440.0 * 2 ** ((e.note - 69) / 12)
            y[a:b] += sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 30) if k * f0 < 16000)
        cut = 200.0 + 6000.0 * float(self.parameters["a_filter_1_cutoff"].raw_value)
        y = lp(y, min(cut, 20000.0))
        return np.stack([y, 0.9 * y]).astype(np.float32)


class _AutoParams(dict):
    def __getitem__(self, k):
        if k not in self:
            dict.__setitem__(self, k, _Param())
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        return self[k]


@pytest.mark.parametrize("select", ["mss", "lexi"])
def test_real_stem_eval_runs_with_the_new_options(tmp_path, monkeypatch, select):
    import json
    import soundfile as sf
    import torch
    import eval_real_stems_consensus_v6 as E
    from test_v6_foundation import _v6_flow
    root = tmp_path / "real_stems_eval"
    (root / "audio").mkdir(parents=True)
    (root / "midi_muscriptor").mkdir()
    notes = [{"note": 33 + (i % 3) * 5, "start": 0.25 * i, "dur": 0.2, "vel": 100} for i in range(8)]
    fake = _FakeSurge()
    fake.parameters["a_filter_1_cutoff"].raw_value = 0.2
    real = fake.process(E.events_from_notes(notes, 2.2), 2.2).T
    sf.write(root / "audio" / "stem_x_real.wav", real, SR)
    (root / "midi_muscriptor" / "stem_x_muscriptor.mid").write_bytes(b"")
    monkeypatch.setattr(E, "resolve_mantu", lambda p: str(root))
    monkeypatch.setattr(E, "STEM_CATALOG", [{"id": "stem_x", "name": "x", "bpm": 120.0, "style": "test"}])
    monkeypatch.setattr(E, "extract_notes_from_midi", lambda p, max_dur=None: [dict(n) for n in notes])
    monkeypatch.setattr(E, "load_clean_pedalboard_events", lambda p, max_dur=None: E.events_from_notes(notes, max_dur))
    monkeypatch.setattr(E.spec_v4, "init_synth", lambda *a, **k: fake)
    flow = _v6_flow()
    monkeypatch.setattr(E, "load_inverter_v4", lambda p, device="cpu": flow)
    torch.manual_seed(0)
    out = tmp_path / "out"
    E.evaluate_models_consensus({"M": "unused.pt"}, str(out), max_notes=2, k_per_note=1, steps=2, occam_tol=0.02,
                                max_eval_freq=4000.0, select=select, midi_scale={"stem_x": 1.0}, square_shape=1.0,
                                freq_decay=1.25 if select == "lexi" else None)
    res = json.loads((out / "consensus_eval_summary.json").read_text())[0]
    m = res["models"]["M"]
    assert res["select"] == select and res["max_eval_freq"] == 4000.0 and res["midi_scale_check"]["best"] is not None
    assert set(m["band_dists"]) == {"sub", "body", "air"} and m["spectral_emd_oct"] >= 0.0
    assert isinstance(m["phrase_mss"], float) and (out / "listen" / "stem_x__M.wav").exists()
    full, _ = sf.read(out / "listen" / "stem_x__real_phrase.wav")
    assert np.allclose(full, np.clip(real, -1, 1), atol=1e-3)                           # saved reference stays full band


def test_decay_stft_loss_and_linear_emd():
    from audio_utils import MultiScaleSTFTLoss
    saw = wave("saw", seconds=0.5)
    closed = lp(saw, 400.0)
    rng = np.random.default_rng(0)
    hiss = am6.lowpass(rng.normal(0, 1, len(saw)), SR, 16000.0)
    hiss = hiss - am6.lowpass(hiss, SR, 4000.0)                       # only above 4 kHz
    target = closed / np.abs(closed).max() + 0.01 * hiss
    a, b = closed / np.abs(closed).max(), saw / np.abs(saw).max()     # right closed bass vs open saw
    plain, flat = MultiScaleSTFTLoss(), am6.DecayMultiScaleSTFTLoss(gamma=0.0)
    assert flat(a, target) == pytest.approx(plain(a, target), rel=1e-4)
    decay = am6.DecayMultiScaleSTFTLoss(knee_hz=900.0, gamma=1.25)
    assert decay(a, target) < decay(b, target)
    assert decay(a, target) / decay(b, target) < plain(a, target) / plain(b, target)   # hiss counts less
    assert am6.spectral_emd_linear(a, a) == pytest.approx(0.0, abs=1e-12)
    assert am6.spectral_emd_linear(a, target) < am6.spectral_emd_linear(b, target)
