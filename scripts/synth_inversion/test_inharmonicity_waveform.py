"""Waveform-extraction contract: additive-synthesis fixtures with KNOWN partials, one condition per test.

PROVIDED BY THE SPEC. Do NOT edit assertions, constants or parametrize lists. If a test fails for a reason that is
not in the extractor (for example a synthesis typo), STOP and report. The noise test prints its table: run with -s
and paste the lines into the report (spec section 4, step S4).
Frequencies are exact FFT bin centres, so peak-picking error is small and failures mean the extractor is wrong.
Run from scripts/synth_inversion:   REQUIRE_ALL=1 python -m pytest test_inharmonicity_waveform.py -q -s
"""
import importlib
import math
import os

import pytest


def need(mod):
    try:
        return importlib.import_module(mod)
    except ImportError:
        if os.environ.get("REQUIRE_ALL") == "1":
            raise
        pytest.skip(f"{mod} not installed", allow_module_level=True)


np = need("numpy")
need("essentia")

from inharmonicity_target import InharmonicityConfig, InharmonicityExtractor, Reason  # noqa: E402

SR = 44100
N = 4096
BIN = SR / N  # 10.7666015625 Hz
F0 = 12 * BIN  # 129.19921875 Hz: a bin centre; with an even bin count every partial k*F0 is a bin centre too
AMPS = [1.0, 0.8, 0.8, 0.8]  # equal-ish energies so one dropped partial removes a clearly measurable share

# TEST VALUES ONLY. They are NOT recommended production values: those are Kim's decision (spec section 3).
BASE = dict(sample_rate=SR, frame_size=N, max_harmonics=20, tolerance=0.2,
            min_selected_partials=2, max_excluded_energy_fraction=0.10, require_fundamental_peak=True,
            silence_peak_abs=1e-4, max_level_change_db=None, attack_guard_s=0.020, min_valid_windows=2,
            min_partial_relative_db=40.0)

RESULT_KEYS = {"value", "valid", "rejection_reason", "all_rejection_reasons", "fundamental_hz",
               "fundamental_source", "analysis_configuration_id", "coverage_diagnostics"}


def tone(freqs, amps, sr=SR, n=N):
    t = np.arange(n) / sr
    x = np.zeros(n, dtype=np.float64)
    for f, a in zip(freqs, amps):
        x += a * np.sin(2 * np.pi * f * t)
    return x.astype(np.float32)


def extract(x, f0, **over):
    d = dict(BASE)
    d.update(over)
    return InharmonicityExtractor(InharmonicityConfig(**d)).process_window(x, f0, fundamental_source="test")


def partials(disp_bins=0.0, k=3, f0=F0):
    fr = [f0 * i for i in (1, 2, 3, 4)]
    fr[k - 1] += disp_bins * BIN
    return fr


def test_an_exact_harmonic_series_is_valid_and_near_zero():
    r = extract(tone(partials(), AMPS), F0)
    assert set(r) == RESULT_KEYS
    assert r["valid"], r["rejection_reason"]
    assert r["value"] < 0.02
    assert r["coverage_diagnostics"]["selected_peak_count"] == 4


def test_displacement_inside_the_tolerance_raises_the_value_monotonically():
    vals = []
    for n_bins in (0, 1, 2):  # distance n/12 of f0: 0, 0.083, 0.167 (tolerance 0.2)
        r = extract(tone(partials(n_bins), AMPS), F0)
        assert r["valid"], (n_bins, r["rejection_reason"])
        vals.append(r["value"])
    assert vals[0] < vals[1] < vals[2]


@pytest.mark.parametrize("n_bins", [3, 4, 5])  # distance 0.25, 0.33, 0.42 of f0: outside the tolerance
def test_selection_dropout_is_detected_in_the_waveform_path(n_bins):
    r = extract(tone(partials(n_bins), AMPS), F0)
    assert r["valid"] is False, "a displaced partial vanished from the selection and the label looked harmonic"
    assert r["rejection_reason"] == Reason.EXCLUDED_ENERGY
    assert 0.15 < r["coverage_diagnostics"]["excluded_energy_fraction"] < 0.30


@pytest.mark.parametrize("offset_bins", [0.0, 0.25, 0.5])
def test_exact_harmonic_series_is_robust_to_the_fundamental_not_being_a_bin_centre(offset_bins):
    f0 = F0 + offset_bins * BIN
    r = extract(tone(partials(0, f0=f0), AMPS), f0)
    assert r["valid"], (offset_bins, r["rejection_reason"])
    assert r["value"] < 0.02


def test_silence_is_invalid_and_not_zero_inharmonicity():
    r = extract(np.zeros(N, dtype=np.float32), F0)
    assert r["valid"] is False and r["rejection_reason"] == Reason.SILENCE and math.isnan(r["value"])


def test_nonfinite_audio_is_invalid_and_does_not_raise():
    x = tone(partials(), AMPS)
    x[10] = np.nan
    r = extract(x, F0)
    assert r["valid"] is False and r["rejection_reason"] == Reason.NONFINITE_INPUT


def test_a_window_of_the_wrong_length_is_rejected_not_cropped_or_padded():
    r = extract(tone(partials(), AMPS, n=N - 1), F0)
    assert r["valid"] is False and r["rejection_reason"] == Reason.WINDOW_LENGTH


def test_a_single_sinusoid_is_not_a_valid_label():
    r = extract(tone([F0], [1.0]), F0)
    assert r["valid"] is False and r["rejection_reason"] == Reason.TOO_FEW_PARTIALS


def test_the_sample_rate_is_honoured_end_to_end():
    sr = 48000
    bin48 = sr / N
    f0 = 12 * bin48
    x = tone([f0 * i for i in (1, 2, 3, 4)], AMPS, sr=sr)
    r = extract(x, f0, sample_rate=sr)
    assert r["valid"], r["rejection_reason"]
    assert r["value"] < 0.02  # a 44.1 kHz peak axis on 48 kHz audio would scale every peak by 0.91875


def test_level_change_is_always_reported_and_gates_only_when_configured():
    x = tone(partials(), AMPS)
    ramp = np.linspace(0.05, 1.0, N).astype(np.float32)
    r_diag = extract(x * ramp, F0)  # max_level_change_db is None: diagnostic only
    assert r_diag["coverage_diagnostics"]["level_change_db"] > 6.0
    r_gate = extract(x * ramp, F0, max_level_change_db=6.0)
    assert r_gate["valid"] is False and r_gate["rejection_reason"] == Reason.LEVEL_UNSTABLE
    flat = extract(x, F0, max_level_change_db=6.0)
    assert abs(flat["coverage_diagnostics"]["level_change_db"]) < 0.5
    assert flat["valid"], flat["rejection_reason"]


@pytest.mark.parametrize("snr_db", [60, 40, 20, 10])
def test_noise_condition_is_reported_per_snr(snr_db):
    rng = np.random.RandomState(0)
    x = tone(partials(), AMPS)
    noise = rng.randn(N).astype(np.float32)
    noise *= (np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(noise ** 2))) * 10 ** (-snr_db / 20)
    r = extract(x + noise, F0)
    assert set(r) == RESULT_KEYS
    if r["valid"]:
        assert r["value"] < 0.05, r
    print(f"SNR {snr_db:2d} dB -> valid={r['valid']} reason={r['rejection_reason']} "
          f"value={r['value']:.5f} excluded={r['coverage_diagnostics']['excluded_energy_fraction']}")
