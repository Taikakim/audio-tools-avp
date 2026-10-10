"""Peak-array contract for the inharmonicity extractor: reference parity, descriptor semantics, selection dropout.

These tests use EXACT peak arrays, so FFT resolution and pitch estimation cannot be mistaken for descriptor errors.
PROVIDED BY THE SPEC. Do NOT edit assertions, constants or parametrize lists. If the installed Essentia disagrees
with a hand-computed constant, the PARITY test is authoritative: STOP and report, do not edit the constant.
Needs numpy and Essentia, and inharmonicity_reference_fixtures.json from make_inharmonicity_fixtures.py.
Run from scripts/synth_inversion:   REQUIRE_ALL=1 python -m pytest test_inharmonicity_peaks.py -q
"""
import importlib
import json
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

from inharmonicity_target import InharmonicityConfig, Reason, inharmonicity_from_peaks  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "inharmonicity_reference_fixtures.json")

# TEST VALUES ONLY. They are NOT recommended production values: those are Kim's decision (spec section 3).
BASE = dict(sample_rate=44100, frame_size=4096, max_harmonics=20, tolerance=0.2,
            min_selected_partials=2, max_excluded_energy_fraction=0.10, require_fundamental_peak=True,
            silence_peak_abs=1e-4, max_level_change_db=None, attack_guard_s=0.020, min_valid_windows=2,
            min_partial_relative_db=40.0)
# Permissive settings that isolate NUMERICAL parity from validity policy (400 dB = no relative-magnitude floor).
PARITY = dict(min_selected_partials=1, max_excluded_energy_fraction=1.0, require_fundamental_peak=False,
              min_partial_relative_db=400.0)

RESULT_KEYS = {"value", "valid", "rejection_reason", "all_rejection_reasons", "fundamental_hz",
               "fundamental_source", "analysis_configuration_id", "coverage_diagnostics"}
DIAG_KEYS = {"input_peak_count", "selected_peak_count", "total_peak_energy", "selected_peak_energy",
             "selected_energy_fraction", "excluded_energy_fraction", "tolerance", "max_harmonics",
             "first_selected_hz", "level_change_db"}


def cfg(**over):
    d = dict(BASE)
    d.update(over)
    return InharmonicityConfig(**d)


def run(freqs, mags, f0=100.0, **over):
    return inharmonicity_from_peaks(freqs, mags, f0, cfg(**over), fundamental_source="test")


def _cases():
    if not os.path.exists(FIXTURES):
        return []
    with open(FIXTURES) as f:
        return json.load(f)["cases"]


CASES = _cases()


def test_fixture_file_is_present_and_versioned():
    assert os.path.exists(FIXTURES), "run make_inharmonicity_fixtures.py on the training machine first"
    with open(FIXTURES) as f:
        data = json.load(f)
    assert data["essentia_version"]
    assert len(data["cases"]) >= 25


# ------------------------------------------------------------------ A. reference parity
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_parity_with_the_installed_essentia(case):
    res = inharmonicity_from_peaks(case["freqs"], case["mags"], case["pitch"], cfg(**PARITY),
                                   fundamental_source="fixture")
    assert set(res) == RESULT_KEYS
    assert set(res["coverage_diagnostics"]) == DIAG_KEYS
    hp = case["harmonic_peaks"]
    ref = case.get("inharmonicity_on_selected")
    finite_input = all(math.isfinite(v) for v in case["freqs"] + case["mags"] + [case["pitch"]])
    if finite_input and hp["ok"] and ref is not None and ref["ok"] and sum(1 for m in hp["hm"] if m > 0) >= 1:
        # the wrapper must reproduce the reference HarmonicPeaks -> Inharmonicity composition exactly
        assert res["valid"], (case["name"], res["rejection_reason"])
        assert math.isclose(res["value"], ref["value"], rel_tol=1e-6, abs_tol=1e-9)
        return
    # reference raised, returned empty/sentinel, or selected no observed partial: NEVER a valid label
    assert res["valid"] is False
    assert math.isnan(res["value"])
    assert res["rejection_reason"] in Reason.ALL


@pytest.mark.parametrize("freqs,mags,f0,reason", [
    ([200.0, 100.0], [0.5, 1.0], 100.0, Reason.UNSORTED_OR_DUPLICATE),
    ([100.0, 100.0, 200.0], [1.0, 1.0, 0.5], 100.0, Reason.UNSORTED_OR_DUPLICATE),
    ([0.0, 100.0], [0.1, 1.0], 100.0, Reason.UNSORTED_OR_DUPLICATE),
    ([100.0, 200.0], [1.0], 100.0, Reason.SIZE_MISMATCH),
    ([100.0, float("nan")], [1.0, 0.5], 100.0, Reason.NONFINITE_INPUT),
    ([100.0, 200.0], [1.0, float("inf")], 100.0, Reason.NONFINITE_INPUT),
    ([], [], 100.0, Reason.NO_PEAKS),
    ([100.0, 200.0], [1.0, 0.5], 0.0, Reason.BAD_FUNDAMENTAL),
    ([100.0, 200.0], [1.0, 0.5], -100.0, Reason.BAD_FUNDAMENTAL),
])
def test_structurally_bad_input_is_invalid_with_a_specific_reason_and_never_raises(freqs, mags, f0, reason):
    r = run(freqs, mags, f0=f0)
    assert r["valid"] is False
    assert r["rejection_reason"] == reason
    assert math.isnan(r["value"])
    assert set(r) == RESULT_KEYS and set(r["coverage_diagnostics"]) == DIAG_KEYS


# ------------------------------------------------------------------ B. descriptor semantics
def test_exact_harmonic_series_is_near_zero_and_valid():
    r = run([100.0, 200.0, 300.0, 400.0], [1.0, 0.5, 0.3, 0.2])
    assert r["valid"] and r["rejection_reason"] is None
    assert abs(r["value"]) < 1e-6
    assert r["coverage_diagnostics"]["selected_peak_count"] == 4


def test_a_missing_intermediate_harmonic_creates_no_false_deviation():
    r = run([100.0, 200.0, 400.0], [1.0, 0.5, 0.2])
    assert r["valid"] and abs(r["value"]) < 1e-6
    assert r["coverage_diagnostics"]["selected_peak_count"] == 3  # zero-magnitude placeholders are not observed partials


def test_one_displaced_partial_matches_the_hand_calculation():
    # Hand-computed from the upstream Essentia source (spec section 1.1): f0 = first selected peak = 100;
    # partial 305 -> ratio round(3.05) = 3, deviation 5 Hz; weights are magnitude squared (1, 0.25, 0.25).
    r = run([100.0, 200.0, 305.0], [1.0, 0.5, 0.5])
    assert r["valid"]
    assert math.isclose(r["value"], 1.25 / (1.5 * 100.0), rel_tol=1e-5)


def test_value_rises_with_displacement_inside_the_tolerance():
    vals = [run([100.0, 200.0, 300.0 + d], [1.0, 0.5, 0.5])["value"] for d in (2.0, 5.0, 10.0, 15.0)]
    assert all(v == v for v in vals)  # no NaN: every one of these is a valid, fully selected series
    assert vals == sorted(vals) and len(set(vals)) == 4


def test_uniform_frequency_scaling_leaves_the_value_unchanged():
    base = run([100.0, 200.0, 305.0], [1.0, 0.5, 0.5])["value"]
    s = 1.5
    scaled = run([100.0 * s, 200.0 * s, 305.0 * s], [1.0, 0.5, 0.5], f0=100.0 * s)["value"]
    assert math.isclose(base, scaled, rel_tol=1e-5)


def test_uniform_amplitude_scaling_leaves_the_value_unchanged():
    base = run([100.0, 200.0, 305.0], [1.0, 0.5, 0.5])["value"]
    scaled = run([100.0, 200.0, 305.0], [7.0, 3.5, 3.5])["value"]
    assert math.isclose(base, scaled, rel_tol=1e-5)


def test_selection_dropout_is_detected_and_not_reported_as_harmonic():
    # partial 3 sits 0.25*f0 away from 300 Hz: outside the 0.2 tolerance, so Essentia silently drops it
    r = run([100.0, 200.0, 325.0], [1.0, 0.5, 0.5])
    assert r["valid"] is False
    assert r["rejection_reason"] == Reason.EXCLUDED_ENERGY
    assert math.isnan(r["value"])
    assert math.isclose(r["coverage_diagnostics"]["excluded_energy_fraction"], 1.0 - 1.25 / 1.5, rel_tol=1e-5)


def test_the_worked_example_that_the_old_policy_labelled_valid_and_harmonic():
    # The previous code accepted this (coverage 0.714 >= 0.5, two selected partials) with value 0.0.
    r = run([100.0, 200.0, 250.0, 350.0], [1.0, 0.5, 0.5, 0.5])
    assert r["valid"] is False and r["rejection_reason"] == Reason.EXCLUDED_ENERGY
    assert math.isclose(r["coverage_diagnostics"]["excluded_energy_fraction"], 1.0 - 1.25 / 1.75, rel_tol=1e-5)


def test_valid_implies_the_excluded_energy_is_within_the_limit_over_a_displacement_sweep():
    limit = 0.10
    for d in range(0, 60, 3):
        r = run([100.0, 200.0, 300.0 + d], [1.0, 0.5, 0.5], max_excluded_energy_fraction=limit)
        if r["valid"]:
            assert r["coverage_diagnostics"]["excluded_energy_fraction"] <= limit + 1e-9, d


def test_a_single_sinusoid_is_never_a_valid_label():
    r = run([100.0], [1.0])
    assert r["valid"] is False and r["rejection_reason"] == Reason.TOO_FEW_PARTIALS and math.isnan(r["value"])


def test_two_unrelated_pitches_do_not_get_a_valid_single_fundamental_label():
    r = run([100.0, 137.0, 211.0, 289.0, 353.0], [1.0] * 5)
    assert r["valid"] is False


def test_a_missing_fundamental_is_rejected_unless_explicitly_opted_out():
    r = run([200.0, 300.0, 400.0], [0.5, 0.3, 0.2])
    assert r["valid"] is False and r["rejection_reason"] == Reason.NO_FUNDAMENTAL_PEAK
    r2 = run([200.0, 300.0, 400.0], [0.5, 0.3, 0.2], require_fundamental_peak=False)
    assert r2["valid"] is True  # explicit opt-out; the value is then relative to the nominal pitch only


def test_every_failing_gate_is_reported_and_the_first_reason_is_never_overwritten():
    r = run([100.0, 250.0, 350.0], [1.0, 0.5, 0.5])
    assert r["valid"] is False
    assert r["rejection_reason"] == Reason.TOO_FEW_PARTIALS
    assert Reason.TOO_FEW_PARTIALS in r["all_rejection_reasons"]
    assert Reason.EXCLUDED_ENERGY in r["all_rejection_reasons"]


def test_an_invalid_result_is_nan_never_zero_and_has_the_complete_schema():
    for freqs, mags in (([], []), ([100.0], [0.0])):
        r = run(freqs, mags)
        assert r["valid"] is False and math.isnan(r["value"])
        assert set(r) == RESULT_KEYS and set(r["coverage_diagnostics"]) == DIAG_KEYS


def test_the_configuration_id_is_stable_and_sensitive_to_every_change():
    a, b, c = cfg(), cfg(), cfg(tolerance=0.15)
    assert a.config_id() == b.config_id()
    assert a.config_id() != c.config_id()
    assert a.config_id() != cfg(max_excluded_energy_fraction=0.20).config_id()
    r = run([100.0, 200.0], [1.0, 0.5])
    assert r["analysis_configuration_id"] == cfg().config_id()


def test_inputs_are_not_mutated():
    f = np.array([100.0, 200.0, 305.0], dtype=np.float64)
    m = np.array([1.0, 0.5, 0.5], dtype=np.float64)
    f0, m0 = f.copy(), m.copy()
    run(f, m)
    assert np.array_equal(f, f0) and np.array_equal(m, m0)


def test_the_config_loader_is_strict(tmp_path):
    good = dict(BASE)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(good))
    assert InharmonicityConfig.from_json(str(p)).tolerance == 0.2
    missing = dict(good)
    del missing["tolerance"]
    p.write_text(json.dumps(missing))
    with pytest.raises(ValueError):
        InharmonicityConfig.from_json(str(p))
    p.write_text(json.dumps(dict(good, bogus=1)))
    with pytest.raises(ValueError):
        InharmonicityConfig.from_json(str(p))
