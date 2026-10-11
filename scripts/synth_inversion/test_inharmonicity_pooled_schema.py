"""Round-2 contract: the pooled (per-phrase) label keeps the full schema and the per-window results.

Why: the pooled result is the label that actually trains the head, and it is what monitoring reads. If it drops
`analysis_configuration_id`, the label cannot be traced to its settings; if it drops `window_results`, nobody can
tell WHY a phrase was rejected (every failure collapses to "too few valid note windows").

PROVIDED BY THE SPEC. Do NOT edit assertions. Run from scripts/synth_inversion:
    REQUIRE_ALL=1 python -m pytest test_inharmonicity_pooled_schema.py -q
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


need("numpy")
need("essentia")

from inharmonicity_target import (InharmonicityConfig, Reason, inharmonicity_from_peaks,  # noqa: E402
                                  pool_note_results)

# TEST VALUES ONLY. They are NOT recommended production values.
BASE = dict(sample_rate=44100, frame_size=4096, max_harmonics=20, tolerance=0.2,
            min_selected_partials=2, max_excluded_energy_fraction=0.10, require_fundamental_peak=True,
            silence_peak_abs=1e-4, max_level_change_db=None, attack_guard_s=0.020, min_valid_windows=2,
            min_partial_relative_db=40.0)
CFG = InharmonicityConfig(**BASE)

FULL_KEYS = {"value", "valid", "rejection_reason", "all_rejection_reasons", "fundamental_hz",
             "fundamental_source", "analysis_configuration_id", "coverage_diagnostics",
             "n_valid_windows", "window_results"}
DIAG_KEYS = {"input_peak_count", "selected_peak_count", "total_peak_energy", "selected_peak_energy",
             "selected_energy_fraction", "excluded_energy_fraction", "tolerance", "max_harmonics",
             "first_selected_hz", "level_change_db"}


def _windows():
    good = inharmonicity_from_peaks([100.0, 200.0, 305.0], [1.0, 0.5, 0.5], 100.0, CFG)       # valid
    bad = inharmonicity_from_peaks([100.0, 200.0, 325.0], [1.0, 0.5, 0.5], 100.0, CFG)        # excluded energy
    assert good["valid"] and not bad["valid"]
    return good, bad


def test_a_valid_pooled_result_has_the_full_schema():
    good, _ = _windows()
    p = pool_note_results([good, good], CFG)
    assert set(p) == FULL_KEYS
    assert p["valid"] is True and p["rejection_reason"] is None and p["all_rejection_reasons"] == []
    assert p["n_valid_windows"] == 2 and math.isclose(p["value"], good["value"])
    assert p["fundamental_hz"] is None and p["fundamental_source"] == "per_note_midi"
    assert p["analysis_configuration_id"] == CFG.config_id()
    d = p["coverage_diagnostics"]
    assert set(d) == DIAG_KEYS
    assert d["tolerance"] == CFG.tolerance and d["max_harmonics"] == CFG.max_harmonics
    assert all(v is None for k, v in d.items() if k not in ("tolerance", "max_harmonics"))
    assert len(p["window_results"]) == 2


def test_a_rejected_pooled_result_keeps_the_reason_each_window_failed():
    good, bad = _windows()
    p = pool_note_results([good, bad], CFG)  # min_valid_windows = 2, only one valid
    assert set(p) == FULL_KEYS
    assert p["valid"] is False and math.isnan(p["value"])
    assert p["rejection_reason"] == Reason.TOO_FEW_VALID_WINDOWS
    assert p["n_valid_windows"] == 1
    assert p["analysis_configuration_id"] == CFG.config_id()
    assert p["window_results"][0]["valid"] is True
    assert p["window_results"][1]["rejection_reason"] == Reason.EXCLUDED_ENERGY  # the ROOT cause is not lost


def test_a_dropped_window_is_recorded_as_none_not_silently_removed():
    good, _ = _windows()
    p = pool_note_results([None, good], CFG)
    assert set(p) == FULL_KEYS
    assert len(p["window_results"]) == 2 and p["window_results"][0] is None
    assert p["valid"] is False and p["rejection_reason"] == Reason.TOO_FEW_VALID_WINDOWS
