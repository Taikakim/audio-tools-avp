"""Alignment contract: label windows come from the SAME note timeline render_patch plays; pooling never counts
invalid windows as zeros; invalid labels become (0.0, mask 0.0) in exactly one place.

PROVIDED BY THE SPEC and VALID ONLY IF Kim approved decision D1 as recommended (spec section 3). The event times
asserted below are what render_patch passes TODAY: they must not change (spec rule R10). Do NOT edit assertions.
Run from scripts/synth_inversion:   REQUIRE_ALL=1 python -m pytest test_inharmonicity_phrase.py -q
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
sp = need("surge_spec_v3")

from inharmonicity_target import (InharmonicityConfig, Reason, RejectionStats, note_windows,  # noqa: E402
                                  pool_note_results, to_training_pair)

# TEST VALUES ONLY. They are NOT recommended production values: those are Kim's decision (spec section 3).
BASE = dict(sample_rate=44100, frame_size=4096, max_harmonics=20, tolerance=0.2,
            min_selected_partials=2, max_excluded_energy_fraction=0.10, require_fundamental_peak=True,
            silence_peak_abs=1e-4, max_level_change_db=None, attack_guard_s=0.020, min_valid_windows=2,
            min_partial_relative_db=40.0)

# (type, note, velocity, absolute time in seconds from the start of the buffer): exactly render_patch's current values
EXPECTED_EVENTS_40 = [("note_on", 40, 105, 0.0), ("note_off", 40, 0, 0.125),
                      ("note_on", 52, 105, 0.125), ("note_off", 52, 0, 0.250)]


def cfg(**over):
    d = dict(BASE)
    d.update(over)
    return InharmonicityConfig(**d)


class FakePlugin:
    def __init__(self):
        self.calls = []

    def reset(self):
        pass

    def process(self, events, duration, sample_rate, num_channels):
        self.calls.append(dict(events=list(events), duration=duration, sample_rate=sample_rate,
                               num_channels=num_channels))
        return np.zeros((num_channels, int(duration * sample_rate)), dtype=np.float32)


# ------------------------------------------------------------------ timeline: one source of truth
def test_phrase_events_are_exactly_what_render_patch_plays_today():
    got = [(t, n, v, round(s, 6)) for (t, n, v, s) in sp.phrase_events(40)]
    assert got == EXPECTED_EVENTS_40


def test_the_second_note_is_clamped_to_127():
    ev = sp.phrase_events(120)
    assert ev[2][1] == 127 and ev[3][1] == 127


def test_spans_are_derived_from_the_same_events():
    assert sp.phrase_note_spans(40) == [{"note": 40, "on_s": 0.0, "off_s": 0.125},
                                        {"note": 52, "on_s": 0.125, "off_s": 0.250}]


def test_render_patch_sends_exactly_phrase_events(monkeypatch):
    need("mido")
    monkeypatch.setattr(sp, "apply_patch", lambda plugin, patch: None)
    fake = FakePlugin()
    out = sp.render_patch(fake, {}, 40, 0.3)
    call = fake.calls[0]
    sent = [(m.type, m.note, m.velocity, round(m.time, 6)) for m in call["events"]]
    assert sent == EXPECTED_EVENTS_40
    assert sent == [(t, n, v, round(s, 6)) for (t, n, v, s) in sp.phrase_events(40)]
    assert call["duration"] == sp.DURATION_S and call["sample_rate"] == sp.SAMPLE_RATE and call["num_channels"] == 2
    assert out.shape == (int(sp.DURATION_S * sp.SAMPLE_RATE),)


# ------------------------------------------------------------------ windows
def test_windows_end_at_note_off_start_after_the_attack_guard_and_use_the_playing_notes_pitch():
    spans = sp.phrase_note_spans(40)
    wins = note_windows(spans, cfg())
    assert len(wins) == 2 and all(w is not None for w in wins)
    for w, s in zip(wins, spans):
        assert w["note"] == s["note"]
        assert w["end_sample"] == round(s["off_s"] * 44100)
        assert w["end_sample"] - w["start_sample"] == 4096
        assert w["start_sample"] >= round((s["on_s"] + 0.020) * 44100)
        assert math.isclose(w["f0_hz"], 440.0 * 2 ** ((s["note"] - 69) / 12), rel_tol=1e-12)


def test_a_window_is_dropped_not_shortened_when_the_guard_leaves_no_room():
    assert note_windows(sp.phrase_note_spans(40), cfg(attack_guard_s=0.050)) == [None, None]


# ------------------------------------------------------------------ pooling
def _res(valid, value, reason=None):
    return {"valid": valid, "value": value, "rejection_reason": reason,
            "all_rejection_reasons": [reason] if reason else []}


def test_pooling_is_the_mean_of_valid_windows_and_never_counts_invalid_as_zero():
    a, b = _res(True, 0.02), _res(True, 0.04)
    bad = _res(False, float("nan"), Reason.EXCLUDED_ENERGY)
    p = pool_note_results([a, b], cfg(min_valid_windows=2))
    assert p["valid"] and math.isclose(p["value"], 0.03) and p["n_valid_windows"] == 2
    p1 = pool_note_results([bad, b], cfg(min_valid_windows=1))
    assert p1["valid"] and math.isclose(p1["value"], 0.04) and p1["n_valid_windows"] == 1
    p2 = pool_note_results([bad, b], cfg(min_valid_windows=2))
    assert p2["valid"] is False and p2["rejection_reason"] == Reason.TOO_FEW_VALID_WINDOWS and math.isnan(p2["value"])
    p3 = pool_note_results([None, None], cfg(min_valid_windows=1))
    assert p3["valid"] is False and p3["rejection_reason"] == Reason.TOO_FEW_VALID_WINDOWS


# ------------------------------------------------------------------ the single invalid -> (0.0, 0.0) mapping
def test_training_pair_maps_invalid_to_zero_with_a_zero_mask_and_refuses_inconsistent_results():
    assert to_training_pair({"valid": True, "value": 0.02}) == (0.02, 1.0)
    assert to_training_pair({"valid": False, "value": float("nan")}) == (0.0, 0.0)
    with pytest.raises(ValueError):
        to_training_pair({"valid": True, "value": float("nan")})


# ------------------------------------------------------------------ monitoring
def test_rejection_stats_count_reasons_and_valid_rate_by_note():
    st = RejectionStats()
    st.add(_res(True, 0.01), 40)
    st.add(_res(False, float("nan"), Reason.TOO_FEW_PARTIALS), 40)
    st.add(_res(False, float("nan"), Reason.EXCLUDED_ENERGY), 50)
    s = st.summary()
    assert s["n"] == 3 and s["n_valid"] == 1 and math.isclose(s["valid_rate"], 1 / 3)
    assert s["by_reason"] == {Reason.TOO_FEW_PARTIALS: 1, Reason.EXCLUDED_ENERGY: 1}
    assert s["valid_rate_by_note"] == {40: 0.5, 50: 0.0}
    st.reset()
    assert st.summary()["n"] == 0
