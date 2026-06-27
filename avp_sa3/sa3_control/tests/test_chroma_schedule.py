"""Pure (CPU, model-free) tests for the chroma development axis.

Covers chord_to_chroma (templates + L2-norm + sharpness), parse_progression,
and ChromaSchedule (shape, per-frame chord resolution, standardization, blend).
The ChromaGuidedGenerator runtime is GPU/model-bound and intentionally untested
here (see module docstring / 'gaps').
"""
import math

import numpy as np
import pytest

from sa3_control.chroma_guided_generator import (
    chord_to_chroma,
    parse_progression,
    ChromaSchedule,
)


# ---------------------------------------------------------------------------
# chord_to_chroma
# ---------------------------------------------------------------------------

def _active_pcs(vec, tol=1e-6):
    return sorted(int(i) for i in range(12) if vec[i] > tol)


def test_chord_to_chroma_minor_triad():
    # Am = A(9), C(0), E(4)
    v = chord_to_chroma("Am")
    assert _active_pcs(v) == [0, 4, 9]
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    assert v.dtype == np.float32


def test_chord_to_chroma_major_triad():
    # C = C(0), E(4), G(7)
    assert _active_pcs(chord_to_chroma("C")) == [0, 4, 7]
    # F# = F#(6), A#(10), C#(1)
    assert _active_pcs(chord_to_chroma("F#")) == [1, 6, 10]


def test_chord_to_chroma_seventh_and_flats():
    # Bb7 = Bb(10), D(2), F(5), Ab(8)
    assert _active_pcs(chord_to_chroma("Bb7")) == [2, 5, 8, 10]
    # Dm7 = D(2), F(5), A(9), C(0)
    assert _active_pcs(chord_to_chroma("Dm7")) == [0, 2, 5, 9]


def test_chord_to_chroma_raw_pcset():
    v = chord_to_chroma("0,4,7")
    assert _active_pcs(v) == [0, 4, 7]
    # equal-weight chord tones, L2-normalized -> each == 1/sqrt(3)
    assert abs(v[0] - 1.0 / math.sqrt(3)) < 1e-5


def test_chord_to_chroma_sharpness_leakage():
    hard = chord_to_chroma("C", sharpness=1.0)
    soft = chord_to_chroma("C", sharpness=0.5)
    # hard: off-tones are exactly zero
    assert hard[1] == 0.0
    # soft: off-tones leak in (non-zero) but stay below chord-tone weight
    assert soft[1] > 0.0
    assert soft[1] < soft[0]
    # both unit norm
    assert abs(float(np.linalg.norm(hard)) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(soft)) - 1.0) < 1e-5


def test_chord_to_chroma_degree_with_key():
    # In key C: vi == Am -> A(9),C(0),E(4)
    assert _active_pcs(chord_to_chroma("vi", key="C")) == [0, 4, 9]
    # In key C: V == G major -> G(7),B(11),D(2)
    assert _active_pcs(chord_to_chroma("V", key="C")) == [2, 7, 11]


def test_chord_to_chroma_empty_is_zero():
    v = chord_to_chroma("")
    assert float(np.linalg.norm(v)) == 0.0


# ---------------------------------------------------------------------------
# parse_progression
# ---------------------------------------------------------------------------

def test_parse_progression_single():
    assert parse_progression("Am") == "Am"


def test_parse_progression_schedule():
    out = parse_progression("0:Am|32:F|64:C|96:G")
    assert out == [(0.0, "Am"), (32.0, "F"), (64.0, "C"), (96.0, "G")]


def test_parse_progression_colon_safe_single():
    # a non-float leading token -> single prompt, not a schedule
    assert parse_progression("key:Cmaj") == "key:Cmaj"


def test_parse_progression_empty():
    assert parse_progression("") == ""


# ---------------------------------------------------------------------------
# ChromaSchedule
# ---------------------------------------------------------------------------

def test_schedule_requires_t0():
    with pytest.raises(ValueError):
        ChromaSchedule([(4.0, "Am")], fps=10.0)


def test_schedule_target_shape_numpy():
    sched = ChromaSchedule([(0.0, "Am")], fps=10.0)
    arr = sched.target_numpy(0.0, 25)
    assert arr.shape == (12, 25)
    assert arr.dtype == np.float32


def test_schedule_target_shape_torch():
    torch = pytest.importorskip("torch")
    sched = ChromaSchedule([(0.0, "Am")], fps=10.0)
    t = sched.target(0.0, 17)
    assert tuple(t.shape) == (1, 12, 17)
    assert t.dtype == torch.float32


def test_schedule_chord_change_resolves_in_window():
    # Am for t<2s, C for t>=2s; fps=10 -> change at frame 20.
    sched = ChromaSchedule([(0.0, "Am"), (2.0, "C")], fps=10.0)
    arr = sched.target_numpy(0.0, 40)  # 0..4s
    am = chord_to_chroma("Am")
    c = chord_to_chroma("C")
    # frame 19 (t=1.9) still Am, frame 20 (t=2.0) is C
    assert np.allclose(arr[:, 19], am, atol=1e-5)
    assert np.allclose(arr[:, 20], c, atol=1e-5)


def test_schedule_change_across_window_boundary():
    # window starting mid-progression picks up the right chord by absolute time
    sched = ChromaSchedule([(0.0, "Am"), (3.0, "F")], fps=10.0)
    arr = sched.target_numpy(3.0, 5)  # starts exactly at the change
    f = chord_to_chroma("F")
    assert np.allclose(arr[:, 0], f, atol=1e-5)


def test_schedule_standardization_applied():
    meta = {"standardized": True, "std_mean": 0.25, "std_std": 0.27}
    sched = ChromaSchedule([(0.0, "Am")], fps=10.0, head_metadata=meta)
    arr = sched.target_numpy(0.0, 4)
    raw = chord_to_chroma("Am")
    expected = (raw - 0.25) / 0.27
    assert np.allclose(arr[:, 0], expected, atol=1e-5)


def test_schedule_no_standardization_without_meta():
    sched = ChromaSchedule([(0.0, "Am")], fps=10.0)
    arr = sched.target_numpy(0.0, 4)
    assert np.allclose(arr[:, 0], chord_to_chroma("Am"), atol=1e-5)


def test_schedule_blend_crossfades():
    sched = ChromaSchedule([(0.0, "Am"), (2.0, "C")], fps=100.0, blend_sec=0.5)
    am = chord_to_chroma("Am")
    c = chord_to_chroma("C")
    # well before the blend window -> pure Am
    assert np.allclose(sched.chroma_at(1.0), am, atol=1e-5)
    # well after -> pure C
    assert np.allclose(sched.chroma_at(3.0), c, atol=1e-5)
    # at the exact change midpoint -> a normalized mix (not equal to either pure)
    mid = sched.chroma_at(2.0)
    assert not np.allclose(mid, am, atol=1e-3)
    assert not np.allclose(mid, c, atol=1e-3)
    assert abs(float(np.linalg.norm(mid)) - 1.0) < 1e-5
