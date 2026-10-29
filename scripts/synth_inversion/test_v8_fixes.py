# Owner: KUANG (created 2026-10-29)
"""CPU tests for the V8 fixes from docs/synth-v8-code-review-2026-10-29.md (avp-audio-craft).

No Surge, pedalboard, laion_clap or demucs: a fake synth honors pedalboard's reset semantics (reset=True clears the
voices, as pedalboard's process() does by default), a stub stands in for pedalboard's effects, and a fake loader for
the CLAP checkpoint. Run from this directory:  python -m pytest test_v8_fixes.py -q
"""

import os
import random
import sys
import types
from collections import defaultdict

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:  # musical_dataset_v8 imports pedalboard's effects at module level
    import pedalboard  # noqa: F401
except ImportError:
    _pb = types.ModuleType("pedalboard")

    class _Board(list):
        def __call__(self, wav, sr):
            return wav

    _pb.Pedalboard = _Board
    for _n in ("Chorus", "Delay", "Distortion", "HighpassFilter", "LowpassFilter", "Reverb"):
        setattr(_pb, _n, lambda *a, **k: None)
    sys.modules["pedalboard"] = _pb

import dynamic_dataset_v8 as dd
import dynamic_inverter_v8 as di
import musical_dataset_v8 as md
import surge_spec_v5 as S5
import surge_spec_v6 as S6


class _Param:
    def __init__(self):
        self.raw_value = 0.0


class FakeSynth:
    """Sine voices that sound from note-on to note-off. process() truncates duration * sample_rate like pedalboard
    and clears all voices when reset=True (pedalboard's default)."""

    def __init__(self):
        self.parameters = defaultdict(_Param)
        self.voices = {}
        self.clock = 0
        self.calls = []

    def reset(self):
        self.voices = {}

    def process(self, events, duration, sample_rate, num_channels=2, reset=True):
        if reset:
            self.voices = {}
        n = int(duration * sample_rate)
        self.calls.append({"reset": reset, "n": n,
                           "cutoff": self.parameters["a_filter_1_cutoff"].raw_value,
                           "drive": self.parameters["a_waveshaper_drive"].raw_value})
        marks = sorted((int(m.time * sample_rate), m.type, m.note) for m in events)
        out = np.zeros(n, np.float32)
        pos = 0
        for at, etype, note in marks + [(n, None, None)]:
            at = min(max(at, 0), n)
            idx = np.arange(pos, at) + self.clock
            for nt in self.voices:
                out[pos:at] += 0.2 * np.sin(2 * np.pi * 110.0 * 2 ** ((nt - 45) / 12) * idx / sample_rate)
            pos = at
            if etype == "note_on":
                self.voices[note] = True
            elif etype == "note_off":
                self.voices.pop(note, None)
        self.clock += n
        return np.stack([out] * num_channels)


def _patch():
    vec = np.full(S6.NUM_PARAMS, 0.5, np.float32)
    return S6.vector_to_patch(vec)


# ---------------------------------------------------------------- C1: one reset per phrase, gapless chunks
def test_renderer_resets_once_and_keeps_held_notes_sounding():
    synth = FakeSynth()
    flat = [np.full(dd.NUM_FRAMES, 0.5, np.float32)] * 4
    wav = dd.render_dynamic_patch(synth, _patch(), 40, *flat, notes=[(40, 100, 0.0, 3.5)])
    assert [c["reset"] for c in synth.calls] == [True] + [False] * (dd.NUM_FRAMES - 1)
    bounds = np.round(np.linspace(0, dd.TOTAL_SAMPLES, dd.NUM_FRAMES + 1)).astype(int)
    held = [k for k in range(dd.NUM_FRAMES) if bounds[k + 1] <= 3.4 * dd.SAMPLE_RATE]
    rms = [np.sqrt(np.mean(wav[bounds[k]:bounds[k + 1]] ** 2)) for k in held]
    assert min(rms) > 0.05  # every chunk under the held note sounds (with a reset per chunk all but the first were silent)


def test_renderer_chunks_follow_the_latent_grid_without_gaps():
    synth = FakeSynth()
    flat = [np.full(dd.NUM_FRAMES, 0.5, np.float32)] * 4
    wav = dd.render_dynamic_patch(synth, _patch(), 40, *flat)
    bounds = np.round(np.linspace(0, dd.TOTAL_SAMPLES, dd.NUM_FRAMES + 1)).astype(int)
    assert [c["n"] for c in synth.calls] == list(np.diff(bounds))
    assert sum(c["n"] for c in synth.calls) == dd.TOTAL_SAMPLES == len(wav)


# ---------------------------------------------------------------- C3: lanes in v6 units, drive mapped to raw
def test_lane_values_reach_the_plugin_in_raw_units():
    synth = FakeSynth()
    cutoff = np.linspace(0.1, 0.9, dd.NUM_FRAMES).astype(np.float32)
    drive = np.full(dd.NUM_FRAMES, 0.6, np.float32)
    zero = np.zeros(dd.NUM_FRAMES, np.float32)
    dd.render_dynamic_patch(synth, _patch(), 40, cutoff, zero, drive, zero)
    assert [c["cutoff"] for c in synth.calls] == pytest.approx(list(cutoff))
    assert all(c["drive"] == pytest.approx(0.8) for c in synth.calls)  # raw = 0.5 + 0.5 * 0.6, as the static vector
    assert dd.lane_to_raw(2, 0.0) == pytest.approx(0.5) and dd.lane_to_raw(0, 0.3) == pytest.approx(0.3)


# ---------------------------------------------------------------- C2: targets describe what was rendered
def test_rendered_static_vector_matches_the_rendered_patch():
    base = np.random.RandomState(0).rand(S6.NUM_PARAMS).astype(np.float32)
    lanes = np.stack([np.linspace(0.2, 0.8, 43), np.full(43, 0.3), np.full(43, 0.1), np.full(43, 0.25)]).astype(np.float32)
    for f_idx in dd.FILTER_CHOICES:
        vec = dd.rendered_static_vector(base, f_idx, lanes)
        patch = S6.vector_to_patch(vec)
        assert patch["filter_idx"] == f_idx and patch["filter_balance"] == 0.0
        for k, idx in enumerate(dd.LANE_V6_INDEX):
            assert vec[idx] == pytest.approx(lanes[k].mean())
    assert vec[S6.PARAM_INDEX["shape"]] == base[S6.PARAM_INDEX["shape"]]  # everything else untouched


def test_lane_active_follows_waveshaper_and_fm_routing():
    assert list(dd.lane_active({"ws_idx": 0, "fm_routing": 0})) == [True, True, False, False]
    assert list(dd.lane_active({"ws_idx": 3, "fm_routing": 2})) == [True, True, True, True]


def test_loss_ignores_lane_params_in_static_and_inactive_lanes():
    torch.manual_seed(0)
    ts, td = torch.rand(2, S6.NUM_PARAMS), torch.rand(2, 4, 43)
    ps, pd = ts.clone(), td.clone()
    ps[:, S6.PARAM_INDEX["cutoff"]] += 0.7         # a lane parameter in the static head: not a static error
    pd[:, 2] += 0.9                                # drive lane, inactive below
    active = torch.tensor([[True, True, False, True]] * 2)
    _, m = di.dynamic_inverter_loss(ps, ts, pd, td, lambda_smooth=0.0, lane_active=active)
    assert m["loss_static"] == pytest.approx(0.0, abs=1e-12) and m["loss_dynamic"] == pytest.approx(0.0, abs=1e-12)
    _, m_all = di.dynamic_inverter_loss(ps, ts, pd, td, lambda_smooth=0.0)
    assert m_all["loss_dynamic"] > 0.1


# ---------------------------------------------------------------- C4: the cue never overlaps the target
def _write_cache(tmp_path, with_ids=True, sr=1000):
    clip = 4 * sr
    k = lambda v, n: np.full((n, clip), v, np.float32)
    np.save(tmp_path / "kicks.npy", k(0.05, 2))
    np.save(tmp_path / "drums.npy", k(0.05, 2))
    # bass: stem 7 has clips 0..2 (values 0.10, 0.11, 0.12), stem 9 one clip (0.50, not usable as a cued target)
    bass = np.stack([np.full(clip, v, np.float32) for v in (0.10, 0.11, 0.12, 0.50)])
    np.save(tmp_path / "bass.npy", bass)
    synths = np.stack([np.full(clip, v, np.float32) for v in (0.20, 0.21, 0.30)])
    np.save(tmp_path / "synths.npy", synths)
    if with_ids:
        np.save(tmp_path / "bass_stem_ids.npy", np.array([7, 7, 7, 9]))
        np.save(tmp_path / "synths_stem_ids.npy", np.array([3, 3, 4]))


def test_cue_comes_from_another_clip_of_the_same_stem(tmp_path):
    _write_cache(tmp_path)
    random.seed(0)
    ds = md.MusicalDatasetV8(archive_root=tmp_path / "none", bank_v6_dir=None, sample_rate=1000, epoch_length=60,
                             cache_dir=tmp_path)
    for i in range(60):
        item = ds[i]
        t = float(item["target"].mean())
        c = float(item["cue_pos"].mean())
        assert bool(item["cue_disjoint"])
        assert item["cue_pos"].shape[-1] == 1000 and item["target"].shape[-1] == 4000
        assert abs(t - c) > 1e-6                         # a different clip, never the target's own audio
        ratio = c / t                                    # both scaled by the same peak factor
        siblings = {(0.10, 0.11), (0.10, 0.12), (0.11, 0.10), (0.11, 0.12), (0.12, 0.10), (0.12, 0.11),
                    (0.20, 0.21), (0.21, 0.20)}
        assert any(abs(ratio - b / a) < 1e-4 for a, b in siblings)  # same stem; stems 9 and 4 never cue-target


def test_cache_without_stem_ids_is_refused(tmp_path):
    _write_cache(tmp_path, with_ids=False)
    with pytest.raises(RuntimeError, match="stem ids"):
        md.MusicalDatasetV8(archive_root=tmp_path / "none", bank_v6_dir=None, sample_rate=1000, cache_dir=tmp_path)


def test_pick_start_avoids_the_target_window():
    random.seed(1)
    for _ in range(500):
        s = md.pick_start(10_000, 1_000, avoid=(4_000, 8_000))
        assert 0 <= s <= 9_000 and (s + 1_000 <= 4_000 or s >= 8_000)
    assert md.pick_start(4_500, 1_000, avoid=(0, 4_000)) is None
    assert md.pick_start(900, 1_000) is None


def test_cache_builder_records_stem_ids_and_the_val_split(tmp_path):
    import soundfile as sf
    import cache_v8_stems as cs
    root = tmp_path / "archive"
    for song in ("Song A", sorted(cs.HOLDOUT_SONGS)[0]):
        (root / song / "bass").mkdir(parents=True)
        for j in range(2):
            sf.write(root / song / "bass" / f"b{j}.wav", 0.1 * np.ones(int(9 * cs.SR), np.float32), cs.SR)
    out = tmp_path / "cache"
    cs.main(["--archive-root", str(root), "--cache-dir", str(out), "--start-s", "0", "--read-s", "9"])
    cs.main(["--split", "val", "--archive-root", str(root), "--cache-dir", str(out), "--start-s", "0", "--read-s", "9"])
    assert list(np.load(out / "bass_stem_ids.npy")) == [0, 0, 1, 1]   # 2 clips per 9 s stem, ids per file
    assert list(np.load(out / "val_bass_stem_ids.npy")) == [0, 0, 1, 1]
    assert "Song A" in (out / "stem_names.txt").read_text(encoding="utf-8")
    assert "Song A" not in (out / "val_stem_names.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------- C6: CLAP weights load or fail loudly
class _Clapish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)
        self.register_buffer("position_ids", torch.arange(4))


def test_clap_load_accepts_a_matching_checkpoint_and_only_position_ids_missing(tmp_path):
    import cgsp_extractor as ce
    f = tmp_path / "clap.pt"
    f.write_bytes(b"x")
    m = _Clapish()
    sd = {k: v for k, v in _Clapish().state_dict().items() if k != "position_ids"}
    assert ce.load_clap_weights(m, str(f), state_dict_loader=lambda p: sd) == 2
    assert torch.equal(m.proj.weight, sd["proj.weight"])


def test_clap_load_refuses_mismatched_or_missing_checkpoints(tmp_path):
    import cgsp_extractor as ce
    f = tmp_path / "clap.pt"
    f.write_bytes(b"x")
    wrong = {"text_branch.layer.weight": torch.zeros(2)}
    with pytest.raises(RuntimeError, match="does not match"):
        ce.load_clap_weights(_Clapish(), str(f), state_dict_loader=lambda p: wrong)
    with pytest.raises(FileNotFoundError):
        ce.load_clap_weights(_Clapish(), str(tmp_path / "missing.pt"), state_dict_loader=lambda p: wrong)
    with pytest.raises(ValueError):
        ce.load_clap_weights(_Clapish(), None)


# ---------------------------------------------------------------- M6: Phase 2 metrics
def test_phase2_summary_reports_family_rho_and_r2_against_a_constant_guess():
    import eval_v8_phase2_dynamic as e2
    T = 43
    t = np.linspace(0, 1, T)
    true = np.zeros((3, 4, T), np.float32)
    true[0, 0] = 0.1 + 0.8 * t                       # rising
    true[1, 0] = 0.5 + 0.4 * np.sin(2 * np.pi * t)   # sinusoidal
    true[2, 0] = 0.5                                 # static: no rho
    true[:, 1] = 0.4
    pred = true.copy()
    pred[0, 0] = t ** 3                              # wrong values, right direction: rho 1 on a monotone family
    pred[1, 0] = 0.5                                 # flat guess on the sinusoid
    pred[:, 1] = 0.4
    active = np.array([[True, True, False, False]] * 3)
    res = e2.summarize(np.zeros((3, S6.NUM_PARAMS)), np.zeros((3, S6.NUM_PARAMS)), pred, true, active,
                       np.array([dd.SWEEP_TYPES.index("rising"), dd.SWEEP_TYPES.index("sinusoidal"), 4]),
                       lane_means=np.array([0.5, 0.5, 0.3, 0.2]))
    assert res["cutoff_rho_by_family"]["rising"]["mean"] == pytest.approx(1.0)
    assert res["cutoff_rho_n"] == 1                  # the flat sinusoid prediction has no rank correlation
    assert res["lanes"]["resonance"]["r2"] == pytest.approx(1.0)  # exact against a 0.1-off constant guess
    assert res["lanes"]["drive"] == {"n_active": 0}
    assert res["lanes"]["cutoff"]["r2"] < 1.0


# ---------------------------------------------------------------- M8: MIDI automation file
def test_daw_midi_has_notes_and_one_cc_per_lane_and_frame():
    import daw_export_v8 as dx
    lanes = np.tile(np.linspace(0.0, 1.0, 43, dtype=np.float32), (4, 1))
    mid = dx.create_daw_automation_midi(lanes, bpm=140.0)
    msgs = [m for m in mid.tracks[0] if not m.is_meta]
    ccs = [m for m in msgs if m.type == "control_change"]
    assert len(ccs) == 4 * 43 and len([m for m in msgs if m.type == "note_on"]) == 16
    assert {m.control for m in ccs} == {74, 71, 1, 11}
    assert [m.value for m in ccs if m.control == 74][-1] == 127 and [m.value for m in ccs if m.control == 74][0] == 0
    end_tick = sum(m.time for m in mid.tracks[0])
    assert end_tick == round((3.75 + 0.18) / ((60.0 / 140.0) / 480))  # last note-off at 3.93 s
