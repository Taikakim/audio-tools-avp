# Owner: KUANG (created 2026-10-19)
"""V6 parameter layout (V6_SPECIFICATION.md section 3.1, V6_REVIEW.md section 5): the note is a conditioning input,
oscillator 1 is fixed at octave 0, oscillators 2 and 3 have octaves RELATIVE to oscillator 1, and the oscillator
2 / 3 fine pitch spans +-0.2 semitones.

Storage (59 values, the ordinal layout every bank / renderer / prior uses):
  index 0 midi_note: the CONDITION (normalised (note - 28) / 22). It stays in the stored vector so banks, datasets
                     and renderers work unchanged, and the JEPA's parameter side still sees it; the flow does NOT
                     predict it (param_codec_v6 leaves it out of the encoding and the flow takes it as an input).
  osc_1_octave:      removed (always 0).
  osc_2_octave, osc_3_octave: 5 classes, -2..+2 octaves relative to osc 1, stored class / 4.
  osc_2_pitch, osc_3_pitch:   [0, 1] -> -0.2..+0.2 semitones (0.5 = in tune).
  everything else:   as surge_spec_v5.

Rendering goes through surge_spec_v5 (to_v5 / from_v5 convert), so the Surge mapping is the one V5 verified.
Assumption to verify on Surge (check_v6_spec.py): surge_spec_v5 maps its osc pitch value p to +-1 semitone
(raw = 0.5 + (p - 0.5) * 2 / 14, i.e. a +-7 semitone raw range); the +-0.2 semitone span here relies on it.
Encoded width (param_codec_v6): 43 continuous + 71 one-hot = 114 (V6_SPECIFICATION.md says 110: 126 - 1 - 7 - 4
is 114).
"""
import numpy as np

import surge_spec_v5 as _v5
from surge_spec_v5 import (  # noqa: F401  (re-exported: renderers import these from the spec module)
    AUDIO_LEN, DEFAULT_PLUGIN_PATH, DURATION_S, FX_SLOTS, LP_FILTERS, NOTE_DUR_RANGE, NOTE_HIGH, NOTE_LOW,
    OCTAVE_THRESHOLDS, OSC_TYPES, SAMPLE_RATE, WAVESHAPER_TYPES, apply_patch, check_glitch, flush_synth, init_synth,
    phrase_events, raw_to_octave_class,
)

PARAM_NAMES = [n for n in _v5.PARAM_NAMES if n != "osc_1_octave"]
NUM_PARAMS = len(PARAM_NAMES)                                   # 59
PARAM_INDEX = {n: i for i, n in enumerate(PARAM_NAMES)}
CONDITION_PARAMS = ("midi_note",)
REL_OCTAVES = [-2, -1, 0, 1, 2]
DETUNE_ST = 0.2                                                 # +-semitones of osc 2 / 3 fine pitch
V5_PITCH_ST = 1.0                                               # surge_spec_v5's osc pitch value spans +-1 semitone

CATEGORICAL = {n: k for n, k in _v5.CATEGORICAL.items() if n != "osc_1_octave"}
CATEGORICAL["osc_2_octave"] = len(REL_OCTAVES)
CATEGORICAL["osc_3_octave"] = len(REL_OCTAVES)
CAT_INDICES = [(PARAM_INDEX[n], k) for n, k in CATEGORICAL.items()]
CONT_INDICES = [i for i, n in enumerate(PARAM_NAMES) if n not in CATEGORICAL]                # storage (JEPA side)
FLOW_CONT_INDICES = [i for i in CONT_INDICES if PARAM_NAMES[i] not in CONDITION_PARAMS]     # predicted by the flow
CONT_BOUNDS = {n: b for n, b in _v5.CONT_BOUNDS.items() if n in PARAM_INDEX}
DOMAIN_WEIGHTS = {n: w for n, w in _v5.DOMAIN_WEIGHTS.items() if n in PARAM_INDEX}

_V5_OCT = {n: _v5.PARAM_INDEX[n] for n in ("osc_1_octave", "osc_2_octave", "osc_3_octave")}


def domain_weights() -> np.ndarray:
    w = np.ones(NUM_PARAMS, dtype=np.float32)
    for name, value in DOMAIN_WEIGHTS.items():
        w[PARAM_INDEX[name]] = value
    return w / w.mean()


def ordinal_to_class(value, n_classes):
    return int(np.clip(round(float(value) * (n_classes - 1)), 0, n_classes - 1))


def to_v5(p):
    """[59] or [B, 59] v6 storage -> surge_spec_v5 storage (osc 1 at octave 0, relative octaves made absolute,
    fine pitch rescaled to v5's +-1 semitone value)."""
    p = np.asarray(p, dtype=np.float32)
    q = p.reshape(-1, NUM_PARAMS)
    out = np.empty((q.shape[0], _v5.NUM_PARAMS), dtype=np.float32)
    for j, name in enumerate(_v5.PARAM_NAMES):
        if name == "osc_1_octave":
            out[:, j] = 3 / 6.0
        elif name in ("osc_2_octave", "osc_3_octave"):
            rel = np.clip(np.round(q[:, PARAM_INDEX[name]] * 4.0), 0, 4) - 2
            out[:, j] = (3 + rel) / 6.0
        elif name in ("osc_2_pitch", "osc_3_pitch"):
            out[:, j] = 0.5 + (q[:, PARAM_INDEX[name]] - 0.5) * (DETUNE_ST / V5_PITCH_ST)
        else:
            out[:, j] = q[:, PARAM_INDEX[name]]
    return out.reshape(p.shape[:-1] + (_v5.NUM_PARAMS,))


def from_v5(p60, midi=None):
    """surge_spec_v5 storage -> (v6 storage, sounding MIDI note). The note moves by osc 1's octave so the patch
    keeps its pitch; it is then folded into NOTE_LOW..NOTE_HIGH by whole octaves (the timbre then belongs to a note
    an octave away: keytracking differs slightly). Octaves of osc 2 / 3 become relative and are clamped to +-2;
    fine pitch beyond +-0.2 semitones is clamped. midi: the note the v5 vector was rendered at (default: its own)."""
    v = np.asarray(p60, dtype=np.float32).reshape(_v5.NUM_PARAMS)
    c = {n: int(raw_to_octave_class(v[i])) for n, i in _V5_OCT.items()}
    note = int(round(NOTE_LOW + v[0] * (NOTE_HIGH - NOTE_LOW))) if midi is None else int(midi)
    note += 12 * (c["osc_1_octave"] - 3)
    while note > NOTE_HIGH and note - 12 >= NOTE_LOW:
        note -= 12
    while note < NOTE_LOW and note + 12 <= NOTE_HIGH:
        note += 12
    out = np.empty(NUM_PARAMS, dtype=np.float32)
    for name, i in PARAM_INDEX.items():
        if name == "midi_note":
            out[i] = np.clip((note - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW), 0.0, 1.0)
        elif name in ("osc_2_octave", "osc_3_octave"):
            out[i] = (np.clip(c[name] - c["osc_1_octave"], -2, 2) + 2) / 4.0
        elif name in ("osc_2_pitch", "osc_3_pitch"):
            out[i] = np.clip(0.5 + (v[_v5.PARAM_INDEX[name]] - 0.5) * (V5_PITCH_ST / DETUNE_ST), 0.0, 1.0)
        else:
            out[i] = v[_v5.PARAM_INDEX[name]]
    return out, int(np.clip(note, NOTE_LOW, NOTE_HIGH))


def canonicalize_vector(p):
    """Inactive dependants to 0, as surge_spec_v5.canonicalize_vector (unison detune with 1 voice, drive with the
    waveshaper off, delay feedback with no delay)."""
    p = np.array(p, dtype=np.float32, copy=True)
    q = p.reshape(-1, NUM_PARAMS)
    q[q[:, PARAM_INDEX["unison"]] < (0.5 / 5.0), PARAM_INDEX["unison_detune"]] = 0.0
    q[q[:, PARAM_INDEX["waveshaper_type"]] < (0.5 / (len(WAVESHAPER_TYPES) - 1)), PARAM_INDEX["drive"]] = 0.0
    q[q[:, PARAM_INDEX["delay_mix"]] <= 0.0, PARAM_INDEX["delay_fb"]] = 0.0
    return q.reshape(p.shape)


def vector_to_patch(p):
    """v6 storage -> the surge_spec_v5 patch dict (plus relative octave fields)."""
    p = np.asarray(p, dtype=np.float32)
    patch = _v5.vector_to_patch(to_v5(p))
    patch["osc_2_octave_rel"] = ordinal_to_class(p[PARAM_INDEX["osc_2_octave"]], 5) - 2
    patch["osc_3_octave_rel"] = ordinal_to_class(p[PARAM_INDEX["osc_3_octave"]], 5) - 2
    return patch


def render_patch(plugin, patch, midi_note, note_dur=0.3, duration=DURATION_S, sample_rate=SAMPLE_RATE):
    return _v5.render_patch(plugin, patch, midi_note, note_dur, duration, sample_rate)
