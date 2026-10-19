# Owner: KUANG (created 2026-10-19)
"""One place that maps a flow checkpoint to its parameter layout (spec module + codec module).

Before this, flow_sampling_v5, render_pool_v5, dpo_pairs_v5, train_flow_dpo_v5 and audio_eval_v5 imported the v3
spec and codec directly, so they could not be used on v4 (108-d) or v5 (126-d) flows; the real-stem eval guessed the
note range. Every layout so far uses NOTE_LOW..NOTE_HIGH = 28..50; take it from here, never hard-code it.

  L = layout_for_dim(flow.param_dim)        # 92 -> v3, 108 -> v4, 126 -> v5, 114 -> v6 (from the codecs)
  L = layout_named("v5")
  L.spec, L.codec, L.name, L.encoded_dim, L.num_params
  L.note_unit(midi)                          # MIDI note -> the [0, 1] value of parameter 0
  fold_note(midi, L)                         # -> (note given to the model, octaves folded) for out-of-range notes
  shift_octaves(vec, k, L)                   # move every oscillator's octave by k (all three layouts)
"""
import importlib
from dataclasses import dataclass

import numpy as np

_BY_NAME = {"v3": ("surge_spec_v3", "param_codec_v3"),
            "v4": ("surge_spec_v4", "param_codec_v4"),
            "v5": ("surge_spec_v5", "param_codec_v5"),
            "v6": ("surge_spec_v6", "param_codec_v6")}
OCTAVE_PARAMS = ("osc_1_octave", "osc_2_octave", "osc_3_octave")


@dataclass(frozen=True)
class Layout:
    name: str
    spec: object
    codec: object

    @property
    def encoded_dim(self):
        return int(self.codec.ENCODED_DIM)

    @property
    def num_params(self):
        return int(self.spec.NUM_PARAMS)

    @property
    def note_range(self):
        return int(self.spec.NOTE_LOW), int(self.spec.NOTE_HIGH)

    @property
    def note_conditioned(self):
        """v6: the note is the flow's input (models_v6), osc 1 is fixed at octave 0; not a predicted parameter."""
        return hasattr(self.spec, "CONDITION_PARAMS")

    def decode(self, x, note_unit=None):
        """Flow output [B, D] -> storage vectors. note_unit [B]: required (and used) for note-conditioned layouts."""
        if self.note_conditioned:
            if note_unit is None:
                raise ValueError(f"layout {self.name} needs the note to decode")
            return self.codec.decode(x, note_unit)
        return self.codec.decode(x)

    def note_unit(self, midi):
        lo, hi = self.note_range
        return np.clip((np.asarray(midi, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def layout_named(name):
    if name not in _BY_NAME:
        raise ValueError(f"unknown layout {name!r}; known: {sorted(_BY_NAME)}")
    spec_name, codec_name = _BY_NAME[name]
    return Layout(name, importlib.import_module(spec_name), importlib.import_module(codec_name))


def dims():
    """{encoded width: layout name}, read from the codecs (currently 92 -> v3, 108 -> v4, 126 -> v5)."""
    return {layout_named(n).encoded_dim: n for n in _BY_NAME}


def layout_for_dim(param_dim):
    """Encoded flow output width -> Layout. Raises for an unknown width instead of guessing (a checkpoint from an
    older v3 parameter set, e.g. the 48-parameter A-series, needs the code of its own commit)."""
    known = dims()
    name = known.get(int(param_dim))
    if name is None:
        raise ValueError(f"no layout has encoded width {param_dim}; known: {known}")
    return layout_named(name)


def layout_for_flow(flow):
    return layout_for_dim(flow.param_dim)


def fold_note(midi, layout):
    """Fold a MIDI note into the trained range by whole octaves. -> (note for the model, k) with
    note_for_model = midi - 12 * k. The model, told the note is 12 k lower than it sounds, raises the oscillators by
    k octaves to match the audio; the patch is then right at note_for_model. To play it at the real note,
    shift_octaves(vec, -k). Notes no whole-octave shift brings into range keep k = 0 (and are clipped)."""
    lo, hi = layout.note_range
    m, k = int(midi), 0
    while m > hi and m - 12 >= lo:
        m, k = m - 12, k + 1
    while m < lo and m + 12 <= hi:
        m, k = m + 12, k - 1
    return m, k


def shift_octaves(vec, k, layout):
    """Shift all three oscillators by k octaves in the ordinal storage layout (v3 / v4 / v5: 7 octave positions,
    stored as class / 6). Clamped at -3 / +3; returns (new vector, number of oscillators that hit the clamp).
    v6: a no-op. Osc 1 is fixed at octave 0 and osc 2 / 3 are relative, so a patch played at the real note
    already sounds at the right pitch."""
    out = np.array(vec, dtype=np.float32, copy=True)
    if k == 0 or layout.note_conditioned:
        return out, 0
    if not all(n in layout.spec.PARAM_INDEX for n in OCTAVE_PARAMS):
        raise ValueError(f"layout {layout.name} has no per-oscillator octave parameters")
    clamped = 0
    for n in OCTAVE_PARAMS:
        i = layout.spec.PARAM_INDEX[n]
        cls = octave_class(out[..., i], layout)
        new = cls + int(k)
        clamped += int(np.any((new < 0) | (new > 6)))
        out[..., i] = np.clip(new, 0, 6) / 6.0
    return out, clamped


def octave_class(value, layout):
    """Stored octave value -> class 0..6 (-3..+3). v3 / v4 pass the stored value to Surge as the raw octave, so
    Surge's own digitisation applies; v5 stores class / 6, which the same thresholds map back exactly."""
    v = np.asarray(value, dtype=np.float64)
    return np.digitize(v, getattr(layout.spec, "OCTAVE_THRESHOLDS", SURGE_OCTAVE_THRESHOLDS)).astype(int)


SURGE_OCTAVE_THRESHOLDS = [0.0876, 0.2526, 0.4176, 0.5826, 0.7475, 0.9125]   # as surge_spec_v5 (measured there)
