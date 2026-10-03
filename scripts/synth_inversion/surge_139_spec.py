"""Surge XT 139-Parameter Specification for Synth-JEPA.

References:
  - Hayes, Tian, Lattner, "Synth-JEPA: Joint Embedding Prediction for Renderer-Free
    Synthesizer Parameter Search", arXiv:2609.31024, Sec. 3.1 & 3.2.
  - Barkan et al., "InverSynth II", ISMIR 2023 [11].

Characteristics:
  * 139 parameters covering 3 Oscillators, Mixer, Dual Filters, Envelopes, and 6 LFOs.
  * Audio effects are strictly disabled (all FX slots set to bypass/0.0).
  * Standard audio rendering: 3.0 seconds, stereo, 44.1 kHz.

v2 (2026-10-03 review). v1 assumed every enum's length (osc type 7, filter type 12, unison
voices 8, LFO type 5 "deterministic", ...) and set raw = class/(n-1). Surge's real lists are
longer (e.g. 34 filter types, see surge_spec.LP_FILTERS), so class k landed on an arbitrary list
entry, and the five "deterministic" LFO classes swept the WHOLE LFO list, including Noise, S&H,
Step Seq and MSEG. It also silently skipped any name the plugin does not expose. Now:
  * verify_param_names() fails loudly if any of the 139 names is missing from the plugin;
  * calibrate_enums() reads each enum's real entries from the plugin (sweeping raw values and
    collecting the distinct display strings), drops entries that make a training sample
    meaningless (LFO shapes other than Sine/Triangle/Square/Sawtooth, which are random or
    sequenced; the Audio Input oscillator, which is silent), and returns class -> raw tables;
  * the mute switches are categorical (on/off), not continuous;
  * render_patch_139() returns None for a silent render (e.g. every oscillator muted), so the
    dataset can redraw instead of training on a normalised noise floor.
Known gap, not fixed: in a default patch the LFOs are not routed to anything, so their 60
parameters have no audible effect; the paper's parameterisation presumably includes routings.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

SAMPLE_RATE = 44100
DURATION_S = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_S)  # 132,300 samples
NUM_PARAMS = 139
DEFAULT_PLUGIN_PATH = "~/.vst3/Surge XT.vst3"
SILENCE_PEAK = 1e-5

# 139 Parameters in canonical order
PARAM_NAMES: List[str] = []

# Oscillators 1, 2, 3 (12 params each = 36 params)
for osc in [1, 2, 3]:
    PARAM_NAMES.extend([
        f"a_osc_{osc}_type",
        f"a_osc_{osc}_octave",
        f"a_osc_{osc}_pitch",
        f"a_osc_{osc}_shape",
        f"a_osc_{osc}_width_1",
        f"a_osc_{osc}_sub_mix",
        f"a_osc_{osc}_sync",
        f"a_osc_{osc}_unison_voices",
        f"a_osc_{osc}_unison_detune",
        f"a_osc_{osc}_volume",
        f"a_osc_{osc}_mute",
        f"a_osc_{osc}_route",
    ])

# Mixer (6 params)
PARAM_NAMES.extend([
    "a_noise_volume",
    "a_noise_color",
    "a_ring_modulation_1x2_volume",
    "a_ring_modulation_1x2_mute",
    "a_ring_modulation_2x3_volume",
    "a_ring_modulation_2x3_mute",
])

# FM & Feedback (4 params)
PARAM_NAMES.extend([
    "a_fm_routing",
    "a_fm_depth",
    "a_feedback",
    "a_pre_filter_gain",
])

# Filter 1 & Filter 2 & Routing (15 params)
PARAM_NAMES.extend([
    "a_filter_1_type",
    "a_filter_1_subtype",
    "a_filter_1_cutoff",
    "a_filter_1_resonance",
    "a_filter_1_keytrack",
    "a_filter_1_feg_mod_amount",
    "a_filter_2_type",
    "a_filter_2_subtype",
    "a_filter_2_cutoff",
    "a_filter_2_resonance",
    "a_filter_2_keytrack",
    "a_filter_2_feg_mod_amount",
    "a_filter_configuration",
    "a_filter_balance",
    "a_highpass",
])

# Waveshaper (2 params)
PARAM_NAMES.extend([
    "a_waveshaper_type",
    "a_waveshaper_drive",
])

# Envelopes (AEG & FEG) (14 params)
PARAM_NAMES.extend([
    "a_amp_eg_attack",
    "a_amp_eg_attack_shape",
    "a_amp_eg_decay",
    "a_amp_eg_decay_shape",
    "a_amp_eg_sustain",
    "a_amp_eg_release",
    "a_amp_eg_release_shape",
    "a_filter_eg_attack",
    "a_filter_eg_attack_shape",
    "a_filter_eg_decay",
    "a_filter_eg_decay_shape",
    "a_filter_eg_sustain",
    "a_filter_eg_release",
    "a_filter_eg_release_shape",
])

# LFO 1 to LFO 6 (10 params each = 60 params)
for lfo in range(1, 7):
    PARAM_NAMES.extend([
        f"a_lfo_{lfo}_type",
        f"a_lfo_{lfo}_rate",
        f"a_lfo_{lfo}_phase",
        f"a_lfo_{lfo}_deform",
        f"a_lfo_{lfo}_amplitude",
        f"a_lfo_{lfo}_attack",
        f"a_lfo_{lfo}_hold",
        f"a_lfo_{lfo}_decay",
        f"a_lfo_{lfo}_sustain",
        f"a_lfo_{lfo}_release",
    ])

# Scene Global (2 params)
PARAM_NAMES.extend([
    "a_pan",
    "a_width",
])

assert len(PARAM_NAMES) == 139, f"Expected 139 parameters, got {len(PARAM_NAMES)}"

PARAM_INDEX: Dict[str, int] = {name: i for i, name in enumerate(PARAM_NAMES)}

# Which parameters are categorical. Their class lists come from the plugin (calibrate_enums).
CATEGORICAL_NAMES = (
    [f"a_osc_{o}_{p}" for o in (1, 2, 3) for p in ("type", "unison_voices", "route", "mute")]
    + ["a_ring_modulation_1x2_mute", "a_ring_modulation_2x3_mute", "a_filter_1_type", "a_filter_2_type",
       "a_filter_configuration", "a_waveshaper_type", "a_fm_routing"]
    + [f"a_lfo_{lfo}_type" for lfo in range(1, 7)]
)
CAT_INDEX_SET = {PARAM_INDEX[n] for n in CATEGORICAL_NAMES}
CONT_INDICES: List[int] = [i for i in range(NUM_PARAMS) if i not in CAT_INDEX_SET]

# Entries excluded from enum classes (lower-case substrings of the display string)
LFO_KEEP = ("sine", "tri", "square", "saw")  # periodic and deterministic
OSC_TYPE_DROP = ("audio in",)                 # silent without an input signal


def verify_param_names(plugin) -> None:
    missing = [n for n in PARAM_NAMES if n not in plugin.parameters]
    if missing:
        raise RuntimeError(f"{len(missing)} of the 139 parameter names are not exposed by this plugin: {missing[:12]}"
                           f"{' ...' if len(missing) > 12 else ''}. Fix PARAM_NAMES in surge_139_spec.py.")


def _sweep_enum(param, steps: int = 2049) -> List[Tuple[str, float]]:
    """Distinct display strings in raw order -> [(display, raw at the middle of its band)]."""
    old = param.raw_value
    bands = []
    for r in np.linspace(0.0, 1.0, steps):
        param.raw_value = float(r)
        s = str(param.string_value)
        if bands and bands[-1][0] == s:
            bands[-1][2] = float(r)
        else:
            bands.append([s, float(r), float(r)])
    param.raw_value = old
    return [(s, 0.5 * (lo + hi)) for s, lo, hi in bands]


def calibrate_enums(plugin) -> Dict[str, List[Tuple[str, float]]]:
    """{categorical name: [(display, raw)] per class}, read from the plugin."""
    tables = {}
    for name in CATEGORICAL_NAMES:
        entries = _sweep_enum(plugin.parameters[name])
        if "_lfo_" in name and name.endswith("_type"):
            entries = [e for e in entries if any(k in e[0].lower() for k in LFO_KEEP)]
        elif name.startswith("a_osc_") and name.endswith("_type"):
            entries = [e for e in entries if not any(k in e[0].lower() for k in OSC_TYPE_DROP)]
        if len(entries) < 2:
            raise RuntimeError(f"{name}: enum calibration found {len(entries)} usable entries ({entries})")
        tables[name] = entries
    return tables


def cat_indices(tables) -> List[Tuple[int, int]]:
    """[(vector index, n_classes)] in PARAM_NAMES order."""
    return [(i, len(tables[n])) for i, n in enumerate(PARAM_NAMES) if n in tables]


def draw_patch_139(seed: int, tables) -> dict:
    """Draw a random 139-parameter patch from the uniform prior. Categoricals are stored as
    class / (n - 1), continuous parameters as their raw value."""
    rng = np.random.default_rng(seed)
    patch = {}
    for name in PARAM_NAMES:
        if name in tables:
            n = len(tables[name])
            patch[name] = float(rng.integers(0, n) / (n - 1))
        else:
            patch[name] = float(rng.uniform(0.0, 1.0))
    patch["midi_note"] = int(rng.integers(24, 96))
    patch["note_dur"] = float(rng.uniform(0.2, 2.5))
    patch["fx_bypass"] = True
    return patch


def patch_to_vector_139(patch: dict) -> np.ndarray:
    vec = np.zeros(NUM_PARAMS, dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        vec[i] = float(patch.get(name, 0.0))
    return vec


def vector_to_patch_139(vec: np.ndarray, midi_note: int = 60, note_dur: float = 1.5) -> dict:
    patch = {"midi_note": midi_note, "note_dur": note_dur, "fx_bypass": True}
    for i, name in enumerate(PARAM_NAMES):
        patch[name] = float(vec[i])
    return patch


def init_synth_139(plugin_path: str = DEFAULT_PLUGIN_PATH, sample_rate: int = SAMPLE_RATE, verify: bool = True):
    import pedalboard
    synth = pedalboard.load_plugin(os.path.expanduser(plugin_path))
    if verify:
        verify_param_names(synth)
    synth.parameters["active_scene"].raw_value = 0.0

    # Explicitly silence / bypass ALL FX slots to eliminate delay / chorus
    for k in synth.parameters.keys():
        if k.startswith("fx_"):
            if "output_mix" in k:
                synth.parameters[k].raw_value = 0.0
            elif "fx_type" in k:
                synth.parameters[k].raw_value = 0.0

    # Prime synth buffer to initialize voice engine
    synth.process(np.zeros((2, 1024), dtype=np.float32), sample_rate)
    return synth


def apply_patch_139(plugin, patch: dict, tables) -> None:
    P = plugin.parameters
    for name in PARAM_NAMES:
        v = float(patch[name])
        if name in tables:
            n = len(tables[name])
            v = tables[name][int(np.clip(round(v * (n - 1)), 0, n - 1))][1]
        P[name].raw_value = v

    # Ensure all FX output mixes are 0.0 (Strictly dry)
    for k in ["fx_a1_output_mix", "fx_a2_output_mix", "fx_a3_output_mix", "fx_a4_output_mix"]:
        if k in P:
            P[k].raw_value = 0.0


def render_patch_139(plugin, patch: dict, midi_note: int, note_dur: float, tables,
                     duration: float = DURATION_S, sample_rate: int = SAMPLE_RATE):
    """Render 3.0 s stereo; peak-normalised float32 [2, NUM_SAMPLES], or None if silent."""
    import mido

    apply_patch_139(plugin, patch, tables)
    plugin.reset()
    events = [
        mido.Message("note_on", note=int(midi_note), velocity=105, time=0.0),
        mido.Message("note_off", note=int(midi_note), velocity=0, time=float(note_dur)),
    ]
    audio = plugin.process(events, duration=duration, sample_rate=sample_rate, num_channels=2)
    peak = float(np.max(np.abs(audio)))
    if not np.isfinite(peak) or peak < SILENCE_PEAK:
        return None
    return (audio / peak).astype(np.float32)
