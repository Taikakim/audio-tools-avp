"""Surge XT 139-Parameter Specification for Synth-JEPA.

References:
  - Hayes, Tian, Lattner, "Synth-JEPA: Joint Embedding Prediction for Renderer-Free
    Synthesizer Parameter Search", arXiv:2609.31024, Sec. 3.1 & 3.2.
  - Barkan et al., "InverSynth II", ISMIR 2023 [11].

Characteristics:
  * 139 active parameters covering 3 Oscillators, Mixer, Dual Filters, Envelopes,
    and 6 LFOs.
  * Audio effects are strictly disabled (all FX slots set to bypass/0.0).
  * Non-deterministic modulators (Random / S&H) are excluded.
  * Standard audio rendering: 3.0 seconds, stereo, 44.1 kHz.
"""

import os
from typing import Dict, List, Tuple

import numpy as np

SAMPLE_RATE = 44100
DURATION_S = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_S)  # 132,300 samples
NUM_PARAMS = 139
DEFAULT_PLUGIN_PATH = "~/.vst3/Surge XT.vst3"

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

# Categorical parameters: (index, n_classes)
# Surge XT parameter enum sizes
CATEGORICAL_MAP = {
    "a_osc_1_type": 7, "a_osc_2_type": 7, "a_osc_3_type": 7,
    "a_osc_1_unison_voices": 8, "a_osc_2_unison_voices": 8, "a_osc_3_unison_voices": 8,
    "a_osc_1_route": 3, "a_osc_2_route": 3, "a_osc_3_route": 3,
    "a_filter_1_type": 12, "a_filter_2_type": 12,
    "a_filter_configuration": 4,
    "a_waveshaper_type": 6,
    "a_fm_routing": 3,
}
for lfo in range(1, 7):
    CATEGORICAL_MAP[f"a_lfo_{lfo}_type"] = 5  # Sine, Tri, Square, Ramp, Noise (deterministic)

CAT_INDICES: List[Tuple[int, int]] = [
    (PARAM_INDEX[name], k) for name, k in CATEGORICAL_MAP.items() if name in PARAM_INDEX
]
CAT_INDEX_SET = {i for i, _ in CAT_INDICES}
CONT_INDICES: List[int] = [i for i in range(NUM_PARAMS) if i not in CAT_INDEX_SET]


def draw_patch_139(seed: int) -> dict:
    """Draw a random 139-parameter patch from the synthesizer prior."""
    rng = np.random.default_rng(seed)

    patch = {}
    for i, name in enumerate(PARAM_NAMES):
        if i in CAT_INDEX_SET:
            n_classes = next(k for idx, k in CAT_INDICES if idx == i)
            patch[name] = float(rng.integers(0, n_classes) / max(1, n_classes - 1))
        else:
            # Continuous raw parameter in [0.0, 1.0]
            patch[name] = float(rng.uniform(0.0, 1.0))

    # Musical pitch & duration
    patch["midi_note"] = int(rng.integers(24, 96))
    patch["note_dur"] = float(rng.uniform(0.2, 2.5))

    # Audio effects explicitly bypassed
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


def init_synth_139(plugin_path: str = DEFAULT_PLUGIN_PATH, sample_rate: int = SAMPLE_RATE):
    import pedalboard
    synth = pedalboard.load_plugin(os.path.expanduser(plugin_path))
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


def apply_patch_139(plugin, patch: dict) -> None:
    P = plugin.parameters
    for name in PARAM_NAMES:
        if name in patch and name in P:
            P[name].raw_value = float(patch[name])

    # Ensure all FX output mixes are 0.0 (Strictly dry)
    for k in ["fx_a1_output_mix", "fx_a2_output_mix", "fx_a3_output_mix", "fx_a4_output_mix"]:
        if k in P:
            P[k].raw_value = 0.0


def render_patch_139(plugin, patch: dict, midi_note: int, note_dur: float,
                     duration: float = DURATION_S, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render 3.0s stereo audio; returns peak-normalised float32 array [2, NUM_SAMPLES]."""
    import mido

    apply_patch_139(plugin, patch)
    plugin.reset()
    events = [
        mido.Message("note_on", note=int(midi_note), velocity=105, time=0.0),
        mido.Message("note_off", note=int(midi_note), velocity=0, time=float(note_dur)),
    ]
    audio = plugin.process(events, duration=duration, sample_rate=sample_rate, num_channels=2)
    # audio shape: [2, NUM_SAMPLES]
    peak = np.max(np.abs(audio)) + 1e-7
    return (audio / peak).astype(np.float32)
