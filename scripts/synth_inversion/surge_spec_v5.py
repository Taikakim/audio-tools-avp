"""Single source of truth for the 60-d Surge XT bass & synth parameter space (v5).

Major upgrades for v5 (2026-10-18):
- 16 categoricals (promoted osc_1_octave, osc_2_octave, osc_3_octave from continuous to 7-class discrete
  one-hot categoricals, completely eliminating continuous sub-octave drift and octave layering).
- 44 continuous parameters.
- Encoded dimension: 44 continuous + 82 categorical channels = 126 dimensions.
- Exact Surge XT octave digitization thresholds: [0.0876, 0.2526, 0.4176, 0.5826, 0.7475, 0.9125].
- Calibrated domain loss weights targeting snappy analog envelopes (feg_decay, feg_amount),
  sharp release (aeg_release, feg_release), sub-osc suppression (sub_mix), and stereo lushness (unison, unison_detune).
"""
import os
import re
import numpy as np

SAMPLE_RATE = 44100
DURATION_S = 0.8
AUDIO_LEN = int(SAMPLE_RATE * DURATION_S)  # 35280
SEED_OFFSET = 5000
NOTE_LOW, NOTE_HIGH = 28, 50  # E1..D3, inclusive
NOTE_DUR_RANGE = (0.18, 0.45)  # seconds

LP_FILTERS = [
    ("LP 12 dB", 0.0303),
    ("LP 24 dB", 0.0606),
    ("LP Legacy Ladder", 0.0909),
    ("LP Vintage Ladder", 0.3030),
    ("LP OB-Xd 12 dB", 0.3333),
    ("LP OB-Xd 24 dB", 0.3636),
    ("LP K35", 0.3939),
    ("LP Diode Ladder", 0.4545),
    ("LP Cutoff Warp", 0.4848),
    ("LP Res Warp", 0.8485),
]

WAVESHAPER_TYPES = [
    ("Off", 0.0000),
    ("Soft", 0.0238),
    ("Hard", 0.0476),
    ("Asymmetric", 0.0714),
    ("Sine", 0.0952),
    ("Fuzz", 0.5714),
    ("Medium", 0.9524),
    ("OJD", 0.9762),
]

OSC_TYPES = [
    ("Classic", 0.0000),
    ("Modern", 8.0 / 11.0),
]

OCTAVE_THRESHOLDS = [0.0876, 0.2526, 0.4176, 0.5826, 0.7475, 0.9125]
OCTAVE_VALUES = [-3, -2, -1, 0, 1, 2, 3]

FX_SLOTS = [("fx_a1_fx_type", "Chorus", 0.3103), ("fx_a2_fx_type", "Delay", 0.0345)]
FX_RENDER_ENABLED = False
FX_TYPE_OFF = 0.005
FX_A1_TYPE_CHORUS = 0.3103
FX_A2_TYPE_DELAY = 0.0345

PARAM_NAMES = [
    "midi_note",        # 0  (NOTE_LOW..NOTE_HIGH) -> [0, 1]
    "filter_type",      # 1  categorical (10)
    "shape",            # 2  saw <-> morph <-> pulse
    "width",            # 3
    "sub_mix",          # 4
    "sync",             # 5
    "fm_depth",         # 6
    "unison",           # 7  categorical (6 voices: 1..6)
    "unison_detune",    # 8  raw / 0.35
    "cutoff",           # 9
    "resonance",        # 10 full range [0, 1]
    "keytrack",         # 11 (raw - 0.5) / 0.5
    "feg_amount",       # 12
    "feg_decay",        # 13
    "feg_sustain",      # 14
    "aeg_decay",        # 15
    "aeg_sustain",      # 16
    "aeg_release",      # 17
    "waveshaper_type",  # 18 categorical (8)
    "drive",            # 19 raw = 0.50 + drive * 0.50 (0 dB .. +24 dB)
    "chorus_mix",       # 20
    "delay_mix",        # 21
    "delay_fb",         # 22
    "osc_1_width_2", "osc_2_shape", "osc_2_width_1", "osc_2_width_2",
    "osc_3_shape", "osc_3_width_1", "osc_3_width_2",
    "osc_1_octave",     # 30 categorical (7: -3..+3)
    "osc_2_octave",     # 31 categorical (7: -3..+3)
    "osc_3_octave",     # 32 categorical (7: -3..+3)
    "noise_color",
    "osc_1_volume", "osc_2_volume", "osc_3_volume",
    "ring_1x2_volume", "ring_2x3_volume", "noise_volume",
    "filter_balance", "filter_configuration", "fm_routing",
    "filter_2_type", "filter_2_cutoff", "filter_2_resonance", "filter_2_keytrack", "filter_2_feg_amount",
    "amp_eg_mode", "filter_eg_mode", "amp_eg_decay_shape", "filter_eg_decay_shape",
    "feg_release",
    "filter_feedback",  # 53 continuous [0, 1] -> raw = p (-100% .. +100%)
    "osc_1_type",       # 54 categorical (2): 0=Classic, 1=Modern
    "osc_2_type",       # 55 categorical (2): 0=Classic, 1=Modern
    "osc_3_type",       # 56 categorical (2): 0=Classic, 1=Modern
    "osc_2_pitch",      # 57 continuous [0, 1] -> raw = 0.5 + (p - 0.5) * (2/14) (+-1 st)
    "osc_3_pitch",      # 58 continuous [0, 1] -> raw = 0.5 + (p - 0.5) * (2/14) (+-1 st)
    "drift"             # 59 continuous [0, 1] -> raw = p (0% .. 100%)
]
NUM_PARAMS = len(PARAM_NAMES)
PARAM_INDEX = {n: i for i, n in enumerate(PARAM_NAMES)}

CATEGORICAL = {
    "filter_type": len(LP_FILTERS),          # 10
    "unison": 6,                             # 6 (1 to 6 voices)
    "waveshaper_type": len(WAVESHAPER_TYPES),# 8
    "filter_configuration": 7,               # 7
    "fm_routing": 4,                         # 4
    "filter_2_type": len(LP_FILTERS),        # 10
    "amp_eg_mode": 2,                        # 2
    "filter_eg_mode": 2,                     # 2
    "amp_eg_decay_shape": 3,                 # 3
    "filter_eg_decay_shape": 3,              # 3
    "osc_1_type": 2,                         # 2
    "osc_2_type": 2,                         # 2
    "osc_3_type": 2,                         # 2
    "osc_1_octave": 7,                       # 7 (-3 to +3)
    "osc_2_octave": 7,                       # 7 (-3 to +3)
    "osc_3_octave": 7,                       # 7 (-3 to +3)
}
CAT_INDICES = [(PARAM_INDEX[n], k) for n, k in CATEGORICAL.items()]
CONT_INDICES = [i for i, n in enumerate(PARAM_NAMES) if n not in CATEGORICAL]

DOMAIN_WEIGHTS = {
    "osc_1_octave": 4.0,   # Eliminates sub-octave & octave drift on main oscillator
    "shape": 3.5,          # Sawtooth vs Square vs Morph
    "cutoff": 2.5,
    "sub_mix": 2.5,        # Heavy penalty on unwanted sub-osc bleed
    "osc_2_octave": 2.5,
    "osc_3_octave": 2.5,
    "resonance": 2.2,
    "feg_amount": 2.0,
    "feg_decay": 2.0,      # Snappy envelope transient
    "aeg_release": 2.0,    # Sharp release definition
    "feg_release": 2.0,
    "waveshaper_type": 2.0,# Non-linear saturation / squelch
    "drive": 2.0,
    "unison": 2.0,
    "unison_detune": 2.0,  # Stereo lushness & spread
    "filter_type": 2.0,
    "filter_configuration": 2.5, # Occam parsimony: penalize spurious dual-filter topologies
    "filter_balance": 2.0,       # Occam parsimony: penalize unwarranted filter split / pan
    "filter_feedback": 2.0,      # Occam parsimony: penalize spurious filter feedback
    "osc_1_type": 2.0,
    "osc_2_type": 1.5,
    "osc_3_type": 1.5,
    "aeg_sustain": 1.8,
}

def domain_weights() -> np.ndarray:
    w = np.ones(NUM_PARAMS, dtype=np.float32)
    for name, value in DOMAIN_WEIGHTS.items():
        if name in PARAM_INDEX:
            w[PARAM_INDEX[name]] = value
    return w / w.mean()

def ordinal_to_class(value: float, n_classes: int) -> int:
    return int(np.clip(round(float(value) * (n_classes - 1)), 0, n_classes - 1))

def raw_to_octave_class(raw: float) -> int:
    return int(np.digitize(raw, OCTAVE_THRESHOLDS))

def octave_class_to_raw(cls: int) -> float:
    return float(np.clip(cls, 0, 6)) / 6.0

CONT_BOUNDS = {
    "midi_note": (0.0, 1.0), "shape": (0.0, 1.0), "width": (0.05, 0.95), "sub_mix": (0.0, 0.85),
    "sync": (0.0, 0.40), "fm_depth": (0.0, 0.45), "unison_detune": (0.0, 1.0), "cutoff": (0.08, 0.95),
    "resonance": (0.0, 1.00), "keytrack": (0.0, 1.0), "feg_amount": (0.1, 0.95), "feg_decay": (0.03, 1.0),
    "feg_sustain": (0.0, 0.70), "aeg_decay": (0.05, 0.70), "aeg_sustain": (0.0, 0.85), "aeg_release": (0.04, 0.45),
    "drive": (0.0, 1.0), "chorus_mix": (0.0, 0.60), "delay_mix": (0.0, 0.45), "delay_fb": (0.0, 0.50),
    "osc_1_width_2": (0.05, 0.95), "osc_2_shape": (0.0, 1.0), "osc_2_width_1": (0.05, 0.95), "osc_2_width_2": (0.05, 0.95),
    "osc_3_shape": (0.0, 1.0), "osc_3_width_1": (0.05, 0.95), "osc_3_width_2": (0.05, 0.95),
    "noise_color": (0.0, 1.0),
    "osc_1_volume": (0.0, 1.0), "osc_2_volume": (0.0, 1.0), "osc_3_volume": (0.0, 1.0),
    "ring_1x2_volume": (0.0, 1.0), "ring_2x3_volume": (0.0, 1.0), "noise_volume": (0.0, 0.30),
    "filter_balance": (0.0, 1.0), "filter_2_cutoff": (0.08, 1.0), "filter_2_resonance": (0.0, 1.0),
    "filter_2_keytrack": (0.0, 1.0), "filter_2_feg_amount": (0.0, 1.0),
    "feg_release": (0.04, 0.75),
    "filter_feedback": (0.0, 1.0), "osc_2_pitch": (0.0, 1.0), "osc_3_pitch": (0.0, 1.0), "drift": (0.0, 1.0),
}

def canonicalize_vector(p: np.ndarray) -> np.ndarray:
    p = np.array(p, dtype=np.float32, copy=True)
    q = p.reshape(-1, NUM_PARAMS)
    # 0 class in unison means 1 voice -> detune is inactive
    q[q[:, PARAM_INDEX["unison"]] < (0.5 / 5.0), PARAM_INDEX["unison_detune"]] = 0.0
    # 0 class in waveshaper means Off -> drive is inactive
    q[q[:, PARAM_INDEX["waveshaper_type"]] < (0.5 / (len(WAVESHAPER_TYPES) - 1)), PARAM_INDEX["drive"]] = 0.0
    q[q[:, PARAM_INDEX["delay_mix"]] <= 0.0, PARAM_INDEX["delay_fb"]] = 0.0
    return q.reshape(p.shape)

def check_glitch(audio_mono: np.ndarray) -> bool:
    """Anti-glitch gate: reject total silence or sterile single-frequency self-oscillations."""
    total_energy = float(np.sum(audio_mono ** 2))
    if total_energy < 1e-4:
        return True
    fft = np.abs(np.fft.rfft(audio_mono)) ** 2
    sum_power = float(np.sum(fft))
    if sum_power <= 1e-9:
        return True
    peak_ratio = float(np.max(fft) / sum_power)
    return peak_ratio > 0.88

def patch_to_vector(patch: dict) -> np.ndarray:
    oct1_cls = patch.get("osc_1_octave_idx", raw_to_octave_class(patch.get("osc_1_octave", 0.5)))
    oct2_cls = patch.get("osc_2_octave_idx", raw_to_octave_class(patch.get("osc_2_octave", 0.5)))
    oct3_cls = patch.get("osc_3_octave_idx", raw_to_octave_class(patch.get("osc_3_octave", 0.5)))

    return np.array([
        (patch["midi_note"] - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW),
        patch["filter_idx"] / (len(LP_FILTERS) - 1),
        patch["shape"],
        patch["width"],
        patch["sub_mix"],
        patch["sync"],
        patch["fm_depth"],
        patch.get("unison_voices_idx", 0) / 5.0 if "unison_voices_idx" in patch else (1.0/5.0 if patch.get("unison", False) else 0.0),
        patch["unison_detune"] / 0.35,
        patch["cutoff"],
        patch["resonance"],
        (patch["keytrack_raw"] - 0.5) / 0.5,
        patch["feg_amount"],
        patch["feg_decay"],
        patch["feg_sustain"],
        patch["aeg_decay"],
        patch["aeg_sustain"],
        patch["aeg_release"],
        patch["ws_idx"] / (len(WAVESHAPER_TYPES) - 1),
        (patch["drive_raw"] - 0.50) / 0.50, # 0 dB to +24 dB
        patch["chorus_mix"],
        patch["delay_mix"],
        patch["delay_fb"],
        patch["osc_1_width_2"], patch["osc_2_shape"], patch["osc_2_width_1"], patch["osc_2_width_2"],
        patch["osc_3_shape"], patch["osc_3_width_1"], patch["osc_3_width_2"],
        oct1_cls / 6.0, oct2_cls / 6.0, oct3_cls / 6.0, patch["noise_color"],
        patch["osc_1_volume"], patch["osc_2_volume"], patch["osc_3_volume"],
        patch["ring_1x2_volume"], patch["ring_2x3_volume"], patch["noise_volume"],
        patch["filter_balance"], patch["filter_configuration"] / 6.0, patch["fm_routing"] / 3.0,
        patch["filter_2_type"] / (len(LP_FILTERS) - 1), patch["filter_2_cutoff"], patch["filter_2_resonance"],
        patch["filter_2_keytrack"], patch["filter_2_feg_amount"],
        float(patch.get("amp_eg_mode", 0)),
        float(patch.get("filter_eg_mode", 0)),
        float(patch.get("amp_eg_decay_shape", 1)) / 2.0,
        float(patch.get("filter_eg_decay_shape", 1)) / 2.0,
        float(patch.get("feg_release", patch.get("aeg_release", 0.23))),
        float(patch.get("filter_feedback", 0.5)),
        float(patch.get("osc_1_type", 0)),
        float(patch.get("osc_2_type", 0)),
        float(patch.get("osc_3_type", 0)),
        float(patch.get("osc_2_pitch", 0.5)),
        float(patch.get("osc_3_pitch", 0.5)),
        float(patch.get("drift", 0.0)),
    ], dtype=np.float32)

def vector_to_patch(p) -> dict:
    p = np.asarray(p, dtype=np.float64).copy()
    for i, name in enumerate(PARAM_NAMES):
        if name in CONT_BOUNDS:
            p[i] = np.clip(p[i], CONT_BOUNDS[name][0], CONT_BOUNDS[name][1])
        else:
            p[i] = np.clip(p[i], 0.0, 1.0)

    unison_cls = ordinal_to_class(p[7], 6)
    unison_voices = unison_cls + 1
    unison = unison_voices > 1

    # Surge's own thresholds, as param_codec_v5.target_classes (KUANG 2026-10-19: rounding disagreed with the codec's
    # labels on raw prior values near a boundary; for stored class / 6 values both give the same class)
    oct1_cls = raw_to_octave_class(p[30])
    oct2_cls = raw_to_octave_class(p[31])
    oct3_cls = raw_to_octave_class(p[32])

    return dict(
        midi_note=int(round(NOTE_LOW + p[0] * (NOTE_HIGH - NOTE_LOW))),
        filter_idx=ordinal_to_class(p[1], len(LP_FILTERS)),
        shape=float(p[2]),
        width=float(p[3]),
        sub_mix=float(p[4]),
        sync=float(p[5]),
        fm_depth=float(p[6]),
        unison=unison,
        unison_voices=unison_voices,
        unison_voices_idx=unison_cls,
        unison_detune=float(p[8] * 0.35) if unison else 0.0,
        cutoff=float(p[9]),
        resonance=float(p[10]),
        keytrack_raw=float(0.5 + p[11] * 0.5),
        feg_amount=float(p[12]),
        feg_decay=float(p[13]),
        feg_sustain=float(p[14]),
        aeg_decay=float(p[15]),
        aeg_sustain=float(p[16]),
        aeg_release=float(p[17]),
        ws_idx=ordinal_to_class(p[18], len(WAVESHAPER_TYPES)),
        drive_raw=float(0.50 + p[19] * 0.50), # 0 dB .. +24 dB
        chorus_mix=float(p[20]),
        delay_mix=float(p[21]),
        delay_fb=float(p[22]),
        osc_1_width_2=float(p[23]),
        osc_2_shape=float(p[24]),
        osc_2_width_1=float(p[25]),
        osc_2_width_2=float(p[26]),
        osc_3_shape=float(p[27]),
        osc_3_width_1=float(p[28]),
        osc_3_width_2=float(p[29]),
        osc_1_octave=octave_class_to_raw(oct1_cls),
        osc_1_octave_idx=oct1_cls,
        osc_1_octave_offset=oct1_cls - 3,
        osc_2_octave=octave_class_to_raw(oct2_cls),
        osc_2_octave_idx=oct2_cls,
        osc_2_octave_offset=oct2_cls - 3,
        osc_3_octave=octave_class_to_raw(oct3_cls),
        osc_3_octave_idx=oct3_cls,
        osc_3_octave_offset=oct3_cls - 3,
        noise_color=float(p[33]),
        osc_1_volume=float(p[34]),
        osc_2_volume=float(p[35]),
        osc_3_volume=float(p[36]),
        ring_1x2_volume=float(p[37]),
        ring_2x3_volume=float(p[38]),
        noise_volume=float(p[39]),
        filter_balance=float(p[40]),
        filter_configuration=ordinal_to_class(p[41], 7),
        fm_routing=ordinal_to_class(p[42], 4),
        filter_2_type=ordinal_to_class(p[43], len(LP_FILTERS)),
        filter_2_cutoff=float(p[44]),
        filter_2_resonance=float(p[45]),
        filter_2_keytrack=float(p[46]),
        filter_2_feg_amount=float(p[47]),
        amp_eg_mode=ordinal_to_class(p[48], 2),
        filter_eg_mode=ordinal_to_class(p[49], 2),
        amp_eg_decay_shape=ordinal_to_class(p[50], 3),
        filter_eg_decay_shape=ordinal_to_class(p[51], 3),
        feg_release=float(p[52]),
        filter_feedback=float(p[53]),
        osc_1_type=ordinal_to_class(p[54], 2),
        osc_2_type=ordinal_to_class(p[55], 2),
        osc_3_type=ordinal_to_class(p[56], 2),
        osc_2_pitch=float(p[57]),
        osc_3_pitch=float(p[58]),
        drift=float(p[59]),
    )

DEFAULT_PLUGIN_PATH = "~/.vst3/Surge XT.vst3"

def flush_synth(plugin, sample_rate: int = SAMPLE_RATE) -> None:
    import mido
    panic_events = [
        mido.Message("control_change", control=120, value=0, time=0.0),
        mido.Message("control_change", control=123, value=0, time=0.0),
    ]
    for n in range(128):
        panic_events.append(mido.Message("note_off", note=n, velocity=0, time=0.0))
    plugin.process(panic_events, duration=0.05, sample_rate=sample_rate, num_channels=2)
    plugin.reset()

def init_synth(plugin_path: str = DEFAULT_PLUGIN_PATH, sample_rate: int = SAMPLE_RATE, verify: bool = False):
    import pedalboard
    plugin = pedalboard.load_plugin(os.path.expanduser(plugin_path))
    plugin.parameters["active_scene"].raw_value = 0.0
    plugin.parameters["a_osc_1_type"].raw_value = 0.0
    plugin.parameters["a_osc_1_octave"].raw_value = 0.5
    for param, _name, raw in FX_SLOTS:
        plugin.parameters[param].raw_value = raw
    plugin.process([], 0.05, sample_rate, 2)
    flush_synth(plugin, sample_rate)
    return plugin

def _set_osc_params(P, prefix, p0, p1, p2, p3):
    p0_obj = P.get(f"{prefix}_shape", P.get(f"{prefix}_sawtooth"))
    if p0_obj is not None:
        p0_obj.raw_value = p0
    p1_obj = P.get(f"{prefix}_width_1", P.get(f"{prefix}_pulse"))
    if p1_obj is not None:
        p1_obj.raw_value = p1
    p2_obj = P.get(f"{prefix}_width_2", P.get(f"{prefix}_triangle"))
    if p2_obj is not None:
        p2_obj.raw_value = p2
    p3_obj = P.get(f"{prefix}_sub_mix", P.get(f"{prefix}_width"))
    if p3_obj is not None:
        p3_obj.raw_value = p3

def apply_patch(plugin, patch: dict) -> None:
    P = plugin.parameters
    
    # 1. Set engine types first
    P["a_osc_1_type"].raw_value = OSC_TYPES[patch.get("osc_1_type", 0)][1]
    P["a_osc_2_type"].raw_value = OSC_TYPES[patch.get("osc_2_type", 0)][1]
    P["a_osc_3_type"].raw_value = OSC_TYPES[patch.get("osc_3_type", 0)][1]
    P = plugin.parameters

    # 2. Oscillators 1, 2, 3 parameters (robust against Classic vs Modern alias)
    _set_osc_params(P, "a_osc_1", patch["shape"], patch["width"], patch["osc_1_width_2"], patch["sub_mix"])
    _set_osc_params(P, "a_osc_2", patch["osc_2_shape"], patch["osc_2_width_1"], patch["osc_2_width_2"], 0.0)
    _set_osc_params(P, "a_osc_3", patch["osc_3_shape"], patch["osc_3_width_1"], patch["osc_3_width_2"], 0.0)

    P["a_osc_1_sync"].raw_value = patch["sync"]
    P["a_fm_depth"].raw_value = patch["fm_depth"]
    
    # Unison: 1 to 6 voices
    u_cls = patch.get("unison_voices_idx", 0 if not patch.get("unison", False) else 1)
    P["a_osc_1_unison_voices"].raw_value = float(u_cls) / 15.0
    P["a_osc_1_unison_detune"].raw_value = patch["unison_detune"] if u_cls > 0 else 0.0

    # Filters & Routing
    P["a_filter_1_type"].raw_value = LP_FILTERS[patch["filter_idx"]][1]
    P["a_filter_1_cutoff"].raw_value = patch["cutoff"]
    P["a_filter_1_resonance"].raw_value = patch["resonance"]
    P["a_filter_1_keytrack"].raw_value = patch["keytrack_raw"]
    P["a_filter_1_feg_mod_amount"].raw_value = patch["feg_amount"]
    P["a_filter_eg_decay"].raw_value = patch["feg_decay"]
    P["a_filter_eg_sustain"].raw_value = patch["feg_sustain"]
    P["a_amp_eg_decay"].raw_value = patch["aeg_decay"]
    P["a_amp_eg_sustain"].raw_value = patch["aeg_sustain"]
    P["a_amp_eg_release"].raw_value = patch["aeg_release"]
    P["a_waveshaper_type"].raw_value = WAVESHAPER_TYPES[patch["ws_idx"]][1]
    P["a_waveshaper_drive"].raw_value = patch["drive_raw"]

    P["a_osc_1_octave"].raw_value = patch["osc_1_octave"]
    P["a_osc_2_octave"].raw_value = patch["osc_2_octave"]
    P["a_osc_3_octave"].raw_value = patch["osc_3_octave"]
    P["a_noise_color"].raw_value = patch["noise_color"]
    P["a_osc_1_mute"].raw_value = 0.0
    P["a_osc_2_mute"].raw_value = 0.0
    P["a_osc_3_mute"].raw_value = 0.0
    P["a_osc_1_volume"].raw_value = patch["osc_1_volume"]
    P["a_osc_2_volume"].raw_value = patch["osc_2_volume"]
    P["a_osc_3_volume"].raw_value = patch["osc_3_volume"]
    P["a_ring_modulation_1x2_volume"].raw_value = patch["ring_1x2_volume"]
    P["a_ring_modulation_2x3_volume"].raw_value = patch["ring_2x3_volume"]
    P["a_noise_volume"].raw_value = patch["noise_volume"]
    P["a_filter_balance"].raw_value = patch["filter_balance"]
    P["a_filter_configuration"].raw_value = patch["filter_configuration"] / 6.0
    P["a_fm_routing"].raw_value = patch["fm_routing"] / 3.0
    P["a_filter_2_type"].raw_value = LP_FILTERS[patch["filter_2_type"]][1]
    P["a_filter_2_cutoff"].raw_value = patch["filter_2_cutoff"]
    P["a_filter_2_resonance"].raw_value = patch["filter_2_resonance"]
    P["a_filter_2_keytrack"].raw_value = patch["filter_2_keytrack"]
    P["a_filter_2_feg_mod_amount"].raw_value = patch["filter_2_feg_amount"]

    # Envelope modes
    P["a_amp_eg_envelope_mode"].raw_value = 0.5 if patch.get("amp_eg_mode", 0) == 1 else 0.0
    P["a_filter_eg_envelope_mode"].raw_value = 0.5 if patch.get("filter_eg_mode", 0) == 1 else 0.0
    decay_shapes = {0: 0.0, 1: 0.26, 2: 0.75}
    P["a_amp_eg_decay_shape"].raw_value = decay_shapes.get(patch.get("amp_eg_decay_shape", 1), 0.26)
    P["a_filter_eg_decay_shape"].raw_value = decay_shapes.get(patch.get("filter_eg_decay_shape", 1), 0.26)
    P["a_filter_eg_release"].raw_value = patch.get("feg_release", patch.get("aeg_release", 0.23))

    # Feedback, microtuning, drift
    P["a_feedback"].raw_value = float(patch.get("filter_feedback", 0.5))
    p2 = float(patch.get("osc_2_pitch", 0.5))
    p3 = float(patch.get("osc_3_pitch", 0.5))
    P["a_osc_2_pitch"].raw_value = float(np.clip(0.5 + (p2 - 0.5) * (2.0 / 14.0), 0.0, 1.0))
    P["a_osc_3_pitch"].raw_value = float(np.clip(0.5 + (p3 - 0.5) * (2.0 / 14.0), 0.0, 1.0))
    P["a_osc_drift"].raw_value = float(np.clip(patch.get("drift", 0.0), 0.0, 1.0))

    # FX
    fx_on = FX_RENDER_ENABLED
    P["fx_a1_fx_type"].raw_value = FX_A1_TYPE_CHORUS if fx_on else FX_TYPE_OFF
    P["fx_a2_fx_type"].raw_value = FX_A2_TYPE_DELAY if fx_on else FX_TYPE_OFF
    if fx_on:
        P = plugin.parameters
        P["fx_a1_output_mix"].raw_value = patch["chorus_mix"]
        P["fx_a2_output_mix"].raw_value = patch["delay_mix"]
        P["fx_a2_feedback_eq_feedback"].raw_value = patch["delay_fb"]

def phrase_events(midi_note: int):
    n1 = int(midi_note)
    n2 = min(n1 + 12, 127)
    return [
        ("note_on", n1, 105, 0.000),
        ("note_off", n1, 0, 0.125),
        ("note_on", n2, 105, 0.250),
        ("note_off", n2, 0, 0.500),
    ]

def render_patch(plugin, patch: dict, midi_note: int, note_dur: float = 0.3,
                 duration: float = DURATION_S, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    import mido
    apply_patch(plugin, patch)
    plugin.reset()
    events = [mido.Message(type, note=n, velocity=v, time=t) 
              for type, n, v, t in phrase_events(midi_note)]
    audio = plugin.process(events, duration=duration, sample_rate=sample_rate, num_channels=2)
    mono = np.mean(audio, axis=0).astype(np.float32)
    return mono / (np.max(np.abs(mono)) + 1e-7)
