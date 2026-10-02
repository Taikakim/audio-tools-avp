"""Single source of truth for the 23-d Surge XT bass parameter space.

Everything that touches Surge (dataset generation, real-stem evaluation, DE matching,
held-out audio scoring) goes through this module, so the parameter layout, the raw-value
mapping and the render call cannot drift apart between scripts again.

numpy-only on purpose: the dataset generator runs in a venv without torch.
"""
import os
import re

import numpy as np

SAMPLE_RATE = 44100
DURATION_S = 0.8
AUDIO_LEN = int(SAMPLE_RATE * DURATION_S)  # 35280
SEED_OFFSET = 5000  # dataset sample i was drawn from np.random.seed(i + SEED_OFFSET)
NOTE_LOW, NOTE_HIGH = 28, 50  # E1..D3, inclusive
NOTE_DUR_RANGE = (0.18, 0.45)  # seconds; training-distribution note lengths

# Surge XT 1.3.4 raw (normalised 0..1) values for enum parameters. These are hard-coded
# positions in Surge's enum lists; verify_surge_mapping() checks them against the
# plugin's own display strings at startup so a Surge update cannot silently remap them.
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
    ("Off", 0.0),
    ("Soft", 0.025),
    ("Hard", 0.050),
    ("Asymmetric", 0.075),
    ("Sine", 0.100),
    ("Fuzz", 0.575),
]
FX_SLOTS = [("fx_a1_fx_type", "Chorus", 0.3103), ("fx_a2_fx_type", "Delay", 0.0345)]
UNISON_VOICES_RAW = {1: 0.0, 2: 0.05}

PARAM_NAMES = [
    "midi_note",        # 0  (NOTE_LOW..NOTE_HIGH) -> [0, 1]
    "filter_type",      # 1  categorical, stored as class / (n - 1)
    "shape",            # 2  saw <-> morph <-> pulse
    "width",            # 3
    "sub_mix",          # 4
    "sync",             # 5
    "fm_depth",         # 6
    "unison",           # 7  categorical (1 or 2 voices), stored 0 / 1
    "unison_detune",    # 8  raw / 0.35
    "cutoff",           # 9
    "resonance",        # 10
    "keytrack",         # 11 (raw - 0.5) / 0.5
    "feg_amount",       # 12
    "feg_decay",        # 13
    "feg_sustain",      # 14
    "aeg_decay",        # 15
    "aeg_sustain",      # 16
    "aeg_release",      # 17
    "waveshaper_type",  # 18 categorical, stored as class / (n - 1)
    "drive",            # 19 (raw - 0.5) / 0.32
    "chorus_mix",       # 20
    "delay_mix",        # 21
    "delay_fb",         # 22
]
NUM_PARAMS = len(PARAM_NAMES)
PARAM_INDEX = {n: i for i, n in enumerate(PARAM_NAMES)}

# Categorical parameters have no meaningful order, so they must be learned as classes,
# not regressed as numbers. The h5 keeps the legacy ordinal storage for compatibility;
# param_codec.py converts at train / inference time.
CATEGORICAL = {"filter_type": len(LP_FILTERS), "unison": 2, "waveshaper_type": len(WAVESHAPER_TYPES)}
CAT_INDICES = [(PARAM_INDEX[n], k) for n, k in CATEGORICAL.items()]
CONT_INDICES = [i for i, n in enumerate(PARAM_NAMES) if n not in CATEGORICAL]

# Domain loss weights (shape, filter and envelope character dominate the sound).
DOMAIN_WEIGHTS = {
    "filter_type": 2.0,
    "shape": 3.5,
    "cutoff": 2.5,
    "resonance": 2.0,
    "feg_amount": 1.8,
    "aeg_decay": 1.5,
    "aeg_release": 1.5,
}


def domain_weights() -> np.ndarray:
    w = np.ones(NUM_PARAMS, dtype=np.float32)
    for name, value in DOMAIN_WEIGHTS.items():
        w[PARAM_INDEX[name]] = value
    return w / w.mean()


def ordinal_to_class(value: float, n_classes: int) -> int:
    return int(np.clip(round(float(value) * (n_classes - 1)), 0, n_classes - 1))


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------
def draw_patch(seed: int, note_dur_range=NOTE_DUR_RANGE) -> dict:
    """Draw one random patch from the training distribution.

    The ORDER of the np.random calls below is frozen: surge_bass_200k.h5 was generated
    with exactly this sequence from np.random.seed(index + SEED_OFFSET), and datasets
    written before note_dur was stored recover it by replaying this function.
    Do not reorder, add or remove draws; append new ones at the very end only.
    """
    np.random.seed(seed)

    filter_choice = int(np.random.randint(0, len(LP_FILTERS)))
    midi_note = int(np.random.randint(NOTE_LOW, NOTE_HIGH + 1))
    shape = float(np.random.uniform(0.0, 1.0))
    width = float(np.random.uniform(0.0, 1.0))
    sub_mix = float(np.random.uniform(0.0, 0.85))
    sync = float(np.random.uniform(0.0, 0.40)) if np.random.rand() < 0.3 else 0.0
    fm_depth = float(np.random.uniform(0.0, 0.45)) if np.random.rand() < 0.4 else 0.0

    use_unison = np.random.rand() < 0.25
    unison_detune = float(np.random.uniform(0.05, 0.35)) if use_unison else 0.0

    cutoff = float(np.random.uniform(0.08, 0.92))
    resonance = float(np.random.uniform(0.0, 0.85))
    keytrack_raw = float(np.random.uniform(0.5, 1.0))  # 0 % .. 100 %
    feg_amount = float(np.random.uniform(0.2, 0.95))

    feg_decay = float(np.random.uniform(0.03, 0.65))
    feg_sustain = float(np.random.uniform(0.0, 0.60))
    aeg_decay = float(np.random.uniform(0.05, 0.65))
    aeg_sustain = float(np.random.uniform(0.0, 0.80))
    aeg_release = float(np.random.uniform(0.01, 0.40))

    if np.random.rand() < 0.45:
        ws_idx = int(np.random.randint(1, len(WAVESHAPER_TYPES)))
        drive_raw = float(np.random.uniform(0.50, 0.82))  # 0 dB .. +15 dB
    else:
        ws_idx = 0
        drive_raw = 0.50

    chorus_mix = float(np.random.uniform(0.1, 0.6)) if np.random.rand() < 0.35 else 0.0
    delay_mix = float(np.random.uniform(0.1, 0.45)) if np.random.rand() < 0.25 else 0.0
    delay_fb = float(np.random.uniform(0.1, 0.5)) if delay_mix > 0.0 else 0.0

    # np.random.uniform(a, b) consumes one draw regardless of (a, b), so a different
    # note_dur_range changes the value but not the RNG stream.
    note_dur = float(np.random.uniform(*note_dur_range))

    return dict(
        midi_note=midi_note,
        filter_idx=filter_choice,
        shape=shape,
        width=width,
        sub_mix=sub_mix,
        sync=sync,
        fm_depth=fm_depth,
        unison=bool(use_unison),
        unison_detune=unison_detune,
        cutoff=cutoff,
        resonance=resonance,
        keytrack_raw=keytrack_raw,
        feg_amount=feg_amount,
        feg_decay=feg_decay,
        feg_sustain=feg_sustain,
        aeg_decay=aeg_decay,
        aeg_sustain=aeg_sustain,
        aeg_release=aeg_release,
        ws_idx=ws_idx,
        drive_raw=drive_raw,
        chorus_mix=chorus_mix,
        delay_mix=delay_mix,
        delay_fb=delay_fb,
        note_dur=note_dur,
    )


def patch_to_vector(patch: dict) -> np.ndarray:
    """Patch dict -> 23-d normalised vector (the h5 'params' layout)."""
    return np.array([
        (patch["midi_note"] - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW),
        patch["filter_idx"] / (len(LP_FILTERS) - 1),
        patch["shape"],
        patch["width"],
        patch["sub_mix"],
        patch["sync"],
        patch["fm_depth"],
        1.0 if patch["unison"] else 0.0,
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
        (patch["drive_raw"] - 0.50) / 0.32,
        patch["chorus_mix"],
        patch["delay_mix"],
        patch["delay_fb"],
    ], dtype=np.float32)


def vector_to_patch(p) -> dict:
    """23-d normalised vector -> patch dict (inverse of patch_to_vector, with clipping)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    unison = ordinal_to_class(p[7], 2) == 1
    return dict(
        midi_note=int(round(NOTE_LOW + p[0] * (NOTE_HIGH - NOTE_LOW))),
        filter_idx=ordinal_to_class(p[1], len(LP_FILTERS)),
        shape=float(p[2]),
        width=float(p[3]),
        sub_mix=float(p[4]),
        sync=float(p[5]),
        fm_depth=float(p[6]),
        unison=unison,
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
        drive_raw=float(0.50 + p[19] * 0.32),
        chorus_mix=float(p[20]),
        delay_mix=float(p[21]),
        delay_fb=float(p[22]),
    )


def describe_patch(patch: dict) -> dict:
    """Human-readable, JSON-serialisable view of a patch (names instead of indices)."""
    out = {k: v for k, v in patch.items() if k not in ("filter_idx", "ws_idx")}
    out["filter_circuit"] = LP_FILTERS[patch["filter_idx"]][0]
    out["waveshaper"] = WAVESHAPER_TYPES[patch["ws_idx"]][0]
    return out


# ---------------------------------------------------------------------------
# Surge XT plugin
# ---------------------------------------------------------------------------
DEFAULT_PLUGIN_PATH = "~/.vst3/Surge XT.vst3"


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def verify_surge_mapping(plugin) -> None:
    """Check every hard-coded enum raw value against the plugin's display string.

    Raises RuntimeError listing every mismatch. Set SURGE_SKIP_MAPPING_CHECK=1 to bypass
    (only if you have confirmed the display strings differ cosmetically).
    """
    if os.environ.get("SURGE_SKIP_MAPPING_CHECK") == "1":
        return
    checks = []
    for name, raw in LP_FILTERS:
        checks.append(("a_filter_1_type", raw, name))
    for name, raw in WAVESHAPER_TYPES:
        checks.append(("a_waveshaper_type", raw, name))
    for param, name, raw in FX_SLOTS:
        checks.append((param, raw, name))
    checks.append(("a_osc_1_unison_voices", UNISON_VOICES_RAW[2], "2"))

    mismatches = []
    for param, raw, expected in checks:
        plugin.parameters[param].raw_value = raw
        actual = plugin.parameters[param].string_value
        ok = _norm(actual).startswith(_norm(expected)) if expected == "2" else _norm(expected) in _norm(actual)
        if not ok:
            mismatches.append(f"  {param} raw={raw}: expected '{expected}', plugin shows '{actual}'")
    if mismatches:
        raise RuntimeError(
            "Surge XT enum mapping does not match this plugin build:\n" + "\n".join(mismatches)
            + "\nFix the raw values in surge_spec.py (or set SURGE_SKIP_MAPPING_CHECK=1 if the"
            " strings differ only cosmetically)."
        )


def init_synth(plugin_path: str = DEFAULT_PLUGIN_PATH, sample_rate: int = SAMPLE_RATE, verify: bool = True):
    import pedalboard

    plugin = pedalboard.load_plugin(os.path.expanduser(plugin_path))
    if verify:
        verify_surge_mapping(plugin)
    plugin.parameters["active_scene"].raw_value = 0.0
    plugin.parameters["a_osc_1_type"].raw_value = 0.0
    plugin.parameters["a_osc_1_octave"].raw_value = 0.5
    for param, _name, raw in FX_SLOTS:
        plugin.parameters[param].raw_value = raw
    plugin.process([], 0.05, sample_rate, 2)
    return plugin


def apply_patch(plugin, patch: dict) -> None:
    """Set EVERY varied parameter, so no value can leak in from a previous render."""
    P = plugin.parameters
    P["a_filter_1_type"].raw_value = LP_FILTERS[patch["filter_idx"]][1]
    P["a_osc_1_shape"].raw_value = patch["shape"]
    P["a_osc_1_width_1"].raw_value = patch["width"]
    P["a_osc_1_sub_mix"].raw_value = patch["sub_mix"]
    P["a_osc_1_sync"].raw_value = patch["sync"]
    P["a_fm_depth"].raw_value = patch["fm_depth"]
    P["a_osc_1_unison_voices"].raw_value = UNISON_VOICES_RAW[2 if patch["unison"] else 1]
    P["a_osc_1_unison_detune"].raw_value = patch["unison_detune"] if patch["unison"] else 0.0
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
    P["fx_a1_output_mix"].raw_value = patch["chorus_mix"]
    P["fx_a2_output_mix"].raw_value = patch["delay_mix"]
    P["fx_a2_feedback_eq_feedback"].raw_value = patch["delay_fb"]


def render_patch(plugin, patch: dict, midi_note: int, note_dur: float,
                 duration: float = DURATION_S, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Render one note; returns peak-normalised mono float32."""
    import mido

    apply_patch(plugin, patch)
    plugin.reset()
    events = [
        mido.Message("note_on", note=int(midi_note), velocity=105, time=0.0),
        mido.Message("note_off", note=int(midi_note), velocity=0, time=float(note_dur)),
    ]
    audio = plugin.process(events, duration=duration, sample_rate=sample_rate, num_channels=2)
    mono = np.mean(audio, axis=0).astype(np.float32)
    return mono / (np.max(np.abs(mono)) + 1e-7)
