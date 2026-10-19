"""Project real Surge XT presets (.fxp) into the 60-d parameter space of surge_spec_v4.py.

Supports:
  --mode bass: 30% manifold (Templates + Factory Basses + NewLoops + Presetshare)
  --mode all:  10% manifold (All Factory + 3rdparty presets across all categories)
"""
import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from surge_spec_v4 import (
    DEFAULT_PLUGIN_PATH, LP_FILTERS, NUM_PARAMS, PARAM_INDEX, PARAM_NAMES,
    WAVESHAPER_TYPES, OSC_TYPES
)

FORMAT_VERSION = 4
SURGE_DATA_DIR = "/home/kim/Downloads/surge-xt-portable-content-1.3.4/Surge Synth Team/SurgeXTData"
TEMPLATES_DIR = "/home/kim/Documents/Surge XT/Patches/Templates"
NEW_LOOPS_ZIP = "/home/kim/Documents/Surge XT/Patches/AI Inversions/kim_bass_.zip"
PRESETSHARE_DIR = "/home/kim/Documents/Surge XT/Patches/Presetshare"

FILTER_POS_TO_CLASS = {
    1: 0,   # LP 12 dB
    2: 1,   # LP 24 dB
    3: 2,   # LP Legacy Ladder
    10: 3,  # LP Vintage Ladder
    11: 4,  # LP OB-Xd 12 dB
    12: 5,  # LP OB-Xd 24 dB
    13: 6,  # LP K35
    15: 7,  # LP Diode Ladder
    16: 8,  # LP Cutoff Warp
    28: 9,  # LP Res Warp
    23: 1,  # BP 24 dB mapped to LP 24 dB
    0: 0,   # Off / bypassed -> LP 12 dB
}

WS_POS_TO_CLASS = {
    0: 0,   # Off
    1: 1,   # Soft
    2: 2,   # Hard
    3: 3,   # Asymmetric
    4: 4,   # Sine
    24: 5,  # Fuzz
    25: 5,  # Fuzz Soft Clip
    26: 5,  # Heavy Fuzz
    27: 5,  # Fuzz Center
    28: 5,  # Fuzz Soft Edge
    40: 6,  # Medium
    41: 7,  # OJD
    6: 1,   # Soft Harmonic 2
    7: 1,   # Soft Harmonic 3
    8: 1,   # Soft Harmonic 4
    9: 1,   # Soft Harmonic 5
    13: 1,  # Soft Rectifier
    42: 1,  # Soft Single Fold
}

CONTINUOUS = {
    "shape":         ("a_osc1_param0",       -1.0,  1.0),
    "width":         ("a_osc1_param1",        0.0,  1.0),
    "sub_mix":       ("a_osc1_param3",        0.0,  1.0),
    "sync":          ("a_osc1_param4",        0.0, 60.0),
    "unison_detune": ("a_osc1_param5",        0.0,  1.0),
    "fm_depth":      ("a_fm_depth",         -48.0, 16.0),
    "cutoff":        ("a_filter1_cutoff",   -60.0, 70.0),
    "resonance":     ("a_filter1_resonance",  0.0,  1.0),
    "keytrack":      ("a_filter1_keytrack",  -1.0,  1.0),
    "feg_amount":    ("a_filter1_envmod",   -96.0, 96.0),
    "feg_decay":     ("a_env2_decay",        -8.0,  5.0),
    "feg_sustain":   ("a_env2_sustain",       0.0,  1.0),
    "aeg_decay":     ("a_env1_decay",        -8.0,  5.0),
    "aeg_sustain":   ("a_env1_sustain",       0.0,  1.0),
    "aeg_release":   ("a_env1_release",      -8.0,  5.0),
    "feg_release":   ("a_env2_release",      -8.0,  5.0),
    "drive":         ("a_ws_drive",           0.0, 24.0), # mapped to 0..24 dB
    "osc_1_width_2": ("a_osc1_param2",        0.0,  1.0),
    "osc_2_shape":   ("a_osc2_param0",       -1.0,  1.0),
    "osc_2_width_1": ("a_osc2_param1",        0.0,  1.0),
    "osc_2_width_2": ("a_osc2_param2",        0.0,  1.0),
    "osc_3_shape":   ("a_osc3_param0",       -1.0,  1.0),
    "osc_3_width_1": ("a_osc3_param1",        0.0,  1.0),
    "osc_3_width_2": ("a_osc3_param2",        0.0,  1.0),
    "osc_1_octave":  ("a_osc1_octave",       -4.0,  4.0),
    "osc_2_octave":  ("a_osc2_octave",       -4.0,  4.0),
    "osc_3_octave":  ("a_osc3_octave",       -4.0,  4.0),
    "noise_color":   ("a_noise_color",       -1.0,  1.0),
    "osc_1_volume":  ("a_level_o1",           0.0,  1.0),
    "osc_2_volume":  ("a_level_o2",           0.0,  1.0),
    "osc_3_volume":  ("a_level_o3",           0.0,  1.0),
    "ring_1x2_volume":("a_level_ring12",      0.0,  1.0),
    "ring_2x3_volume":("a_level_ring23",      0.0,  1.0),
    "noise_volume":  ("a_level_noise",        0.0,  1.0),
    "filter_balance":("a_f_balance",         -1.0,  1.0),
    "filter_2_cutoff":("a_filter2_cutoff",  -60.0, 70.0),
    "filter_2_resonance":("a_filter2_resonance", 0.0, 1.0),
    "filter_2_keytrack":("a_filter2_keytrack", -1.0, 1.0),
    "filter_2_feg_amount":("a_filter2_envmod", -96.0, 96.0),
    "filter_feedback":("a_feedback",         -1.0,  1.0),
    "osc_2_pitch":   ("a_osc2_pitch",        -1.0,  1.0), # +-1 st mapped to 0..1
    "osc_3_pitch":   ("a_osc3_pitch",        -1.0,  1.0), # +-1 st mapped to 0..1
    "drift":         ("a_osc_drift",          0.0,  1.0),
}

def parse_fxp(data: bytes):
    start = data.find(b"<?xml")
    end = data.rfind(b"</patch>")
    if start == -1 or end == -1:
        return None
    try:
        root = ET.fromstring(data[start:end + len(b"</patch>")].decode("utf-8", errors="replace"))
    except ET.ParseError:
        return None
    params = root.find("parameters")
    if params is None:
        return None
    out = {}
    for elem in params:
        if "value" in elem.attrib:
            try:
                out[elem.tag] = float(elem.attrib["value"])
            except ValueError:
                pass
    return out

def is_val_row(vec: np.ndarray, val_frac: float = 0.1) -> bool:
    h = hashlib.sha256(np.round(vec, 4).tobytes()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < val_frac

def preset_to_vector(xml_p: dict):
    # Oscillator 1 type: 0 = Classic, 8 = Modern, 2 = Wavetable (mapped to Modern)
    o1_pos = int(round(xml_p.get("a_osc1_type", 0.0)))
    if o1_pos == 0:
        o1_cls = 0
    elif o1_pos in (8, 2):
        o1_cls = 1
    else:
        return None, None, f"osc 1 type {o1_pos} not subtractive (Classic/Modern)"

    f1_pos = int(round(xml_p.get("a_filter1_type", 1.0)))
    if f1_pos not in FILTER_POS_TO_CLASS:
        return None, None, f"filter 1 type {f1_pos} not recognized"
    f1_cls = FILTER_POS_TO_CLASS[f1_pos]

    ws_pos = int(round(xml_p.get("a_ws_type", 0.0)))
    if ws_pos not in WS_POS_TO_CLASS:
        return None, None, f"waveshaper {ws_pos} not recognized"
    ws_cls = WS_POS_TO_CLASS[ws_pos]

    vec = np.zeros(NUM_PARAMS, dtype=np.float32)
    vec[PARAM_INDEX["midi_note"]] = 0.5  # drawn at runtime
    vec[PARAM_INDEX["filter_type"]] = f1_cls / (len(LP_FILTERS) - 1)
    vec[PARAM_INDEX["waveshaper_type"]] = ws_cls / (len(WAVESHAPER_TYPES) - 1)

    # Unison: 1 to 6 voices
    voices = int(round(xml_p.get("a_osc1_param6", 1.0)))
    u_cls = min(max(voices, 1), 6) - 1
    vec[PARAM_INDEX["unison"]] = u_cls / 5.0

    # Osc types
    vec[PARAM_INDEX["osc_1_type"]] = float(o1_cls)
    o2_pos = int(round(xml_p.get("a_osc2_type", 0.0)))
    vec[PARAM_INDEX["osc_2_type"]] = 1.0 if o2_pos in (8, 2) else 0.0
    o3_pos = int(round(xml_p.get("a_osc3_type", 0.0)))
    vec[PARAM_INDEX["osc_3_type"]] = 1.0 if o3_pos in (8, 2) else 0.0

    # Filter routing & config
    vec[PARAM_INDEX["filter_configuration"]] = min(max((xml_p.get("a_fb_config", 4.0) - 4.0) / 3.0, 0.0), 1.0)
    vec[PARAM_INDEX["fm_routing"]] = min(max(xml_p.get("a_fm_routing", 0.0) / 3.0, 0.0), 1.0)
    f2_pos = int(round(xml_p.get("a_filter2_type", f1_pos)))
    vec[PARAM_INDEX["filter_2_type"]] = FILTER_POS_TO_CLASS.get(f2_pos, f1_cls) / (len(LP_FILTERS) - 1)

    # EG modes & shapes
    vec[PARAM_INDEX["amp_eg_mode"]] = 1.0 if int(round(xml_p.get("a_env1_mode", 0.0))) == 1 else 0.0
    vec[PARAM_INDEX["filter_eg_mode"]] = 1.0 if int(round(xml_p.get("a_env2_mode", 0.0))) == 1 else 0.0
    vec[PARAM_INDEX["amp_eg_decay_shape"]] = min(max(int(round(xml_p.get("a_env1_decay_shape", 1.0))), 0), 2) / 2.0
    vec[PARAM_INDEX["filter_eg_decay_shape"]] = min(max(int(round(xml_p.get("a_env2_decay_shape", 1.0))), 0), 2) / 2.0

    # Continuous params
    clipped = []
    for name, (tag, lo, hi) in CONTINUOUS.items():
        val = xml_p.get(tag, 0.0)
        # Handle special conversions
        if name == "drive":
            # val is dB (-24..24). We map 0..24 dB to [0, 1]
            raw_norm = (val - 0.0) / 24.0
        elif name in ("keytrack", "filter_2_keytrack", "filter_balance", "filter_feedback", "noise_color", "shape", "osc_2_shape", "osc_3_shape"):
            raw_norm = (val - lo) / (hi - lo)
        elif name in ("osc_2_pitch", "osc_3_pitch"):
            # pitch offset in semitones (-1..1) mapped to (0..1)
            raw_norm = 0.5 + 0.5 * min(max(val, -1.0), 1.0)
        else:
            raw_norm = (val - lo) / (hi - lo)
        
        v = min(max(raw_norm, 0.0), 1.0)
        if raw_norm < -1e-4 or raw_norm > 1.0 + 1e-4:
            clipped.append(name)
        vec[PARAM_INDEX[name]] = v

    if u_cls == 0:
        vec[PARAM_INDEX["unison_detune"]] = 0.0
    if ws_cls == 0:
        vec[PARAM_INDEX["drive"]] = 0.0
    fm_off = int(round(xml_p.get("a_fm_switch", 1.0))) == 0
    if fm_off:
        vec[PARAM_INDEX["fm_depth"]] = 0.0

    return vec.astype(np.float32), clipped, None

def collect_presets(mode: str, surge_data_dir: str, templates_dir: str, new_loops_zip: str, presetshare_dir: str):
    out = []
    if templates_dir and os.path.isdir(templates_dir):
        for f in os.listdir(templates_dir):
            if f.endswith(".fxp"):
                with open(os.path.join(templates_dir, f), "rb") as fh:
                    out.append(("Templates/" + f, fh.read()))

    if mode == "bass":
        basses_dir = os.path.join(surge_data_dir, "patches_factory", "Basses")
        if os.path.isdir(basses_dir):
            for root, _dirs, files in os.walk(basses_dir):
                for f in files:
                    if f.endswith(".fxp"):
                        with open(os.path.join(root, f), "rb") as fh:
                            out.append((os.path.relpath(os.path.join(root, f), surge_data_dir), fh.read()))
        if new_loops_zip and os.path.exists(new_loops_zip):
            with zipfile.ZipFile(new_loops_zip) as z:
                for info in z.infolist():
                    if "bass" in info.filename.lower() and info.filename.endswith(".fxp"):
                        out.append(("NewLoops/" + info.filename, z.read(info)))
        if presetshare_dir and os.path.isdir(presetshare_dir):
            for root, _dirs, files in os.walk(presetshare_dir):
                for f in files:
                    if f.endswith(".fxp") and "bass" in (root + "/" + f).lower():
                        with open(os.path.join(root, f), "rb") as fh:
                            out.append(("Presetshare/" + os.path.relpath(os.path.join(root, f), presetshare_dir), fh.read()))
    else: # mode == "all"
        for folder in ("patches_factory", "patches_3rdparty"):
            pdir = os.path.join(surge_data_dir, folder)
            if os.path.isdir(pdir):
                for root, _dirs, files in os.walk(pdir):
                    for f in files:
                        if f.endswith(".fxp"):
                            with open(os.path.join(root, f), "rb") as fh:
                                out.append((os.path.relpath(os.path.join(root, f), surge_data_dir), fh.read()))
    return sorted(out)

def build_manifold(presets, val_frac=0.1):
    rows, names, skips = [], [], {}
    clip_counts = np.zeros(NUM_PARAMS)
    seen = set()
    for name, data in presets:
        xml_p = parse_fxp(data)
        if xml_p is None:
            reason = "unparseable"
            vec = None
        else:
            vec, clipped, reason = preset_to_vector(xml_p)
        if vec is None:
            skips[reason] = skips.get(reason, 0) + 1
            continue
        key = np.round(vec, 4).tobytes()
        if key in seen:
            skips["duplicate vector"] = skips.get("duplicate vector", 0) + 1
            continue
        seen.add(key)
        for c in (clipped or []):
            if c in PARAM_INDEX:
                clip_counts[PARAM_INDEX[c]] += 1
        rows.append(vec)
        names.append(name)
    if not rows:
        raise RuntimeError(f"no usable presets (skips: {skips})")
    params = np.stack(rows)
    is_val = np.array([is_val_row(v, val_frac) for v in params])
    return params, names, is_val, skips, clip_counts / len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bass", "all"), default="bass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--surge_data_dir", default=SURGE_DATA_DIR)
    ap.add_argument("--templates_dir", default=TEMPLATES_DIR)
    ap.add_argument("--new_loops_zip", default=NEW_LOOPS_ZIP)
    ap.add_argument("--presetshare_dir", default=PRESETSHARE_DIR)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    presets = collect_presets(args.mode, args.surge_data_dir, args.templates_dir, args.new_loops_zip, args.presetshare_dir)
    print(f"Collected {len(presets)} presets for mode '{args.mode}'.")
    params, names, is_val, skips, clip_frac = build_manifold(presets, args.val_frac)
    print(f"Kept {len(params)} presets ({int(is_val.sum())} held out). Skips:")
    for k, v in sorted(skips.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez_compressed(
        tmp, params=params, preset_names=np.array(names), is_val=is_val,
        param_names=np.array(PARAM_NAMES), clip_frac=clip_frac.astype(np.float32),
        skip_reasons=json.dumps(skips), format_version=FORMAT_VERSION
    )
    os.replace(tmp, args.out)
    print(f"Saved {args.out} (shape {params.shape})")

if __name__ == "__main__":
    main()
