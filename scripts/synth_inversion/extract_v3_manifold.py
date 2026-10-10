"""Project real Surge XT bass presets (.fxp) into the 23-d parameter space of surge_spec.py.

Output (default /run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz, format_version 2):
  params       [N, 23] float32, the h5 'params' layout (surge_spec_v3.patch_to_vector conventions)
  preset_names [N]     source file of each row
  is_val       [N]     held-out presets (split by a hash of the row's parameter vector, so a preset
                       that appears in both sources cannot land on both sides)
  param_names, clip_frac [23] (fraction of rows clipped into the renderable range per parameter),
  skip_reasons (JSON), ranges (JSON, the native ranges used), calibrated (bool), format_version

How a preset value becomes a vector value (v2, 2026-10-02 review; v1 guessed every conversion):
  1. Surge stores each parameter in its NATIVE unit in the patch XML (semitones, log2 seconds,
     dB, fractions). The VST parameter the renderer sets (pedalboard raw_value) is that value
     mapped linearly over the parameter's native range: raw = (x - min) / (max - min).
  2. The native ranges below come from Surge's Parameter.cpp. calibrate_ranges() CHECKS them
     against the plugin itself: it sets raw values, reads the plugin's display strings, converts
     them back to native units and compares. A mismatch stops the run and prints the measured
     range (pass --use_measured_ranges to adopt it once you have looked at it).
  3. raw -> vector value with exactly the conventions vector_to_patch() inverts (detune = raw/0.35,
     keytrack = (raw-0.5)/0.5, drive = (raw-0.5)/0.32); anything outside [0, 1] is clipped and
     counted in clip_frac.
  4. Enum parameters (filter, waveshaper) are integer positions in Surge's lists; the raw values in
     surge_spec_v3.LP_FILTERS / WAVESHAPER_TYPES are those positions / (list length - 1), which
     verify_surge_mapping() checks against the plugin's names. A preset whose filter is not one of
     our 10 lowpass circuits, whose waveshaper is not one of our 6, or whose oscillator 1 is not
     the Classic oscillator (the only one whose param0..6 mean shape/width/.../voices) is SKIPPED
     and counted, not relabelled.
  5. Only scene A, oscillator 1, filter 1 and the two envelopes are read; oscillators 2/3, LFOs,
     modulation routings, filter 2 and FX are not part of the 23-d space. The projected patch
     therefore does not sound like the preset; it is a prior over OUR space, informed by where
     real bass presets put these 20 controls.

Needs pedalboard (the data venv), the Surge XT VST3, and the preset files. --no_plugin skips the
calibration (prints a warning and records calibrated=False); use it only for a dry look.
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

from surge_spec_v3 import (DEFAULT_PLUGIN_PATH, LP_FILTERS, NUM_PARAMS, PARAM_INDEX, PARAM_NAMES, UNISON_VOICES_RAW,
                        WAVESHAPER_TYPES, verify_surge_mapping)

FORMAT_VERSION = 2
OUTPUT_PATH = "/run/media/kim/Mantu2/surge_200k_models/real_bass_manifold_v3.npz"
SURGE_DATA_DIR = "/home/kim/Downloads/surge-xt-portable-content-1.3.4/Surge Synth Team/SurgeXTData"
NEW_LOOPS_ZIP = "/home/kim/Documents/Surge XT/Patches/AI Inversions/kim_bass_.zip"

# Surge enum list lengths - 1 (the raw value of position i is i / N). surge_spec's enum raw
# values are exact multiples of these, which _enum_positions() asserts.
FILTER_TYPE_DENOM = 33
WAVESHAPER_DENOM = 40
CLASSIC_OSC_TYPE = 0  # a_osc1_type value of the Classic oscillator (surge_spec_v3.init_synth sets raw 0.0)

# Continuous parameters: vector name -> (patch XML tag, plugin parameter, native min, native max, display unit kind)
CONTINUOUS = {
    "shape":         ("a_osc1_param0",       "a_osc_1_shape",             -1.0,  1.0, "pct"),
    "width":         ("a_osc1_param1",       "a_osc_1_width_1",            0.0,  1.0, "pct"),
    "sub_mix":       ("a_osc1_param3",       "a_osc_1_sub_mix",            0.0,  1.0, "pct"),
    "sync":          ("a_osc1_param4",       "a_osc_1_sync",               0.0, 60.0, "semitones"),
    "unison_detune": ("a_osc1_param5",       "a_osc_1_unison_detune",      0.0,  1.0, "cents"),
    "fm_depth":      ("a_fm_depth",          "a_fm_depth",               -48.0, 16.0, "db"),
    "cutoff":        ("a_filter1_cutoff",    "a_filter_1_cutoff",        -60.0, 70.0, "hz"),
    "resonance":     ("a_filter1_resonance", "a_filter_1_resonance",       0.0,  1.0, "pct"),
    "keytrack":      ("a_filter1_keytrack",  "a_filter_1_keytrack",       -1.0,  1.0, "pct"),
    "feg_amount":    ("a_filter1_envmod",    "a_filter_1_feg_mod_amount", -96.0, 96.0, "semitones"),
    "feg_decay":     ("a_env2_decay",        "a_filter_eg_decay",          -8.0,  5.0, "seconds"),
    "feg_sustain":   ("a_env2_sustain",      "a_filter_eg_sustain",        0.0,  1.0, "pct"),
    "aeg_decay":     ("a_env1_decay",        "a_amp_eg_decay",             -8.0,  5.0, "seconds"),
    "aeg_sustain":   ("a_env1_sustain",      "a_amp_eg_sustain",           0.0,  1.0, "pct"),
    "aeg_release":   ("a_env1_release",      "a_amp_eg_release",           -8.0,  5.0, "seconds"),
    "drive":         ("a_ws_drive",          "a_waveshaper_drive",       -24.0, 24.0, "db"),
    "osc_1_width_2": ("a_osc1_param2", "a_osc_1_width_2", 0.0, 1.0, "pct"),
    "osc_2_shape": ("a_osc2_param0", "a_osc_2_shape", -1.0, 1.0, "pct"),
    "osc_2_width_1": ("a_osc2_param1", "a_osc_2_width_1", 0.0, 1.0, "pct"),
    "osc_2_width_2": ("a_osc2_param2", "a_osc_2_width_2", 0.0, 1.0, "pct"),
    "osc_3_shape": ("a_osc3_param0", "a_osc_3_shape", -1.0, 1.0, "pct"),
    "osc_3_width_1": ("a_osc3_param1", "a_osc_3_width_1", 0.0, 1.0, "pct"),
    "osc_3_width_2": ("a_osc3_param2", "a_osc_3_width_2", 0.0, 1.0, "pct"),
    "osc_1_octave": ("a_osc1_octave", "a_osc_1_octave", -4.0, 4.0, "semitones"),
    "osc_2_octave": ("a_osc2_octave", "a_osc_2_octave", -4.0, 4.0, "semitones"),
    "osc_3_octave": ("a_osc3_octave", "a_osc_3_octave", -4.0, 4.0, "semitones"),
    "noise_color": ("a_noise_color", "a_noise_color", -1.0, 1.0, "pct"),
    "osc_1_volume": ("a_level_o1", "a_osc_1_level", 0.0, 1.0, "pct"),
    "osc_2_volume": ("a_level_o2", "a_osc_2_level", 0.0, 1.0, "pct"),
    "osc_3_volume": ("a_level_o3", "a_osc_3_level", 0.0, 1.0, "pct"),
    "ring_1x2_volume": ("a_level_ring12", "a_ring_1x2_level", 0.0, 1.0, "pct"),
    "ring_2x3_volume": ("a_level_ring23", "a_ring_2x3_level", 0.0, 1.0, "pct"),
    "noise_volume": ("a_level_noise", "a_noise_level", 0.0, 1.0, "pct"),
    "filter_balance": ("a_f_balance", "a_filter_balance", -1.0, 1.0, "pct"),
    "filter_2_cutoff": ("a_filter2_cutoff", "a_filter_2_cutoff", -60.0, 70.0, "hz"),
    "filter_2_resonance": ("a_filter2_resonance", "a_filter_2_resonance", 0.0, 1.0, "pct"),
    "filter_2_keytrack": ("a_filter2_keytrack", "a_filter_2_keytrack", -1.0, 1.0, "pct"),
    "filter_2_feg_amount": ("a_filter2_envmod", "a_filter_2_feg_mod_amount", -96.0, 96.0, "semitones"),
}
ENUM_TAGS = {"osc_type": "a_osc1_type", "voices": "a_osc1_param6", "filter_type": "a_filter1_type",
             "ws_type": "a_ws_type", "fm_switch": "a_fm_switch",
             "filter_configuration": "a_fb_config", "fm_routing": "a_fm_routing", "filter_2_type": "a_filter2_type"}

# Raw -> vector value, inverse of surge_spec_v3.vector_to_patch
_RAW_TO_VEC = {
    "unison_detune": lambda r: r / 0.35,
    "keytrack": lambda r: (r - 0.5) / 0.5,
    "drive": lambda r: (r - 0.5) / 0.32,
}


# --------------------------------------------------------------------------- display parsing
_NUM_UNIT = re.compile(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z%]*)")


def display_to_native(text: str, kind: str) -> float:
    """Plugin display string -> native unit. Raises ValueError when it cannot be read."""
    m = _NUM_UNIT.search(str(text).replace(",", ""))
    if not m:
        raise ValueError(f"no number in {text!r}")
    v, unit = float(m.group(1)), m.group(2).lower()
    if kind == "pct":
        if unit != "%":
            raise ValueError(f"expected %, got {text!r}")
        return v / 100.0
    if kind == "semitones":
        if unit not in ("", "st", "semi", "semis", "semitone", "semitones"):
            raise ValueError(f"expected semitones, got {text!r}")
        return v
    if kind == "cents":
        if unit not in ("c", "ct", "cent", "cents"):
            raise ValueError(f"expected cents, got {text!r}")
        return v / 100.0
    if kind == "db":
        if unit != "db":
            raise ValueError(f"expected dB, got {text!r}")
        return v
    if kind == "hz":
        hz = v * 1000.0 if unit == "khz" else v if unit == "hz" else None
        if hz is None or hz <= 0:
            raise ValueError(f"expected Hz/kHz, got {text!r}")
        return 12.0 * math.log2(hz / 440.0)
    if kind == "seconds":
        sec = v / 1000.0 if unit == "ms" else v if unit == "s" else None
        if sec is None or sec <= 0:
            raise ValueError(f"expected s/ms, got {text!r}")
        return math.log2(sec)
    raise ValueError(f"unknown display kind {kind}")


def calibrate_ranges(plugin, use_measured=False, points=(0.2, 0.5, 0.8), tol=0.02):
    """Check every native range in CONTINUOUS against the plugin's own display strings.
    Returns {name: (min, max)} to use. Raises RuntimeError listing each mismatch."""
    ranges, problems = {}, []
    for name, (_tag, pname, lo, hi, kind) in CONTINUOUS.items():
        param = plugin.parameters[pname]
        old = param.raw_value
        xs, ys = [], []
        try:
            for r in points:
                param.raw_value = r
                xs.append(r)
                ys.append(display_to_native(param.string_value, kind))
        except ValueError as e:
            problems.append(f"  {name} ({pname}): cannot read display: {e}")
            continue
        finally:
            param.raw_value = old
        slope, icpt = np.polyfit(xs, ys, 1)
        measured = (float(icpt), float(icpt + slope))
        expected = np.array([lo + r * (hi - lo) for r in xs])
        worst = float(np.max(np.abs(np.array(ys) - expected)))
        if worst > tol * (hi - lo):
            problems.append(f"  {name} ({pname}): source range [{lo}, {hi}] but the plugin reads as "
                            f"[{measured[0]:.3f}, {measured[1]:.3f}] (worst error {worst:.3f})")
            ranges[name] = measured
        else:
            ranges[name] = (lo, hi)
    if problems and not use_measured:
        raise RuntimeError("Native ranges do not match this Surge build:\n" + "\n".join(problems)
                           + "\nCheck them, then rerun with --use_measured_ranges to adopt the measured ones.")
    for p in problems:
        print("WARNING (adopting measured range):", p.strip())
    return ranges


def _enum_positions():
    """Surge list position -> our class index, from surge_spec's verified raw values."""
    def positions(table, denom):
        out = {}
        for k, (name, raw) in enumerate(table):
            pos = raw * denom
            if abs(pos - round(pos)) > 0.02:
                raise RuntimeError(f"{name}: raw {raw} is not a multiple of 1/{denom}; "
                                   "the enum list length assumption is wrong")
            out[int(round(pos))] = k
        return out
    return positions(LP_FILTERS, FILTER_TYPE_DENOM), positions(WAVESHAPER_TYPES, WAVESHAPER_DENOM)


FILTER_POS_TO_CLASS, WS_POS_TO_CLASS = _enum_positions()


# --------------------------------------------------------------------------- preset parsing
def parse_fxp(data: bytes):
    """Surge .fxp bytes -> {xml tag: float value} for the patch's <parameters>, or None."""
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


def preset_to_vector(xml_p: dict, ranges: dict):
    """-> (23-d vector, clipped-parameter names, None) or (None, None, skip reason)."""
    if int(round(xml_p.get(ENUM_TAGS["osc_type"], 0))) != CLASSIC_OSC_TYPE:
        return None, None, "osc 1 not Classic"
    f_pos = int(round(xml_p.get(ENUM_TAGS["filter_type"], 0)))
    if f_pos not in FILTER_POS_TO_CLASS:
        return None, None, f"filter type {f_pos} not one of our lowpass circuits"
    ws_pos = int(round(xml_p.get(ENUM_TAGS["ws_type"], 0)))
    if ws_pos not in WS_POS_TO_CLASS:
        return None, None, f"waveshaper type {ws_pos} not one of our six"

    vec = np.zeros(NUM_PARAMS, dtype=np.float64)
    vec[PARAM_INDEX["midi_note"]] = 0.5  # placeholder; the sampler draws the note
    vec[PARAM_INDEX["filter_type"]] = FILTER_POS_TO_CLASS[f_pos] / (len(LP_FILTERS) - 1)
    ws_class = WS_POS_TO_CLASS[ws_pos]
    vec[PARAM_INDEX["waveshaper_type"]] = ws_class / (len(WAVESHAPER_TYPES) - 1)
    
    vec[PARAM_INDEX["filter_configuration"]] = min(max((xml_p.get(ENUM_TAGS["filter_configuration"], 4.0) - 4.0) / 3.0, 0.0), 1.0)
    vec[PARAM_INDEX["fm_routing"]] = min(max(xml_p.get(ENUM_TAGS["fm_routing"], 0) / 3.0, 0.0), 1.0)
    f2_pos = int(round(xml_p.get(ENUM_TAGS["filter_2_type"], f_pos)))
    vec[PARAM_INDEX["filter_2_type"]] = FILTER_POS_TO_CLASS.get(f2_pos, 0) / (len(LP_FILTERS) - 1)
    voices = int(round(xml_p.get(ENUM_TAGS["voices"], 1)))
    unison = voices > 1  # our space has 1 or 2 voices; 3+ voice presets become 2 voices
    vec[PARAM_INDEX["unison"]] = 1.0 if unison else 0.0

    clipped = []
    for name, (tag, _pname, _lo, _hi, _kind) in CONTINUOUS.items():
        lo, hi = ranges[name]
        raw = (xml_p.get(tag, 0.0) - lo) / (hi - lo)
        v = _RAW_TO_VEC.get(name, lambda r: r)(raw)
        if v < -1e-6 or v > 1 + 1e-6:
            clipped.append(name)
        vec[PARAM_INDEX[name]] = min(max(v, 0.0), 1.0)
    # Inactive controls are exactly 0, as in surge_spec.draw_patch / canonicalize_vector
    if not unison:
        vec[PARAM_INDEX["unison_detune"]] = 0.0
    if ws_class == 0:
        vec[PARAM_INDEX["drive"]] = 0.0
    fm_off = int(round(xml_p.get(ENUM_TAGS["fm_switch"], 1))) == 0  # tag absent: keep the depth
    if fm_off:
        vec[PARAM_INDEX["fm_depth"]] = 0.0
    # chorus_mix / delay_mix / delay_fb stay 0: preset FX are not projected
    inactive = {"unison_detune": not unison, "drive": ws_class == 0,
                "fm_depth": fm_off}
    return vec.astype(np.float32), [c for c in clipped if not inactive.get(c, False)], None


def is_val_row(vec: np.ndarray, val_frac: float) -> bool:
    h = hashlib.md5(np.round(vec, 5).astype(np.float32).tobytes()).hexdigest()
    return (int(h[:8], 16) % 10000) < val_frac * 10000


def collect_presets(surge_data_dir, new_loops_zip):
    """-> list of (name, bytes) for every .fxp under a path containing 'bass'."""
    out = []
    if surge_data_dir and os.path.isdir(surge_data_dir):
        for root, _dirs, files in os.walk(surge_data_dir):
            if "bass" in root.lower():
                for f in files:
                    if f.endswith(".fxp"):
                        with open(os.path.join(root, f), "rb") as fh:
                            out.append((os.path.relpath(os.path.join(root, f), surge_data_dir), fh.read()))
    if new_loops_zip and os.path.exists(new_loops_zip):
        with zipfile.ZipFile(new_loops_zip) as z:
            for info in z.infolist():
                if "bass" in info.filename.lower() and info.filename.endswith(".fxp"):
                    out.append(("NewLoops/" + info.filename, z.read(info)))
    return sorted(out)


def build_manifold(presets, ranges, val_frac=0.1):
    rows, names, skips = [], [], {}
    clip_counts = np.zeros(NUM_PARAMS)
    seen = set()
    for name, data in presets:
        xml_p = parse_fxp(data)
        if xml_p is None:
            reason = "unparseable"
            vec = None
        else:
            vec, clipped, reason = preset_to_vector(xml_p, ranges)
        if vec is None:
            skips[reason] = skips.get(reason, 0) + 1
            continue
        key = np.round(vec, 5).tobytes()
        if key in seen:
            skips["duplicate vector"] = skips.get("duplicate vector", 0) + 1
            continue
        seen.add(key)
        for c in clipped:
            clip_counts[PARAM_INDEX[c]] += 1
        rows.append(vec)
        names.append(name)
    if not rows:
        raise RuntimeError(f"no usable presets (skips: {skips})")
    params = np.stack(rows)
    is_val = np.array([is_val_row(v, val_frac) for v in params])
    return params, names, is_val, skips, clip_counts / len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--surge_data_dir", default=SURGE_DATA_DIR)
    ap.add_argument("--new_loops_zip", default=NEW_LOOPS_ZIP)
    ap.add_argument("--out", default=OUTPUT_PATH)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--val_frac", type=float, default=0.1, help="held-out preset fraction")
    ap.add_argument("--use_measured_ranges", action="store_true")
    ap.add_argument("--no_plugin", action="store_true", help="skip calibration (unverified; dry look only)")
    args = ap.parse_args()

    if args.no_plugin:
        print("WARNING: --no_plugin: native ranges are UNVERIFIED against the plugin.")
        ranges = {n: (c[2], c[3]) for n, c in CONTINUOUS.items()}
    else:
        import pedalboard
        plugin = pedalboard.load_plugin(os.path.expanduser(args.plugin))
        verify_surge_mapping(plugin)  # enum raw values <-> names
        ranges = calibrate_ranges(plugin, use_measured=args.use_measured_ranges)
        print("Native ranges verified against the plugin.")

    presets = collect_presets(args.surge_data_dir, args.new_loops_zip)
    print(f"Found {len(presets)} bass preset files.")
    params, names, is_val, skips, clip_frac = build_manifold(presets, ranges, args.val_frac)
    print(f"Kept {len(params)} presets ({int(is_val.sum())} held out). Skipped:")
    for k, v in sorted(skips.items(), key=lambda kv: -kv[1]):
        print(f"  {v:4d}  {k}")
    print(f"\n{'parameter':16s} {'mean':>6s} {'p10':>6s} {'p50':>6s} {'p90':>6s} {'zero%':>6s} {'clip%':>6s}")
    for i, n in enumerate(PARAM_NAMES):
        col = params[:, i]
        print(f"{n:16s} {col.mean():6.3f} {np.percentile(col, 10):6.3f} {np.percentile(col, 50):6.3f} "
              f"{np.percentile(col, 90):6.3f} {100 * np.mean(col == 0):6.1f} {100 * clip_frac[i]:6.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(dir=os.path.dirname(os.path.abspath(args.out)), suffix=".npz", delete=False)
    tmp.close()
    np.savez_compressed(tmp.name, params=params, preset_names=np.array(names), is_val=is_val,
                        param_names=np.array(PARAM_NAMES), clip_frac=clip_frac.astype(np.float32),
                        skip_reasons=json.dumps(skips), ranges=json.dumps(ranges),
                        calibrated=not args.no_plugin, format_version=FORMAT_VERSION,
                        unison_voices_raw=json.dumps(UNISON_VOICES_RAW))
    os.replace(tmp.name, args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
