"""Surge XT v4 Parameter Registry, Bounds, and Manifold Union (v4 Spec Section 2).

Unifies:
1. Existing 122-preset manifold (real_bass_manifold.npz)
2. Kim's 13 handcrafted presets (~/Documents/Surge XT/Patches/AI Inversions/kim_bass_*.fxp)
into a comprehensive parameter registry with bounds and archetype priors.
"""
import os
import glob
import json
from typing import Dict, List, Tuple, Any
import numpy as np
import surgepy

V4_PARAM_NAMES: List[str] = [
    # Pitch & Global (2)
    "midi_note",
    "osc_drift",

    # Osc 1 (13)
    "osc1_mute",
    "osc1_type",
    "osc1_volume",
    "osc1_octave",
    "osc1_pitch",
    "osc1_shape",
    "osc1_width",
    "osc1_sub_mix",
    "osc1_sync",
    "osc1_unison_voices",
    "osc1_unison_detune",
    "osc1_retrigger",
    "osc1_route",

    # Osc 2 (13)
    "osc2_mute",
    "osc2_type",
    "osc2_volume",
    "osc2_octave",
    "osc2_pitch",
    "osc2_shape",
    "osc2_width",
    "osc2_sub_mix",
    "osc2_sync",
    "osc2_unison_voices",
    "osc2_unison_detune",
    "osc2_retrigger",
    "osc2_route",

    # Osc 3 (13)
    "osc3_mute",
    "osc3_type",
    "osc3_volume",
    "osc3_octave",
    "osc3_pitch",
    "osc3_shape",
    "osc3_width",
    "osc3_sub_mix",
    "osc3_sync",
    "osc3_unison_voices",
    "osc3_unison_detune",
    "osc3_retrigger",
    "osc3_route",

    # Mixer & Ring Mod (4)
    "ringmod_12_mute",
    "ringmod_12_volume",
    "ringmod_23_mute",
    "ringmod_23_volume",

    # Filter Global (3)
    "filter_config",
    "filter_balance",
    "feedback",

    # Filter 1 (5)
    "filter_1_type",
    "filter_1_cutoff",
    "filter_1_resonance",
    "filter_1_keytrack",
    "filter_1_feg_amount",

    # Filter 2 (5)
    "filter_2_type",
    "filter_2_cutoff",
    "filter_2_resonance",
    "filter_2_keytrack",
    "filter_2_feg_amount",

    # Waveshaper (2)
    "waveshaper_type",
    "waveshaper_drive",

    # Envelopes (10)
    "aeg_attack",
    "aeg_decay",
    "aeg_sustain",
    "aeg_release",
    "aeg_mode",
    "feg_attack",
    "feg_decay",
    "feg_sustain",
    "feg_release",
    "feg_mode",

    # Modulation & FX (2)
    "feg_to_pw",
    "chorus_mix",
]

NUM_V4_PARAMS = len(V4_PARAM_NAMES)
PARAM_INDEX = {name: i for i, name in enumerate(V4_PARAM_NAMES)}

# Categorical parameter specifications (name -> number of classes)
V4_CATEGORICAL = {
    "osc1_type": 12,
    "osc2_type": 12,
    "osc3_type": 12,
    "osc1_route": 3,
    "osc2_route": 3,
    "osc3_route": 3,
    "osc1_unison_voices": 16,
    "osc2_unison_voices": 16,
    "osc3_unison_voices": 16,
    "filter_config": 8,
    "filter_1_type": 36,
    "filter_2_type": 36,
    "waveshaper_type": 43,
    "aeg_mode": 2,
    "feg_mode": 2,
}

# Boolean parameters
V4_BOOLEAN = {
    "osc1_mute", "osc1_retrigger",
    "osc2_mute", "osc2_retrigger",
    "osc3_mute", "osc3_retrigger",
    "ringmod_12_mute", "ringmod_23_mute",
}

# Default continuous bounds in native Surge units
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "midi_note": (28.0, 50.0),
    "osc_drift": (0.0, 1.0),
    "osc1_volume": (0.0, 1.0),
    "osc1_octave": (-3.0, 3.0),
    "osc1_pitch": (-7.0, 7.0),
    "osc1_shape": (-1.0, 1.0),
    "osc1_width": (0.0, 1.0),
    "osc1_sub_mix": (0.0, 1.0),
    "osc1_sync": (0.0, 60.0),
    "osc1_unison_detune": (0.0, 1.0),
    "osc2_volume": (0.0, 1.0),
    "osc2_octave": (-3.0, 3.0),
    "osc2_pitch": (-7.0, 7.0),
    "osc2_shape": (-1.0, 1.0),
    "osc2_width": (0.0, 1.0),
    "osc2_sub_mix": (0.0, 1.0),
    "osc2_sync": (0.0, 60.0),
    "osc2_unison_detune": (0.0, 1.0),
    "osc3_volume": (0.0, 1.0),
    "osc3_octave": (-3.0, 3.0),
    "osc3_pitch": (-7.0, 7.0),
    "osc3_shape": (-1.0, 1.0),
    "osc3_width": (0.0, 1.0),
    "osc3_sub_mix": (0.0, 1.0),
    "osc3_sync": (0.0, 60.0),
    "osc3_unison_detune": (0.0, 1.0),
    "ringmod_12_volume": (0.0, 1.0),
    "ringmod_23_volume": (0.0, 1.0),
    "filter_balance": (-1.0, 1.0),
    "feedback": (-1.0, 1.0),
    "filter_1_cutoff": (-60.0, 70.0),
    "filter_1_resonance": (0.0, 1.0),
    "filter_1_keytrack": (-1.0, 1.0),
    "filter_1_feg_amount": (-96.0, 96.0),
    "filter_2_cutoff": (-60.0, 70.0),
    "filter_2_resonance": (0.0, 1.0),
    "filter_2_keytrack": (-1.0, 1.0),
    "filter_2_feg_amount": (-96.0, 96.0),
    "waveshaper_drive": (-24.0, 24.0),
    "aeg_attack": (-8.0, 5.0),
    "aeg_decay": (-8.0, 5.0),
    "aeg_sustain": (0.0, 1.0),
    "aeg_release": (-8.0, 5.0),
    "feg_attack": (-8.0, 5.0),
    "feg_decay": (-8.0, 5.0),
    "feg_sustain": (0.0, 1.0),
    "feg_release": (-8.0, 5.0),
    "feg_to_pw": (-1.0, 1.0),
    "chorus_mix": (0.0, 1.0),
}

def extract_patch_from_synth(synth: surgepy.SurgeSynthesizer) -> Dict[str, Any]:
    """Reads all v4 parameters from a live Surge instance into a dict."""
    p = {}
    cg_osc = synth.getControlGroup(2)
    cg_mix = synth.getControlGroup(3)
    cg_filt = synth.getControlGroup(4)
    cg_env = synth.getControlGroup(5)
    cg_global = synth.getControlGroup(0)

    def find_p(cg, name):
        for e in cg.getEntries():
            for param in e.getParams():
                if param.getName() == name:
                    return param
        return None

    # Global
    p["midi_note"] = 36
    p["osc_drift"] = float(synth.getParamVal(find_p(cg_global, "Osc Drift")))
    p["feedback"] = float(synth.getParamVal(find_p(cg_global, "Feedback")))
    p["filter_config"] = int(synth.getParamVal(find_p(cg_global, "Filter Configuration")))
    p["filter_balance"] = float(synth.getParamVal(find_p(cg_global, "Filter Balance")))
    p["waveshaper_type"] = int(synth.getParamVal(find_p(cg_global, "Waveshaper Type")))
    p["waveshaper_drive"] = float(synth.getParamVal(find_p(cg_global, "Waveshaper Drive")))

    # Oscillators
    for osc in (1, 2, 3):
        p[f"osc{osc}_mute"] = bool(synth.getParamVal(find_p(cg_mix, f"Osc {osc} Mute")) > 0.5)
        p[f"osc{osc}_volume"] = float(synth.getParamVal(find_p(cg_mix, f"Osc {osc} Volume")))
        p[f"osc{osc}_route"] = int(synth.getParamVal(find_p(cg_mix, f"Osc {osc} Route")))
        p[f"osc{osc}_type"] = int(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Type")))
        p[f"osc{osc}_octave"] = int(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Octave")))
        p[f"osc{osc}_pitch"] = float(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Pitch")))
        p[f"osc{osc}_shape"] = float(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Shape")))
        p[f"osc{osc}_width"] = float(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Width 1")))
        p[f"osc{osc}_sub_mix"] = float(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Sub Mix")))
        p[f"osc{osc}_sync"] = float(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Sync")))
        p[f"osc{osc}_unison_voices"] = int(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Unison Voices")))
        p[f"osc{osc}_unison_detune"] = float(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Unison Detune")))
        p[f"osc{osc}_retrigger"] = bool(synth.getParamVal(find_p(cg_osc, f"Osc {osc} Retrigger")) > 0.5)

    # Ring mod
    p["ringmod_12_mute"] = bool(synth.getParamVal(find_p(cg_mix, "Ring Modulation 1x2 Mute")) > 0.5)
    p["ringmod_12_volume"] = float(synth.getParamVal(find_p(cg_mix, "Ring Modulation 1x2 Volume")))
    p["ringmod_23_mute"] = bool(synth.getParamVal(find_p(cg_mix, "Ring Modulation 2x3 Mute")) > 0.5)
    p["ringmod_23_volume"] = float(synth.getParamVal(find_p(cg_mix, "Ring Modulation 2x3 Volume")))

    # Filters
    for f in (1, 2):
        p[f"filter_{f}_type"] = int(synth.getParamVal(find_p(cg_filt, f"Filter {f} Type")))
        p[f"filter_{f}_cutoff"] = float(synth.getParamVal(find_p(cg_filt, f"Filter {f} Cutoff")))
        p[f"filter_{f}_resonance"] = float(synth.getParamVal(find_p(cg_filt, f"Filter {f} Resonance")))
        p[f"filter_{f}_keytrack"] = float(synth.getParamVal(find_p(cg_filt, f"Filter {f} Keytrack")))
        p[f"filter_{f}_feg_amount"] = float(synth.getParamVal(find_p(cg_filt, f"Filter {f} FEG Mod Amount")))

    # Envelopes
    p["aeg_attack"] = float(synth.getParamVal(find_p(cg_env, "Amp EG Attack")))
    p["aeg_decay"] = float(synth.getParamVal(find_p(cg_env, "Amp EG Decay")))
    p["aeg_sustain"] = float(synth.getParamVal(find_p(cg_env, "Amp EG Sustain")))
    p["aeg_release"] = float(synth.getParamVal(find_p(cg_env, "Amp EG Release")))
    p["aeg_mode"] = int(synth.getParamVal(find_p(cg_env, "Amp EG Envelope Mode")))

    p["feg_attack"] = float(synth.getParamVal(find_p(cg_env, "Filter EG Attack")))
    p["feg_decay"] = float(synth.getParamVal(find_p(cg_env, "Filter EG Decay")))
    p["feg_sustain"] = float(synth.getParamVal(find_p(cg_env, "Filter EG Sustain")))
    p["feg_release"] = float(synth.getParamVal(find_p(cg_env, "Filter EG Release")))
    p["feg_mode"] = int(synth.getParamVal(find_p(cg_env, "Filter EG Envelope Mode")))

    # Modulation routings
    p["feg_to_pw"] = 0.0
    mod_dict = synth.getAllModRoutings()
    for sc in mod_dict.get("scene", []):
        for k, r_list in sc.items():
            for r in r_list:
                if "Filter EG" in r.getSource().getName() and "Width 1" in r.getDest().getName():
                    p["feg_to_pw"] = float(r.getDepth())

    # FX
    p["chorus_mix"] = 0.0
    return p

def build_v4_archetype_dataset(
    kim_dir: str = "/home/kim/Documents/Surge XT/Patches/AI Inversions",
    out_path: str = "/home/kim/sao_runs/v4_kim_archetypes.json",
) -> Dict[str, Any]:
    """Extracts Kim's 13 presets into v4 archetype dataset."""
    synth = surgepy.createSurge(44100)
    kim_files = sorted(glob.glob(os.path.join(kim_dir, "kim_bass_*.fxp")))
    print(f"Loading {len(kim_files)} Kim presets...")
    
    kim_patches = []
    for kf in kim_files:
        ok = synth.loadPatch(kf)
        if ok:
            p = extract_patch_from_synth(synth)
            p["preset_name"] = os.path.basename(kf)
            kim_patches.append(p)

    print(f"Extracted {len(kim_patches)} patches from Kim presets.")

    # Save Kim archetypes to JSON
    with open(out_path, "w") as f:
        json.dump(kim_patches, f, indent=2)

    with open("v4_bounds.json", "w") as f:
        json.dump(DEFAULT_BOUNDS, f, indent=2)

    return {
        "num_params": NUM_V4_PARAMS,
        "param_names": V4_PARAM_NAMES,
        "bounds": DEFAULT_BOUNDS,
        "kim_patches": kim_patches,
    }

if __name__ == "__main__":
    res = build_v4_archetype_dataset()
    print(f"V4 Parameter Registry initialized with {res['num_params']} parameters!")
