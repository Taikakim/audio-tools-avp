"""Extracts empirical parameter distributions from all 430 real curated Surge XT bass presets.

Saves:
  /run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz
Containing:
  - 'params': [N, D] array of real bass normalized parameter vectors
  - 'param_names': list of parameter names
  - 'mean': empirical mean vector
  - 'cov': empirical covariance matrix
  - 'p10', 'p50', 'p90': percentiles
  - 'zero_prob': fraction of presets where parameter is 0 / off
"""

import os
import glob
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

OUTPUT_PATH = "/run/media/kim/Mantu/surge_200k_models/real_bass_manifold.npz"
SURGE_DATA_DIR = "/home/kim/Downloads/surge-xt-portable-content-1.3.4/Surge Synth Team/SurgeXTData"
NEW_LOOPS_ZIP = "/home/kim/Downloads/New_Loops-Surge_Presets.zip"
NL_EXTRACT_DIR = "/tmp/new_loops_bass"

def collect_fxp_files():
    files = []
    # 1. SurgeXTData
    for root, dirs, f_list in os.walk(SURGE_DATA_DIR):
        if "bass" in root.lower():
            for f in f_list:
                if f.endswith(".fxp"):
                    files.append(os.path.join(root, f))
    # 2. New Loops
    os.makedirs(NL_EXTRACT_DIR, exist_ok=True)
    if os.path.exists(NEW_LOOPS_ZIP):
        with zipfile.ZipFile(NEW_LOOPS_ZIP, "r") as z:
            for info in z.infolist():
                if "bass" in info.filename.lower() and info.filename.endswith(".fxp"):
                    z.extract(info, NL_EXTRACT_DIR)
                    files.append(os.path.join(NL_EXTRACT_DIR, info.filename))
    return sorted(list(set(files)))

def parse_fxp_to_xml(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        start = data.find(b"<?xml")
        if start == -1: return None
        end = data.rfind(b"</patch>") + 8
        root = ET.fromstring(data[start:end].decode("utf-8", errors="ignore"))
        params = root.find("parameters")
        if params is None: return None
        d = {}
        for elem in params:
            try:
                d[elem.tag] = float(elem.attrib.get("value", 0.0))
            except:
                pass
        return d
    except:
        return None

def extract_preset_vector(xml_p: dict) -> np.ndarray:
    """Map XML parameters into the canonical normalized 23-d vector space."""
    # 0: midi_note (default around 36 = C2, normalized [0, 1] for 28..50)
    midi_note = (36.0 - 28.0) / (50.0 - 28.0)

    # 1: filter_type (LP 12dB=0, LP 24dB=1, Legacy=2, Vintage=3, OB12=4, OB24=5, K35=6, Diode=7, Warp=8, Res=9)
    f_type_raw = xml_p.get("a_filter1_type", 1.0)
    filter_type = np.clip(f_type_raw / 9.0, 0.0, 1.0)

    # 2: shape (saw <-> morph <-> pulse). XML a_osc1_param0: -1 to +1 or 0 to 1
    p0 = xml_p.get("a_osc1_param0", 0.0)
    if p0 < 0:
        shape = (p0 + 1.0) / 2.0
    elif p0 <= 1.0:
        shape = p0
    else:
        shape = 0.5
    shape = float(np.clip(shape, 0.0, 1.0))

    # 3: width (pulse width). XML a_osc1_param1: 0.0 to 1.0
    p1 = xml_p.get("a_osc1_param1", 0.5)
    if -1.0 <= p1 <= 1.0:
        width = (p1 + 1.0) / 2.0 if p1 < 0 else p1
    else:
        width = 0.5
    width = float(np.clip(width, 0.0, 1.0))

    # 4: sub_mix. XML a_osc1_param2: 0.0 to 1.0, clamp max to 0.50
    p2 = xml_p.get("a_osc1_param2", 0.0)
    sub_mix = float(np.clip(p2 if 0.0 <= p2 <= 1.0 else 0.0, 0.0, 0.50))

    # 5: sync. XML a_osc1_param3: 0.0 to 60.0 semitones
    p3 = xml_p.get("a_osc1_param3", 0.0)
    sync = float(np.clip(p3 / 60.0 if 0.0 <= p3 <= 60.0 else 0.0, 0.0, 0.40))

    # 6: fm_depth. XML a_fm_depth: -48 dB to +24 dB
    fm = xml_p.get("a_fm_depth", -48.0)
    fm_raw = (fm + 48.0) / 72.0 if -48.0 <= fm <= 24.0 else 0.0
    fm_depth = float(np.clip(fm_raw, 0.0, 0.45))

    # 7: unison (0 or 1). XML a_osc1_param6: voices
    u_voices = xml_p.get("a_osc1_param6", 1.0)
    unison = 1.0 if u_voices > 1.0 else 0.0

    # 8: unison_detune (raw / 0.35)
    u_detune = xml_p.get("a_osc1_param5", 0.0)
    unison_detune = float(np.clip(u_detune / 0.35 if unison > 0.5 and 0 <= u_detune <= 0.35 else 0.0, 0.0, 1.0))

    # 9: cutoff (-60 to +70 semitones -> 0..1)
    f_cut = xml_p.get("a_filter1_cutoff", -8.5)
    cutoff = float(np.clip((f_cut + 60.0) / 130.0, 0.05, 0.90))

    # 10: resonance (0..1)
    res = xml_p.get("a_filter1_resonance", 0.0)
    resonance = float(np.clip(res if 0 <= res <= 1.0 else 0.0, 0.0, 0.95))

    # 11: keytrack (-100% to +100% -> 0..1)
    kt = xml_p.get("a_filter1_keytrack", 1.0)
    keytrack = float(np.clip((kt + 1.0) / 2.0 if -1 <= kt <= 1.0 else 0.5, 0.0, 1.0))

    # 12: feg_amount (-96 to +96 -> 0..1)
    f_env = xml_p.get("a_filter1_envmod", 0.0)
    feg_amount = float(np.clip((f_env + 96.0) / 192.0, 0.0, 1.0))

    # 13: feg_decay (-8 to +5 -> 0..1)
    fe_d = xml_p.get("a_env2_decay", -2.0)
    feg_decay = float(np.clip((fe_d + 8.0) / 13.0, 0.02, 0.85))

    # 14: feg_sustain (0..1)
    fe_s = xml_p.get("a_env2_sustain", 0.0)
    feg_sustain = float(np.clip(fe_s if 0 <= fe_s <= 1.0 else 0.0, 0.0, 1.0))

    # 15: aeg_decay (-8 to +5 -> 0..1)
    ae_d = xml_p.get("a_env1_decay", -0.75)
    aeg_decay = float(np.clip((ae_d + 8.0) / 13.0, 0.02, 0.85))

    # 16: aeg_sustain (0..1)
    ae_s = xml_p.get("a_env1_sustain", 1.0)
    aeg_sustain = float(np.clip(ae_s if 0 <= ae_s <= 1.0 else 1.0, 0.0, 1.0))

    # 17: aeg_release (-8 to +5 -> 0..1)
    ae_r = xml_p.get("a_env1_release", -3.75)
    aeg_release = float(np.clip((ae_r + 8.0) / 13.0, 0.01, 0.85))

    # 18: waveshaper_type (0 to 5)
    ws = xml_p.get("a_ws_type", 0.0)
    ws_idx = float(np.clip(ws / 5.0 if 0 <= ws <= 5.0 else 0.0, 0.0, 1.0))

    # 19: drive (0..1)
    drv = xml_p.get("a_ws_drive", 0.0)
    drive = float(np.clip(drv / 24.0 if 0 <= drv <= 24.0 else 0.0, 0.0, 1.0))

    # 20: chorus_mix, 21: delay_mix, 22: delay_fb (dry)
    chorus_mix = 0.0
    delay_mix = 0.0
    delay_fb = 0.0

    return np.array([
        midi_note, filter_type, shape, width, sub_mix, sync, fm_depth,
        unison, unison_detune, cutoff, resonance, keytrack, feg_amount,
        feg_decay, feg_sustain, aeg_decay, aeg_sustain, aeg_release,
        ws_idx, drive, chorus_mix, delay_mix, delay_fb
    ], dtype=np.float32)

def main():
    files = collect_fxp_files()
    print(f"Collected {len(files)} bass preset files.")
    vectors = []
    names = []
    for f in files:
        xml_p = parse_fxp_to_xml(f)
        if xml_p is not None:
            vec = extract_preset_vector(xml_p)
            vectors.append(vec)
            names.append(os.path.basename(f))

    vectors = np.array(vectors, dtype=np.float32)
    print(f"Extracted parameter matrix shape: {vectors.shape}")

    mean = np.mean(vectors, axis=0)
    cov = np.cov(vectors, rowvar=False)
    p10 = np.percentile(vectors, 10, axis=0)
    p50 = np.percentile(vectors, 50, axis=0)
    p90 = np.percentile(vectors, 90, axis=0)
    zero_prob = (vectors == 0).mean(axis=0)

    PARAM_NAMES = [
        "midi_note", "filter_type", "shape", "width", "sub_mix", "sync", "fm_depth",
        "unison", "unison_detune", "cutoff", "resonance", "keytrack", "feg_amount",
        "feg_decay", "feg_sustain", "aeg_decay", "aeg_sustain", "aeg_release",
        "waveshaper_type", "drive", "chorus_mix", "delay_mix", "delay_fb"
    ]

    print("\n--- Empirical Bass Manifold Statistics ---")
    for i, name in enumerate(PARAM_NAMES):
        print(f"  {name:18s}: mean={mean[i]:.3f}, std={np.sqrt(cov[i, i]):.3f}, p10={p10[i]:.3f}, p50={p50[i]:.3f}, p90={p90[i]:.3f}, zero_pct={zero_prob[i]*100:5.1f}%")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        params=vectors,
        preset_names=names,
        param_names=PARAM_NAMES,
        mean=mean,
        cov=cov,
        p10=p10,
        p50=p50,
        p90=p90,
        zero_prob=zero_prob
    )
    print(f"\nSaved empirical bass manifold to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
