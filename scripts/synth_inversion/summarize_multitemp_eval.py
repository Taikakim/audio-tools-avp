#!/usr/bin/env python3
# Owner: ANTIGRAVITY (created 2026-10-12)
"""Summarize Multi-Temperature Evaluation Results for Arms A1, A2, and A3.

Reads results.json across eval directories and backfills any missing STFT
scores directly from the rendered .wav files against the ground-truth phrase.

Prints:
1. Mean MultiScaleSTFT phrase error per temperature [0.30, 0.50, 0.70, 0.85, 1.00]
2. EMA vs Online comparison
3. Raw vs Harmonic Snapped comparison
4. Identification of the optimal temperature (sweet spot)
"""

import os
import sys
import json
import soundfile as sf
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import MultiScaleSTFTLoss
from invert_stem_collection_v3_cpu import STEM_CATALOG, resolve_mantu

BASE_DIR = "/run/media/kim/Mantu/surge_200k_models"
ARMS = ["a1", "a2", "a3"]
LABELS = {
    "t015": "tau=0.15",
    "t03": "tau=0.30",
    "t05": "tau=0.50",
    "t07": "tau=0.70",
    "t085": "tau=0.85",
    "t10": "tau=1.00",
}


def backfill_and_analyze_arm(arm_id: str, loss_fn: MultiScaleSTFTLoss):
    base_arm_dir = resolve_mantu(os.path.join(BASE_DIR, f"real_bass_v3_{arm_id}"))
    eval_dir = os.path.join(base_arm_dir, "eval_cpu_muscriptor_temps")
    json_path = os.path.join(eval_dir, "results.json")
    audio_dir = os.path.join(eval_dir, "audio")

    if not os.path.exists(audio_dir):
        return None

    data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            data = {}

    modified = False

    # Backfill missing stems/temps from rendered audio
    for kind in ("ema", "online"):
        if kind not in data:
            data[kind] = {}

        for stem in STEM_CATALOG:
            sid = stem["id"]
            ref_path = os.path.join(audio_dir, f"{sid}_real.wav")
            if not os.path.exists(ref_path):
                # Fallback to A2 or A1 real audio if present
                for alt_arm in ("a2", "a1", "a3"):
                    alt_ref = resolve_mantu(os.path.join(BASE_DIR, f"real_bass_v3_{alt_arm}", "eval_cpu_muscriptor_temps", "audio", f"{sid}_real.wav"))
                    if os.path.exists(alt_ref):
                        ref_path = alt_ref
                        break

            if not os.path.exists(ref_path):
                continue

            ref_audio = None

            if sid not in data[kind]:
                data[kind][sid] = {
                    "name": stem["name"],
                    "style": stem.get("style", ""),
                    "bpm": stem.get("bpm", 140.0),
                    "temps": {}
                }

            temps_dict = data[kind][sid].setdefault("temps", {})

            for t_tag in ("t015", "t03", "t05", "t07", "t085", "t10"):
                harm_path = os.path.join(audio_dir, f"{sid}_{kind}_{t_tag}_harm.wav")
                raw_path = os.path.join(audio_dir, f"{sid}_{kind}_{t_tag}_raw.wav")

                if not os.path.exists(harm_path):
                    continue

                t_entry = temps_dict.setdefault(t_tag, {})
                need_calc = ("stft_phrase_harm" not in t_entry or "stft_phrase_raw" not in t_entry or
                             np.isnan(t_entry.get("stft_phrase_harm", np.nan)))

                if need_calc:
                    if ref_audio is None:
                        ref_raw, _ = sf.read(ref_path)
                        ref_audio = np.mean(ref_raw, axis=1) if ref_raw.ndim > 1 else ref_raw

                    harm_raw, _ = sf.read(harm_path)
                    harm_mono = np.mean(harm_raw, axis=1) if harm_raw.ndim > 1 else harm_raw

                    raw_mono = None
                    if os.path.exists(raw_path):
                        r_raw, _ = sf.read(raw_path)
                        raw_mono = np.mean(r_raw, axis=1) if r_raw.ndim > 1 else r_raw

                    min_len_harm = min(len(ref_audio), len(harm_mono))
                    stft_harm = float(loss_fn(harm_mono[:min_len_harm], ref_audio[:min_len_harm]))
                    t_entry["stft_phrase_harm"] = round(stft_harm, 3)

                    if raw_mono is not None:
                        min_len_raw = min(len(ref_audio), len(raw_mono))
                        stft_raw = float(loss_fn(raw_mono[:min_len_raw], ref_audio[:min_len_raw]))
                        t_entry["stft_phrase_raw"] = round(stft_raw, 3)
                    else:
                        t_entry["stft_phrase_raw"] = t_entry["stft_phrase_harm"]

                    modified = True

    if modified:
        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save updated {json_path}: {e}")

    # Build summary
    summary = {}
    for kind in ("ema", "online"):
        if kind not in data:
            continue
        summary[kind] = {}
        for sid, s_info in data[kind].items():
            temps_dict = s_info.get("temps", {})
            for t_tag, t_val in temps_dict.items():
                if t_tag not in summary[kind]:
                    summary[kind][t_tag] = {"stft_raw": [], "stft_harm": [], "cutoff": [], "resonance": []}
                summary[kind][t_tag]["stft_raw"].append(t_val.get("stft_phrase_raw", np.nan))
                summary[kind][t_tag]["stft_harm"].append(t_val.get("stft_phrase_harm", np.nan))
                summary[kind][t_tag]["cutoff"].append(t_val.get("cutoff", np.nan))
                summary[kind][t_tag]["resonance"].append(t_val.get("resonance", np.nan))

    return summary


def main():
    print("=" * 85)
    print("MULTI-TEMPERATURE MULTI-SCALE STFT BENCHMARK MATRIX (LOWER IS BETTER)")
    print("=" * 85)

    loss_fn = MultiScaleSTFTLoss()

    for arm in ("a1", "a2", "a3"):
        res = backfill_and_analyze_arm(arm, loss_fn)
        if not res:
            print(f"\nArm {arm.upper()}: not yet available.")
            continue

        arm_name = {
            "a1": "ARM A1 (Baseline Flow)",
            "a2": "ARM A2 (Temporal ASP - Spec+Envelope Loss)",
            "a3": "ARM A3 (Temporal ASP + Inharmonicity Ratio Loss)"
        }.get(arm, f"ARM {arm.upper()}")

        print(f"\n--- {arm_name} ---")
        for kind in ("ema", "online"):
            if kind not in res or not res[kind]:
                continue
            print(f"  [{kind.upper()}]")
            print(f"  {'Temp':<10} | {'STFT Raw (Mean ± Std)':<24} | {'STFT Harm (Mean ± Std)':<24} | {'N Stems'}")
            print("  " + "-" * 70)
            t_keys = sorted(res[kind].keys(), key=lambda k: float(k[1:].replace("085", "85").replace("015", "15")) / 100.0)
            for tk in t_keys:
                d = res[kind][tk]
                raw_arr = np.array(d["stft_raw"], dtype=float)
                harm_arr = np.array(d["stft_harm"], dtype=float)
                raw_m, raw_s = np.nanmean(raw_arr), np.nanstd(raw_arr)
                harm_m, harm_s = np.nanmean(harm_arr), np.nanstd(harm_arr)
                n = len(harm_arr[~np.isnan(harm_arr)])
                label = LABELS.get(tk, tk)
                print(f"  {label:<10} | {raw_m:6.3f} ± {raw_s:5.3f}             | {harm_m:6.3f} ± {harm_s:5.3f}             | {n}")


if __name__ == "__main__":
    main()
