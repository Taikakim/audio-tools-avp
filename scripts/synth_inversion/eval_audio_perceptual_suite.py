#!/usr/bin/env python3
# Owner: ANTIGRAVITY (created 2026-10-13)
"""Perceptual Audio Evaluation Suite for Surge XT Neural Inversion.

Computes objective perceptual and DSP metrics comparing neural synth inversions
against real ground-truth audio stems (24 test stems):
1. 9-Band Spectral Balance Cosine Distance (SonicMaster EQ balance)
2. Sub-Bass Energy Difference (20-60 Hz)
3. Bass Energy Difference (20-250 Hz)
4. Mel-SSIM (Spectrogram structural similarity on 128 mel bins, [0, 1])
5. Transient Punch Error (librosa onset strength error)
6. Dynamic Range RMS Std Error (dynamic consistency)
7. Modulation Spectrum Distance (envelope modulation frequency match)
8. Spectral Flatness Error (harmonic-to-noise ratio consistency)
"""

import os
import sys
import glob
import json
import argparse
import multiprocessing as mp
import soundfile as sf
import numpy as np
import librosa
from scipy.spatial.distance import cosine, euclidean
from scipy.signal import stft as scipy_stft
from skimage.metrics import structural_similarity as ssim

# Default evaluation arms and directories
DEFAULT_ARMS = {
    "A0_base": "/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a0_seeded/eval_cpu_muscriptor_temps/audio",
    "A1_inharm": "/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a1/eval_cpu_muscriptor_temps/audio",
    "A2_asp": "/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a2/eval_cpu_muscriptor_temps/audio",
    "A3_asp_inh": "/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a3/eval_cpu_muscriptor_temps/audio",
    "A3_prime": "/run/media/kim/Mantu/surge_200k_models/real_bass_v3_a3_prime/eval_cpu_muscriptor_temps/audio",
}

SPECTRAL_BANDS = [
    (20, 60),      # Sub-bass
    (60, 250),     # Bass
    (250, 500),    # Low-mid / warmth
    (500, 2000),   # Mid
    (2000, 4000),  # High-mid / presence
    (4000, 6000),  # Low-treble
    (6000, 10000), # Treble / bite
    (10000, 16000),# High treble / sizzle
    (16000, 20000) # Air
]


def modulation_spectrum_distance(x1, x2, fs=44100, n_fft=1024, hop_length=512, n_mod_bins=20):
    """Modulation spectrum distance between two signals (envelope modulation FFT)."""
    def get_modulation_spectrum(x):
        _, _, Zxx = scipy_stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)
        mod_spec = []
        for band in mag:
            env = band - np.mean(band)
            spec = np.abs(np.fft.fft(env))[:n_mod_bins]
            mod_spec.append(spec)
        mod_spec = np.array(mod_spec)
        mod_spec /= (np.linalg.norm(mod_spec) + 1e-10)
        return mod_spec.flatten()

    mod1 = get_modulation_spectrum(x1)
    mod2 = get_modulation_spectrum(x2)
    return float(euclidean(mod1, mod2))


def evaluate_audio_pair(real_path, cand_path):
    """Compute full perceptual suite for a single (real, cand) pair."""
    y_real, sr_real = sf.read(real_path)
    y_cand, sr_cand = sf.read(cand_path)

    if sr_real != sr_cand:
        y_cand = librosa.resample(y_cand, orig_sr=sr_cand, target_sr=sr_real)

    sr = sr_real

    # Convert to mono
    if y_real.ndim > 1:
        y_real = y_real.mean(axis=1)
    if y_cand.ndim > 1:
        y_cand = y_cand.mean(axis=1)

    min_len = min(len(y_real), len(y_cand))
    if min_len < 2048:
        raise ValueError(f"Audio too short: {min_len} samples")

    y_real = y_real[:min_len].astype(np.float32)
    y_cand = y_cand[:min_len].astype(np.float32)

    # 1. 9-band spectral profile & distances
    S_real = np.abs(librosa.stft(y_real, n_fft=2048, hop_length=512)) ** 2
    S_cand = np.abs(librosa.stft(y_cand, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    p_real = [np.sum(S_real[np.logical_and(freqs >= lo, freqs < hi), :]) for lo, hi in SPECTRAL_BANDS]
    p_cand = [np.sum(S_cand[np.logical_and(freqs >= lo, freqs < hi), :]) for lo, hi in SPECTRAL_BANDS]
    tot_real = np.sum(p_real) + 1e-10
    tot_cand = np.sum(p_cand) + 1e-10
    p_real_norm = np.array(p_real) / tot_real
    p_cand_norm = np.array(p_cand) / tot_cand

    spectral_cosine = float(cosine(p_real_norm, p_cand_norm))
    spectral_euclidean = float(euclidean(p_real_norm, p_cand_norm))

    sub_bass_real = float(p_real_norm[0])
    sub_bass_cand = float(p_cand_norm[0])
    sub_bass_err = abs(sub_bass_real - sub_bass_cand)

    bass_real = float(p_real_norm[0] + p_real_norm[1])
    bass_cand = float(p_cand_norm[0] + p_cand_norm[1])
    bass_err = abs(bass_real - bass_cand)

    # 2. Mel-SSIM (log power mel spectrogram)
    mel_real = librosa.power_to_db(librosa.feature.melspectrogram(y=y_real, sr=sr, n_fft=2048, hop_length=512, n_mels=128), ref=np.max)
    mel_cand = librosa.power_to_db(librosa.feature.melspectrogram(y=y_cand, sr=sr, n_fft=2048, hop_length=512, n_mels=128), ref=np.max)
    m_min = min(mel_real.shape[1], mel_cand.shape[1])
    mel_real = mel_real[:, :m_min]
    mel_cand = mel_cand[:, :m_min]

    norm_real = (mel_real - mel_real.min()) / (mel_real.max() - mel_real.min() + 1e-8)
    norm_cand = (mel_cand - mel_cand.min()) / (mel_cand.max() - mel_cand.min() + 1e-8)
    mel_ssim = float(ssim(norm_real, norm_cand, data_range=1.0))

    # 3. Transient Punch (Onset strength)
    onset_real = float(np.mean(librosa.onset.onset_strength(y=y_real, sr=sr)))
    onset_cand = float(np.mean(librosa.onset.onset_strength(y=y_cand, sr=sr)))
    punch_err = abs(onset_real - onset_cand) / (onset_real + 1e-10)

    # 4. Dynamic Range (RMS standard deviation)
    frames_real = librosa.util.frame(y_real, frame_length=2048, hop_length=1024)
    frames_cand = librosa.util.frame(y_cand, frame_length=2048, hop_length=1024)
    rms_real_std = float(np.std(np.sqrt(np.mean(frames_real ** 2, axis=0))))
    rms_cand_std = float(np.std(np.sqrt(np.mean(frames_cand ** 2, axis=0))))
    dyn_range_err = abs(rms_real_std - rms_cand_std) / (rms_real_std + 1e-10)

    # 5. Spectral Flatness
    flat_real = float(np.mean(librosa.feature.spectral_flatness(y=y_real)))
    flat_cand = float(np.mean(librosa.feature.spectral_flatness(y=y_cand)))
    flatness_err = abs(flat_real - flat_cand) / (flat_real + 1e-10)

    # 6. Modulation Spectrum Distance
    mod_dist = modulation_spectrum_distance(y_real, y_cand, fs=sr)

    return {
        "spectral_cosine": spectral_cosine,
        "spectral_euclidean": spectral_euclidean,
        "sub_bass_err": sub_bass_err,
        "bass_err": bass_err,
        "mel_ssim": mel_ssim,
        "punch_err": punch_err,
        "dyn_range_err": dyn_range_err,
        "flatness_err": flatness_err,
        "mod_dist": mod_dist,
    }


def worker_eval_job(task):
    arm, stem_id, model_type, temp, variant, real_p, cand_p = task
    try:
        metrics = evaluate_audio_pair(real_p, cand_p)
        return {
            "success": True,
            "arm": arm,
            "stem_id": stem_id,
            "model_type": model_type,
            "temp": temp,
            "variant": variant,
            "metrics": metrics,
        }
    except Exception as e:
        return {
            "success": False,
            "arm": arm,
            "stem_id": stem_id,
            "model_type": model_type,
            "temp": temp,
            "variant": variant,
            "error": str(e),
        }


def collect_jobs(arms, temps=("t07",), model_types=("ema",), variants=("harm",)):
    jobs = []
    for arm_name, arm_dir in arms.items():
        if not os.path.isdir(arm_dir):
            print(f"WARNING: Directory for {arm_name} does not exist: {arm_dir}")
            continue

        real_files = glob.glob(os.path.join(arm_dir, "*_real.wav"))
        for real_p in sorted(real_files):
            stem_base = os.path.basename(real_p).replace("_real.wav", "")
            for m in model_types:
                for t in temps:
                    for v in variants:
                        cand_name = f"{stem_base}_{m}_{t}_{v}.wav"
                        cand_p = os.path.join(arm_dir, cand_name)
                        if os.path.exists(cand_p):
                            jobs.append((arm_name, stem_base, m, t, v, real_p, cand_p))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temps", nargs="+", default=["t03", "t05", "t07", "t085", "t10"], help="Temperatures to evaluate")
    parser.add_argument("--primary_temp", default="t07", help="Primary temperature for main comparison table")
    parser.add_argument("--model_types", nargs="+", default=["ema", "online"], help="Model weights (ema, online)")
    parser.add_argument("--variants", nargs="+", default=["harm", "raw"], help="Variants (harm, raw)")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel worker threads")
    parser.add_argument("--out_json", default=os.path.join(os.path.dirname(__file__), "audio_perceptual_eval_results.json"), help="Path to save detailed JSON")
    parser.add_argument("--out_csv", default=os.path.join(os.path.dirname(__file__), "audio_perceptual_eval_summary.csv"), help="Path to save summary CSV")
    args = parser.parse_args()

    print(f"Collecting evaluation jobs across {len(DEFAULT_ARMS)} arms...")
    jobs = collect_jobs(DEFAULT_ARMS, temps=args.temps, model_types=args.model_types, variants=args.variants)
    print(f"Total comparison pairs to compute: {len(jobs)}")

    if not jobs:
        print("No audio pairs found!")
        return

    print(f"Executing with {args.num_workers} parallel workers...")
    with mp.Pool(processes=args.num_workers) as pool:
        results = pool.map(worker_eval_job, jobs)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"WARNING: {len(failures)} jobs failed:")
        for f in failures[:5]:
            print(f"  {f['arm']} {f['stem_id']} {f['temp']}: {f.get('error')}")

    print(f"Successfully computed metrics for {len(successes)} audio pairs.\n")

    # Aggregate by (arm, model_type, temp, variant)
    grouped = {}
    for r in successes:
        key = (r["arm"], r["model_type"], r["temp"], r["variant"])
        grouped.setdefault(key, []).append(r)

    summary_rows = []
    for (arm, m, t, v), items in sorted(grouped.items()):
        row = {
            "arm": arm,
            "model_type": m,
            "temp": t,
            "variant": v,
            "count": len(items),
        }
        for metric_name in [
            "mel_ssim", "spectral_cosine", "sub_bass_err", "bass_err",
            "punch_err", "dyn_range_err", "flatness_err", "mod_dist"
        ]:
            vals = [item["metrics"][metric_name] for item in items]
            row[f"{metric_name}_mean"] = float(np.mean(vals))
            row[f"{metric_name}_std"] = float(np.std(vals))
        summary_rows.append(row)

    # Save detailed JSON and CSV
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary_rows, "details": successes}, f, indent=2)
    print(f"Detailed results saved to: {args.out_json}")

    # CSV Export
    import csv
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV saved to: {args.out_csv}\n")

    # Format Markdown Table for Primary Temp (EMA, harm)
    print(f"=== PRIMARY COMPARISON TABLE: {args.primary_temp.upper()} (EMA, Harmonic Snapped) ===")
    print(f"{'Arm':12s} | {'Mel-SSIM (↑)':12s} | {'Spec-Cos (↓)':12s} | {'Sub-Bass Err (↓)':16s} | {'Bass Err (↓)':12s} | {'Punch Err (↓)':13s} | {'Mod Dist (↓)':12s}")
    print("-" * 96)

    for arm in DEFAULT_ARMS.keys():
        key = (arm, "ema", args.primary_temp, "harm")
        match = next((r for r in summary_rows if (r["arm"], r["model_type"], r["temp"], r["variant"]) == key), None)
        if match:
            print(f"{arm:12s} | {match['mel_ssim_mean']:6.4f} ±{match['mel_ssim_std']:<4.2f} | "
                  f"{match['spectral_cosine_mean']:6.4f} ±{match['spectral_cosine_std']:<4.2f} | "
                  f"{match['sub_bass_err_mean']:6.4f} ±{match['sub_bass_err_std']:<4.2f}         | "
                  f"{match['bass_err_mean']:6.4f} ±{match['bass_err_std']:<4.2f} | "
                  f"{match['punch_err_mean']:6.4f} ±{match['punch_err_std']:<4.2f} | "
                  f"{match['mod_dist_mean']:6.4f} ±{match['mod_dist_std']:<4.2f}")

    print("\n=== TEMPERATURE SENSITIVITY CURVE (Mel-SSIM ↑ across arms) ===")
    header_temps = " | ".join([f"{t:^8s}" for t in args.temps])
    print(f"{'Arm':12s} | {header_temps}")
    print("-" * (15 + 11 * len(args.temps)))
    for arm in DEFAULT_ARMS.keys():
        row_str = f"{arm:12s} | "
        cols = []
        for t in args.temps:
            key = (arm, "ema", t, "harm")
            match = next((r for r in summary_rows if (r["arm"], r["model_type"], r["temp"], r["variant"]) == key), None)
            if match:
                cols.append(f"{match['mel_ssim_mean']:6.4f}  ")
            else:
                cols.append("  N/A    ")
        row_str += " | ".join(cols)
        print(row_str)


if __name__ == "__main__":
    main()
