#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Phase 1: Asymmetric Target Sound Extractor Evaluation & Separation Tax Gate.

Evaluates trained Asymmetric TSE (tse_best.pt) vs Demucs baseline on held-out songs:
1. Loads 20 known test patches from bank_v6.
2. Mixes with authentic held-out stems (kicks, drums, synths) at 0 dB SNR.
3. Performs extraction with V8 Asymmetric TSE (using 1.0s clean cue).
4. Performs separation with Demucs (htdemucs).
5. Inverts continuous parameters using V6 C flow inverter across:
   - (a) Clean Ground Truth (Ceiling)
   - (b) V8 TSE Extracted
   - (c) Demucs Separated (Baseline)
   - (d) Raw Mixture (Floor)
6. Computes Mel-SSIM distance, parameter MSE, cutoff error, and Separation Tax.
7. Evaluates the Phase 1 Gate: Mel-SSIM error <= 1.25x of clean stem.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from demucs.apply import apply_model
from demucs.pretrained import get_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_metrics_v5 import mel_ssim_dist, octave_db_l1
from audio_utils import ExactGpuMel
from models_v4 import load_inverter_v4
import param_codec_v6 as codec
from surge_spec_v6 import FLOW_CONT_INDICES, NUM_PARAMS, PARAM_INDEX
from train_v8_asymmetric_tse import AsymmetricTSEModel


import subprocess

GUARD_SCRIPT = "/home/kim/Projects/SAO/Misc/gpu_guard.sh"


def acquire_gpu_guard() -> bool:
    pid = str(os.getpid())
    env = os.environ.copy()
    env["TARGET_GPU"] = "0"
    res = subprocess.run(["bash", GUARD_SCRIPT, "acquire", "ANTIGRAVITY", pid], env=env)
    return res.returncode == 0


def release_gpu_guard():
    subprocess.run(["bash", GUARD_SCRIPT, "release", "ANTIGRAVITY"])


def run_eval(device: str):
    # 1. Load V8 Asymmetric TSE Model
    tse_ckpt_path = "/home/kim/sao_runs/v8_phase1_tse/tse_best.pt"
    if not os.path.exists(tse_ckpt_path):
        print(f"Error: TSE checkpoint not found at {tse_ckpt_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading V8 TSE model from {tse_ckpt_path}...")
    tse_model = AsymmetricTSEModel().to(device).eval()
    ckpt = torch.load(tse_ckpt_path, map_location=device, weights_only=False)
    tse_model.load_state_dict(ckpt["model"])
    print(f"Loaded TSE model (Epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 0.0):.4f}).")

    # 2. Load V6 C Inverter
    v6c_path = "/home/kim/sao_runs/v6_C/flow_best.pt"
    print(f"Loading V6 C inverter from {v6c_path}...")
    flow = load_inverter_v4(v6c_path, device=device).eval()
    mel_fn = ExactGpuMel(device=device)

    # 3. Load Demucs Baseline
    print("Loading Demucs (htdemucs)...")
    demucs_model = get_model("htdemucs").to(device).eval()
    bass_idx = demucs_model.sources.index("bass")

    # 4. Load Held-Out Stems from NVMe Cache
    cache_dir = Path("/home/kim/sao_runs/v8_stem_cache")
    val_kicks = np.load(cache_dir / "val_kicks.npy", mmap_mode="r")
    val_drums = np.load(cache_dir / "val_drums.npy", mmap_mode="r")
    val_synths = np.load(cache_dir / "val_synths.npy", mmap_mode="r")
    print(f"Loaded validation caches: {len(val_kicks)} kicks, {len(val_drums)} drums, {len(val_synths)} synths.")

    # 5. Load 20 known test patches from bank_v6
    bank_dir = "/home/kim/sao_runs/bank_v6"
    audio_raw = np.load(os.path.join(bank_dir, "main_00000_audio.npy"), mmap_mode="r")
    meta = np.load(os.path.join(bank_dir, "main_00000_meta.npz"))

    n_test = 20
    test_audio = audio_raw[:n_test].astype(np.float32) / 32768.0  # [N, 35280]
    true_vecs = torch.from_numpy(meta["vec"][:n_test].astype(np.float32))  # [N, 59]
    midi_notes = meta["midi"][:n_test]
    norm_notes = torch.from_numpy(((midi_notes - 28) / 22.0).astype(np.float32))

    clip_samples = 176400  # 4.0s
    cue_samples = 44100    # 1.0s
    cutoff_idx = PARAM_INDEX["cutoff"]

    results = {
        "mse_clean": [],
        "mse_tse": [],
        "mse_demucs": [],
        "mse_mix": [],
        "cutoff_clean": [],
        "cutoff_tse": [],
        "cutoff_demucs": [],
        "cutoff_mix": [],
        "ssim_tse": [],
        "ssim_demucs": [],
        "ssim_mix": [],
        "octave_tse": [],
        "octave_demucs": [],
        "octave_mix": [],
    }

    print(f"\nEvaluating {n_test} test patches across 4 conditions...\n")

    for i in range(n_test):
        patch_08s = test_audio[i]
        target_vec = true_vecs[i]
        note = norm_notes[i : i + 1].to(device)

        # 4.0s tiled target
        repeats = (clip_samples // len(patch_08s)) + 1
        clean_4s = np.tile(patch_08s, repeats)[:clip_samples]

        # 1.0s audio cue
        cue_wav = clean_4s[:cue_samples]

        # Sample held-out backing arrangement
        kick_bg = val_kicks[i % len(val_kicks)]
        drum_bg = val_drums[i % len(val_drums)]
        synth_bg = val_synths[i % len(val_synths)]

        backing = drum_bg * 0.6 + kick_bg * 0.4 + synth_bg * 0.3
        b_rms = np.sqrt(np.mean(backing ** 2)) + 1e-6
        t_rms = np.sqrt(np.mean(clean_4s ** 2)) + 1e-6

        # Mix at 0 dB SNR
        scaled_backing = backing * (t_rms / b_rms)
        mix_4s = clean_4s + scaled_backing

        # Peak normalize
        max_val = max(1.0, float(np.abs(mix_4s).max()))
        mix_4s = mix_4s / max_val
        clean_4s_norm = clean_4s / max_val
        cue_wav_norm = cue_wav / max_val

        # (a) V8 TSE Extraction
        t_mix = torch.from_numpy(mix_4s).float().unsqueeze(0).unsqueeze(0).to(device)
        t_cue = torch.from_numpy(cue_wav_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                pred_tse, rel_mask = tse_model(t_mix, t_cue)
        tse_4s = pred_tse.squeeze().float().cpu().numpy()

        # (b) Demucs Separation
        mix_stereo = torch.from_numpy(mix_4s).unsqueeze(0).repeat(2, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            sources = apply_model(demucs_model, mix_stereo, shifts=0, split=True, overlap=0.25)
        demucs_4s = sources[0, bass_idx].mean(dim=0).cpu().numpy()

        # Slice 0.8s for V6 Inversion
        slice_clean = clean_4s_norm[:35280]
        slice_tse = tse_4s[:35280]
        slice_demucs = demucs_4s[:35280]
        slice_mix = mix_4s[:35280]

        # Audio perceptual metrics
        d_ssim_tse = mel_ssim_dist(slice_clean, slice_tse, sr=44100)
        d_ssim_dem = mel_ssim_dist(slice_clean, slice_demucs, sr=44100)
        d_ssim_mix = mel_ssim_dist(slice_clean, slice_mix, sr=44100)

        d_oct_tse = octave_db_l1(slice_clean, slice_tse, sr=44100)
        d_oct_dem = octave_db_l1(slice_clean, slice_demucs, sr=44100)
        d_oct_mix = octave_db_l1(slice_clean, slice_mix, sr=44100)

        results["ssim_tse"].append(d_ssim_tse)
        results["ssim_demucs"].append(d_ssim_dem)
        results["ssim_mix"].append(d_ssim_mix)

        results["octave_tse"].append(d_oct_tse)
        results["octave_demucs"].append(d_oct_dem)
        results["octave_mix"].append(d_oct_mix)

        # Inversion with V6 C Flow
        def invert(wav_arr, seed):
            mel = mel_fn(torch.from_numpy(wav_arr).to(device)).unsqueeze(0)
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                pred_enc = flow.sample(mel, note=note, num_steps=25, generator=gen)
                pred = codec.decode(pred_enc, note)[0].cpu()
            return pred

        p_clean = invert(slice_clean, 1337 + i)
        p_tse = invert(slice_tse, 1337 + i)
        p_dem = invert(slice_demucs, 1337 + i)
        p_mix = invert(slice_mix, 1337 + i)

        cont_true = target_vec[FLOW_CONT_INDICES]
        e_c = (p_clean[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        e_t = (p_tse[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        e_d = (p_dem[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        e_m = (p_mix[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()

        results["mse_clean"].append(e_c)
        results["mse_tse"].append(e_t)
        results["mse_demucs"].append(e_d)
        results["mse_mix"].append(e_m)

        c_c = abs(p_clean[cutoff_idx].item() - target_vec[cutoff_idx].item())
        c_t = abs(p_tse[cutoff_idx].item() - target_vec[cutoff_idx].item())
        c_d = abs(p_dem[cutoff_idx].item() - target_vec[cutoff_idx].item())
        c_m = abs(p_mix[cutoff_idx].item() - target_vec[cutoff_idx].item())

        results["cutoff_clean"].append(c_c)
        results["cutoff_tse"].append(c_t)
        results["cutoff_demucs"].append(c_d)
        results["cutoff_mix"].append(c_m)

        print(
            f"Patch {i+1:02d} | Mel-SSIM: TSE={d_ssim_tse:.3f}, Demucs={d_ssim_dem:.3f} | "
            f"Param MSE: Clean={e_c:.4f}, TSE={e_t:.4f}, Demucs={e_d:.4f}"
        )

    # Aggregate summaries
    mean_mse_clean = np.mean(results["mse_clean"])
    mean_mse_tse = np.mean(results["mse_tse"])
    mean_mse_dem = np.mean(results["mse_demucs"])
    mean_mse_mix = np.mean(results["mse_mix"])

    mean_ssim_tse = np.mean(results["ssim_tse"])
    mean_ssim_dem = np.mean(results["ssim_demucs"])
    mean_ssim_mix = np.mean(results["ssim_mix"])

    mean_oct_tse = np.mean(results["octave_tse"])
    mean_oct_dem = np.mean(results["octave_demucs"])
    mean_oct_mix = np.mean(results["octave_mix"])

    mean_cut_clean = np.mean(results["cutoff_clean"])
    mean_cut_tse = np.mean(results["cutoff_tse"])
    mean_cut_dem = np.mean(results["cutoff_demucs"])
    mean_cut_mix = np.mean(results["cutoff_mix"])

    tax_ratio_mse = mean_mse_tse / mean_mse_clean
    tax_demucs_mse = mean_mse_dem / mean_mse_clean

    print("\n" + "=" * 80)
    print("PHASE 1 EVALUATION: ASYMMETRIC TSE VS DEMUCS BENCHMARK")
    print("=" * 80)
    print(f"Metric                     | Clean Ceiling | V8 TSE       | Demucs       | Raw Mixture")
    print("-" * 80)
    print(f"Continuous Param MSE       | {mean_mse_clean:.5f}       | {mean_mse_tse:.5f}      | {mean_mse_dem:.5f}      | {mean_mse_mix:.5f}")
    print(f"Separation Tax (MSE Ratio) | 1.000x        | {tax_ratio_mse:.3f}x       | {tax_demucs_mse:.3f}x       | {mean_mse_mix/mean_mse_clean:.3f}x")
    print(f"Mel-SSIM Distance (1-SSIM) | 0.000         | {mean_ssim_tse:.4f}       | {mean_ssim_dem:.4f}       | {mean_ssim_mix:.4f}")
    print(f"Octave Energy L1 (dB)      | 0.000         | {mean_oct_tse:.3f} dB      | {mean_oct_dem:.3f} dB      | {mean_oct_mix:.3f} dB")
    print(f"Cutoff Absolute Error      | {mean_cut_clean:.4f}        | {mean_cut_tse:.4f}       | {mean_cut_dem:.4f}       | {mean_cut_mix:.4f}")
    print("=" * 80)

    # Gate verification:
    gate_pass = tax_ratio_mse <= 1.25
    print(f"\nSeparation Tax Gate Status (Tax <= 1.25x): {'PASSED' if gate_pass else 'DID NOT PASS 1.25x (Tax=' + f'{tax_ratio_mse:.3f}x)'}")
    if mean_mse_tse < mean_mse_dem:
        print(f"--> TSE Outperforms Demucs: Parameter MSE reduced by {(1 - mean_mse_tse/mean_mse_dem)*100:.1f}%!")

    # Save output JSON
    out_dir = Path("/home/kim/Projects/SAO/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "v8_phase1_tse_results.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "mean_mse_clean": float(mean_mse_clean),
                "mean_mse_tse": float(mean_mse_tse),
                "mean_mse_demucs": float(mean_mse_dem),
                "mean_mse_mix": float(mean_mse_mix),
                "tax_ratio_mse": float(tax_ratio_mse),
                "tax_demucs_mse": float(tax_demucs_mse),
                "mean_ssim_tse": float(mean_ssim_tse),
                "mean_ssim_demucs": float(mean_ssim_dem),
                "mean_ssim_mix": float(mean_ssim_mix),
                "mean_oct_tse": float(mean_oct_tse),
                "mean_oct_demucs": float(mean_oct_dem),
                "mean_oct_mix": float(mean_oct_mix),
                "mean_cut_clean": float(mean_cut_clean),
                "mean_cut_tse": float(mean_cut_tse),
                "mean_cut_demucs": float(mean_cut_dem),
                "mean_cut_mix": float(mean_cut_mix),
                "gate_pass": bool(gate_pass),
            },
            f,
            indent=2,
        )
    print(f"Saved results to {summary_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("V8 Phase 1: Asymmetric TSE Evaluation & Separation Tax Benchmark")
    print(f"Device: {device}")
    print("=" * 80)

    if "cuda" in device:
        print("Checking GPU lock via gpu_guard.sh...")
        if not acquire_gpu_guard():
            print("GPU is currently busy. Refusing to run.")
            sys.exit(0)

    try:
        run_eval(device)
    finally:
        if "cuda" in device:
            release_gpu_guard()


if __name__ == "__main__":
    main()
