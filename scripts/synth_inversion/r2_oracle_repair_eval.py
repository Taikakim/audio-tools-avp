#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Pre-Flight Check R2: Oracle Repair Benchmark.

Tests whether contested sub-bass cells (40-150 Hz under kick onsets) are the true
bottleneck for Demucs degradation on synth inversion:
1. Mixes 20 known Surge patches with authentic Goa/psytrance kick stems at 0 dB SNR.
2. Separates with Demucs (htdemucs).
3. Forms Oracle-Repaired stems by replacing separated audio with clean ground truth
   EXCLUSIVELY in the contested time-frequency bins (kick-on frames, 40-150 Hz).
4. Runs V6 C flow inverter across all 4 conditions:
   - (a) Clean Ground Truth (Ceiling)
   - (b) Demucs Separated (Current baseline)
   - (c) Oracle Repaired (Pillar 2 theoretical ceiling)
   - (d) Raw Mixture (Floor)
5. Computes continuous parameter MSE, cutoff error, and the Repair Recovery Ratio (R).
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa

from demucs.pretrained import get_model
from demucs.apply import apply_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v4 import load_inverter_v4
from audio_utils import ExactGpuMel
import param_codec_v6 as codec
from surge_spec_v6 import NUM_PARAMS, PARAM_INDEX, FLOW_CONT_INDICES


def create_oracle_repair(
    clean_wav: np.ndarray,
    demucs_wav: np.ndarray,
    kick_wav: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 256,
    sub_band_range: tuple = (40.0, 150.0),
    kick_window_ms: float = 30.0,
) -> np.ndarray:
    """Replaces Demucs STFT with clean STFT only in contested sub-bass bins during kick transients."""
    # Detect kick onsets
    onset_samples = librosa.onset.onset_detect(
        y=kick_wav, sr=sr, hop_length=hop_length, backtrack=False, units="samples"
    )

    # Compute complex STFTs
    S_clean = librosa.stft(clean_wav, n_fft=n_fft, hop_length=hop_length)
    S_demucs = librosa.stft(demucs_wav, n_fft=n_fft, hop_length=hop_length)
    
    # Frequency mask (40 - 150 Hz)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    f_mask = (freqs >= sub_band_range[0]) & (freqs <= sub_band_range[1])

    # Time mask (+- kick_window_ms around each kick onset)
    t_mask = np.zeros(S_demucs.shape[1], dtype=bool)
    half_win_samples = int(kick_window_ms * 1e-3 * sr)
    for onset in onset_samples:
        start_samp = max(0, onset - half_win_samples)
        end_samp = min(len(kick_wav), onset + half_win_samples)
        start_frame = start_samp // hop_length
        end_frame = min(S_demucs.shape[1], (end_samp // hop_length) + 1)
        t_mask[start_frame:end_frame] = True

    # 2D Contested Mask: where frequency is 40-150 Hz AND time is within kick transient
    contested_mask = np.outer(f_mask, t_mask)

    # Oracle splicing
    S_oracle = S_demucs.copy()
    S_oracle[contested_mask] = S_clean[contested_mask]

    # Invert back to time-domain
    oracle_wav = librosa.istft(S_oracle, hop_length=hop_length, length=len(clean_wav))
    return oracle_wav.astype(np.float32)


def main():
    device = "cpu"
    print("=" * 80)
    print("Pre-Flight Check R2: Oracle Repair Benchmark (Is Sub-Bass the Bottleneck?)")
    print("=" * 80)

    # 1. Load V6 C inverter
    v6c_path = "/home/kim/sao_runs/v6_C/flow_best.pt"
    if not os.path.exists(v6c_path):
        print(f"Error: V6 C checkpoint not found at {v6c_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading V6 C flow model from {v6c_path}...")
    flow = load_inverter_v4(v6c_path, device=device).eval()

    mel_fn = ExactGpuMel(device=device)

    # 2. Load Demucs
    print("Loading Demucs (htdemucs)...")
    demucs_model = get_model("htdemucs").to(device).eval()
    bass_idx = demucs_model.sources.index("bass")

    # 3. Load 20 known test patches from bank_v6
    bank_dir = "/home/kim/sao_runs/bank_v6"
    audio_path = os.path.join(bank_dir, "main_00000_audio.npy")
    meta_path = os.path.join(bank_dir, "main_00000_meta.npz")

    audio_raw = np.load(audio_path, mmap_mode="r")
    meta = np.load(meta_path)

    n_test = 20
    clean_audio = audio_raw[:n_test].astype(np.float32) / 32768.0  # [N, 35280]
    true_vecs = torch.from_numpy(meta["vec"][:n_test].astype(np.float32))  # [N, 59]
    midi_notes = meta["midi"][:n_test]
    norm_notes = torch.from_numpy(((midi_notes - 28) / 22.0).astype(np.float32))

    # 4. Load authentic kick stems from AVP multitrack archive
    kick_path = "/run/media/kim/Mantu/avp-stems-original-classified/Summamutikka - Acid Alien aggressive powerful psychedelic melodic goa trance/kick/KICK_downmix.flac"
    kick_wav, sr = sf.read(kick_path, dtype="float32")
    if kick_wav.ndim > 1:
        kick_wav = kick_wav.mean(axis=1)
    if sr != 44100:
        kick_wav = librosa.resample(kick_wav, orig_sr=sr, target_sr=44100)

    # Crop an active 0.8s slice with multiple kick hits
    kick_slice = kick_wav[44100 * 48 : 44100 * 48 + 35280]
    kick_rms = np.sqrt(np.mean(kick_slice ** 2)) + 1e-6

    print(f"Loaded {n_test} ground truth test patches and authentic kick stem.")
    print("Evaluating across 4 conditions: Clean, Demucs, Oracle Repaired, Raw Mix...\n")

    cutoff_idx = PARAM_INDEX["cutoff"]

    mse_clean = []
    mse_demucs = []
    mse_oracle = []
    mse_mix = []

    cutoff_clean = []
    cutoff_demucs = []
    cutoff_oracle = []
    cutoff_mix = []

    for i in range(n_test):
        c_np = clean_audio[i]
        c_audio = torch.from_numpy(c_np)
        target_vec = true_vecs[i]
        note = norm_notes[i : i + 1]

        # Match kick RMS to synth RMS for 0 dB SNR
        synth_rms = np.sqrt(np.mean(c_np ** 2)) + 1e-6
        k_scaled = kick_slice * (synth_rms / kick_rms)
        mix_np = c_np + k_scaled

        # Normalize mixture
        max_val = max(1.0, float(np.abs(mix_np).max()))
        mix_np = mix_np / max_val
        c_np_scaled = c_np / max_val
        k_np_scaled = k_scaled / max_val

        # (a) Clean Mel
        mel_clean = mel_fn(torch.from_numpy(c_np_scaled)).unsqueeze(0)

        # (b) Demucs Separation
        mix_stereo = torch.from_numpy(mix_np).unsqueeze(0).repeat(2, 1).unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(demucs_model, mix_stereo, shifts=0, split=True, overlap=0.25)
        demucs_np = sources[0, bass_idx].mean(dim=0).numpy()
        mel_demucs = mel_fn(torch.from_numpy(demucs_np)).unsqueeze(0)

        # (c) Oracle Repaired Audio
        oracle_np = create_oracle_repair(c_np_scaled, demucs_np, k_np_scaled)
        mel_oracle = mel_fn(torch.from_numpy(oracle_np)).unsqueeze(0)

        # (d) Raw Mix Mel
        mel_mix = mel_fn(torch.from_numpy(mix_np)).unsqueeze(0)

        # Invert with V6 C Flow
        def run_inverter(mel_input, seed):
            gen = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                pred_enc = flow.sample(mel_input, note=note, num_steps=25, generator=gen)
                pred = codec.decode(pred_enc, note)[0]
            return pred

        pred_c = run_inverter(mel_clean, 1337 + i)
        pred_d = run_inverter(mel_demucs, 1337 + i)
        pred_o = run_inverter(mel_oracle, 1337 + i)
        pred_m = run_inverter(mel_mix, 1337 + i)

        cont_true = target_vec[FLOW_CONT_INDICES]
        err_c = (pred_c[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        err_d = (pred_d[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        err_o = (pred_o[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        err_m = (pred_m[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()

        mse_clean.append(err_c)
        mse_demucs.append(err_d)
        mse_oracle.append(err_o)
        mse_mix.append(err_m)

        cutoff_clean.append(abs(pred_c[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_demucs.append(abs(pred_d[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_oracle.append(abs(pred_o[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_mix.append(abs(pred_m[cutoff_idx].item() - target_vec[cutoff_idx].item()))

        print(f"Sample {i+1:02d} | Clean MSE: {err_c:.4f} | Demucs: {err_d:.4f} | Oracle: {err_o:.4f} | Mix: {err_m:.4f}")

    # Summary
    mean_c = np.mean(mse_clean)
    mean_d = np.mean(mse_demucs)
    mean_o = np.mean(mse_oracle)
    mean_m = np.mean(mse_mix)

    total_damage = mean_d - mean_c
    recovered = mean_d - mean_o
    recovery_ratio = (recovered / (total_damage + 1e-8)) * 100.0

    mean_cut_c = np.mean(cutoff_clean)
    mean_cut_d = np.mean(cutoff_demucs)
    mean_cut_o = np.mean(cutoff_oracle)
    mean_cut_m = np.mean(cutoff_mix)

    print("\n" + "=" * 80)
    print("ORACLE REPAIR BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Condition                  | Continuous Parameter MSE | Cutoff Absolute Error")
    print("-" * 80)
    print(f"(a) Clean Ground Truth     | {mean_c:.5f}                  | {mean_cut_c:.4f}")
    print(f"(b) Demucs Separation      | {mean_d:.5f} (+{(mean_d/mean_c - 1)*100:+.1f}%)         | {mean_cut_d:.4f} (+{(mean_cut_d/mean_cut_c - 1)*100:+.1f}%)")
    print(f"(c) Oracle Repaired        | {mean_o:.5f}                  | {mean_cut_o:.4f}")
    print(f"(d) Raw Mixture (Floor)    | {mean_m:.5f}                  | {mean_cut_m:.4f}")
    print("-" * 80)
    print(f"Demucs Damage (Demucs - Clean): {total_damage:+.5f}")
    print(f"Oracle Recovered:               {recovered:+.5f}")
    print(f"Repair Recovery Ratio (R):      {recovery_ratio:5.1f}%")
    print("=" * 80)

    verdict = "PASSED (Sub-bass masking under kick is confirmed dominant bottleneck)" if recovery_ratio >= 50.0 else "INCONCLUSIVE / FAILED (Damage is driven by other factors: bleed, phase, or high bands)"
    print(f"Verdict: {verdict}")

    out_file = Path("/home/kim/Projects/SAO/eval/r2_oracle_repair_results.json")
    with open(out_file, "w") as f:
        json.dump({
            "mean_mse_clean": mean_c,
            "mean_mse_demucs": mean_d,
            "mean_mse_oracle": mean_o,
            "mean_mse_mix": mean_m,
            "recovery_ratio_pct": recovery_ratio,
            "mean_cutoff_clean": mean_cut_c,
            "mean_cutoff_demucs": mean_cut_d,
            "mean_cutoff_oracle": mean_cut_o,
            "mean_cutoff_mix": mean_cut_m,
            "verdict": verdict,
        }, f, indent=2)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
