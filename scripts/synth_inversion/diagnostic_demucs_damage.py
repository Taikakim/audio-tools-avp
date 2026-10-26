# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Diagnostic Step 0: Measuring the exact parameter error cost of Demucs separation on ground truth.

Tests the core premise of V7 before spending GPU compute:
  Does Demucs separation actually damage Surge XT parameter recovery on known patches,
  or is the degradation negligible compared to clean renders?

Evaluates V6 C (flow_best.pt) on:
  (a) Clean Surge render (theoretical ceiling)
  (b) Demucs-separated render (what V6 currently gets on real stems)
  (c) Raw mixture directly (floor)
"""

import os
import sys
import time
import numpy as np
import torch
import torchaudio

from demucs.pretrained import get_model
from demucs.apply import apply_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v4 import load_inverter_v4
from audio_utils import ExactGpuMel
import param_codec_v6 as codec
from surge_spec_v6 import NUM_PARAMS, PARAM_INDEX, FLOW_CONT_INDICES


def main():
    device = "cpu"
    print(f"=== Step 0 Diagnostic: Measuring Demucs Degradation on Ground Truth ===")

    # 1. Load V6 C model
    v6c_path = "/home/kim/sao_runs/v6_C/flow_best.pt"
    if not os.path.exists(v6c_path):
        raise FileNotFoundError(f"V6 C checkpoint not found at {v6c_path}")
    print(f"Loading V6 C flow model from {v6c_path}...")
    flow = load_inverter_v4(v6c_path, device=device).eval()

    # Mel extractor
    mel_fn = ExactGpuMel(device=device)

    # 2. Load Demucs model
    print("Loading Demucs (htdemucs)...")
    demucs_model = get_model("htdemucs").to(device).eval()
    bass_idx = demucs_model.sources.index("bass")

    # 3. Load 20 known patches from bank_v6
    bank_dir = "/home/kim/sao_runs/bank_v6"
    audio_path = os.path.join(bank_dir, "main_00000_audio.npy")
    meta_path = os.path.join(bank_dir, "main_00000_meta.npz")

    audio_raw = np.load(audio_path, mmap_mode="r")
    meta = np.load(meta_path)

    n_test = 20
    clean_audio = audio_raw[:n_test].astype(np.float32) / 32768.0  # [N, 35280]
    true_vecs = torch.from_numpy(meta["vec"][:n_test].astype(np.float32))  # [N, 59]
    midi_notes = meta["midi"][:n_test]  # [N]
    norm_notes = torch.from_numpy(((midi_notes - 28) / 22.0).astype(np.float32))

    # 4. Load a real drum loop from classified multitracks
    drum_stem_path = "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance/drums/punchy noisy snare hard medium bright room reverb 141 BPM.flac"
    drum_wav, sr = torchaudio.load(drum_stem_path)
    if sr != 44100:
        drum_wav = torchaudio.functional.resample(drum_wav, sr, 44100)
    drum_wav = drum_wav.mean(dim=0)  # mono

    # Crop an active 0.8s slice (35280 samples)
    drum_slice = drum_wav[44100 * 10 : 44100 * 10 + 35280]
    drum_rms = drum_slice.pow(2).mean().sqrt().item()

    print(f"Loaded {n_test} ground truth test patches and drum stem.")
    print("Running inversion across the 3 conditions (Clean, Demucs, Raw Mix)...")

    cutoff_idx = PARAM_INDEX["cutoff"]
    res_idx = PARAM_INDEX["resonance"]

    mse_clean_list = []
    mse_demucs_list = []
    mse_mix_list = []

    cutoff_err_clean = []
    cutoff_err_demucs = []
    cutoff_err_mix = []

    start_time = time.time()

    for i in range(n_test):
        c_audio = torch.from_numpy(clean_audio[i])
        target_vec = true_vecs[i]
        note = norm_notes[i : i + 1]

        # Match drum RMS to synth RMS for 0 dB SNR
        synth_rms = c_audio.pow(2).mean().sqrt().item() + 1e-6
        d_scaled = drum_slice * (synth_rms / (drum_rms + 1e-6))
        mix_audio = c_audio + d_scaled

        # Normalize mixture
        max_val = mix_audio.abs().max().item()
        if max_val > 1.0:
            mix_audio = mix_audio / max_val
            c_audio_scaled = c_audio / max_val
        else:
            c_audio_scaled = c_audio

        # (a) Clean Mel
        mel_clean = mel_fn(c_audio_scaled).unsqueeze(0)  # [1, 128, 81]

        # (b) Demucs Separation
        mix_stereo = mix_audio.unsqueeze(0).repeat(2, 1).unsqueeze(0)  # [1, 2, 35280]
        with torch.no_grad():
            sources = apply_model(demucs_model, mix_stereo, shifts=0, split=True, overlap=0.25)
        demucs_bass = sources[0, bass_idx].mean(dim=0)  # mono [35280]
        mel_demucs = mel_fn(demucs_bass).unsqueeze(0)

        # (c) Raw Mix Mel
        mel_mix = mel_fn(mix_audio).unsqueeze(0)

        # Invert with V6 C (best-of-1 draw with fixed seed)
        gen = torch.Generator(device=device).manual_seed(1337 + i)
        with torch.no_grad():
            pred_clean_enc = flow.sample(mel_clean, note=note, num_steps=25, generator=gen)
            pred_clean = codec.decode(pred_clean_enc, note)[0]

            gen = torch.Generator(device=device).manual_seed(1337 + i)
            pred_demucs_enc = flow.sample(mel_demucs, note=note, num_steps=25, generator=gen)
            pred_demucs = codec.decode(pred_demucs_enc, note)[0]

            gen = torch.Generator(device=device).manual_seed(1337 + i)
            pred_mix_enc = flow.sample(mel_mix, note=note, num_steps=25, generator=gen)
            pred_mix = codec.decode(pred_mix_enc, note)[0]

        # Compute MSE across continuous parameters
        cont_true = target_vec[FLOW_CONT_INDICES]
        err_c = (pred_clean[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        err_d = (pred_demucs[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()
        err_m = (pred_mix[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item()

        mse_clean_list.append(err_c)
        mse_demucs_list.append(err_d)
        mse_mix_list.append(err_m)

        # Cutoff absolute error
        cutoff_err_clean.append(abs(pred_clean[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_err_demucs.append(abs(pred_demucs[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_err_mix.append(abs(pred_mix[cutoff_idx].item() - target_vec[cutoff_idx].item()))

        sys.stdout.write(f"\rProcessed {i+1}/{n_test} patches... (last MSE Clean: {err_c:.4f}, Demucs: {err_d:.4f})")
        sys.stdout.flush()

    print("\n")
    elapsed = time.time() - start_time

    mean_mse_clean = np.mean(mse_clean_list)
    mean_mse_demucs = np.mean(mse_demucs_list)
    mean_mse_mix = np.mean(mse_mix_list)

    mean_cutoff_clean = np.mean(cutoff_err_clean)
    mean_cutoff_demucs = np.mean(cutoff_err_demucs)
    mean_cutoff_mix = np.mean(cutoff_err_mix)

    pct_degrad = ((mean_mse_demucs - mean_mse_clean) / max(1e-6, mean_mse_clean)) * 100.0

    print("=" * 65)
    print("STEP 0 DIAGNOSTIC RESULTS (20 Ground-Truth Surge Patches at 0 dB SNR)")
    print("=" * 65)
    print(f"Condition                     Continuous Param MSE   Cutoff Abs Error")
    print(f"-----------------------------------------------------------------")
    print(f"(a) Clean Render (Ceiling)    {mean_mse_clean:.4f}                 {mean_cutoff_clean:.4f}")
    print(f"(b) Demucs-Separated Bass     {mean_mse_demucs:.4f}                 {mean_cutoff_demucs:.4f}")
    print(f"(c) Raw Mixture Directly      {mean_mse_mix:.4f}                 {mean_cutoff_mix:.4f}")
    print("=" * 65)
    print(f"Demucs Degradation over Clean: {pct_degrad:+.2f}%")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print("=" * 65)

    if pct_degrad > 15.0:
        print("VERDICT: PREMISE CONFIRMED (Degradation > 15%).")
        print("Demucs significantly damages ground-truth parameter recovery.")
        print("Proceeding with Step 1 (Cue-gated feature extractor) is justified!")
    elif pct_degrad < 5.0:
        print("VERDICT: PREMISE REJECTED (Degradation < 5%).")
        print("Demucs does NOT meaningfully degrade parameter recovery on ground truth.")
        print("The bottleneck is the synthetic-to-real domain gap, NOT source separation.")
    else:
        print(f"VERDICT: MARGINAL EFFECT (Degradation {pct_degrad:.1f}% is in 5-15% gray zone).")


if __name__ == "__main__":
    main()
