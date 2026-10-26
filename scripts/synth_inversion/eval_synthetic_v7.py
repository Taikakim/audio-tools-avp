# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Evaluation of Step 1 Cue-Gated ASP on ground-truth synthetic mixtures.

Compares:
  (a) Clean Render (Ceiling)
  (b) Demucs-Separated Bass
  (c) Raw Mixture Directly (Floor)
  (d) Raw Mix + Cue-Gated ASP (Demucs Bass as Cue)
  (e) Raw Mix + Cue-Gated ASP (Clean Audio as Cue - Oracle)
"""

import argparse
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
from cue_gated_temporal_asp import CueGatedTemporalAudioEncoder
from audio_utils import ExactGpuMel
import param_codec_v6 as codec
from surge_spec_v6 import NUM_PARAMS, PARAM_INDEX, FLOW_CONT_INDICES


def sample_with_emb(flow, audio_emb, note, num_steps=25, generator=None):
    B = audio_emb.shape[0]
    x = torch.randn(B, flow.param_dim, device=audio_emb.device, generator=generator)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((B,), i / num_steps, device=audio_emb.device, dtype=torch.float32)
        x = x + flow(x, t, audio_emb=audio_emb, note=note) * dt
    return torch.clamp(x, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate V7 Cue-Gated ASP on ground truth")
    parser.add_argument("--v6c-ckpt", type=str, default="/home/kim/sao_runs/v6_C/flow_best.pt")
    parser.add_argument("--gated-ckpt", type=str, default="/home/kim/sao_runs/cue_gated_asp_v7/cue_gated_asp_best.pt")
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"=== Evaluating Step 1 Cue-Gated ASP on {device} ===")

    # 1. Load V6 C model
    print(f"Loading V6 C flow model from {args.v6c_ckpt}...")
    flow = load_inverter_v4(args.v6c_ckpt, device=str(device)).eval()
    base_encoder = flow.encoder
    mel_fn = ExactGpuMel(device=str(device))

    # 2. Load trained CueGatedTemporalAudioEncoder
    print(f"Loading Cue-Gated ASP checkpoint from {args.gated_ckpt}...")
    gated_encoder = CueGatedTemporalAudioEncoder(
        base_encoder=base_encoder,
        cue_dim=512,
        proj_dim=128,
        hidden_dim=512,
        freeze_base=True,
    ).to(device)

    ckpt = torch.load(args.gated_ckpt, map_location=device, weights_only=False)
    gated_encoder.load_state_dict(ckpt["model_state"])
    gated_encoder.eval()

    # 3. Load Demucs model
    print("Loading Demucs (htdemucs)...")
    demucs_model = get_model("htdemucs").to(device).eval()
    bass_idx = demucs_model.sources.index("bass")

    # 4. Load ground truth patches
    bank_dir = "/home/kim/sao_runs/bank_v6"
    audio_path = os.path.join(bank_dir, "main_00000_audio.npy")
    meta_path = os.path.join(bank_dir, "main_00000_meta.npz")

    audio_raw = np.load(audio_path, mmap_mode="r")
    meta = np.load(meta_path)

    n_test = args.n_test
    clean_audio = audio_raw[:n_test].astype(np.float32) / 32768.0  # [N, 35280]
    true_vecs = torch.from_numpy(meta["vec"][:n_test].astype(np.float32)).to(device)  # [N, 59]
    midi_notes = meta["midi"][:n_test]  # [N]
    norm_notes = torch.from_numpy(((midi_notes - 28) / 22.0).astype(np.float32)).to(device)

    # 5. Load drum stem
    drum_stem_path = "/run/media/kim/Mantu/avp-stems-original-classified/Aavepyörä - Goddess Guerrilla  goa morning bright energetic trance/drums/punchy noisy snare hard medium bright room reverb 141 BPM.flac"
    drum_wav, sr = torchaudio.load(drum_stem_path)
    if sr != 44100:
        drum_wav = torchaudio.functional.resample(drum_wav, sr, 44100)
    drum_wav = drum_wav.mean(dim=0).to(device)

    drum_slice = drum_wav[44100 * 10 : 44100 * 10 + 35280]
    drum_rms = drum_slice.pow(2).mean().sqrt().item()

    print(f"Loaded {n_test} ground truth test patches and drum stem.")

    cutoff_idx = PARAM_INDEX["cutoff"]

    mse_clean = []
    mse_demucs = []
    mse_mix = []
    mse_gated_demucs = []
    mse_gated_clean = []

    cutoff_clean = []
    cutoff_demucs = []
    cutoff_mix = []
    cutoff_gated_demucs = []
    cutoff_gated_clean = []

    start_time = time.time()

    for i in range(n_test):
        c_audio = torch.from_numpy(clean_audio[i]).to(device)
        target_vec = true_vecs[i]
        note = norm_notes[i : i + 1]

        synth_rms = c_audio.pow(2).mean().sqrt().item() + 1e-6
        d_scaled = drum_slice * (synth_rms / (drum_rms + 1e-6))
        mix_audio = c_audio + d_scaled

        max_val = mix_audio.abs().max().item()
        if max_val > 1.0:
            mix_audio = mix_audio / max_val
            c_audio_scaled = c_audio / max_val
        else:
            c_audio_scaled = c_audio

        # Mels
        mel_clean = mel_fn(c_audio_scaled).unsqueeze(0)
        mix_stereo = mix_audio.unsqueeze(0).repeat(2, 1).unsqueeze(0)
        with torch.no_grad():
            sources = apply_model(demucs_model, mix_stereo, shifts=0, split=True, overlap=0.25)
        demucs_bass = sources[0, bass_idx].mean(dim=0)
        mel_demucs = mel_fn(demucs_bass).unsqueeze(0)
        mel_mix = mel_fn(mix_audio).unsqueeze(0)

        # Invert with V6 C and Cue-Gated ASP
        with torch.no_grad():
            # (a) Clean
            gen = torch.Generator(device=str(device)).manual_seed(1337 + i)
            emb_c = base_encoder(mel_clean)
            p_clean = codec.decode(sample_with_emb(flow, emb_c, note, generator=gen), note)[0]

            # (b) Demucs
            gen = torch.Generator(device=str(device)).manual_seed(1337 + i)
            emb_d = base_encoder(mel_demucs)
            p_demucs = codec.decode(sample_with_emb(flow, emb_d, note, generator=gen), note)[0]

            # (c) Raw Mix
            gen = torch.Generator(device=str(device)).manual_seed(1337 + i)
            emb_m = base_encoder(mel_mix)
            p_mix = codec.decode(sample_with_emb(flow, emb_m, note, generator=gen), note)[0]

            # (d) Mix + Cue-Gated ASP (Demucs Cue)
            cue_d = emb_d[:, :512]
            emb_gated_d = gated_encoder(mel_mix, cue_d)
            gen = torch.Generator(device=str(device)).manual_seed(1337 + i)
            p_gated_d = codec.decode(sample_with_emb(flow, emb_gated_d, note, generator=gen), note)[0]

            # (e) Mix + Cue-Gated ASP (Clean Cue - Oracle)
            cue_c = emb_c[:, :512]
            emb_gated_c = gated_encoder(mel_mix, cue_c)
            gen = torch.Generator(device=str(device)).manual_seed(1337 + i)
            p_gated_c = codec.decode(sample_with_emb(flow, emb_gated_c, note, generator=gen), note)[0]

        cont_true = target_vec[FLOW_CONT_INDICES]

        mse_clean.append((p_clean[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item())
        mse_demucs.append((p_demucs[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item())
        mse_mix.append((p_mix[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item())
        mse_gated_demucs.append((p_gated_d[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item())
        mse_gated_clean.append((p_gated_c[FLOW_CONT_INDICES] - cont_true).pow(2).mean().item())

        cutoff_clean.append(abs(p_clean[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_demucs.append(abs(p_demucs[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_mix.append(abs(p_mix[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_gated_demucs.append(abs(p_gated_d[cutoff_idx].item() - target_vec[cutoff_idx].item()))
        cutoff_gated_clean.append(abs(p_gated_c[cutoff_idx].item() - target_vec[cutoff_idx].item()))

        sys.stdout.write(f"\rProcessed {i+1}/{n_test} patches...")
        sys.stdout.flush()

    print("\n")
    elapsed = time.time() - start_time

    print("=" * 75)
    print(f"STEP 1 EVALUATION RESULTS ({n_test} Ground-Truth Surge Patches at 0 dB SNR)")
    print("=" * 75)
    print(f"Condition                               Continuous Param MSE   Cutoff Abs Error")
    print(f"---------------------------------------------------------------------------")
    print(f"(a) Clean Render (Ceiling)              {np.mean(mse_clean):.4f}                 {np.mean(cutoff_clean):.4f}")
    print(f"(b) Demucs-Separated Bass (Baseline)    {np.mean(mse_demucs):.4f}                 {np.mean(cutoff_demucs):.4f}")
    print(f"(c) Raw Mixture Directly (Floor)        {np.mean(mse_mix):.4f}                 {np.mean(cutoff_mix):.4f}")
    print(f"(d) Raw Mix + Cue-Gated (Demucs Cue)    {np.mean(mse_gated_demucs):.4f}                 {np.mean(cutoff_gated_demucs):.4f}")
    print(f"(e) Raw Mix + Cue-Gated (Clean Cue)     {np.mean(mse_gated_clean):.4f}                 {np.mean(cutoff_gated_clean):.4f}")
    print("=" * 75)

    gap_demucs = np.mean(mse_demucs) - np.mean(mse_clean)
    gap_gated = np.mean(mse_gated_demucs) - np.mean(mse_clean)
    pct_recovery = ((gap_demucs - gap_gated) / max(1e-6, gap_demucs)) * 100.0
    print(f"Demucs Degradation Gap: +{gap_demucs:.4f}")
    print(f"Cue-Gated Degradation Gap: +{gap_gated:.4f}")
    print(f"Gap Closed by Cue Gate: {pct_recovery:+.1f}%")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print("=" * 75)


if __name__ == "__main__":
    main()
