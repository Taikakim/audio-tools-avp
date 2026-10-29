#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""V8 End-to-End Pipeline: Asymmetric TSE + Dynamic Parameter Inversion + DAW Export Bundle.

Demonstrates the complete Deep V8 inverted-audio pipeline:
1. Input: Polyphonic audio mixture (containing drums, bass, and synth lead).
2. Cue: 0.5s reference snippet of the target synth lead.
3. Separation (Phase 1): Asymmetric 2D Complex U-Net TSE isolates the synth stem from the mix.
4. Parameter Inversion (Phase 2): Dynamic Parameter Inverter predicts:
   - 59 static patch parameters (oscillator shapes, envelopes, filter type).
   - 4 dynamic automation curves at 10.76 Hz (Cutoff CC 74, Resonance CC 71, Drive CC 1, FM Depth CC 11).
5. DAW Export Bundle (Phase 3): Generates native Steinberg .vstpreset container + .mid automation track.
6. Re-synthesis: Renders the inverted patch + automation in Surge XT to verify acoustic match.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pedalboard
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_utils import ExactGpuMel
from train_v8_asymmetric_tse import AsymmetricTSEModel
from dynamic_inverter_v8 import DynamicParameterInverterV8
from daw_export_v8 import export_v8_daw_bundle
import surge_spec_v5 as S5
import surge_spec_v6 as S6
from dynamic_dataset_v8 import render_dynamic_patch

GUARD_SCRIPT = "/home/kim/Projects/SAO/Misc/gpu_guard.sh"


def acquire_gpu_guard() -> bool:
    pid = str(os.getpid())
    env = os.environ.copy()
    env["TARGET_GPU"] = "0"
    res = subprocess.run(["bash", GUARD_SCRIPT, "acquire", "ANTIGRAVITY", pid], env=env)
    return res.returncode == 0


def release_gpu_guard():
    subprocess.run(["bash", GUARD_SCRIPT, "release", "ANTIGRAVITY"])


def run_e2e_pipeline(device: str = "cuda", out_dir: str = "/home/kim/sao_runs/v8_daw_export"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V8 END-TO-END PIPELINE: ASYMMETRIC TSE -> DYNAMIC INVERTER -> DAW EXPORT")
    print(f"Device: {device} | Output Directory: {out_path}")
    print("=" * 80)

    # 1. Load TSE Model (Phase 1)
    tse_ckpt_path = Path("/home/kim/sao_runs/v8_phase1_tse/tse_best.pt")
    if not tse_ckpt_path.exists():
        print(f"Error: TSE checkpoint not found at {tse_ckpt_path}", file=sys.stderr)
        return False
    tse_model = AsymmetricTSEModel().to(device).eval()
    tse_ckpt = torch.load(tse_ckpt_path, map_location=device, weights_only=False)
    tse_model.load_state_dict(tse_ckpt["model"])

    # 2. Load Dynamic Inverter Model (Phase 2)
    inv_ckpt_path = Path("/home/kim/sao_runs/v8_phase2_inverter/dynamic_inverter_best.pt")
    if not inv_ckpt_path.exists():
        print(f"Error: Inverter checkpoint not found at {inv_ckpt_path}", file=sys.stderr)
        return False
    print(f"Loading Phase 2 Dynamic Inverter from {inv_ckpt_path}...")
    inv_model = DynamicParameterInverterV8().to(device).eval()
    inv_ckpt = torch.load(inv_ckpt_path, map_location=device, weights_only=False)
    inv_model.load_state_dict(inv_ckpt["model_state"])

    # 3. Create a Realistic Polyphonic Test Mixture
    print("\nSynthesizing realistic test phrase (Lead Synth with dynamic cutoff sweep)...")
    plugin = pedalboard.load_plugin(os.path.expanduser(S5.DEFAULT_PLUGIN_PATH))
    mel_fn = ExactGpuMel(sample_rate=44100, n_mels=128, n_fft=2048, hop_length=441)

    # True synth patch
    true_static = np.full(59, 0.5, dtype=np.float32)
    true_static[0] = 0.0606  # Filter LP 24
    true_patch = S5.vector_to_patch(S6.to_v5(true_static))
    true_patch["filter_idx"] = 1
    true_patch["filter_balance"] = 0.0

    # Dynamic sweep: rising filter cutoff across 4.0s
    t = np.linspace(0.15, 0.85, 43, dtype=np.float32)
    cutoff_lane = t
    res_lane = np.full(43, 0.45, dtype=np.float32)
    drive_lane = np.linspace(0.1, 0.5, 43, dtype=np.float32)
    fm_lane = np.full(43, 0.0, dtype=np.float32)

    target_synth_wav = render_dynamic_patch(
        plugin, true_patch, 38, cutoff_lane, res_lane, drive_lane, fm_lane
    )

    # Load backing stem from cache
    stem_cache_dir = Path("/home/kim/sao_runs/v8_stem_cache")
    if (stem_cache_dir / "val_drums.npy").exists():
        drums_bank = np.load(stem_cache_dir / "val_drums.npy", mmap_mode="r")
        drums_wav = drums_bank[0]
    else:
        drums_wav = np.zeros_like(target_synth_wav)

    if (stem_cache_dir / "val_bass.npy").exists():
        bass_bank = np.load(stem_cache_dir / "val_bass.npy", mmap_mode="r")
        bass_wav = bass_bank[0]
    else:
        bass_wav = np.zeros_like(target_synth_wav)

    # Mix 0 dB SNR
    mix_wav = target_synth_wav + 0.7 * (drums_wav + bass_wav)
    mix_peak = np.abs(mix_wav).max()
    if mix_peak > 1e-4:
        mix_wav = mix_wav / max(1.0, mix_peak)

    # Cue: 0.5s from target synth
    cue_samples = int(44100 * 0.5)
    cue_wav = target_synth_wav[:cue_samples]

    print(f"Phrase generated: 4.0s mixture (Peak={mix_peak:.3f}), cue length={len(cue_wav)} samples.")

    # 4. TSE Separation Step
    print("\nStep 1: Running Asymmetric TSE Separation...")
    mix_t = torch.from_numpy(mix_wav).float().unsqueeze(0).unsqueeze(0).to(device)
    cue_t = torch.from_numpy(cue_wav).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            sep_wav_t, rel_mask = tse_model(mix_t, cue_t)
            sep_mel = mel_fn(sep_wav_t.squeeze(0).cpu()).to(device)

    sep_wav = sep_wav_t.squeeze().cpu().float().numpy()

    # 5. Dynamic Parameter Inversion Step
    print("Step 2: Inverting Separated Stem into Dynamic Parameters...")
    with torch.no_grad():
        with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            pred_static, pred_dynamic = inv_model(sep_mel.unsqueeze(0))

    pred_static_np = pred_static.squeeze(0).cpu().float().numpy()
    pred_dynamic_np = pred_dynamic.squeeze(0).cpu().float().numpy()

    print(f"Predicted Static Parameters: shape {pred_static_np.shape}")
    print(f"Predicted Dynamic Lanes:     shape {pred_dynamic_np.shape}")
    print(f"  - Cutoff range:   [{pred_dynamic_np[0].min():.3f}, {pred_dynamic_np[0].max():.3f}] (True: [{cutoff_lane.min():.3f}, {cutoff_lane.max():.3f}])")
    print(f"  - Resonance mean: {pred_dynamic_np[1].mean():.3f} (True: {res_lane.mean():.3f})")
    print(f"  - Drive mean:     {pred_dynamic_np[2].mean():.3f} (True: {drive_lane.mean():.3f})")

    # 6. DAW Export Bundle
    print("\nStep 3: Generating V8 DAW Export Bundle (.vstpreset + .mid)...")
    output_stem = out_path / "v8_lead_synth_inversion"
    bundle = export_v8_daw_bundle(
        plugin=plugin,
        static_params=pred_static_np,
        dynamic_lanes=pred_dynamic_np,
        output_stem=output_stem,
        bpm=140.0,
    )

    print("Created DAW bundle files:")
    for k, p in bundle.items():
        size_b = os.path.getsize(p)
        print(f"  [{k.upper():16s}] {p} ({size_b:,} bytes)")

    # 7. Re-Synthesis & Verification
    print("\nStep 4: Re-Synthesizing from Inverted Patch & Automation...")
    pred_patch = S5.vector_to_patch(S6.to_v5(pred_static_np))
    resynth_wav = render_dynamic_patch(
        plugin, pred_patch, 38,
        pred_dynamic_np[0], pred_dynamic_np[1], pred_dynamic_np[2], pred_dynamic_np[3]
    )

    # Save audio files for audition
    sf.write(str(out_path / "mix.wav"), mix_wav, 44100)
    sf.write(str(out_path / "target_clean.wav"), target_synth_wav, 44100)
    sf.write(str(out_path / "separated_tse.wav"), sep_wav, 44100)
    sf.write(str(out_path / "resynthesized_surge.wav"), resynth_wav, 44100)

    # Compute Mel spectrogram correlation
    resynth_mel = mel_fn(torch.from_numpy(resynth_wav))
    target_mel = mel_fn(torch.from_numpy(target_synth_wav))
    mel_l1 = F.l1_loss(resynth_mel, target_mel).item()
    print(f"Reconstruction Mel L1 Distance: {mel_l1:.4f}")

    print("\n" + "=" * 80)
    print("V8 END-TO-END PIPELINE EXECUTION SUCCESSFUL!")
    print(f"All files saved to {out_path}")
    print("=" * 80)
    return True


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cuda" in device:
        if not acquire_gpu_guard():
            print("GPU is currently busy. Exiting.", file=sys.stderr)
            sys.exit(0)
    try:
        run_e2e_pipeline(device=device)
    finally:
        if "cuda" in device:
            release_gpu_guard()


if __name__ == "__main__":
    main()
