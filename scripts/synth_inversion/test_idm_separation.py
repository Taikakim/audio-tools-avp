#!/usr/bin/env python3
# Owner: KUANG / KIM / ANTIGRAVITY (created 2026-10-21)
"""Phase 1 Drum Source Separation & Transcription Runner (Tier 2A).

Uses the pretrained Inverse Drum Machine (IDM; Torres et al., IEEE TASLP 2025) to decompose
audio into 9 kit pieces at 44.1 kHz, extracting symbolic onset timestamps and velocities.
Can run entirely on CPU without GPU contention.

Usage:
  python test_idm_separation.py --input /path/to/drum_loop.wav --out_dir ~/sao_runs/idm_test --device cpu
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Ensure inverse-drum-machine is importable
IDM_ROOT = "/home/kim/Projects/inverse-drum-machine"
if IDM_ROOT not in sys.path:
    sys.path.insert(0, IDM_ROOT)

from idm.inference import load_model, separate
def pick_peaks(x, threshold=0.5):
    """Local maxima peak-picking across time dimension."""
    padded = torch.nn.functional.pad(x, (1, 1), mode='constant', value=0)
    return (x > padded[..., :-2]) & (x >= padded[..., 2:]) & (x > threshold)


DRUM_LABELS = {
    "KD": "Kick Drum",
    "SD": "Snare Drum",
    "HH_CHH": "Closed Hi-Hat",
    "HH_OHH": "Open Hi-Hat",
    "CY_CR": "Crash Cymbal",
    "CY_RD": "Ride Cymbal",
    "TT_HFT": "High Floor Tom",
    "TT_HMT": "High Mid Tom",
    "TT_LMT": "Low Mid Tom",
}


def extract_symbolic_transcription(model, audio_tensor, threshold=0.5):
    """Extracts timestamped onsets and velocities from the IDM encoder activations."""
    with torch.no_grad():
        encoder_outs = model.encoder(audio_tensor.unsqueeze(0))
        activations = encoder_outs["activations"]
        onset_logits = activations["onset"]
        onsets = torch.sigmoid(onset_logits)
        peaks = pick_peaks(onsets, threshold=threshold)
        velocities = activations["velocity"]

        activation_rate = getattr(model.encoder, "frame_rate", 172.26)
        train_classes = getattr(model, "train_classes", list(DRUM_LABELS.keys()))

        events = []
        B, K, T = peaks.shape
        for k_idx in range(K):
            class_name = train_classes[k_idx]
            peak_indices = torch.nonzero(peaks[0, k_idx] > 0.5).squeeze(-1).cpu().numpy()
            for idx in peak_indices:
                t_sec = float(idx / activation_rate)
                vel = float(velocities[0, k_idx, idx].cpu().numpy())
                conf = float(onsets[0, k_idx, idx].cpu().numpy())
                events.append({
                    "timestamp": t_sec,
                    "class": class_name,
                    "label": DRUM_LABELS.get(class_name, class_name),
                    "velocity": vel,
                    "confidence": conf
                })

        events.sort(key=lambda e: e["timestamp"])
        return events, encoder_outs


def run_idm_eval(input_path: str, out_dir: str, device: str = "cpu", threshold: float = 0.5):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("IDM 44.1 kHz 9-CLASS DRUM SEPARATION & TRANSCRIPTION")
    print(f"Input: {input_path}")
    print(f"Output: {out_dir}")
    print(f"Device: {device}")
    print("================================================================================")

    t0 = time.time()
    print("Loading pretrained IDM model ('idm-44-train-kits')...", flush=True)
    model, name = load_model("idm-44-train-kits", device=device, log_dir=os.path.join(IDM_ROOT, "pretrained"))
    print(f"Model loaded in {time.time() - t0:.2f}s.\n", flush=True)

    # 1. Load audio
    audio, sr = sf.read(input_path, always_2d=True)
    audio_t = torch.from_numpy(audio.T).float().to(device)
    dur = len(audio) / sr
    print(f"Input audio: {len(audio)} samples, {dur:.2f}s @ {sr} Hz ({audio.shape[1]} ch)")

    # 2. Extract symbolic events
    t_trans = time.time()
    events, _ = extract_symbolic_transcription(model, audio_t, threshold=threshold)
    print(f"Extracted {len(events)} drum events in {time.time() - t_trans:.2f}s.")

    # Save transcription text file
    txt_path = out_path / "transcription.txt"
    with open(txt_path, "w") as f:
        f.write("# Timestamp(s)\tClass\tVelocity\tConfidence\tLabel\n")
        for e in events:
            f.write(f"{e['timestamp']:.4f}\t{e['class']}\t{e['velocity']:.3f}\t{e['confidence']:.3f}\t{e['label']}\n")
    print(f"Saved transcription table -> {txt_path}")

    # Summary count per class
    counts = {}
    for e in events:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  - {k} ({DRUM_LABELS.get(k, k)}): {v} hits")

    # 3. Audio Source Separation via Wiener Masking
    t_sep = time.time()
    print("\nRunning analysis-by-synthesis & Wiener filtering...", flush=True)
    stems = separate(input_path, model, output_dir=str(out_path / "stems"), masking="wiener", alpha=1.0)
    print(f"Separation completed in {time.time() - t_sep:.2f}s!")
    print(f"Separated 9 stems saved to -> {out_path / 'stems'}")

    return events, stems


def main():
    ap = argparse.ArgumentParser(description="Test IDM Drum Separation & Transcription")
    ap.add_argument("--input", default="/home/kim/Projects/inverse-drum-machine/demo/mix.wav", help="Input WAV file")
    ap.add_argument("--out_dir", default="/home/kim/sao_runs/idm_drum_eval", help="Output directory")
    ap.add_argument("--device", default="cpu", help="Device (cpu or cuda:0)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Onset detection threshold")
    args = ap.parse_args()

    run_idm_eval(args.input, args.out_dir, device=args.device, threshold=args.threshold)


if __name__ == "__main__":
    main()
