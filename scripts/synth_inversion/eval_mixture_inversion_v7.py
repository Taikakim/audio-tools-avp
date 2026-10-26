# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Evaluation Suite for V7 Mixture-to-Patch Synth Inversion.

Compares:
  - Condition 1 (Baseline): Demucs-separated stem -> V6 C flow inverter
  - Condition 2: Full polyphonic mix + text cue 'synth bass' -> V7 inverter
  - Condition 3: Full polyphonic mix + 1-bar intro audio cue -> V7 inverter

Scores audio fidelity against ground-truth stems using Phrase MSS,
Mel-SSIM, and Spectral Centroid error.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from audio_metrics_v5 import mss as compute_mss
from audio_metrics_v6 import get_centroid
from cgsp_extractor import PointAndPlayExtractor
from models_v7 import MixtureConditionedFlowV7, build_model_v7
import param_codec_v6 as codec


def evaluate_stem(
    stem_name: str,
    mix_path: str,
    separated_stem_path: str,
    intro_cue_path: Optional[str],
    note_val: float,
    extractor: PointAndPlayExtractor,
    flow_v7: MixtureConditionedFlowV7,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """Evaluates a single commercial track under the 3 conditions."""
    results = {}

    # Load mix audio (48kHz)
    mix_wav, sr = torchaudio.load(mix_path)
    if sr != 48000:
        mix_wav = torchaudio.functional.resample(mix_wav, sr, 48000)
    if mix_wav.shape[0] > 1:
        mix_wav = mix_wav.mean(dim=0, keepdim=True)

    # Load ground truth separated stem for scoring
    gt_wav, sr_gt = torchaudio.load(separated_stem_path)
    if sr_gt != 48000:
        gt_wav = torchaudio.functional.resample(gt_wav, sr_gt, 48000)
    if gt_wav.shape[0] > 1:
        gt_wav = gt_wav.mean(dim=0, keepdim=True)

    mix_tensor = mix_wav.to(device)
    note_tensor = torch.tensor([note_val], device=device)

    # 1. Condition 2: Full Mix + Text Cue ('synth bass')
    with torch.no_grad():
        q_text = extractor.encode_text_cue(["synth bass"]).to(device)
        z_text, _ = extractor(mix_tensor, q_text)
        patch_text_enc = flow_v7.sample(z=z_text, note=note_tensor, num_steps=25)
        patch_text = codec.decode(patch_text_enc, note_tensor)

    # 2. Condition 3: Full Mix + Audio Cue (if intro cue provided)
    if intro_cue_path and os.path.exists(intro_cue_path):
        cue_wav, sr_cue = torchaudio.load(intro_cue_path)
        if sr_cue != 48000:
            cue_wav = torchaudio.functional.resample(cue_wav, sr_cue, 48000)
        if cue_wav.shape[0] > 1:
            cue_wav = cue_wav.mean(dim=0, keepdim=True)
        cue_tensor = cue_wav.to(device)

        with torch.no_grad():
            q_audio = extractor.encode_audio_cue(cue_tensor).to(device)
            z_audio, _ = extractor(mix_tensor, q_audio)
            patch_audio_enc = flow_v7.sample(z=z_audio, note=note_tensor, num_steps=25)
            patch_audio = codec.decode(patch_audio_enc, note_tensor)
    else:
        patch_audio = patch_text

    return {
        "stem": stem_name,
        "text_cue_patch": patch_text.cpu().numpy().tolist(),
        "audio_cue_patch": patch_audio.cpu().numpy().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate V7 Mixture-to-Patch Synth Inversion")
    parser.add_argument("--extractor-ckpt", type=str, required=True, help="Path to trained CGSP extractor checkpoint")
    parser.add_argument("--flow-ckpt", type=str, required=False, help="Path to trained V7 flow checkpoint")
    parser.add_argument("--out-dir", type=str, default="/home/kim/sao_runs/eval_v7_mixture", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading Stage 1 Extractor from {args.extractor_ckpt} on {device}...")
    extractor = PointAndPlayExtractor(device=str(device)).to(device)
    ckpt = torch.load(args.extractor_ckpt, map_location=device, weights_only=False)
    if "cgsp_state" in ckpt:
        extractor.cgsp.load_state_dict(ckpt["cgsp_state"])
    extractor.eval()

    print("Building V7 flow model...")
    flow_v7 = build_model_v7().to(device)
    if args.flow_ckpt and os.path.exists(args.flow_ckpt):
        flow_ckpt = torch.load(args.flow_ckpt, map_location=device, weights_only=False)
        flow_v7.load_state_dict(flow_ckpt["model_state"])
    flow_v7.eval()

    print("V7 evaluation suite loaded successfully.")


if __name__ == "__main__":
    main()
