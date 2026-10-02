#!/usr/bin/env python3
"""Render evaluation audio clips across epoch checkpoints.

Renders:
1. Target isolated note: untitled.wav (C2, 80ms)
2. Target 303 pluck: target_acid_pluck_a2.wav (A2, 220ms)
3. Target rolling disco bass: target_disco_bass_f2.wav (F2, 280ms)

Generates audio .wav files and a comprehensive trajectory summary.
"""
import os
import glob
import re
import json
import numpy as np
import soundfile as sf
import torch

from audio_utils import prepare_target, make_mel_spec, MultiScaleSTFTLoss, spectral_centroid
from inference import load_inverter, predict_vectors, vectors_to_patches, is_flow
from surge_spec import init_synth, render_patch, SAMPLE_RATE, LP_FILTERS, WAVESHAPER_TYPES

STEMS = [
    ("untitled_c2", "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/untitled.wav", 36, 0.08),
    ("acid_pluck_a2", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_acid_pluck_a2.wav", 45, 0.22),
    ("disco_bass_f2", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_disco_bass_f2.wav", 41, 0.28),
]

def main():
    suite_dir = "/run/media/kim/Mantu/surge_200k_models/v2_suite"
    out_dir = os.path.join(suite_dir, "epoch_eval_clips")
    os.makedirs(out_dir, exist_ok=True)
    
    synth = init_synth()
    loss_fn = MultiScaleSTFTLoss()
    
    # Find all checkpoints
    ckpts = glob.glob(os.path.join(suite_dir, "*.pt"))
    ckpts.sort()
    
    print(f"Found {len(ckpts)} checkpoints in {suite_dir}")
    
    # Pre-render targets and extract mels
    targets = {}
    for name, wav_path, midi_note, note_dur in STEMS:
        if not os.path.exists(wav_path):
            continue
        y, info = prepare_target(wav_path, note_dur=note_dur)
        mel_t = torch.from_numpy(make_mel_spec(y)).unsqueeze(0)
        targets[name] = {
            "y": y,
            "mel": mel_t,
            "midi": midi_note,
            "dur": note_dur,
            "info": info,
            "centroid": spectral_centroid(y),
        }
        # Save reference target wav
        sf.write(os.path.join(out_dir, f"target_{name}.wav"), y, SAMPLE_RATE)
        
    results = {}
    
    for ckpt_path in ckpts:
        base = os.path.basename(ckpt_path).replace(".pt", "")
        # Filter for best or epoch_ checkpoints
        if not ("_best" in base or "_epoch_" in base):
            continue
            
        print(f"\nEvaluating: {base}")
        model = load_inverter(ckpt_path, device="cpu")
        flow = is_flow(model)
        
        results[base] = {}
        
        for stem_name, t in targets.items():
            vecs, _ = predict_vectors(model, t["mel"], n_draws=8 if flow else 1, steps=20)
            patches = vectors_to_patches(vecs[:, 0])
            audios = [render_patch(synth, p, t["midi"], t["dur"]) for p in patches]
            
            L = t["info"]["score_len"]
            losses = [loss_fn(a[:L], t["y"][:L]) for a in audios]
            best_idx = int(np.argmin(losses))
            
            best_patch = patches[best_idx]
            best_audio = audios[best_idx]
            best_loss = losses[best_idx]
            sc = spectral_centroid(best_audio)
            
            out_wav = os.path.join(out_dir, f"{base}_{stem_name}.wav")
            sf.write(out_wav, best_audio, SAMPLE_RATE)
            
            results[base][stem_name] = {
                "filter": LP_FILTERS[best_patch["filter_idx"]][0],
                "waveshaper": WAVESHAPER_TYPES[best_patch["ws_idx"]][0],
                "stft_loss": best_loss,
                "centroid": sc,
                "out_wav": out_wav,
            }
            print(f"  {stem_name:15s} | Filter: {results[base][stem_name]['filter']:18s} | STFT: {best_loss:.3f} | Centroid: {sc:6.1f} Hz")
            
    with open(os.path.join(out_dir, "epoch_clips_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAll evaluation clips written to: {out_dir}")

if __name__ == "__main__":
    main()
