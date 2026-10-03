#!/usr/bin/env python3
"""Render evaluation audio clips across Synth-JEPA checkpoints for target stems.

Renders:
1. Target isolated note: untitled.wav (C2, 80ms)
2. Target 303 pluck: target_acid_pluck_a2.wav (A2, 220ms)
3. Target rolling disco bass: target_disco_bass_f2.wav (F2, 280ms)
"""
import glob
import json
import os
import numpy as np
import soundfile as sf
import torch

from audio_utils import MultiScaleSTFTLoss, make_mel_spec, prepare_target, save_patch, spectral_centroid
from surge_spec import LP_FILTERS, SAMPLE_RATE, WAVESHAPER_TYPES, init_synth, render_patch
from synth_jepa_search import SynthJEPASearcher, checkpoint_bounds, load_synth_jepa

STEMS = [
    ("untitled_c2", "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/untitled.wav", 36, 0.08),
    ("acid_pluck_a2", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_acid_pluck_a2.wav", 45, 0.22),
    ("disco_bass_f2", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_disco_bass_f2.wav", 41, 0.28),
]


def main():
    jepa_dir = "/run/media/kim/Mantu/surge_200k_models/synth_jepa_runs/jepa_v2_20ep"
    out_dir = os.path.join(jepa_dir, "eval_clips")
    os.makedirs(out_dir, exist_ok=True)

    synth = init_synth(verify=False)
    loss_fn = MultiScaleSTFTLoss()

    ckpts = glob.glob(os.path.join(jepa_dir, "checkpoint_*.pt"))
    ckpts.sort()
    print(f"Found {len(ckpts)} checkpoints in {jepa_dir}:")
    for c in ckpts:
        print(" ", os.path.basename(c))

    # Pre-render targets
    targets = {}
    for name, wav_path, midi_note, note_dur in STEMS:
        if not os.path.exists(wav_path):
            continue
        y, info = prepare_target(wav_path, note_dur=note_dur)
        mel_t = torch.from_numpy(make_mel_spec(y))
        targets[name] = {
            "y": y,
            "mel": mel_t,
            "midi": midi_note,
            "dur": note_dur,
            "info": info,
            "centroid": spectral_centroid(y),
        }
        sf.write(os.path.join(out_dir, f"target_{name}.wav"), y, SAMPLE_RATE)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning renderer-free search on {device} (budget=1024)...")

    results = {}
    for ckpt_path in ckpts:
        base = os.path.basename(ckpt_path).replace(".pt", "")
        print(f"\n=== Evaluating Checkpoint: {base} ===")
        model, normalizer = load_synth_jepa(ckpt_path, device=device)
        searcher = SynthJEPASearcher(model, normalizer, device=device, seed=42,
                                     bounds=checkpoint_bounds(ckpt_path))

        results[base] = {}
        for stem_name, t in targets.items():
            res = searcher.search(t["mel"].to(device), midi_note=t["midi"], total_eval_budget=1024)
            patch = res["patch"]

            audio = render_patch(synth, patch, t["midi"], t["dur"])
            L = t["info"]["score_len"]
            stft_loss = loss_fn(audio[:L], t["y"][:L])
            sc = spectral_centroid(audio)

            out_wav = os.path.join(out_dir, f"{base}_{stem_name}.wav")
            sf.write(out_wav, audio, SAMPLE_RATE)

            # Export native VST3 preset
            path_stem = os.path.join(out_dir, f"{base}_{stem_name}")
            save_patch(synth, patch, path_stem, extra={"stft_loss": stft_loss, "centroid": sc})

            filt_name = LP_FILTERS[patch["filter_idx"]][0]
            ws_name = WAVESHAPER_TYPES[patch["ws_idx"]][0]
            results[base][stem_name] = {
                "filter": filt_name,
                "waveshaper": ws_name,
                "stft_loss": float(stft_loss),
                "centroid": float(sc),
                "out_wav": out_wav,
                "vstpreset": f"{path_stem}.vstpreset",
                "n_evals": res["n_evals"],
            }
            print(f"  {stem_name:15s} | Filter: {filt_name:18s} | STFT: {stft_loss:.3f} | Centroid: {sc:6.1f} Hz")

    with open(os.path.join(out_dir, "jepa_clips_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll clips and .vstpresets saved to: {out_dir}")


if __name__ == "__main__":
    main()
