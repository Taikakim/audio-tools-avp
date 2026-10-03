#!/usr/bin/env python3
"""Evaluate Envelope-Guided Synth-JEPA Search against Vanilla Search.

Compares:
1. Vanilla Synth-JEPA Search (Hayes et al. arXiv:2609.31024)
2. Envelope-Guided Synth-JEPA Search (AEG/FEG decoupled prior + seeded JADE + regularized Adam)

Measures:
- Spectral fidelity: MultiScale STFT loss
- Temporal/groove fidelity: Envelope loss (AEG sub-bass decay & sustain, FEG filter decay)
- VST preset generation for DAW playback
"""

import argparse
import json
import os
import soundfile as sf
import torch

from audio_utils import MultiScaleSTFTLoss, make_mel_spec, prepare_target, save_patch, spectral_centroid
from envelope_extractor import compute_envelope_loss_np, plot_envelope_profile, profile_note_envelope
from surge_spec import LP_FILTERS, SAMPLE_RATE, WAVESHAPER_TYPES, init_synth, render_patch
from synth_jepa_search import SynthJEPASearcher, checkpoint_bounds, load_synth_jepa

STEMS = [
    ("untitled_c2", "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results/untitled.wav", 36, 0.08),
    ("acid_pluck_a2", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_acid_pluck_a2.wav", 45, 0.22),
    ("disco_bass_f2", "/run/media/kim/Mantu/surge_stem_inversion/targets/target_disco_bass_f2.wav", 41, 0.28),
]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Envelope-Guided Search")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint. Default: best in online run")
    parser.add_argument("--budget", type=int, default=1024, help="Search evaluation budget")
    parser.add_argument("--bpm", type=float, default=143.0, help="Tempo in BPM (default: 143.0)")
    parser.add_argument("--out_dir", type=str, default="/run/media/kim/Mantu/surge_200k_models/synth_jepa_runs/envelope_eval",
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        # Check online run first, fallback to v2
        cand1 = "/run/media/kim/Mantu/surge_200k_models/synth_jepa_runs/jepa_online_infinite_v1/checkpoint_latest.pt"
        cand2 = "/run/media/kim/Mantu/surge_200k_models/synth_jepa_runs/jepa_v2_20ep/checkpoint_latest.pt"
        args.ckpt = cand1 if os.path.exists(cand1) else cand2

    print(f"Loading checkpoint: {args.ckpt} on {device}")
    model, normalizer = load_synth_jepa(args.ckpt, device=device)
    searcher = SynthJEPASearcher(model, normalizer, device=device, seed=42, bounds=checkpoint_bounds(args.ckpt))
    synth = init_synth(verify=False)
    loss_stft = MultiScaleSTFTLoss()

    summary = {
        "checkpoint": args.ckpt,
        "budget": args.budget,
        "bpm": args.bpm,
        "targets": {},
    }

    print("\n" + "=" * 80)
    print(f"{'Stem':15s} | {'Method':12s} | {'STFT Loss':10s} | {'AEG Loss':10s} | {'FEG Loss':10s} | {'Centroid (Hz)':14s}")
    print("=" * 80)

    for stem_name, wav_path, midi_note, note_dur in STEMS:
        if not os.path.exists(wav_path):
            continue

        y_target, info = prepare_target(wav_path, note_dur=note_dur)
        mel_target = torch.from_numpy(make_mel_spec(y_target)).to(device)
        L = info["score_len"]
        y_eval = y_target[:L]

        # 1. Extract ground truth envelope profile
        profile = profile_note_envelope(y_eval, fs=SAMPLE_RATE, bpm=args.bpm)
        plot_path = os.path.join(args.out_dir, f"{stem_name}_envelope_profile.png")
        plot_envelope_profile(y_eval, profile, fs=SAMPLE_RATE, out_path=plot_path)

        # 2. Vanilla Synth-JEPA Search
        res_vanilla = searcher.search(mel_target, midi_note=midi_note, total_eval_budget=args.budget)
        audio_vanilla = render_patch(synth, res_vanilla["patch"], midi_note, note_dur)
        stft_v = float(loss_stft(audio_vanilla[:L], y_eval))
        env_v = compute_envelope_loss_np(y_eval, audio_vanilla[:L], fs=SAMPLE_RATE)
        sc_v = float(spectral_centroid(audio_vanilla[:L]))

        # Save audio & preset
        stem_v_path = os.path.join(args.out_dir, f"{stem_name}_vanilla")
        sf.write(f"{stem_v_path}.wav", audio_vanilla, SAMPLE_RATE)
        save_patch(synth, res_vanilla["patch"], stem_v_path,
                   extra={"method": "vanilla", "stft_loss": stft_v, "env_loss": env_v["loss_env_total"]})

        # 3. Envelope-Guided Synth-JEPA Search
        res_guided = searcher.search(
            mel_target,
            midi_note=midi_note,
            total_eval_budget=args.budget,
            envelope_target=profile.suggested_params,
            envelope_weight=0.08,
        )
        audio_guided = render_patch(synth, res_guided["patch"], midi_note, note_dur)
        stft_g = float(loss_stft(audio_guided[:L], y_eval))
        env_g = compute_envelope_loss_np(y_eval, audio_guided[:L], fs=SAMPLE_RATE)
        sc_g = float(spectral_centroid(audio_guided[:L]))

        # Save audio & preset
        stem_g_path = os.path.join(args.out_dir, f"{stem_name}_env_guided")
        sf.write(f"{stem_g_path}.wav", audio_guided, SAMPLE_RATE)
        save_patch(synth, res_guided["patch"], stem_g_path,
                   extra={"method": "envelope_guided", "stft_loss": stft_g, "env_loss": env_g["loss_env_total"]})

        # Print comparison
        target_sc = float(spectral_centroid(y_eval))
        print(f"{stem_name:15s} | {'Vanilla':12s} | {stft_v:10.3f} | {env_v['loss_aeg']:10.5f} | {env_v['loss_feg']:10.5f} | {sc_v:6.1f} (tgt {target_sc:.1f})")
        print(f"{'':15s} | {'Env-Guided':12s} | {stft_g:10.3f} | {env_g['loss_aeg']:10.5f} | {env_g['loss_feg']:10.5f} | {sc_g:6.1f} (tgt {target_sc:.1f})")
        print("-" * 80)

        summary["targets"][stem_name] = {
            "target_centroid": target_sc,
            "gate_duration_target_ms": profile.gate_off_s * 1000.0,
            "vanilla": {
                "stft_loss": stft_v,
                "aeg_loss": env_v["loss_aeg"],
                "feg_loss": env_v["loss_feg"],
                "env_total_loss": env_v["loss_env_total"],
                "centroid": sc_v,
                "patch": res_vanilla["patch"],
                "wav": f"{stem_v_path}.wav",
                "vstpreset": f"{stem_v_path}.vstpreset",
            },
            "guided": {
                "stft_loss": stft_g,
                "aeg_loss": env_g["loss_aeg"],
                "feg_loss": env_g["loss_feg"],
                "env_total_loss": env_g["loss_env_total"],
                "centroid": sc_g,
                "patch": res_guided["patch"],
                "wav": f"{stem_g_path}.wav",
                "vstpreset": f"{stem_g_path}.vstpreset",
            },
        }

    sum_path = os.path.join(args.out_dir, "envelope_eval_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation complete. Full report written to: {sum_path}")
    print(f"Patches exported to DAW directory: ~/Documents/Surge XT/Patches/AI Inversions/")


if __name__ == "__main__":
    main()
