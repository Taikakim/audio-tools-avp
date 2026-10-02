"""Invert real stems with trained models, render the result through Surge, score the audio.

v2 (2026-10-01 review):
  * Targets go through audio_utils.prepare_target(): resampled to 44.1 kHz (v1 fed e.g.
    48 kHz audio straight into a 44.1 kHz mel front-end; the CNN's adaptive pooling hid it),
    leading silence trimmed so the onset sits at t~0 like the training renders, and
    out-of-distribution warnings (short clip, note length) recorded in the results.
  * One shared scoring metric (audio_utils.MultiScaleSTFTLoss) — v1's copy differed from
    match_untitled_note.py's, so their losses were not comparable.
  * Models load through models.load_inverter (v1 and v2 checkpoints).
  * Flow: draw 0 is the headline number; --flow_draws K>1 adds a best-of-K number that
    peeks at the target (renderer-in-the-loop) and is labelled as such.
  * Patches are exported as JSON + pedalboard state (audio_utils.save_patch); v1 called a
    JSON dict a "preset".
  * Paths are CLI arguments (defaults = the v1 hard-coded paths).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from audio_utils import MultiScaleSTFTLoss, make_mel_spec, prepare_target, save_patch, spectral_centroid
from inference import is_flow, load_inverter, parse_ckpt_specs, predict_vectors, vectors_to_patches
from surge_spec import DEFAULT_PLUGIN_PATH, LP_FILTERS, SAMPLE_RATE, WAVESHAPER_TYPES, init_synth, render_patch

DEFAULT_CKPTS = [
    "200k_resmlp=/run/media/kim/Mantu/surge_200k_models/G01_resmlp_deep_200k_best.pt",
    "200k_deepflow=/run/media/kim/Mantu/surge_200k_models/G02_deepflow_200k_best.pt",
]
DEFAULT_STEMS = [
    "acid_303_pluck:/run/media/kim/Mantu/surge_stem_inversion/targets/target_acid_pluck_a2.wav:45:0.22",
    "goa_disco_bass:/run/media/kim/Mantu/surge_stem_inversion/targets/target_disco_bass_f2.wav:41:0.28",
]


def parse_stem(spec):
    if spec.count(":") < 3:
        raise ValueError(f"--stem expects NAME:PATH:MIDI_NOTE:NOTE_DUR_S, got {spec!r}")
    head, midi, note_dur = spec.rsplit(":", 2)
    name, path = head.split(":", 1)
    return name, path, int(midi), float(note_dur)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", help="NAME=PATH, repeatable (default: G01 + G02)")
    ap.add_argument("--stem", action="append", help="NAME:PATH:MIDI_NOTE:NOTE_DUR_S, repeatable")
    ap.add_argument("--flow_draws", type=int, default=1, help=">1 adds a best-of-K (target-peeking) number")
    ap.add_argument("--flow_steps", type=int, default=20)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--out_dir", default="/run/media/kim/Mantu/surge_200k_models/stem_inversion_results")
    args = ap.parse_args()

    ckpts = parse_ckpt_specs(args.ckpt or DEFAULT_CKPTS)
    stems = [parse_stem(s) for s in (args.stem or DEFAULT_STEMS)]
    os.makedirs(args.out_dir, exist_ok=True)

    models = [(name, load_inverter(path)) for name, path in ckpts]
    for name, m in models:
        print(f"Loaded {name}: {type(m).__name__} ({m.encoding})")
    synth = init_synth(args.plugin)
    loss_fn = MultiScaleSTFTLoss()
    results = []

    for stem_name, wav_path, midi_note, note_dur in stems:
        print(f"\n{'=' * 70}\nEvaluating on: {stem_name} (MIDI {midi_note}, note {note_dur*1000:.0f} ms)\n{'=' * 70}")
        y, info = prepare_target(wav_path, note_dur=note_dur)
        mel_t = torch.from_numpy(make_mel_spec(y)).unsqueeze(0)
        sc_target = spectral_centroid(y)
        rendered = {}

        for m_name, model in models:
            flow = is_flow(model)
            vecs, lat_ms = predict_vectors(model, mel_t, n_draws=args.flow_draws if flow else 1, steps=args.flow_steps)
            patches = vectors_to_patches(vecs[:, 0])
            audios = [render_patch(synth, p, midi_note, note_dur) for p in patches]
            L = info["score_len"]
            losses = [loss_fn(a[:L], y[:L]) for a in audios]
            picks = [("draw0" if flow else "point", 0)]
            if flow and len(patches) > 1:
                picks.append((f"best_of_{len(patches)}", int(np.argmin(losses))))
            for label, k in picks:
                key = f"{m_name}/{label}"
                patch, audio = patches[k], audios[k]
                render_patch(synth, patch, midi_note, note_dur)  # leave synth in this patch's state for export
                stem_path = os.path.join(args.out_dir, f"{stem_name}_{m_name}_{label}")
                sf.write(f"{stem_path}.wav", audio, SAMPLE_RATE)
                save_patch(synth, patch, stem_path, extra={"stem": stem_name, "model": key, "latency_ms": lat_ms,
                                                            "stft_loss": losses[k], "target": info})
                rendered[key] = {"audio": audio, "filter": LP_FILTERS[patch["filter_idx"]][0],
                                 "waveshaper": WAVESHAPER_TYPES[patch["ws_idx"]][0], "stft_loss": losses[k],
                                 "centroid": spectral_centroid(audio), "latency_ms": lat_ms}
                r = rendered[key]
                print(f"{key:28s} | Filter: {r['filter']:18s} | Shaper: {r['waveshaper']:10s} | STFT Loss: {r['stft_loss']:.3f} "
                      f"| Centroid: {r['centroid']:6.1f} Hz | Latency: {lat_ms:.1f} ms")
        print(f"Target centroid: {sc_target:.1f} Hz | OOD flags: {info['out_of_distribution'] or 'none'}")

        results.append({"stem": stem_name, "target": info, "target_centroid": sc_target,
                        "methods": {k: {kk: vv for kk, vv in v.items() if kk != "audio"} for k, v in rendered.items()}})

        rows = [("Target Stem", y, "royalblue", 0.0)] + [
            (f"{k} ({v['filter']}, {v['waveshaper']})", v["audio"], "crimson", v["stft_loss"]) for k, v in rendered.items()]
        fig, axs = plt.subplots(len(rows), 2, figsize=(14, 2.7 * len(rows)), squeeze=False)
        t_axis = np.arange(len(y)) / SAMPLE_RATE
        for idx, (title, sig, col, loss) in enumerate(rows):
            axs[idx, 0].plot(t_axis, sig, color=col, alpha=0.8)
            axs[idx, 0].set_title(f"{title} - Waveform (STFT={loss:.2f})", fontsize=9)
            axs[idx, 0].set_ylim(-1.05, 1.05)
            axs[idx, 0].grid(True, alpha=0.3)
            S = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=1024, hop_length=128)), ref=np.max)
            librosa.display.specshow(S, sr=SAMPLE_RATE, hop_length=128, x_axis="time", y_axis="hz",
                                     ax=axs[idx, 1], cmap="magma", vmin=-60, vmax=0)
            axs[idx, 1].set_title(f"{title} - Spectrogram", fontsize=9)
            axs[idx, 1].set_ylim(0, 5000)
        plt.tight_layout()
        plot_p = os.path.join(args.out_dir, f"{stem_name}_comparison.png")
        plt.savefig(plot_p, dpi=150)
        plt.close()
        print(f"Saved comparison figure: {plot_p}")

    with open(os.path.join(args.out_dir, "stem_inversion_results.json"), "w") as f:
        json.dump({"metric": "audio_utils.MultiScaleSTFTLoss (lower = better)", "results": results}, f, indent=2)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
