"""Single-note match: neural inversion vs a closed-loop Differential Evolution (DE) search.

v2 (2026-10-01 review) — what changed and why:
  * DE state leak fixed. v1 tuned 8 parameters but rendered the ResMLP prediction just
    before the search and never reset the rest, so width, sync, FM depth, unison, filter-EG
    sustain and amp sustain were silently the ResMLP's values in every DE candidate — and
    the reported "narrow pulse (shape = 0.919)" depended on a pulse width DE never touched.
    Now every render sets the FULL patch (surge_spec.apply_patch) from an explicit base
    patch, and DE also tunes width, filter-EG sustain and amp sustain (11 dims).
  * The DE result is a REFERENCE, not ground truth: a finite search that optimises the very
    metric it is then scored on. Its budget (--de_maxiter, --de_popsize, --filters) is
    printed and saved with the result.
  * Out-of-distribution flags: the target is prepared like training data (44.1 kHz, onset
    at t~0, 0.8 s window) and a warning is recorded when the clip or note length falls
    outside what the models were trained on (v1's 99 ms / 80 ms note was, silently).
  * Scoring uses only the target's real content (info['score_len']); past the end of a clip
    cut from a track the truth is unknown, not silence.
  * One shared metric (audio_utils.MultiScaleSTFTLoss) for every number.
  * Patches export as JSON + .pedalboard_state. v1 wrote pedalboard's raw_state with a
    .vstpreset extension, but it is not in VST3 preset format.

Usage:
  python match_untitled_note.py --target untitled.wav --midi_note 36 --note_dur 0.08 \
      --ckpt resmlp=/path/G01_best.pt --ckpt flow=/path/G02_best.pt
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.optimize import differential_evolution

from audio_utils import MultiScaleSTFTLoss, make_mel_spec, prepare_target, save_patch, spectral_centroid
from inference import is_flow, load_inverter, parse_ckpt_specs, predict_vectors, vectors_to_patches
from surge_spec import DEFAULT_PLUGIN_PATH, LP_FILTERS, SAMPLE_RATE, init_synth, render_patch

RESULTS_DIR = "/run/media/kim/Mantu/surge_200k_models/stem_inversion_results"
DEFAULT_CKPTS = [
    "resmlp_200k=/run/media/kim/Mantu/surge_200k_models/G01_resmlp_deep_200k_best.pt",
    "deepflow_200k=/run/media/kim/Mantu/surge_200k_models/G02_deepflow_200k_best.pt",
]

# Explicit base patch for DE: everything not searched is pinned here, never inherited.
DE_BASE_PATCH = dict(
    filter_idx=0, shape=0.5, width=0.5, sub_mix=0.0, sync=0.0, fm_depth=0.0, unison=False, unison_detune=0.0,
    cutoff=0.5, resonance=0.3, keytrack_raw=0.77, feg_amount=0.6, feg_decay=0.3, feg_sustain=0.0,
    aeg_decay=0.3, aeg_sustain=0.0, aeg_release=0.1, ws_idx=0, drive_raw=0.50,
    chorus_mix=0.0, delay_mix=0.0, delay_fb=0.0,
)
# Searched parameters, bounded by the training distribution (surge_spec.draw_patch).
DE_SPACE = [
    ("shape", 0.0, 1.0),
    ("width", 0.0, 1.0),
    ("sub_mix", 0.0, 0.85),
    ("cutoff", 0.08, 0.92),
    ("resonance", 0.0, 0.85),
    ("feg_amount", 0.2, 0.95),
    ("feg_decay", 0.03, 0.65),
    ("feg_sustain", 0.0, 0.60),
    ("aeg_decay", 0.05, 0.65),
    ("aeg_sustain", 0.0, 0.80),
    ("aeg_release", 0.01, 0.40),
]


def de_patch(x, filter_idx):
    patch = dict(DE_BASE_PATCH, filter_idx=filter_idx)
    patch.update({name: float(v) for (name, _lo, _hi), v in zip(DE_SPACE, x)})
    return patch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default=os.path.join(RESULTS_DIR, "untitled.wav"))
    ap.add_argument("--midi_note", type=int, default=36)
    ap.add_argument("--note_dur", type=float, default=0.08, help="seconds; training range is 0.18-0.45")
    ap.add_argument("--ckpt", action="append", help="NAME=PATH, repeatable (default: G01 + G02)")
    ap.add_argument("--flow_draws", type=int, default=8, help="flow: draw 0 is headline; best-of-K also reported")
    ap.add_argument("--flow_steps", type=int, default=20)
    ap.add_argument("--filters", default="all", help="'all' or comma-separated circuit names for DE")
    ap.add_argument("--de_maxiter", type=int, default=15)
    ap.add_argument("--de_popsize", type=int, default=8, help="scipy popsize multiplier (x 11 dims)")
    ap.add_argument("--de_seed", type=int, default=42)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--out_dir", default=os.path.join(RESULTS_DIR, "untitled_note_match"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    y, info = prepare_target(args.target, note_dur=args.note_dur)
    L = info["score_len"]
    print(f"Target: {args.target} | {info['content_len_s']*1000:.1f} ms content, scored over {L} samples "
          f"| orig SR {info['orig_sr']} | OOD flags: {info['out_of_distribution'] or 'none'}")
    loss_fn = MultiScaleSTFTLoss()
    synth = init_synth(args.plugin)

    def render(patch):
        return render_patch(synth, patch, args.midi_note, args.note_dur)

    def score(audio):
        return loss_fn(audio[:L], y[:L])

    results = {"target": info, "midi_note": args.midi_note, "note_dur": args.note_dur,
               "metric": "audio_utils.MultiScaleSTFTLoss over the target's content (lower = better)", "methods": {}}
    plot_rows = [("Target", y, "royalblue", 0.0)]

    # 1. Neural inversion
    mel = torch.from_numpy(make_mel_spec(y)).unsqueeze(0)
    for name, path in parse_ckpt_specs(args.ckpt or DEFAULT_CKPTS):
        model = load_inverter(path)
        flow = is_flow(model)
        vecs, lat_ms = predict_vectors(model, mel, n_draws=args.flow_draws if flow else 1, steps=args.flow_steps)
        patches = vectors_to_patches(vecs[:, 0])
        losses = [score(render(p)) for p in patches]
        picks = [("draw0" if flow else "point", 0)]
        if flow and len(patches) > 1:
            picks.append((f"best_of_{len(patches)}", int(np.argmin(losses))))
        for label, k in picks:
            key = f"{name}/{label}"
            audio = render(patches[k])  # also leaves the synth in this patch's state for export
            stem = os.path.join(args.out_dir, f"untitled_{name}_{label}")
            sf.write(f"{stem}.wav", audio, SAMPLE_RATE)
            save_patch(synth, patches[k], stem, extra={"method": key, "stft_loss": losses[k], "latency_ms": lat_ms})
            results["methods"][key] = {"stft_loss": losses[k], "filter": LP_FILTERS[patches[k]["filter_idx"]][0],
                                       "centroid": spectral_centroid(audio[:L]), "latency_ms": lat_ms}
            plot_rows.append((f"{key} ({LP_FILTERS[patches[k]['filter_idx']][0]})", audio, "forestgreen", losses[k]))
            print(f"{key:28s} | Filter: {LP_FILTERS[patches[k]['filter_idx']][0]:18s} | STFT Loss: {losses[k]:.3f}")

    # 2. Closed-loop DE reference
    filters = (list(range(len(LP_FILTERS))) if args.filters == "all"
               else [[n for n, _ in LP_FILTERS].index(f.strip()) for f in args.filters.split(",")])
    budget = {"maxiter": args.de_maxiter, "popsize": args.de_popsize, "dims": len(DE_SPACE),
              "renders_per_filter_max": (args.de_maxiter + 1) * args.de_popsize * len(DE_SPACE),
              "filters": [LP_FILTERS[i][0] for i in filters], "seed": args.de_seed,
              "base_patch": DE_BASE_PATCH, "space": DE_SPACE}
    print(f"\nDE reference search: {len(filters)} circuits x <= {budget['renders_per_filter_max']} renders each")
    t_de = time.time()
    best = None
    per_filter = {}
    for fi in filters:
        res = differential_evolution(lambda x: score(render(de_patch(x, fi))), bounds=[(lo, hi) for _, lo, hi in DE_SPACE],
                                     maxiter=args.de_maxiter, popsize=args.de_popsize, seed=args.de_seed, polish=False)
        per_filter[LP_FILTERS[fi][0]] = float(res.fun)
        print(f"  {LP_FILTERS[fi][0]:18s}: best STFT loss {res.fun:.3f} ({res.nfev} renders)")
        if best is None or res.fun < best[0]:
            best = (float(res.fun), de_patch(res.x, fi))
    de_loss, de_best = best
    budget["seconds"] = time.time() - t_de
    de_audio = render(de_best)
    stem = os.path.join(args.out_dir, "untitled_de_reference")
    sf.write(f"{stem}.wav", de_audio, SAMPLE_RATE)
    save_patch(synth, de_best, stem, extra={"method": "de_reference", "stft_loss": de_loss, "budget": budget})
    results["methods"]["de_reference"] = {"stft_loss": de_loss, "filter": LP_FILTERS[de_best["filter_idx"]][0],
                                          "centroid": spectral_centroid(de_audio[:L]), "per_filter": per_filter,
                                          "budget": budget, "patch": {n: de_best[n] for n, _, _ in DE_SPACE}}
    plot_rows.insert(1, (f"DE reference ({LP_FILTERS[de_best['filter_idx']][0]})", de_audio, "crimson", de_loss))
    print(f"\nDE reference: {LP_FILTERS[de_best['filter_idx']][0]} | STFT loss {de_loss:.3f} | {budget['seconds']:.0f} s")
    print("  " + ", ".join(f"{n}={de_best[n]:.3f}" for n, _, _ in DE_SPACE))

    results["target_centroid"] = spectral_centroid(y[:L])
    sf.write(os.path.join(args.out_dir, "untitled_target_prepared.wav"), y, SAMPLE_RATE)
    with open(os.path.join(args.out_dir, "untitled_match_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, axs = plt.subplots(len(plot_rows), 2, figsize=(14, 2.5 * len(plot_rows)), squeeze=False)
    t_ms = np.arange(L) / SAMPLE_RATE * 1000
    for idx, (title, sig, col, loss_val) in enumerate(plot_rows):
        axs[idx, 0].plot(t_ms, sig[:L], color=col, alpha=0.85)
        axs[idx, 0].set_title(f"{title} - Waveform (STFT={loss_val:.2f})", fontsize=10)
        axs[idx, 0].set_ylim(-1.05, 1.05)
        axs[idx, 0].set_xlabel("Time (ms)")
        axs[idx, 0].grid(True, alpha=0.3)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(sig[:L], n_fft=512, hop_length=64)), ref=np.max)
        librosa.display.specshow(S, sr=SAMPLE_RATE, hop_length=64, x_axis="time", y_axis="hz",
                                 ax=axs[idx, 1], cmap="magma", vmin=-50, vmax=0)
        axs[idx, 1].set_title(f"{title} - Spectrogram", fontsize=10)
        axs[idx, 1].set_ylim(0, 4500)
    plt.tight_layout()
    plot_p = os.path.join(args.out_dir, "untitled_match_comparison.png")
    plt.savefig(plot_p, dpi=150)
    plt.close()
    print(f"\nSaved {plot_p} and untitled_match_results.json in {args.out_dir}/")


if __name__ == "__main__":
    main()
