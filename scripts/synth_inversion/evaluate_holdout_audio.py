"""Audio-domain comparison of inverters on held-out synthetic notes.

Why this exists: parameter-space scores cannot see that two different patches can sound
the same (and categorical/continuous errors are not commensurable), so the only fair
cross-model comparison is to RENDER each prediction through Surge and score the AUDIO
against the held-out target with one shared metric (audio_utils.MultiScaleSTFTLoss).

Every run also renders two reference arms so the metric's dynamic range is visible:
  oracle      true params re-rendered  -> the floor (render reproducibility; ~0 expected)
  mean_patch  the training-set mean patch for every target -> the trivial baseline
A model that does not clearly beat mean_patch has learned nothing audible.

Methods per checkpoint: ResMLP -> its point estimate; flow -> draw 0, and best-of-K
(K draws rendered, the closest to the target kept — this peeks at the target, so it is a
renderer-in-the-loop number, reported separately, never mixed with draw 0).
All renders use the TRUE midi note and note length (pitch is not what is being tested).

Usage:
  python evaluate_holdout_audio.py --ckpt resmlp=/path/G01_best.pt --ckpt flow=/path/G02_best.pt --n 200
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np
import torch

from audio_utils import MultiScaleSTFTLoss
from inference import is_flow, load_inverter, parse_ckpt_specs, predict_vectors, vectors_to_patches
from surge_spec import (DEFAULT_PLUGIN_PATH, NOTE_DUR_RANGE, SEED_OFFSET, draw_patch, init_synth, patch_to_vector,
                        render_patch, vector_to_patch)


def note_durations(h5, indices):
    """note_dur per index: stored (generator v2) or recovered by replaying the frozen RNG
    stream (v1 files) — the replay is verified against the stored params, sample by sample."""
    if "note_dur" in h5:
        return np.array([h5["note_dur"][i] for i in indices], dtype=np.float32)
    seed_offset = int(h5.attrs.get("seed_offset", SEED_OFFSET))
    nd_range = tuple(h5.attrs.get("note_dur_range", NOTE_DUR_RANGE))
    durs, bad = [], []
    for i in indices:
        patch = draw_patch(int(i) + seed_offset, nd_range)
        if not np.allclose(patch_to_vector(patch), h5["params"][i], atol=1e-6):
            bad.append(int(i))
        durs.append(patch["note_dur"])
    if bad:
        raise RuntimeError(f"RNG replay does not reproduce stored params for {len(bad)}/{len(indices)} samples "
                           f"(first: {bad[:5]}); this h5 was not made by the frozen draw_patch stream, "
                           "so note_dur cannot be recovered. Regenerate with generator v2.")
    return np.array(durs, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5")
    ap.add_argument("--ckpt", action="append", required=True, help="NAME=PATH, repeatable")
    ap.add_argument("--n", type=int, default=200, help="held-out notes to render")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="must match training's split")
    ap.add_argument("--flow_draws", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=20)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out_dir", default="/run/media/kim/Mantu/surge_200k_models/holdout_audio_eval")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    loss_fn = MultiScaleSTFTLoss()
    synth = init_synth(args.plugin)

    with h5py.File(args.h5, "r") as h5:
        total = len(h5["params"])
        val_start = total - int(total * args.val_ratio)
        indices = np.unique(np.linspace(val_start, total - 1, args.n).astype(int))
        params = np.stack([h5["params"][i] for i in indices])
        audio = np.stack([h5["audio"][i] for i in indices])
        mels = torch.from_numpy(np.stack([h5["mel"][i] for i in indices]))
        durs = note_durations(h5, list(indices))
        mean_vec = np.asarray(h5["params"][: min(val_start, 20000)]).mean(axis=0)

    true_patches = [vector_to_patch(p) for p in params]
    notes = [p["midi_note"] for p in true_patches]
    mean_patch = vector_to_patch(mean_vec)

    rows = []  # (index, method, stft_loss)

    def score(patch, k):
        return loss_fn(render_patch(synth, patch, notes[k], float(durs[k])), audio[k])

    print(f"Rendering reference arms on {len(indices)} held-out notes...")
    for k, idx in enumerate(indices):
        rows.append((int(idx), "oracle", score(true_patches[k], k)))
        rows.append((int(idx), "mean_patch", score(mean_patch, k)))

    for name, path in parse_ckpt_specs(args.ckpt):
        model = load_inverter(path, device=args.device)
        flow = is_flow(model)
        print(f"{name}: {type(model).__name__} ({model.encoding}) from {path}")
        for start in range(0, len(indices), 64):
            vecs, _ = predict_vectors(model, mels[start:start + 64], n_draws=args.flow_draws if flow else 1,
                                      steps=args.flow_steps, seed=start)
            for j in range(vecs.shape[1]):
                k = start + j
                losses = [score(p, k) for p in vectors_to_patches(vecs[:, j])]
                if flow:
                    rows.append((int(indices[k]), f"{name}/draw0", losses[0]))
                    rows.append((int(indices[k]), f"{name}/best_of_{args.flow_draws}", min(losses)))
                else:
                    rows.append((int(indices[k]), f"{name}/point", losses[0]))

    with open(os.path.join(args.out_dir, "holdout_audio_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h5_index", "method", "stft_loss"])
        w.writerows(rows)

    methods = list(dict.fromkeys(m for _, m, _ in rows))
    by = {m: np.array([l for _, mm, l in rows if mm == m]) for m in methods}
    summary = {"n": int(len(indices)), "h5": args.h5, "metric": "audio_utils.MultiScaleSTFTLoss (lower = better)",
               "methods": {}, "win_rate_vs_mean_patch": {}}
    print(f"\n{'method':28s} {'mean':>8s} {'median':>8s} {'sem':>7s} {'beats mean_patch':>17s}")
    for m in methods:
        v = by[m]
        win = float(np.mean(v < by["mean_patch"]))
        summary["methods"][m] = {"mean": float(v.mean()), "median": float(np.median(v)),
                                 "sem": float(v.std(ddof=1) / np.sqrt(len(v)))}
        summary["win_rate_vs_mean_patch"][m] = win
        print(f"{m:28s} {v.mean():8.3f} {np.median(v):8.3f} {summary['methods'][m]['sem']:7.3f} {win:16.1%}")
    model_methods = [m for m in methods if m not in ("oracle", "mean_patch") and "/best_of_" not in m]
    summary["pairwise_win_rate"] = {f"{a} < {b}": float(np.mean(by[a] < by[b]))
                                    for a in model_methods for b in model_methods if a != b}
    with open(os.path.join(args.out_dir, "holdout_audio_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out_dir}/holdout_audio_scores.csv and holdout_audio_summary.json")


if __name__ == "__main__":
    main()
