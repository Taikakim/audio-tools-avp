# Owner: KUANG (created 2026-10-13)
"""Offline preference data for Flow-DPO (V5_PROPOSAL_REVIEW.md, sec. 6): for each condition, K flow candidates are
sampled, rendered by Surge and scored against the target audio. train_flow_dpo_v5.py turns the scores into
winner / loser pairs.

Conditions:
  --source synth   targets rendered from TRAINING-split prior draws (never the held-out presets the evaluation uses),
                   drawn from their own seed stream (SeedSequence([seed, 2, i])); the true vector is stored too, and
                   the true vector is rendered a second time to measure Surge's render-to-render noise per condition
                   (the floor below which a score difference means nothing).
  --source audio   real, unlabeled clips: a CSV with columns path,midi_note (mono or stereo wav, cut or zero-padded
                   to 0.8 s at 44.1 kHz and peak-normalised like the renders). Keep the evaluation stems OUT of this set.

Why offline: rendering is CPU-bound; doing it once, before training, keeps the DPO run at full GPU speed and makes
the data reusable for several beta / learning-rate settings. The policy drifts from the sampler during DPO (off-policy
by the end); for another round, sample again from the DPO checkpoint.

Output DIR: shard_NNNNN.npz (target int16 audio, midi, candidate vectors [n, K, P], scores per metric [n, K],
true vectors and noise floors for synth) and manifest.json (flow file + md5 + step, K, steps, tau, metrics, seeds,
and the HEADROOM summary: mean score of candidate 0 vs best-of-K vs worst-of-K. If best-of-K is barely better than
candidate 0, there is little for DPO to learn).

Layouts (KUANG 2026-10-19): the flow's layout (layout_v6: 92 -> v3, 108 -> v4, 126 -> v5) picks the spec Surge
renders with and the prior. v3 flows use --manifold (realistic_bass_prior_v3); v4 / v5 flows use the soundset prior
of their trainer (realistic_prior_v4.UnifiedSoundsetPrior, --bass_manifold and --all_manifold).

Usage (desktop):
  python dpo_pairs_v5.py --flow RUN/flow_ema_latest.pt --manifold M.npz --plugin "Surge XT.vst3" \\
      --source synth --n 8192 --k 8 --tau 1.0 --workers 14 --out dpo_pairs_a3p_s0 --device cuda:0
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import audio_metrics_v5 as am  # noqa: E402

N_AUDIO = 35280


def load_clip(path):
    import librosa
    y, _ = librosa.load(path, sr=44100, mono=True)
    y = y[:N_AUDIO] if len(y) >= N_AUDIO else np.pad(y, (0, N_AUDIO - len(y)))
    return (y / (np.max(np.abs(y)) + 1e-7)).astype(np.float32)


def synth_conditions(prior, start, n, seed):
    from render_bank_v5 import item_rng
    vecs, midi = [], []
    for i in range(start, start + n):
        _, v, m, _ = prior.sample_patch_and_midi(item_rng(seed, 2, i))
        vecs.append(np.asarray(v, np.float32))
        midi.append(int(m))
    return np.stack(vecs), np.array(midi)


def build_shard(flow, pool, device, targets, midi, k, steps, tau, seed, metrics, true_vecs=None):
    """-> dict with candidate vectors [n, K, P], scores {metric: [n, K]} and the noise floor (synth only)."""
    from flow_sampling_v5 import mel_of, sample_candidates
    mel = mel_of(targets, device)
    cands = sample_candidates(flow, mel, midi, k=k, steps=steps, seed=seed, tau=tau)
    n, K, P = cands.shape
    audio = pool.render(cands.reshape(n * K, P), np.repeat(midi, K))
    tgt = [targets[i] for i in range(n) for _ in range(K)]
    sc = am.score_many(tgt, list(audio), metrics)
    out = {"cand_vecs": cands, **{f"score__{m}": v.reshape(n, K) for m, v in sc.items()}}
    if true_vecs is not None:
        again = pool.render(true_vecs, midi)                    # render noise: same vector, second render
        fl = am.score_many(list(targets), list(again), metrics)
        out.update({f"noise__{m}": v for m, v in fl.items()})
    return out


def make_prior(layout_name, a):
    """-> (prior with sample_patch_and_midi(rng), md5 string) for the flow's layout, training split."""
    if layout_name == "v3":
        from realistic_bass_prior_v3 import RealisticBassPrior, manifold_md5
        if not a.manifold:
            raise SystemExit("--source synth with a v3 flow needs --manifold")
        return RealisticBassPrior(a.manifold, split="train"), manifold_md5(a.manifold)
    from realistic_prior_v4 import UnifiedSoundsetPrior, manifold_md5
    if not (a.bass_manifold and a.all_manifold):
        raise SystemExit(f"--source synth with a {layout_name} flow needs --bass_manifold and --all_manifold")
    if layout_name == "v6":
        from soundset_prior_v6 import SoundsetPriorV6
        cls = SoundsetPriorV6
    else:
        cls = UnifiedSoundsetPrior
    return (cls(a.bass_manifold, a.all_manifold, split="train"),
            manifold_md5(a.bass_manifold) + "+" + manifold_md5(a.all_manifold))


def headroom(shards, metric):
    s = np.concatenate([sh[f"score__{metric}"] for sh in shards])
    res = {"metric": metric, "n": int(len(s)), "draw0": float(np.nanmean(s[:, 0])), "best": float(np.nanmean(np.nanmin(s, 1))),
           "worst": float(np.nanmean(np.nanmax(s, 1))), "median_spread": float(np.nanmedian(np.nanmax(s, 1) - np.nanmin(s, 1)))}
    noise = [sh[f"noise__{metric}"] for sh in shards if f"noise__{metric}" in sh]
    if noise:
        nz = np.concatenate(noise)
        res["noise_floor_median"] = float(np.nanmedian(nz))
        res["share_spread_gt_noise"] = float(np.mean((np.nanmax(s, 1) - np.nanmin(s, 1)) > nz))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow", required=True, help="flow_*.pt to sample from (the DPO reference and initial policy)")
    ap.add_argument("--source", choices=("synth", "audio"), default="synth")
    ap.add_argument("--manifold", help="prior for --source synth (v3 flows)")
    ap.add_argument("--bass_manifold", help="soundset prior for --source synth (v4 / v5 flows)")
    ap.add_argument("--all_manifold", help="soundset prior for --source synth (v4 / v5 flows)")
    ap.add_argument("--audio_csv", help="path,midi_note rows for --source audio")
    ap.add_argument("--plugin", default=None)
    ap.add_argument("--n", type=int, default=8192, help="conditions (synth)")
    ap.add_argument("--k", type=int, default=8, help="candidates per condition")
    ap.add_argument("--steps", type=int, default=25, help="Euler steps")
    ap.add_argument("--tau", type=float, default=1.0, help="condition noise level (cond_noise models); 1 = clean")
    ap.add_argument("--metrics", nargs="+", default=["logmel_l1", "mss", "mel_ssim_gated_dist"],
                    help="first one is what train_flow_dpo_v5.py ranks by unless told otherwise")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import torch
    from flow_sampling_v5 import file_md5
    from layout_v6 import layout_for_flow
    from models_v4 import load_inverter_v4
    from render_pool_v5 import RenderPool, to_int16
    os.makedirs(a.out, exist_ok=True)
    flow = load_inverter_v4(a.flow, device=a.device)
    layout = layout_for_flow(flow)
    meta_f = torch.load(a.flow, map_location="cpu", weights_only=False)
    prior_md5 = None
    if a.source == "synth":
        prior, prior_md5 = make_prior(layout.name, a)
        n_total, rows = a.n, None
    else:
        if not a.audio_csv:
            raise SystemExit("--source audio needs --audio_csv")
        with open(a.audio_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        n_total = len(rows)
    manifest = {"flow": a.flow, "flow_md5": file_md5(a.flow), "flow_step": meta_f.get("step"), "source": a.source,
                "layout": layout.name, "manifold": a.manifold, "bass_manifold": a.bass_manifold,
                "all_manifold": a.all_manifold, "prior_md5": prior_md5,
                "audio_csv": a.audio_csv, "n": n_total, "k": a.k, "steps": a.steps, "tau": a.tau, "metrics": a.metrics,
                "seed": a.seed, "shard": a.shard, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    shards, t0 = [], time.time()
    with RenderPool(a.plugin, workers=a.workers, spec=layout.name) as pool:
        for s in range(math.ceil(n_total / a.shard)):
            path = os.path.join(a.out, f"shard_{s:05d}.npz")
            if os.path.exists(path):
                with np.load(path) as z:
                    shards.append({key: z[key] for key in z.files})
                continue
            lo, hi = s * a.shard, min((s + 1) * a.shard, n_total)
            if a.source == "synth":
                true_vecs, midi = synth_conditions(prior, lo, hi - lo, a.seed)
                targets = pool.render(true_vecs, midi)
            else:
                true_vecs = None
                midi = np.array([int(r["midi_note"]) for r in rows[lo:hi]])
                targets = np.stack([load_clip(r["path"]) for r in rows[lo:hi]])
            sh = build_shard(flow, pool, a.device, targets, midi, a.k, a.steps, a.tau, a.seed * 100_003 + s, a.metrics,
                             true_vecs)
            sh.update({"target_audio": to_int16(targets), "midi": midi, "cond_index": np.arange(lo, hi)})
            if true_vecs is not None:
                sh["true_vecs"] = true_vecs
            np.savez(path + ".tmp.npz", **sh)
            os.replace(path + ".tmp.npz", path)
            shards.append(sh)
            print(f"shard {s + 1}: {hi} conditions in {time.time() - t0:.0f} s", flush=True)
    manifest["headroom"] = {m: headroom(shards, m) for m in a.metrics}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    for m, h in manifest["headroom"].items():
        print(f"{m}: candidate 0 {h['draw0']:.4f} | best of {a.k} {h['best']:.4f} | worst {h['worst']:.4f}"
              + (f" | render noise {h['noise_floor_median']:.4f}, spread > noise in {100 * h['share_spread_gt_noise']:.0f}%"
                 if "noise_floor_median" in h else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
