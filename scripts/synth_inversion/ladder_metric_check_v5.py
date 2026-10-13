# Owner: KUANG (created 2026-10-13)
"""Gate step 2 for audio metrics (V5_PROPOSAL_REVIEW.md, sec. 3): does a metric's distance grow along a one-parameter
sweep of real Surge renders?

For held-out anchors (the prior's 'val' split) each axis is swept in n_steps with everything else fixed
(training_controls_v3.make_ladder for the four training axes, plus new axes below). Every candidate metric scores
distance(step 0, step k) for k = 1..n_steps-1; a usable metric rises monotonically, so we report per metric and axis
  rho          mean Spearman(k, distance) over ladders, with a 95 % bootstrap CI over ladders
  rho>=0.8     share of ladders with Spearman >= 0.8
  step1/noise  median over ladders of distance(step 0, step 1) / distance(step 0, a second render of step 0):
               > 1 means a single step is larger than Surge's render-to-render noise for that metric
New axes (beyond cutoff_low, aeg_sustain, feg_amount, aeg_decay):
  unison_detune  unison on (2 voices), detune swept over its range        -> beating (modulation spectrum)
  drive          waveshaper "Soft", drive swept over its range            -> noisiness / flatness
  resonance      cutoff held at 0.25..0.55, resonance swept               -> spectral peak
  fm_depth       FM routing "2 > 1", depth swept                          -> sidebands
  noise_volume   noise oscillator level swept 0..0.25                     -> flatness
Caveat: monotonic in the PARAMETER is not the same as monotonic in perception (a sweep can cross a perceptual
plateau). This is a necessary condition for a metric, not a sufficient one; the ear decides (listening_test_v5.py).

Usage (desktop, real Surge):
  python ladder_metric_check_v5.py --manifold M.npz --plugin "Surge XT.vst3" --out ladder_gate_v5 --workers 8
  python ladder_metric_check_v5.py --dry_run ...    # print the plan only
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import audio_metrics_v5 as am  # noqa: E402
from surge_spec_v3 import CONT_BOUNDS, PARAM_INDEX, WAVESHAPER_TYPES, canonicalize_vector  # noqa: E402
from training_controls_v3 import AXES as TRAIN_AXES  # noqa: E402
from training_controls_v3 import make_ladder  # noqa: E402

NEW_AXES = ("unison_detune", "drive", "resonance", "fm_depth", "noise_volume")
AXES_V5 = tuple(TRAIN_AXES) + NEW_AXES
AXIS_PARAM_V5 = {"cutoff_low": "cutoff", "aeg_sustain": "aeg_sustain", "feg_amount": "feg_amount",
                 "aeg_decay": "aeg_decay", "unison_detune": "unison_detune", "drive": "drive",
                 "resonance": "resonance", "fm_depth": "fm_depth", "noise_volume": "noise_volume"}
WS_SOFT = 1.0 / (len(WAVESHAPER_TYPES) - 1)        # class 1 ("Soft") in the ordinal storage
FM_ROUTING_2_TO_1 = 1.0 / 3.0                       # class 1 of Off, 2>1, 3>2>1, 2>1<3


def make_ladder_v5(base_vec, axis, n_steps, rng):
    """[n_steps, P] vectors: base swept along `axis` (ascending), canonicalised like every training sample."""
    if axis in TRAIN_AXES:
        return make_ladder(base_vec, axis, n_steps, rng)
    I = PARAM_INDEX
    v = np.array(base_vec, dtype=np.float32, copy=True)
    name = AXIS_PARAM_V5[axis]
    lo, hi = CONT_BOUNDS[name]
    if axis == "unison_detune":
        v[I["unison"]] = 1.0
    elif axis == "drive":
        v[I["waveshaper_type"]] = WS_SOFT
    elif axis == "resonance":
        v[I["cutoff"]] = float(np.clip(rng.uniform(0.25, 0.55), *CONT_BOUNDS["cutoff"]))
    elif axis == "fm_depth":
        v[I["fm_routing"]] = FM_ROUTING_2_TO_1
    elif axis != "noise_volume":
        raise ValueError(f"unknown axis {axis!r}; known: {AXES_V5}")
    out = np.repeat(v[None], n_steps, axis=0)
    out[:, I[name]] = np.linspace(lo, hi, n_steps)
    return canonicalize_vector(out)


def build_ladders(prior, axes, n_anchors, n_steps, seed):
    rng = np.random.RandomState(seed)
    vecs, midi, axis_ids = [], [], []
    for ax, axis in enumerate(axes):
        for _ in range(n_anchors):
            _, base, midi_note, _ = prior.sample_patch_and_midi(rng)
            vecs.append(make_ladder_v5(base, axis, n_steps, rng))
            midi.append(int(midi_note))
            axis_ids.append(ax)
    return {"vecs": np.stack(vecs).astype(np.float32), "midi": np.array(midi), "axis": np.array(axis_ids),
            "axes": list(axes), "n_steps": int(n_steps)}


def render_ladders(lad, pool):
    """-> audio [L, S, N] and a second render of every step 0, [L, N] (render-noise reference)."""
    L, S, P = lad["vecs"].shape
    flat = pool.render(lad["vecs"].reshape(L * S, P), np.repeat(lad["midi"], S))
    rep = pool.render(lad["vecs"][:, 0], lad["midi"])
    return flat.reshape(L, S, -1), rep


def ladder_distances(audio, rep, metrics, workers=0):
    """-> {metric: (d [L, S-1], d_rep [L])}: distance(step 0, step k) and distance(step 0, re-render of step 0)."""
    L, S, _ = audio.shape
    tgt = [audio[l, 0] for l in range(L) for _ in range(1, S)] + [audio[l, 0] for l in range(L)]
    prd = [audio[l, k] for l in range(L) for k in range(1, S)] + [rep[l] for l in range(L)]
    sc = am.score_many(tgt, prd, metrics, workers=workers)
    n = L * (S - 1)
    return {m: (v[:n].reshape(L, S - 1), v[n:]) for m, v in sc.items()}


def _spearman(x, y):
    from scipy.stats import spearmanr
    if np.any(~np.isfinite(y)) or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def summarize(dists, axis_ids, axes, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for m, (d, d_rep) in dists.items():
        S1 = d.shape[1]
        rho = np.array([_spearman(np.arange(1, S1 + 1), row) for row in d])
        ratio = d[:, 0] / np.maximum(d_rep, 1e-12)
        per_axis = {}
        for ax, axis in enumerate(axes):
            r = rho[(axis_ids == ax) & np.isfinite(rho)]
            q = ratio[(axis_ids == ax) & np.isfinite(ratio)]
            if len(r) == 0:
                per_axis[axis] = {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                                  "share_ge_0.8": float("nan"), "step1_over_noise": float("nan")}
                continue
            boots = r[rng.integers(0, len(r), size=(n_boot, len(r)))].mean(axis=1)
            per_axis[axis] = {"rho": float(r.mean()), "lo": float(np.quantile(boots, 0.025)),
                              "hi": float(np.quantile(boots, 0.975)), "n": int(len(r)),
                              "share_ge_0.8": float(np.mean(r >= 0.8)),
                              "step1_over_noise": float(np.median(q)) if len(q) else float("nan")}
        out[m] = {"family": am.METRICS[m][1], "axes": per_axis,
                  "rho_mean_over_axes": float(np.nanmean([v["rho"] for v in per_axis.values()]))}
    return out


def report_md(summary, axes):
    head = "| metric | family | " + " | ".join(axes) + " | mean |\n|---|---|" + "---|" * (len(axes) + 1) + "\n"
    rows = []
    for m, s in sorted(summary.items(), key=lambda kv: -np.nan_to_num(kv[1]["rho_mean_over_axes"], nan=-9)):
        cells = []
        for a in axes:
            v = s["axes"][a]
            cells.append("n/a" if not np.isfinite(v["rho"]) else
                         f"{v['rho']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}] ×{v['step1_over_noise']:.1f}")
        rows.append(f"| {m} | {s['family']} | " + " | ".join(cells) + f" | {s['rho_mean_over_axes']:.2f} |")
    return ("Mean Spearman(step, distance to step 0) [95 % CI over ladders] ×(step-1 distance / re-render noise).\n\n"
            + head + "\n".join(rows) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifold", required=True)
    ap.add_argument("--plugin", default=None)
    ap.add_argument("--split", default="val", help="prior split for the anchors (default: held-out presets)")
    ap.add_argument("--axes", nargs="+", default=list(AXES_V5), choices=AXES_V5)
    ap.add_argument("--anchors", type=int, default=8, help="ladders per axis")
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--metrics", nargs="+", default=sorted(am.METRICS), choices=sorted(am.METRICS))
    ap.add_argument("--workers", type=int, default=0, help="render and scoring processes (0 = in-process)")
    ap.add_argument("--save_audio", action="store_true", help="also write ladders_audio.npz (int16) for listening")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)
    from realistic_bass_prior_v3 import RealisticBassPrior, manifold_md5
    from render_pool_v5 import RenderPool, to_int16
    prior = RealisticBassPrior(a.manifold, split=a.split)
    lad = build_ladders(prior, a.axes, a.anchors, a.steps, a.seed)
    n_render = lad["vecs"].shape[0] * (a.steps + 1)
    print(f"{len(a.axes)} axes x {a.anchors} anchors x {a.steps} steps (+1 re-render) = {n_render} renders; "
          f"{len(a.metrics)} metrics")
    if a.dry_run:
        return 0
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with RenderPool(a.plugin, workers=a.workers) as pool:
        audio, rep = render_ladders(lad, pool)
    t_render = time.time() - t0
    dists = ladder_distances(audio, rep, a.metrics, workers=a.workers)
    summary = summarize(dists, lad["axis"], a.axes)
    meta = {"manifold": a.manifold, "prior_md5": manifold_md5(a.manifold), "split": a.split, "seed": a.seed,
            "anchors": a.anchors, "steps": a.steps, "axes": a.axes, "render_s": t_render,
            "renders_per_s": n_render / max(t_render, 1e-9), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "ladder_gate.json"), "w") as f:
        json.dump({"meta": meta, "summary": summary}, f, indent=1)
    with open(os.path.join(a.out, "ladder_gate.md"), "w", encoding="utf-8") as f:
        f.write(report_md(summary, a.axes))
    np.savez_compressed(os.path.join(a.out, "ladder_distances.npz"),
                        **{f"{m}__d": d for m, (d, _) in dists.items()}, **{f"{m}__rep": r for m, (_, r) in dists.items()},
                        axis=lad["axis"], midi=lad["midi"], vecs=lad["vecs"])
    if a.save_audio:
        np.savez_compressed(os.path.join(a.out, "ladders_audio.npz"), audio=to_int16(audio), rep=to_int16(rep))
    print(report_md(summary, a.axes))
    print(f"rendered {n_render} clips in {t_render:.0f} s ({meta['renders_per_s']:.1f}/s); results in {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
