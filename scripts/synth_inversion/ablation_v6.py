# Owner: KUANG (created 2026-10-19)
"""Counterfactual ("ablation") tools for V6: label what a synth feature does to the SOUND, simplify inverted patches
when a feature is inaudible, and measure whether a feature is detectable from audio at all (V6_REVIEW.md).

Why not the V6 proposal's parameter labels (e.g. y_dist = waveshaper on or drive > 0.15): a waveshaper at low drive,
a sub-oscillator under a closed filter or a second oscillator at unison pitch can change the sound by less than
Surge's own render-to-render noise. A label from the setting then teaches the probe to "hear" something that is not
there, and a veto built on it is uncalibrated. Here a label is the measured effect of switching the feature off:

    delta(feature) = d(render(v), render(v with the feature off))      d = an audio_metrics_v5 distance
    noise          = d(render(v), render(v) again)                     Surge's own render-to-render noise
    audible        = delta > k * noise  (and delta > a small absolute floor)

Library:
  SIMPLIFICATIONS                      name -> {parameter: value that switches the feature off}
  simplify(vec, name, spec)            the vector with that feature off
  active(vec, name, spec)              whether switching it off changes the vector at all
  counterfactual(vecs, midi, render, score)          -> base audio, render noise [n]
  counterfactual_deltas(vecs, midi, base, render, score, names, spec) -> {name: delta [n], nan where inactive}
  occam(vecs, targets, midi, render, score, names, tol_abs, tol_rel, rounds)
        greedy: per round, try every active simplification of every item in one render batch; keep the one that
        raises the item's distance to its target least, if the rise is within max(tol_abs, tol_rel * distance);
        repeat until nothing is accepted. -> simplified vectors, per-item log.
  auroc(scores, labels), features_mel(audio), probe_cv(X, y, folds, l2, seed)

CLI (desktop; Surge needed for labels / occam):
  python ablation_v6.py labels --layout v5 --bass_manifold B.npz --all_manifold A.npz --n 4096 \\
      --enrich waveshaper_off=0.5 --workers 14 --out abl_labels_v5
  python ablation_v6.py probe --labels abl_labels_v5 --target waveshaper_off [--jepa RUN/jepa_ema_latest.pt]
  python ablation_v6.py occam --items audio_eval_s1/items.npz --system A3p --layout v5 --workers 14 --out occam_s1

labels writes labels.npz (vecs, midi, audio int16, delta__<name>, noise) and summary.json: per feature, how often it
is on in the prior and how often it is audible when on. probe reports cross-validated AUROC (0.5 = chance) of
"audible" against log-mel statistics (and the JEPA audio embedding with --jepa), overall and per stratum, next to the
AUROC of the parameter label, so it shows where the feature can be detected and whether the parameter label is
worse. occam re-renders audio_eval_v5 candidates with inaudible features switched off and reports how many features
were removed and what that did to every metric.

Status: CPU-tested with a fake renderer (test_v6.py). Not run on Surge.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# name -> {parameter: "off" value in the 60-d ordinal storage (v4 / v5; v3 where the names exist)}
SIMPLIFICATIONS = {
    "waveshaper_off": {"waveshaper_type": 0.0, "drive": 0.0},
    "osc2_off": {"osc_2_volume": 0.0},
    "osc3_off": {"osc_3_volume": 0.0},
    "sub_off": {"sub_mix": 0.0},
    "unison_1": {"unison": 0.0, "unison_detune": 0.0},
    "ring_off": {"ring_1x2_volume": 0.0, "ring_2x3_volume": 0.0},
    "noise_off": {"noise_volume": 0.0},
    "fm_off": {"fm_depth": 0.0},
    "sync_off": {"sync": 0.0},
    "feedback_off": {"filter_feedback": 0.5},        # stored 0.5 = 0 % (raw = p, -100 % .. +100 %)
    "detune_off": {"osc_2_pitch": 0.5, "osc_3_pitch": 0.5},
    "drift_off": {"drift": 0.0},
}
DEFAULT_OCCAM = ("waveshaper_off", "osc2_off", "osc3_off", "sub_off", "unison_1", "ring_off", "noise_off",
                 "feedback_off", "detune_off", "drift_off")


# --------------------------------------------------------------------------------------------- simplifications
def _targets(name, spec):
    if name not in SIMPLIFICATIONS:
        raise ValueError(f"unknown simplification {name!r}; known: {sorted(SIMPLIFICATIONS)}")
    return [(spec.PARAM_INDEX[p], v) for p, v in SIMPLIFICATIONS[name].items() if p in spec.PARAM_INDEX]


def simplify(vec, name, spec):
    out = np.array(vec, dtype=np.float32, copy=True)
    for i, v in _targets(name, spec):
        out[..., i] = v
    return out


def active(vec, name, spec, eps=1e-4):
    """True if switching the feature off changes the rendered patch: compared after vector_to_patch, so values the
    spec already treats as off (e.g. unison_detune with 1 voice) do not count."""
    t = _targets(name, spec)
    if not t:
        return False
    a, b = spec.vector_to_patch(np.asarray(vec)), spec.vector_to_patch(simplify(vec, name, spec))
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, (bool, int, np.integer)) or isinstance(y, (bool, int, np.integer)):
            if x != y:
                return True
        elif isinstance(x, (float, np.floating)) and abs(float(x) - float(y)) > eps:
            return True
    return False


# --------------------------------------------------------------------------------------------- counterfactuals
def counterfactual(vecs, midi, render, score):
    """render(vecs [m, P], midi [m]) -> audio [m, N]; score(targets list, preds list) -> [m] distances.
    -> (base audio [n, N], render noise [n]: the distance between two renders of the same vector)."""
    vecs = np.asarray(vecs, np.float32)
    midi = np.asarray(midi)
    base = render(vecs, midi)
    again = render(vecs, midi)
    noise = np.asarray(score(list(base), list(again)), np.float64)
    return base, noise


def counterfactual_deltas(vecs, midi, base, render, score, names, spec):
    """-> {name: delta [n]}: distance between the base render and the render with that feature off (nan where
    the feature is inactive, so nothing is rendered for it)."""
    vecs = np.asarray(vecs, np.float32)
    deltas = {}
    for name in names:
        on = np.array([active(v, name, spec) for v in vecs])
        d = np.full(len(vecs), np.nan)
        if on.any():
            idx = np.where(on)[0]
            off_audio = render(np.stack([simplify(vecs[i], name, spec) for i in idx]), np.asarray(midi)[idx])
            d[idx] = score([base[i] for i in idx], list(off_audio))
        deltas[name] = d
    return deltas


def audible(delta, noise, k=2.0, floor=0.0):
    """delta > k * noise and > floor; False where the feature is inactive (nan)."""
    d = np.asarray(delta, np.float64)
    return np.isfinite(d) & (d > k * np.asarray(noise, np.float64)) & (d > floor)


# --------------------------------------------------------------------------------------------- occam
def occam(vecs, targets, midi, render, score, spec, names=DEFAULT_OCCAM, tol_abs=0.0, tol_rel=0.02, rounds=6):
    """Greedy simplification of each item toward its target audio, batched over items per round.
    -> (vectors [n, P], start scores [n], final scores [n], log: per item a list of (round, name, score))."""
    cur = np.array(vecs, np.float32, copy=True)
    midi = np.asarray(midi)
    targets = list(targets)
    start = np.asarray(score(targets, list(render(cur, midi))), np.float64)
    best = start.copy()
    log = [[] for _ in range(len(cur))]
    live = np.ones(len(cur), bool)
    for r in range(rounds):
        jobs = [(i, name) for i in np.where(live)[0] for name in names if active(cur[i], name, spec)]
        if not jobs:
            break
        cand = np.stack([simplify(cur[i], name, spec) for i, name in jobs])
        sc = np.asarray(score([targets[i] for i, _ in jobs], list(render(cand, midi[[i for i, _ in jobs]]))), np.float64)
        accepted = np.zeros(len(cur), bool)
        for i in sorted(set(i for i, _ in jobs)):
            rows = [j for j, (ii, _) in enumerate(jobs) if ii == i]
            j = min(rows, key=lambda q: sc[q])
            tol = max(tol_abs, tol_rel * best[i])
            if sc[j] <= best[i] + tol:
                cur[i] = cand[j]
                best[i] = sc[j]
                log[i].append((r, jobs[j][1], float(sc[j])))
                accepted[i] = True
        live &= accepted
        if not live.any():
            break
    return cur, start, best, log


# --------------------------------------------------------------------------------------------- probe
def auroc(scores, labels):
    """Area under the ROC curve via ranks (ties averaged); nan if one class is missing."""
    s = np.asarray(scores, np.float64)
    y = np.asarray(labels, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def features_mel(audio, sr=44100, n_mels=64):
    """Clip-level log-mel statistics: per band mean, std, and the mean absolute frame-to-frame change.
    Deliberately simple and model-free: a floor for what the audio shows without any learned encoder."""
    import torch
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, hop_length=512, n_mels=n_mels, f_min=20.0)
    x = torch.as_tensor(np.asarray(audio, np.float32))
    with torch.no_grad():
        m = torch.log(mel(x) + 1e-6)                     # [n, mels, frames]
    d = (m[..., 1:] - m[..., :-1]).abs().mean(-1)
    return torch.cat([m.mean(-1), m.std(-1), d], dim=1).numpy()


def features_jepa(audio, jepa_path, device="cpu", chunk=128):
    """JEPA audio embedding z_a (the encoder V6 would put its heads on), with the checkpoint's own normalizer."""
    import torch
    from flow_sampling_v5 import mel_of
    from synth_jepa_search_v3 import load_synth_jepa
    model, norm = load_synth_jepa(jepa_path, device=device)
    mel = mel_of(np.asarray(audio, np.float32), device)
    out = []
    with torch.no_grad():
        for s in range(0, len(mel), chunk):
            out.append(model.encode_audio(norm.normalize(mel[s:s + chunk].to(device))).float().cpu())
    return torch.cat(out).numpy()


def probe_cv(X, y, folds=5, l2=1e-3, steps=300, lr=0.05, seed=0):
    """Out-of-fold probabilities of an L2 logistic regression (features standardised on each training fold)."""
    import torch
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(len(y)) % folds
    p = np.full(len(y), np.nan)
    for f in range(folds):
        tr, te = fold != f, fold == f
        if len(np.unique(y[tr])) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=lr * 20, max_iter=steps, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xt @ w + b, yt) + l2 * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            p[te] = torch.sigmoid(torch.tensor((X[te] - mu) / sd, dtype=torch.float32) @ w + b).numpy()
    return p


def strata(vecs, spec):
    """Named boolean masks over items: where should a feature be detectable from audio at all?"""
    I = spec.PARAM_INDEX
    v = np.asarray(vecs)
    multi = (v[:, I["unison"]] > 0.1) | (v[:, I["osc_2_volume"]] > 0.1) | (v[:, I["osc_3_volume"]] > 0.1)
    if "drift" in I:
        multi |= v[:, I["drift"]] > 0.05
    sweep = v[:, I["feg_amount"]] >= 0.5
    return {"all": np.ones(len(v), bool), "single_osc": ~multi, "multi_or_unison": multi,
            "filter_sweep": sweep, "static_filter": ~sweep}


# --------------------------------------------------------------------------------------------- CLI
def _score_fn(metric, workers):
    import audio_metrics_v5 as am

    def score(targets, preds):
        return am.score_many(list(targets), list(preds), [metric], workers=workers)[metric]
    return score


def _draw(prior, n, seed, spec, enrich):
    """n prior draws; enrich {name: share} keeps drawing until that share of items has the feature active."""
    rng = np.random.RandomState(seed)
    quota = {k: int(round(v * n)) for k, v in enrich.items()}
    vecs, midi = [], []
    tries = 0
    while len(vecs) < n:
        tries += 1
        if tries > 200 * n:
            raise SystemExit(f"could not meet --enrich {enrich} in {200 * n} draws")
        v, m = prior.sample_vector(rng)[:2]
        need = {k: q for k, q in quota.items() if q > 0}
        left = n - len(vecs)
        if need and sum(need.values()) >= left and not any(active(v, k, spec) for k in need):
            continue
        for k in need:
            if active(v, k, spec):
                quota[k] -= 1
        vecs.append(np.asarray(v, np.float32))
        midi.append(int(m))
    return np.stack(vecs), np.array(midi)


def cmd_labels(a, pool=None, prior=None):
    from layout_v6 import layout_named
    from render_pool_v5 import RenderPool, to_int16
    lay = layout_named(a.layout)
    if prior is None:
        from realistic_prior_v4 import UnifiedSoundsetPrior
        from soundset_prior_v6 import SoundsetPriorV6
        prior = (SoundsetPriorV6 if lay.name == "v6" else UnifiedSoundsetPrior)(a.bass_manifold, a.all_manifold,
                                                                                 split=a.split)
    enrich = dict((kv.split("=")[0], float(kv.split("=")[1])) for kv in a.enrich)
    names = a.features or list(SIMPLIFICATIONS)
    vecs, midi = _draw(prior, a.n, a.seed, lay.spec, enrich)
    score = _score_fn(a.metric, a.workers)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with (pool if pool is not None else RenderPool(a.plugin, workers=a.workers, spec=lay.name)) as pl:
        base, noise = counterfactual(vecs, midi, pl.render, score)
        deltas = counterfactual_deltas(vecs, midi, base, pl.render, score, names, lay.spec)
    summary = {"layout": lay.name, "n": int(len(vecs)), "metric": a.metric, "k_noise": a.k, "floor": a.floor,
               "noise_median": float(np.median(noise)), "seed": a.seed, "split": a.split, "enrich": enrich,
               "time_s": time.time() - t0, "features": {}}
    for name, d in deltas.items():
        on = np.isfinite(d)
        aud = audible(d, noise, a.k, a.floor)
        summary["features"][name] = {"on_share": float(on.mean()), "audible_share_of_on": float(aud[on].mean()) if on.any() else None,
                                     "delta_over_noise_median": float(np.median(d[on] / np.maximum(noise[on], 1e-12))) if on.any() else None}
    np.savez_compressed(os.path.join(a.out, "labels.npz"), vecs=vecs, midi=midi, audio=to_int16(base), noise=noise,
                        **{f"delta__{k}": v for k, v in deltas.items()})
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for name, s in summary["features"].items():
        print(f"{name:15s} on {100 * s['on_share']:5.1f}% | audible when on "
              + (f"{100 * s['audible_share_of_on']:5.1f}% | median delta/noise {s['delta_over_noise_median']:.2f}"
                 if s["audible_share_of_on"] is not None else "-"))
    return summary


def cmd_probe(a, pool=None, prior=None):
    from layout_v6 import layout_named
    from render_pool_v5 import from_int16
    z = np.load(os.path.join(a.labels, "labels.npz"))
    with open(os.path.join(a.labels, "summary.json")) as f:
        meta = json.load(f)
    lay = layout_named(meta["layout"])
    d = z[f"delta__{a.target}"]
    k = a.k if a.k is not None else meta["k_noise"]
    floor = a.floor if a.floor is not None else meta["floor"]
    y_audible = audible(d, z["noise"], k, floor)
    y_param = np.isfinite(d)
    audio = from_int16(z["audio"])
    feats = {"mel_stats": features_mel(audio)}
    if a.jepa:
        feats["jepa_za"] = features_jepa(audio, a.jepa, device=a.device)
    st = strata(z["vecs"], lay.spec)
    out = {"target": a.target, "k_noise": k, "floor": floor, "n": int(len(d)),
           "audible_share": float(y_audible.mean()), "param_on_share": float(y_param.mean()), "results": {}}
    for fname, X in feats.items():
        p_aud = probe_cv(X, y_audible, seed=a.seed)
        p_par = probe_cv(X, y_param, seed=a.seed)
        res = {}
        for sname, mask in st.items():
            res[sname] = {"n": int(mask.sum()), "audible_share": float(y_audible[mask].mean()) if mask.any() else None,
                          "auroc_audible": auroc(p_aud[mask], y_audible[mask]),
                          "auroc_param_label": auroc(p_par[mask], y_param[mask]),
                          "auroc_param_vs_audible": auroc(p_par[mask], y_audible[mask])}
        out["results"][fname] = res
    with open(os.path.join(a.labels, f"probe_{a.target}.json"), "w") as f:
        json.dump(out, f, indent=1)
    for fname, res in out["results"].items():
        for sname, r in res.items():
            print(f"{fname:9s} {sname:16s} n {r['n']:5d} | audible {r['audible_share'] if r['audible_share'] is None else round(100 * r['audible_share'], 1)}% "
                  f"| AUROC audible {r['auroc_audible']:.3f} | param-label probe on audible {r['auroc_param_vs_audible']:.3f}")
    return out


def cmd_occam(a, pool=None, prior=None):
    import audio_metrics_v5 as am
    from layout_v6 import layout_named
    from render_pool_v5 import RenderPool, from_int16
    lay = layout_named(a.layout)
    z = np.load(a.items)
    targets = from_int16(z["target_audio"])
    midi = z["midi"]
    vecs_all = z[f"{a.system}__vecs"]                                 # [n, K, P]
    sel = z[f"{a.system}__score__{a.select}"]                         # [n, K]
    idx = np.nanargmin(np.where(np.isfinite(sel), sel, np.inf), axis=1)
    vecs = vecs_all[np.arange(len(idx)), idx]
    score = _score_fn(a.select, a.workers)
    os.makedirs(a.out, exist_ok=True)
    with (pool if pool is not None else RenderPool(a.plugin, workers=a.workers, spec=lay.name)) as pl:
        new, start, final, log = occam(vecs, targets, midi, pl.render, score, lay.spec, a.features or DEFAULT_OCCAM,
                                       a.tol_abs, a.tol_rel, a.rounds)
        before = pl.render(vecs, midi)
        after = pl.render(new, midi)
    metrics = a.metrics
    sb = am.score_many(list(targets), list(before), metrics, workers=a.workers)
    sa = am.score_many(list(targets), list(after), metrics, workers=a.workers)
    removed = {}
    for entries in log:
        for _, name, _ in entries:
            removed[name] = removed.get(name, 0) + 1
    rep = {"system": a.system, "select": a.select, "n": int(len(vecs)), "tol_abs": a.tol_abs, "tol_rel": a.tol_rel,
           "items_simplified": int(sum(1 for e in log if e)), "removed": removed,
           "metric_before": {m: float(np.nanmean(sb[m])) for m in metrics},
           "metric_after": {m: float(np.nanmean(sa[m])) for m in metrics}}
    np.savez_compressed(os.path.join(a.out, "occam.npz"), vecs_before=vecs, vecs_after=new, start=start, final=final)
    with open(os.path.join(a.out, "occam.json"), "w") as f:
        json.dump({**rep, "log": [[list(e) for e in entries] for entries in log]}, f, indent=1)
    print(json.dumps(rep, indent=1))
    return rep


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lb = sub.add_parser("labels")
    lb.add_argument("--layout", default="v5", choices=("v4", "v5", "v6"),
                    help="the soundset prior: v4 / v5 = realistic_prior_v4, v6 = soundset_prior_v6 (simplicity prior)")
    lb.add_argument("--bass_manifold")
    lb.add_argument("--all_manifold")
    lb.add_argument("--split", default="train")
    lb.add_argument("--n", type=int, default=4096)
    lb.add_argument("--seed", type=int, default=0)
    lb.add_argument("--enrich", nargs="*", default=[], metavar="NAME=SHARE")
    lb.add_argument("--features", nargs="*", default=None, help="simplifications to measure (default: all)")
    lb.add_argument("--metric", default="mss")
    lb.add_argument("--k", type=float, default=2.0, help="audible = delta > k * render noise")
    lb.add_argument("--floor", type=float, default=0.0, help="and delta > floor (metric units)")
    lb.add_argument("--plugin", default=None)
    lb.add_argument("--workers", type=int, default=8)
    lb.add_argument("--out", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--labels", required=True)
    pr.add_argument("--target", default="waveshaper_off")
    pr.add_argument("--jepa", default=None)
    pr.add_argument("--k", type=float, default=None)
    pr.add_argument("--floor", type=float, default=None)
    pr.add_argument("--device", default="cpu")
    pr.add_argument("--seed", type=int, default=0)
    oc = sub.add_parser("occam")
    oc.add_argument("--items", required=True, help="audio_eval_v5 items.npz")
    oc.add_argument("--system", required=True)
    oc.add_argument("--layout", default="v5", choices=("v3", "v4", "v5", "v6"))
    oc.add_argument("--select", default="logmel_l1")
    oc.add_argument("--metrics", nargs="+", default=["logmel_l1", "mss", "modspec_dist"])
    oc.add_argument("--features", nargs="*", default=None)
    oc.add_argument("--tol_abs", type=float, default=0.0)
    oc.add_argument("--tol_rel", type=float, default=0.02)
    oc.add_argument("--rounds", type=int, default=6)
    oc.add_argument("--plugin", default=None)
    oc.add_argument("--workers", type=int, default=8)
    oc.add_argument("--out", required=True)
    return ap


def main(argv=None, pool=None, prior=None):
    """pool / prior replace Surge and the manifold prior (tests)."""
    a = build_parser().parse_args(argv)
    return {"labels": cmd_labels, "probe": cmd_probe, "occam": cmd_occam}[a.cmd](a, pool=pool, prior=prior)


if __name__ == "__main__":
    main()
