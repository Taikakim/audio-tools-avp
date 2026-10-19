# Owner: KUANG (created 2026-10-19)
"""Paired statistics for real-stem evaluations (eval_real_stems_consensus_v5.py consensus_eval_summary.json).

Twenty-four stems is a small sample: a difference in mean phrase MSS or in "wins" can be one or two stems. This
reports, per model, the mean and median phrase score with a 95 % bootstrap CI over stems, and for every model
against --ref the paired mean difference (same stems) with its CI and p, plus a sign test on per-stem wins / losses.
Lower phrase score is better. With --metric mel_ssim_dist it uses 1 - SSIM (lower = closer) and reads old files,
which stored that distance under "mel_ssim", correctly.

  python stem_stats_v6.py ~/sao_runs/stems_v6_dpo_occam/consensus_eval_summary.json --ref S5_16.5h
  python stem_stats_v6.py RUN_A/consensus_eval_summary.json RUN_B/consensus_eval_summary.json --ref A:S5_16.5h
Several files: models are named "<run dir>:<model>" so the same model in two runs can be compared (only stems
present in both enter a pair). The same stems with different MIDI (hand-fixed vs raw) are NOT comparable: compare
within one run, or across runs that used the same MIDI.
"""
import argparse
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import eval_stats_v4 as st  # noqa: E402


def value(d, metric):
    if metric == "mel_ssim_dist":
        return d["mel_ssim_dist"] if "mel_ssim_dist" in d else d["mel_ssim"]     # old files stored the distance
    if metric == "mel_ssim":
        return d["mel_ssim"] if "mel_ssim_dist" in d else 1.0 - d["mel_ssim"]
    return d[metric]


def load(paths):
    """-> {model: {stem_id: record}} (models prefixed with the run directory name when several files are given)."""
    out = {}
    for p in paths:
        with open(p) as f:
            rows = json.load(f)
        run = os.path.basename(os.path.dirname(os.path.abspath(p)))
        for r in rows:
            for m, d in r["models"].items():
                name = f"{run}:{m}" if len(paths) > 1 else m
                out.setdefault(name, {})[r["stem_id"]] = d
    return out


def table(models, metric="phrase_mss", ref=None, n_boot=10000, higher_is_better=False):
    names = list(models)
    stems = sorted(set().union(*[set(v) for v in models.values()]))
    res = {"metric": metric, "n_stems": len(stems), "models": {}, "paired": []}
    for m in names:
        x = np.array([value(models[m][s], metric) for s in stems if s in models[m]], np.float64)
        ci = st.mean_ci(x, np.arange(len(x)), n_boot=n_boot)
        res["models"][m] = {"n": int(len(x)), "mean": ci["estimate"], "lo": ci["lo"], "hi": ci["hi"],
                            "median": float(np.median(x))}
    common = [s for s in stems if all(s in models[m] for m in names)]
    wins = {m: 0 for m in names}
    for s in common:
        vals = {m: value(models[m][s], metric) for m in names}
        best = max(vals.values()) if higher_is_better else min(vals.values())
        for m in names:
            wins[m] += int(vals[m] == best)
    for m in names:
        res["models"][m]["wins_on_common"] = wins[m]
    res["n_common"] = len(common)
    if ref is not None:
        if ref not in models:
            raise SystemExit(f"--ref {ref!r} not among {names}")
        for m in names:
            if m == ref:
                continue
            shared = [s for s in stems if s in models[m] and s in models[ref]]
            a = np.array([value(models[m][s], metric) for s in shared], np.float64)
            b = np.array([value(models[ref][s], metric) for s in shared], np.float64)
            d = st.paired_delta(a, b, np.arange(len(shared)), n_boot=n_boot)
            better = int(np.sum(a > b)) if higher_is_better else int(np.sum(a < b))
            worse = int(np.sum(a < b)) if higher_is_better else int(np.sum(a > b))
            res["paired"].append({"model": m, "ref": ref, "n": len(shared), "delta": d["estimate"], "lo": d["lo"],
                                  "hi": d["hi"], "p": d["p"], "better": better, "worse": worse,
                                  "sign_test_p": st.sign_test(better, worse)})
    return res


def report(res, higher_is_better=False):
    arrow = "higher is better" if higher_is_better else "lower is better"
    lines = [f"{res['metric']} ({arrow}), {res['n_stems']} stems; wins counted on the {res['n_common']} stems every "
             "model has (ties count for each)", "",
             "| model | n | mean [95 % CI] | median | wins |", "|---|---|---|---|---|"]
    for m, r in res["models"].items():
        lines.append(f"| {m} | {r['n']} | {r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | {r['median']:.3f} | "
                     f"{r['wins_on_common']} |")
    if res["paired"]:
        lines += ["", f"Paired against {res['paired'][0]['ref']} (model - ref, same stems):", "",
                  "| model | n | delta [95 % CI] | p | stems better / worse | sign test p |", "|---|---|---|---|---|---|"]
        for r in res["paired"]:
            lines.append(f"| {r['model']} | {r['n']} | {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.3f} | "
                         f"{r['better']} / {r['worse']} | {r['sign_test_p']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summaries", nargs="+")
    ap.add_argument("--metric", default="phrase_mss", help="phrase_mss (default), mel_ssim_dist, mel_ssim, ...")
    ap.add_argument("--ref", default=None, help="model to pair every other model against")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", default=None, help="also write the report (markdown) here")
    a = ap.parse_args(argv)
    hib = a.metric == "mel_ssim"
    res = table(load(a.summaries), a.metric, a.ref, a.n_boot, higher_is_better=hib)
    text = report(res, hib)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    return res


if __name__ == "__main__":
    main()
