# Owner: KUANG (created 2026-10-12)
"""Paired comparison of inversion systems on the REAL stems, with a dev/test split for anything chosen by score.

Why (EVAL_PARADIGM_V4.md, tier T2): real stems are the end task, but there are only ~24 of them, every system is
scored on the same stems, and several "systems" are really settings picked by looking at the scores (which tau,
online or EMA, snapped or raw). The A1 multi-tau sweep, for example, has 5 tau x 2 kinds x raw/snapped = 20 systems
scored on the same 10 stems it reports: the best of 20 noisy numbers is optimistic (winner's curse). So:
  * selection happens on DEV stems only and is reported on TEST stems (the H5 rule in EXPERIMENTS.md);
  * comparisons are paired by stem: per-stem mean over the PRE-REGISTERED seeds, then wins/losses with the exact
    sign test (24 test-or-all stems need 18 wins for p < 0.05) and a bootstrap over stems for the mean difference;
  * the seed-to-seed spread within a stem is reported next to the between-system difference.

Input, one of:
  --csv FILE        long format with columns stem,system,seed,metric,value
  --multitemp FILE  results.json from render_a1_multitemp_eval.py: systems '<kind>/tau<t>/<raw|harm>', metric 'mss'
Metrics are lower-is-better unless --higher_is_better.

Usage:
  python stem_compare_v4.py --multitemp .../eval_cpu_muscriptor_temps/results.json --metric mss \\
      --baseline ema/tau1.00/raw --select "ema_tau=ema/tau" --split_seed 0 --out stems_report.md
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import eval_stats_v4 as st  # noqa: E402


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"stem": r["stem"], "system": r["system"], "seed": int(r.get("seed") or 0),
                         "metric": r["metric"], "value": float(r["value"])})
    return rows


def from_multitemp(path):
    """render_a1_multitemp_eval.py results.json: {kind: {stem: {"temps": {tag: {tau, stft_phrase_raw/harm}}}}}."""
    with open(path) as f:
        res = json.load(f)
    rows = []
    for kind, stems in res.items():
        if not isinstance(stems, dict):
            continue
        for stem, rec in stems.items():
            for tag, r in (rec.get("temps") or {}).items():
                tau = float(r["tau"])
                for variant, key in (("raw", "stft_phrase_raw"), ("harm", "stft_phrase_harm")):
                    if key in r and r[key] is not None:
                        rows.append({"stem": stem, "system": f"{kind}/tau{tau:.2f}/{variant}", "seed": 0,
                                     "metric": "mss", "value": float(r[key])})
    return rows


def pivot(rows, metric):
    """{system: {stem: (mean over seeds, n_seeds, seed sd)}} for one metric."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["metric"] == metric and np.isfinite(r["value"]):
            acc[r["system"]][r["stem"]].append(r["value"])
    out = {}
    for sysname, stems in acc.items():
        out[sysname] = {s: (float(np.mean(v)), len(v), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))
                        for s, v in stems.items()}
    return out


def split_stems(stems, seed=0, dev_list=None):
    stems = sorted(stems)
    if dev_list:
        dev = sorted(set(dev_list) & set(stems))
    else:
        rng = np.random.default_rng(seed)
        perm = list(rng.permutation(stems))
        dev = sorted(perm[: len(stems) // 2])
    return dev, sorted(set(stems) - set(dev))


def select_best(table, prefix, dev, lower=True):
    """The system of the family (name starts with prefix) with the best mean on the dev stems that all of the
    family's systems cover."""
    fam = {k: v for k, v in table.items() if k.startswith(prefix)}
    if not fam:
        raise KeyError(f"no system starts with {prefix!r}")
    common = [s for s in dev if all(s in v for v in fam.values())]
    if not common:
        raise ValueError(f"family {prefix!r} has no dev stem in common")
    means = {k: np.mean([v[s][0] for s in common]) for k, v in fam.items()}
    best = min(means, key=means.get) if lower else max(means, key=means.get)
    return best, means, common


def compare(table, a, b, stems, lower=True, n_boot=10000):
    common = [s for s in stems if s in table[a] and s in table[b]]
    if not common:
        return None
    va = np.array([table[a][s][0] for s in common])
    vb = np.array([table[b][s][0] for s in common])
    d = va - vb
    good = d < 0 if lower else d > 0
    bad = d > 0 if lower else d < 0
    wins, losses = int(good.sum()), int(bad.sum())
    res = st.paired_delta(va, vb, np.arange(len(common)), n_boot=n_boot, seed=0)
    seed_sd = [table[x][s][2] for x in (a, b) for s in common if np.isfinite(table[x][s][2])]
    return {"a": a, "b": b, "n": len(common), "wins": wins, "losses": losses, "ties": len(common) - wins - losses,
            "sign_p": st.sign_test(wins, losses), "wins_needed": st.min_wins(wins + losses) if wins + losses else None,
            "mean_a": float(va.mean()), "mean_b": float(vb.mean()), "delta": res["estimate"], "lo": res["lo"],
            "hi": res["hi"], "median_delta": float(np.median(d)),
            "median_seed_sd": float(np.median(seed_sd)) if seed_sd else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--multitemp")
    ap.add_argument("--metric", default="mss")
    ap.add_argument("--higher_is_better", action="store_true")
    ap.add_argument("--baseline", required=True, help="system every other system is compared with")
    ap.add_argument("--systems", nargs="*", default=None, help="systems to compare (default: all fixed systems)")
    ap.add_argument("--select", action="append", default=[],
                    help="NAME=PREFIX: pick the best system of the family PREFIX on DEV stems, report it on TEST")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--dev_stems", nargs="*", default=None, help="explicit dev stems (pre-registered)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    rows = load_csv(a.csv) if a.csv else from_multitemp(a.multitemp)
    table = pivot(rows, a.metric)
    if a.baseline not in table:
        raise SystemExit(f"baseline {a.baseline!r} not found; systems: {sorted(table)}")
    lower = not a.higher_is_better
    stems = sorted({s for v in table.values() for s in v})
    dev, test = split_stems(stems, a.split_seed, a.dev_stems)
    L = [f"# Real-stem comparison ({a.metric}, {'lower' if lower else 'higher'} is better)", "",
         f"{len(stems)} stems: dev {len(dev)} ({', '.join(dev)}), test {len(test)} ({', '.join(test)}).",
         "Selections are made on dev and reported on test. Fixed systems are reported on all stems and on test.", ""]
    selected = {}
    for spec in a.select:
        name, _, prefix = spec.partition("=")
        best, means, common = select_best(table, prefix, dev, lower)
        selected[name] = best
        L.append(f"- selection {name}: family `{prefix}*` -> **{best}** (dev mean over {len(common)} stems; "
                 + ", ".join(f"{k} {v:.3f}" for k, v in sorted(means.items(), key=lambda kv: kv[1])) + ")")
    L += ["", "| system vs baseline | stems | wins/losses/ties | sign p (wins needed) | mean a | mean b | "
              "a - b [95% CI] | median a - b | median seed sd |", "|---|---|---|---|---|---|---|---|---|"]
    fixed = a.systems if a.systems is not None else [s for s in table if s != a.baseline]
    comparisons = [(s, "all", stems) for s in fixed] + [(s, "test", test) for s in fixed]
    comparisons += [(f"{n}: {s}", "test", test) for n, s in selected.items()]
    for label, which, stem_set in comparisons:
        sysname = label.split(": ", 1)[1] if ": " in label else label
        c = compare(table, sysname, a.baseline, stem_set, lower)
        if c is None:
            continue
        need = c["wins_needed"] if c["wins_needed"] is not None else "-"
        L.append(f"| {label} [{which}] | {c['n']} | {c['wins']}/{c['losses']}/{c['ties']} | {c['sign_p']:.3f} ({need}) | "
                 f"{c['mean_a']:.3f} | {c['mean_b']:.3f} | {c['delta']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] | "
                 f"{c['median_delta']:.3f} | {c['median_seed_sd']:.3f} |")
    md = "\n".join(L) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
