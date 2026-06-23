"""Aggregate onset-density eval runs: rank checkpoints + soups by clean-range
Audiobox quality AND density-control authority, to find the real operating point
and test whether weight-averaging (soups) beats any single checkpoint.

Reads, for each run dir, pq_scores.json (Audiobox PQ/CE/CU/PC per gain×density)
and onset_eval.json (requested vs measured density per gain×density).

    python aggregate_audiobox_eval.py <run_dir1> <run_dir2> ...
"""
import glob
import json
import os
import sys

import numpy as np


def load_run(d):
    pq = json.load(open(os.path.join(d, "pq_scores.json"))) if os.path.exists(os.path.join(d, "pq_scores.json")) else []
    oe = json.load(open(os.path.join(d, "onset_eval.json"))) if os.path.exists(os.path.join(d, "onset_eval.json")) else []
    if not pq:
        return None
    CE = np.array([r["CE"] for r in pq]); PQ = np.array([r["PQ"] for r in pq])
    CU = np.array([r["CU"] for r in pq]); PC = np.array([r["PC"] for r in pq])
    # best cell = max CE among cells whose PQ isn't disintegrating (>= median PQ - 0.5)
    pq_ok = PQ >= (np.median(PQ) - 0.5)
    bi = int(np.argmax(np.where(pq_ok, CE, -1)))
    best = pq[bi]
    # density control: per gain, slope of measured vs requested
    by_gain = {}
    for r in oe:
        by_gain.setdefault(r["gain"], []).append((r["requested"], r["measured"]))
    best_slope, best_corr, best_g = 0.0, 0.0, None
    for g, pts in by_gain.items():
        req = np.array([p[0] for p in pts]); meas = np.array([p[1] for p in pts])
        if len(set(meas)) < 2:
            continue
        slope = np.polyfit(req, meas, 1)[0]
        corr = np.corrcoef(req, meas)[0, 1]
        if abs(slope) > abs(best_slope):
            best_slope, best_corr, best_g = slope, corr, g
    return {
        "CE_mean": round(float(CE.mean()), 3), "CE_med": round(float(np.median(CE)), 3),
        "PQ_med": round(float(np.median(PQ)), 3), "CU_mean": round(float(CU.mean()), 3),
        "PC_mean": round(float(PC.mean()), 3),
        "frac_broke": round(float(np.mean([r["broke"] for r in pq])), 3),
        "best_cell": {"gain": best["gain"], "density": best["density"], "CE": best["CE"], "PQ": best["PQ"], "PC": best["PC"]},
        "ctrl_slope": round(float(best_slope), 3), "ctrl_corr": round(float(best_corr), 3), "ctrl_gain": best_g,
    }


def label(d):
    b = os.path.basename(d.rstrip("/"))
    return b.replace("step", "ep" + str(int(b[4:]) // 5400) + ":") if b.startswith("step") else b


runs = []
for d in sys.argv[1:]:
    r = load_run(d)
    if r:
        runs.append((label(d), r))

print(f"{'run':<20} {'CE_mn':>6} {'CE_md':>6} {'PQ_md':>6} {'CU':>5} {'PC':>5} {'broke':>6} | {'best g/d':>10} {'CE*':>5} | {'slope':>6} {'g':>4}")
print("-" * 100)
# sort by CE_mean (the disintegration-sensitive 'good music' axis) desc
for lab, r in sorted(runs, key=lambda x: -x[1]["CE_mean"]):
    bc = r["best_cell"]
    print(f"{lab:<20} {r['CE_mean']:>6} {r['CE_med']:>6} {r['PQ_med']:>6} {r['CU_mean']:>5} {r['PC_mean']:>5} "
          f"{r['frac_broke']:>6} | g{bc['gain']}/d{bc['density']:<5} {bc['CE']:>5} | {r['ctrl_slope']:>6} {str(r['ctrl_gain']):>4}")

# verdict: best soup vs best checkpoint
ck = [(l, r) for l, r in runs if l.startswith("ep")]
sp = [(l, r) for l, r in runs if l.startswith("soup")]
if ck and sp:
    bck = max(ck, key=lambda x: x[1]["CE_mean"]); bsp = max(sp, key=lambda x: x[1]["CE_mean"])
    print(f"\nBest checkpoint by CE_mean: {bck[0]} = {bck[1]['CE_mean']}")
    print(f"Best soup       by CE_mean: {bsp[0]} = {bsp[1]['CE_mean']}")
    print(f"=> soup {'BEATS' if bsp[1]['CE_mean'] > bck[1]['CE_mean'] else 'does NOT beat'} best checkpoint "
          f"(Δ {bsp[1]['CE_mean'] - bck[1]['CE_mean']:+.3f} CE_mean)")
