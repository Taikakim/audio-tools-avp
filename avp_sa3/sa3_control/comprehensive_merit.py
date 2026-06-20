"""Comprehensive MERIT scoring of EVERY rendered clip in a bracket dir.

Pure MERT post-pass — no SA3, no re-render. Each `*_refonly_gain*.wav` is scored against the
two saved references (`_REF_*.wav`) → per-factor transfer / margin, across *all* models,
checkpoints, AND gains (not just the collapse-check gain 0.8). Light: MERT-330M only.

Run on GPU when the card is free:
    MERIT_DEVICE=cuda python comprehensive_merit.py [bracket_dir]
~1 min for ~250 clips; writes comprehensive_merit.{csv,json} into the dir.
"""
import os
import sys
import re
import glob
import csv
import json

sys.path.append("/home/kim/Projects/SAO/stable-audio-tools/avp_sa3")
import soundfile as sf
import torch
from sa3_control.merit_eval import MeritScorer, FACTORS

DIR = sys.argv[1] if len(sys.argv) > 1 else "/run/media/kim/Lehto/sa3_control_runs/bracket6"
DEV = os.environ.get("MERIT_DEVICE", "cuda")
REFS = ("hallucinogen", "morphem")
PAT = re.compile(r"(.+)_step(\d+)_(hallucinogen|morphem)_refonly_gain([0-9.]+)\.wav$")

print(f"[merit] loading MERT-330M on {DEV} …", flush=True)
scorer = MeritScorer(device=DEV)


def load(path):
    w, sr = sf.read(path, dtype="float32")    # (T,) mono or (T,C)
    t = torch.from_numpy(w)
    if t.ndim == 2:
        t = t.T                               # (T,C) -> (C,T) for embed's mean(0)
    return t, sr


def cos(a, b):
    return {f: float((a[f] * b[f]).sum().item()) for f in FACTORS}


ref_emb = {}
for n in REFS:
    p = f"{DIR}/_REF_{n}_75pct.wav"
    if not os.path.exists(p):
        sys.exit(f"missing reference {p} — run the bracket render first")
    w, sr = load(p)
    ref_emb[n] = scorer.embed(w, sr)
print("[merit] references embedded; scoring clips …", flush=True)

clips = sorted(glob.glob(f"{DIR}/*_refonly_gain*.wav"))
rows = []
for p in clips:
    m = PAT.search(os.path.basename(p))
    if not m:
        continue
    model, step, ref, gain = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
    other = "morphem" if ref == "hallucinogen" else "hallucinogen"
    w, sr = load(p)
    e = scorer.embed(w, sr)
    tr, lk = cos(e, ref_emb[ref]), cos(e, ref_emb[other])
    row = {"model": model, "step": step, "ref": ref, "gain": gain}
    for f in FACTORS:
        row[f"transfer_{f}"] = round(tr[f], 4)          # clip vs its OWN reference (raw)
        row[f"margin_{f}"] = round(tr[f] - lk[f], 4)    # own − other; >0 = specific transfer of factor
    rows.append(row)
    print(f"  {model:14s} s{step:<5d} {ref[:4]} g{gain:<5} "
          f"margin mel {row['margin_mel']:+.2f} rhy {row['margin_rhy']:+.2f} tim {row['margin_tim']:+.2f}", flush=True)

if not rows:
    sys.exit("no *_refonly_gain*.wav clips found")
keys = list(rows[0].keys())
with open(f"{DIR}/comprehensive_merit.csv", "w", newline="") as f:
    cw = csv.DictWriter(f, fieldnames=keys)
    cw.writeheader()
    cw.writerows(rows)
json.dump(rows, open(f"{DIR}/comprehensive_merit.json", "w"), indent=2)

# quick top-of-table: best (model, step, gain) by averaged margin per factor
print(f"\n[merit] scored {len(rows)} clips → {DIR}/comprehensive_merit.csv", flush=True)
from collections import defaultdict
agg = defaultdict(lambda: {f: 0.0 for f in FACTORS} | {"n": 0})
for r in rows:
    k = (r["model"], r["step"], r["gain"])
    for f in FACTORS:
        agg[k][f] += r[f"margin_{f}"]
    agg[k]["n"] += 1
for f in FACTORS:
    best = max(agg.items(), key=lambda kv: kv[1][f] / max(kv[1]["n"], 1))
    print(f"  best {f}: {best[0]}  margin {best[1][f]/best[1]['n']:+.3f}", flush=True)
print("COMPREHENSIVE MERIT DONE", flush=True)
