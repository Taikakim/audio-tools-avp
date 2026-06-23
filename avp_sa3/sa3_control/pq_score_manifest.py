"""Audiobox-score a multi_eval.py output dir (reads manifest.json, joins scores).
Writes pq_scores.json with per-clip prompt/seed/gain/density/measured + PQ/CE/CU/PC.
Runs in the mir venv (audiobox_aesthetics + WavLM).

    /home/kim/Projects/mir/mir/bin/python pq_score_manifest.py <dir>
"""
import json
import os
import sys

sys.path.insert(0, "/home/kim/Projects/mir/src")
from timbral.audiobox_aesthetics import analyze_audiobox_aesthetics

D = sys.argv[1]
man = json.load(open(os.path.join(D, "manifest.json")))
out = []
for m in man:
    p = os.path.join(D, m["file"])
    if not os.path.exists(p):
        continue
    try:
        s = analyze_audiobox_aesthetics(p)
        row = dict(m)
        row.update({
            "PQ": round(float(s.get("PQ") or s.get("production_quality")), 3),
            "CE": round(float(s.get("CE") or s.get("content_enjoyment")), 3),
            "CU": round(float(s.get("CU") or s.get("content_usefulness", 0) or 0), 3),
            "PC": round(float(s.get("PC") or s.get("production_complexity", 0) or 0), 3),
        })
        out.append(row)
        print(f"  {m['file']:<28} CE {row['CE']:.2f} PQ {row['PQ']:.2f} PC {row['PC']:.2f}", flush=True)
    except Exception as e:
        print(f"  [skip] {m['file']}: {e}", flush=True)
json.dump(out, open(os.path.join(D, "pq_scores.json"), "w"), indent=1)
print(f"[pq] wrote {len(out)} -> {D}/pq_scores.json", flush=True)
