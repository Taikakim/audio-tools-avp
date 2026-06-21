"""Score onset-eval clips with Meta Audiobox Aesthetics Production Quality (PQ).

Maps PQ across (gain, density) so we can SEE where a bracket disintegrates and
stop it there. Runs in the mir venv (audiobox_aesthetics + WavLM live there):

    /home/kim/Projects/mir/mir/bin/python pq_score.py <eval_dir> [--pq-floor 6.0]

Prints a gain x density PQ grid + flags cells below the floor (disintegrating).
Writes pq_scores.json into the dir.
"""
import os, sys, re, glob, json

sys.path.insert(0, "/home/kim/Projects/mir/src")
from timbral.audiobox_aesthetics import analyze_audiobox_aesthetics

D = sys.argv[1]
FLOOR = float(sys.argv[sys.argv.index("--pq-floor") + 1]) if "--pq-floor" in sys.argv else 6.0
PAT = re.compile(r"onset_g([0-9.]+)_d([0-9.]+)\.wav$")

clips = sorted(glob.glob(f"{D}/onset_g*_d*.wav"))
rows = []
for p in clips:
    m = PAT.search(os.path.basename(p))
    if not m:
        continue
    g, d = float(m.group(1)), float(m.group(2))
    try:
        s = analyze_audiobox_aesthetics(p)
        pq = float(s.get("PQ") or s.get("production_quality"))
        ce = float(s.get("CE") or s.get("content_enjoyment"))
        cu = float(s.get("CU") or s.get("content_usefulness", 0) or 0)
        pc = float(s.get("PC") or s.get("production_complexity", 0) or 0)
    except Exception as e:
        print(f"  [skip] {os.path.basename(p)}: {e}", flush=True)
        continue
    broke = pq < FLOOR or ce < FLOOR          # disintegration on EITHER axis (PQ can hold while CE drops)
    rows.append({"gain": g, "density": d, "PQ": round(pq, 3), "CE": round(ce, 3),
                 "CU": round(cu, 3), "PC": round(pc, 3), "broke": broke})
    print(f"  g{g:<4} d{d:<5} PQ {pq:.2f}  CE {ce:.2f}{'  <-- DISINTEGRATING' if broke else ''}", flush=True)

json.dump(rows, open(f"{D}/pq_scores.json", "w"), indent=2)

# grids for PQ and CE (the two thresholds we watch)
gains = sorted({r["gain"] for r in rows})
dens = sorted({r["density"] for r in rows})
for metric in ("PQ", "CE", "PC"):     # PC (Production Complexity) may catch high-gain style/distortion CE misses
    val = {(r["gain"], r["density"]): r[metric] for r in rows}
    print(f"\n{metric} grid (rows=gain, cols=density; floor={FLOOR}):")
    print("  gain\\dens " + " ".join(f"{d:>6g}" for d in dens))
    for g in gains:
        cells = " ".join((f"{val[(g,d)]:>6.2f}" if (g, d) in val else "   -  ") for d in dens)
        print(f"  {g:<8} {cells}")
print(f"\n[pq] wrote {D}/pq_scores.json  ({len(rows)} clips). Disintegrating cells: "
      f"{sum(1 for r in rows if r['broke'])}/{len(rows)} (PQ or CE < {FLOOR})")
