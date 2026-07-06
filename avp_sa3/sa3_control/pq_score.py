"""Score onset-eval clips with Meta Audiobox Aesthetics Production Quality (PQ)
plus a spectral-balance brightness metric (lowpass-cheat detector).

Maps PQ across (gain, density) so we can SEE where a bracket disintegrates and
stop it there. Runs in the mir venv (audiobox_aesthetics + WavLM live there):

    /home/kim/Projects/mir/mir/bin/python pq_score.py <eval_dir> [--pq-floor 6.0]

Prints a gain x density PQ grid + flags cells below the floor (disintegrating).
Writes pq_scores.json into the dir.

Clip-name conventions handled (2026-07-05, eval-grid-rich-renderer spec):
  - single-prompt legacy:  onset_g{g}_d{d}.wav              (no prompt_idx/seed)
  - multi-prompt legacy:   onset_p{p}_g{g}_d{d}.wav          (prompt_idx, no seed)
  - composed-sweep:        {run}_p{p}_s{seed}_g{g}_d{d}.wav  (prompt_idx + seed)
  - DoRA audition (2026-07-06, eval-tables-human-first spec):
                           {runid}_epoch{N}-step{M}__p{p}_seed{seed}.wav
                           (no gain/density -- carries `checkpoint`/`epoch`/`step` instead)
  - prompt-style A/B (2026-07-07, newcap8_promptstyle):
                           {arm}__{promptkey}_{plain|styled}_s{seed}.wav
                           (no gain/density -- carries `checkpoint`=arm, `prompt_key`,
                           `style` instead; see Misc/eval_grid.py's render_style_compare_page)
Each row carries prompt_idx/seed as None when the filename convention doesn't
have them, so eval_grid.py's onset_eval.json join degrades to (gain, density).
DoRA rows carry gain/density as None instead and add checkpoint/epoch/step.

spectral_balance = mean spectral centroid / Nyquist (0-1ish; higher = brighter).
A clip that fakes higher onset density by lowpassing reads high on `measured`
but LOW on spectral_balance -- that mismatch is the cheat signal.
"""
import os, sys, re, glob, json

sys.path.insert(0, "/home/kim/Projects/mir/src")
from timbral.audiobox_aesthetics import analyze_audiobox_aesthetics
import librosa
import numpy as np

D = sys.argv[1]
FLOOR = float(sys.argv[sys.argv.index("--pq-floor") + 1]) if "--pq-floor" in sys.argv else 6.0

# Tried in order, most-specific first (composed-sweep names also end in
# ..._g{g}_d{d}.wav so the multi-prompt/single-prompt patterns must anchor
# on the literal "onset_" prefix to avoid mismatching them).
_COMPOSED = re.compile(r"_p(?P<p>\d+)_s(?P<seed>\d+)_g(?P<g>[0-9.]+)_d(?P<d>[0-9.]+)\.wav$")
_MULTI_PROMPT = re.compile(r"^onset_p(?P<p>\d+)_g(?P<g>[0-9.]+)_d(?P<d>[0-9.]+)\.wav$")
_SINGLE_PROMPT = re.compile(r"^onset_g(?P<g>[0-9.]+)_d(?P<d>[0-9.]+)\.wav$")
_DORA = re.compile(r"^(?P<runid>[a-z0-9]+)_epoch(?P<epoch>\d+)-step(?P<step>\d+)__p(?P<p>\d+)_seed(?P<seed>\d+)\.wav$")
_PROMPTSTYLE = re.compile(r"^(?P<arm>[a-zA-Z0-9]+)__(?P<promptkey>[a-z]+)_(?P<style>plain|styled)_s(?P<seed>\d+)\.wav$")


def parse_clip_name(basename: str):
    m = _PROMPTSTYLE.match(basename)
    if m:
        return {"prompt_idx": None, "seed": int(m["seed"]),
                "gain": None, "density": None,
                "checkpoint": m["arm"], "prompt_key": m["promptkey"], "style": m["style"]}
    m = _DORA.match(basename)
    if m:
        return {"prompt_idx": int(m["p"]), "seed": int(m["seed"]),
                "gain": None, "density": None,
                "checkpoint": f"{m['runid']}_epoch{m['epoch']}-step{m['step']}",
                "epoch": int(m["epoch"]), "step": int(m["step"])}
    m = _COMPOSED.search(basename)
    if m:
        return {"prompt_idx": int(m["p"]), "seed": int(m["seed"]),
                "gain": float(m["g"]), "density": float(m["d"])}
    m = _MULTI_PROMPT.match(basename)
    if m:
        return {"prompt_idx": int(m["p"]), "seed": None,
                "gain": float(m["g"]), "density": float(m["d"])}
    m = _SINGLE_PROMPT.match(basename)
    if m:
        return {"prompt_idx": None, "seed": None,
                "gain": float(m["g"]), "density": float(m["d"])}
    return None


def spectral_balance(path: str) -> float | None:
    """Mean spectral centroid normalized by Nyquist. None on load failure."""
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        if y.size == 0:
            return None
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        return round(float(np.mean(centroid)) / (sr / 2.0), 4)
    except Exception:
        return None


clips = sorted(glob.glob(f"{D}/**/*.wav", recursive=True))  # recursive: DoRA auditions nest per-run subdirs
rows = []
for p in clips:
    base = os.path.basename(p)
    parsed = parse_clip_name(base)
    if parsed is None:
        continue
    g, d = parsed["gain"], parsed["density"]
    try:
        s = analyze_audiobox_aesthetics(p)
        pq = float(s.get("PQ") or s.get("production_quality"))
        ce = float(s.get("CE") or s.get("content_enjoyment"))
        cu = float(s.get("CU") or s.get("content_usefulness", 0) or 0)
        pc = float(s.get("PC") or s.get("production_complexity", 0) or 0)
    except Exception as e:
        print(f"  [skip] {base}: {e}", flush=True)
        continue
    sb = spectral_balance(p)
    broke = pq < FLOOR or ce < FLOOR          # disintegration on EITHER axis (PQ can hold while CE drops)
    row = {"prompt_idx": parsed["prompt_idx"], "seed": parsed["seed"],
           "gain": g, "density": d, "PQ": round(pq, 3), "CE": round(ce, 3),
           "CU": round(cu, 3), "PC": round(pc, 3), "spectral_balance": sb,
           "broke": broke}
    if "epoch" in parsed:
        row.update(checkpoint=parsed["checkpoint"], epoch=parsed["epoch"], step=parsed["step"],
                   rel_path=os.path.relpath(p, D))
    elif "prompt_key" in parsed:
        row.update(checkpoint=parsed["checkpoint"], prompt_key=parsed["prompt_key"],
                   style=parsed["style"], rel_path=os.path.relpath(p, D))
    rows.append(row)
    tag = f"g{g:<4} d{d:<5}" if g is not None else f"{row.get('checkpoint','?'):<28}"
    label = f"{row.get('prompt_key')}/{row.get('style')}" if "prompt_key" in parsed else f"p{row['prompt_idx']}"
    print(f"  {tag} {label} s{row['seed']} PQ {pq:.2f}  CE {ce:.2f}"
          f"  SB {sb if sb is not None else '-'}"
          f"{'  <-- DISINTEGRATING' if broke else ''}", flush=True)

json.dump(rows, open(f"{D}/pq_scores.json", "w"), indent=2)

# grids for PQ and CE (the two thresholds we watch) -- gain/density grids only; DoRA
# auditions (no gain/density) skip straight to the summary line below.
if rows and rows[0].get("gain") is not None:
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
