# Owner: KUANG (created 2026-10-12)
"""Blind, reference-conditioned A/B listening test for inversion systems (EVAL_PARADIGM_V4.md, tier T3).

Why: every automatic real-stem metric we use (MSS, wMFCC, envelope cosine, the reranker's composite) is unvalidated
against what we actually want, which is "sounds closer to the reference". This builds a small, defensible
preference test (research report Q10, adopted with our changes):
  * question per trial: "Which of A and B sounds closer to the reference?" (forced choice);
  * the clips are produced by a PRE-REGISTERED protocol (fixed seeds, fixed selector): the listener never picks
    a seed or a candidate, otherwise the listener becomes part of the algorithm;
  * A/B sides and trial order are randomised; files are anonymised; the key is kept in a separate file;
  * every clip of a trial is RMS-matched (pyloudnorm LUFS when installed), so loudness cannot decide;
  * attention checks pair the reference with a clearly degraded copy of itself;
  * analysis resamples STEMS (several trials share a stem), and reports side bias and the check pass rate.
The same answers later validate the automatic metrics (which metric agrees with the ear, EVAL_PARADIGM_V4.md T4).

Layout:  --clips DIR with <stem>__<system>.wav (one clip per stem and system), --refs DIR with <stem>.wav
Usage:
  python listening_ab_v4.py make --clips clips/ --refs refs/ --pair A3_ema A1_ema --pair A3_ema A2_ema \\
      --checks 4 --seed 0 --out ab_test_01
  (open ab_test_01/index.html, listen, press "Download answers")
  python listening_ab_v4.py analyze --key ab_test_01/key.json --answers answers.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import eval_stats_v4 as st  # noqa: E402

TARGET_RMS_DBFS = -20.0


def _read(path):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=True)
    return x.astype(np.float64), sr


def _write(path, x, sr):
    import soundfile as sf
    sf.write(path, x, sr, subtype="PCM_16")


def loudness_match(x, sr):
    """Scale to TARGET_RMS_DBFS (LUFS when pyloudnorm is available); never clip."""
    try:
        import pyloudnorm as pyln
        lufs = pyln.Meter(sr).integrated_loudness(x)
        gain_db = TARGET_RMS_DBFS - lufs if np.isfinite(lufs) else 0.0
    except Exception:
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        gain_db = TARGET_RMS_DBFS - 20.0 * np.log10(rms)
    y = x * 10.0 ** (gain_db / 20.0)
    peak = np.max(np.abs(y))
    return y / peak * 0.98 if peak > 0.98 else y


def degrade(x, sr, rng):
    """An attention-check foil: the reference with broadband noise at 6 dB SNR and a 4 kHz low-pass."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, 4000.0, btype="low", fs=sr, output="sos")
    y = sosfilt(sos, x, axis=0)
    noise = rng.standard_normal(y.shape)
    noise *= np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * 10 ** (-6.0 / 20.0)
    return y + noise


def make(a):
    rng = np.random.default_rng(a.seed)
    out_audio = os.path.join(a.out, "audio")
    os.makedirs(out_audio, exist_ok=True)
    clips = {}
    for fn in sorted(os.listdir(a.clips)):
        if fn.endswith(".wav") and "__" in fn:
            stem, system = fn[:-4].split("__", 1)
            clips.setdefault(stem, {})[system] = os.path.join(a.clips, fn)
    trials = []
    for sa, sb in a.pair:
        for stem in sorted(clips):
            if sa in clips[stem] and sb in clips[stem] and os.path.exists(os.path.join(a.refs, f"{stem}.wav")):
                trials.append({"stem": stem, "x": sa, "y": sb, "check": False})
    if not trials:
        raise SystemExit("no trial: check --clips naming (<stem>__<system>.wav), --refs and --pair")
    check_stems = list(rng.choice(sorted({t["stem"] for t in trials}), size=min(a.checks, len(trials)), replace=False))
    trials += [{"stem": s, "x": "__reference__", "y": "__degraded__", "check": True} for s in check_stems]
    order = rng.permutation(len(trials))
    public, key = [], {}
    for n, i in enumerate(order):
        t = trials[i]
        tid = f"t{n + 1:03d}"
        flip = bool(rng.integers(0, 2))
        ref, sr = _read(os.path.join(a.refs, f"{t['stem']}.wav"))
        if t["check"]:
            xa, xb = ref.copy(), degrade(ref, sr, rng)
        else:
            xa, _ = _read(clips[t["stem"]][t["x"]])
            xb, _ = _read(clips[t["stem"]][t["y"]])
        A, B = (xb, xa) if flip else (xa, xb)
        names = {}
        for role, sig in (("ref", ref), ("A", A), ("B", B)):
            names[role] = f"{tid}_{role}.wav"
            _write(os.path.join(out_audio, names[role]), loudness_match(sig, sr), sr)
        public.append({"trial": tid, **{k: f"audio/{v}" for k, v in names.items()}})
        key[tid] = {"stem": t["stem"], "A": t["y"] if flip else t["x"], "B": t["x"] if flip else t["y"],
                    "check": t["check"]}
    with open(os.path.join(a.out, "trials.json"), "w") as f:
        json.dump(public, f, indent=1)
    with open(os.path.join(a.out, "key.json"), "w") as f:
        json.dump({"seed": a.seed, "pairs": a.pair, "trials": key}, f, indent=1)
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(_html(public))
    print(f"{len(public)} trials ({len(check_stems)} attention checks) in {a.out}; key.json is the answer key: "
          "do not open it before listening.")
    return 0


def _html(public):
    rows = "\n".join(
        f'<div class="t" id="{t["trial"]}"><h3>{t["trial"]}</h3>'
        f'<p>Reference <audio controls preload="none" src="{t["ref"]}"></audio></p>'
        f'<p>A <audio controls preload="none" src="{t["A"]}"></audio></p>'
        f'<p>B <audio controls preload="none" src="{t["B"]}"></audio></p>'
        f'<p>Closer to the reference: <label><input type="radio" name="{t["trial"]}" value="A"> A</label> '
        f'<label><input type="radio" name="{t["trial"]}" value="B"> B</label></p></div>' for t in public)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Blind A/B</title>
<style>body{{font-family:sans-serif;max-width:760px;margin:2em auto;padding:0 16px}}.t{{border:1px solid #ccc;
padding:8px 12px;margin:12px 0;border-radius:6px}}audio{{vertical-align:middle;width:70%}}</style></head><body>
<h1>Which sounds closer to the reference?</h1><p>Listen to the reference, then A and B. Choose one for every trial.
Answers are kept in this browser until you download them.</p>
<label>Listener <input id="who" value=""></label>
{rows}
<p><button onclick="dl()">Download answers</button> <span id="n"></span></p>
<script>
const K='ab_answers_'+location.pathname;
const saved=JSON.parse(localStorage.getItem(K)||'{{}}');
document.querySelectorAll('input[type=radio]').forEach(r=>{{
  if(saved[r.name]===r.value) r.checked=true;
  r.addEventListener('change',()=>{{saved[r.name]=r.value;localStorage.setItem(K,JSON.stringify(saved));upd();}});}});
function upd(){{document.getElementById('n').textContent=Object.keys(saved).length+' answered';}}
function dl(){{const who=document.getElementById('who').value||'listener';
  let s='trial,choice,listener\\n';for(const [t,c] of Object.entries(saved)) s+=t+','+c+','+who+'\\n';
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}}));
  a.download='answers.csv';a.click();}}
upd();
</script></body></html>
"""


def analyze_answers(key, answers):
    """key: {"trials": {tid: {stem, A, B, check}}}; answers: list of {trial, choice, listener}.
    Returns per system pair the preference with a stem-clustered CI, plus check pass rate and side bias."""
    trials = key["trials"]
    checks, picks_a, pairs = [], [], {}
    for r in answers:
        t = trials.get(r["trial"])
        if t is None or r["choice"] not in ("A", "B"):
            continue
        chosen = t[r["choice"]]
        picks_a.append(r["choice"] == "A")
        if t["check"]:
            checks.append(chosen == "__reference__")
            continue
        x, y = sorted((t["A"], t["B"]))
        pairs.setdefault((x, y), []).append({"stem": t["stem"], "x_wins": float(chosen == x),
                                             "listener": r.get("listener", "")})
    out = {"n_answers": len(picks_a), "side_bias_A": float(np.mean(picks_a)) if picks_a else float("nan"),
           "checks_passed": int(sum(checks)), "checks_total": len(checks), "pairs": []}
    for (x, y), recs in sorted(pairs.items()):
        v = np.array([q["x_wins"] for q in recs])
        stems = np.array([q["stem"] for q in recs])
        ci = st.mean_ci(v - 0.5, stems, n_boot=10000, seed=0)  # centred: CI excludes 0 -> a preference
        per_stem = {}
        for q in recs:
            per_stem.setdefault(q["stem"], []).append(q["x_wins"])
        maj = [np.mean(w) for w in per_stem.values()]
        wins, losses = sum(m > 0.5 for m in maj), sum(m < 0.5 for m in maj)
        out["pairs"].append({"x": x, "y": y, "n_trials": len(v), "n_stems": len(per_stem), "x_pref": float(v.mean()),
                             "lo": 0.5 + ci["lo"], "hi": 0.5 + ci["hi"], "stem_wins_x": int(wins),
                             "stem_wins_y": int(losses), "sign_p": st.sign_test(wins, losses)})
    return out


def analyze(a):
    with open(a.key) as f:
        key = json.load(f)
    answers = []
    for path in a.answers:
        with open(path, newline="") as f:
            answers += list(csv.DictReader(f))
    res = analyze_answers(key, answers)
    print(f"{res['n_answers']} answers; attention checks {res['checks_passed']}/{res['checks_total']}; "
          f"chose A {100 * res['side_bias_A']:.0f}% of the time (50% = no side bias)")
    if res["checks_total"] and res["checks_passed"] < res["checks_total"]:
        print("WARNING: failed attention checks: treat this session's preferences with caution")
    for p in res["pairs"]:
        print(f"{p['x']} vs {p['y']}: {p['x']} preferred {100 * p['x_pref']:.0f}% "
              f"[{100 * p['lo']:.0f}, {100 * p['hi']:.0f}] over {p['n_trials']} trials / {p['n_stems']} stems; "
              f"stem majorities {p['stem_wins_x']}-{p['stem_wins_y']}, sign p {p['sign_p']:.3f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--clips", required=True)
    m.add_argument("--refs", required=True)
    m.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SYSTEM_X", "SYSTEM_Y"))
    m.add_argument("--checks", type=int, default=4)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--key", required=True)
    z.add_argument("--answers", nargs="+", required=True)
    z.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return make(a) if a.cmd == "make" else analyze(a)


if __name__ == "__main__":
    sys.exit(main())
