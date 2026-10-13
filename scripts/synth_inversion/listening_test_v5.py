# Owner: KUANG (created 2026-10-13)
"""Paper-style blind listening test for sound matching, several listeners (V5_PROPOSAL_REVIEW.md, sec. 4).

Protocol of the Synth-JEPA paper (arXiv 2609.31024, sec. 4.3), as read from the arXiv HTML on October 13, 2026:
reference-conditioned pairwise trials ("compared a reference to two sound matching attempts and selected the closest"),
16 targets (8 in-domain, 4 NSynth, 4 FSD50K) x 3 system pairs = 48 trials per participant, 18 participants, trial order
and A/B sides randomised, every trial's stimuli normalised to a shared -23 LUFS, headphones or good speakers, four
attention checks "containing the ground truth as an option" (all participants passed), and 95 % confidence intervals
from a bootstrap over PARTICIPANTS; no significance tests beyond the intervals.

What this tool adds to listening_ab_v4.py (one page, one order, one listener):
  * one page per listener (listener_XX/index.html) with its OWN trial order and A/B sides, all pages sharing one
    pool of anonymised audio files (random names, so a file name never reveals a system);
  * ground-truth attention checks as in the paper: one option is the reference itself (in its own anonymised file),
    the other a system's output; choosing the system is a failed check;
  * every file normalised to --lufs (default -23, BS.1770 integrated loudness: pyloudnorm if installed, otherwise
    the built-in meter below). Files that would clip are turned down and listed in the key;
  * domains (e.g. synthetic held-out notes vs real stems), reported separately as in the paper;
  * analysis: per system pair the preference with a 95 % CI from a bootstrap over participants (the paper's
    interval) AND over participants x targets (conservative: targets are a sample too), per domain, plus
    --focus SYSTEM: that system's preference over all trials it appears in (the paper's "preferred in 85.1 % of
    trials overall").
Note on power: with a handful of listeners the participant bootstrap is unstable; report the two-way interval and
read the per-target majorities (stem_wins) too.

Usage:
  python listening_test_v5.py make --clips listen --refs listen/refs --pair A3pDPO_best8 A3p_best8 \\
      --pair A3pXattn_best8 A3p_best8 --domain "syn*=synthetic" --domain "stem_*=real" --listeners 6 --checks 4 \\
      --seed 0 --out lt_v5_01
  (send listener_01/ .. listener_06/ together with audio/; each listener presses "Download answers")
  python listening_test_v5.py analyze --key lt_v5_01/key.json --answers answers_*.csv --focus A3pDPO_best8
"""
import argparse
import csv
import fnmatch
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ---------------------------------------------------------------------------------------------- loudness
def _biquad(kind, G, Q, fc, fs):
    """RBJ cookbook biquads with the K-weighting stage parameters pyloudnorm uses (high shelf +4 dB at 1500 Hz,
    high pass at 38 Hz), so the meter works at any sample rate."""
    A = 10 ** (G / 40.0)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * Q)
    c = np.cos(w0)
    if kind == "high_shelf":
        b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * alpha), -2 * A * ((A - 1) + (A + 1) * c),
             A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * alpha)]
        a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * alpha, 2 * ((A - 1) - (A + 1) * c),
             (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * alpha]
    else:
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def integrated_loudness(x, fs):
    """ITU-R BS.1770-4 integrated loudness (LUFS) of a mono or [n, ch] signal: K-weighting, 400 ms blocks with 75 %
    overlap, -70 LUFS absolute and -10 LU relative gates. -inf for silence."""
    try:
        import pyloudnorm as pyln
        return float(pyln.Meter(fs).integrated_loudness(x))
    except ImportError:
        pass
    from scipy.signal import lfilter
    y = np.asarray(x, dtype=np.float64)
    y = y[:, None] if y.ndim == 1 else y
    for kind, G, Q, fc in (("high_shelf", 4.0, 1 / np.sqrt(2), 1500.0), ("high_pass", 0.0, 0.5, 38.0)):
        b, a = _biquad(kind, G, Q, fc, fs)
        y = lfilter(b, a, y, axis=0)
    n_blk, step = int(round(0.4 * fs)), int(round(0.1 * fs))
    if len(y) < n_blk:
        return float("-inf")
    starts = range(0, len(y) - n_blk + 1, step)
    z = np.array([np.mean(y[s:s + n_blk] ** 2, axis=0).sum() for s in starts])   # channel weights 1 (L, R, C)
    with np.errstate(divide="ignore"):
        lj = -0.691 + 10 * np.log10(z)
    za = z[lj > -70.0]
    if len(za) == 0:
        return float("-inf")
    gamma_r = -0.691 + 10 * np.log10(za.mean()) - 10.0
    zr = z[(lj > -70.0) & (lj > gamma_r)]
    return float(-0.691 + 10 * np.log10(zr.mean()))


def normalize_loudness(x, fs, target):
    """-> (y, gain_db, limited): x scaled to `target` LUFS; turned down further (and limited=True) if it would clip."""
    lufs = integrated_loudness(x, fs)
    g = target - lufs if np.isfinite(lufs) else 0.0
    y = x * 10 ** (g / 20.0)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
        return y, g + 20 * np.log10(0.99 / peak), True
    return y, g, False


# ------------------------------------------------------------------------------------------------------ make
def _read(path):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=False)
    return np.asarray(x, dtype=np.float64), sr


def _write(path, x, sr):
    import soundfile as sf
    sf.write(path, x, sr, subtype="PCM_16")


def domain_of(stem, rules):
    for pattern, name in rules:
        if fnmatch.fnmatch(stem, pattern):
            return name
    return "all"


def make(a):
    rng = np.random.default_rng(a.seed)
    clips = {}
    for fn in sorted(os.listdir(a.clips)):
        if fn.endswith(".wav") and "__" in fn:
            stem, system = fn[:-4].split("__", 1)
            clips.setdefault(stem, {})[system] = os.path.join(a.clips, fn)
    systems = sorted({s for p in a.pair for s in p})
    stems = [s for s in sorted(clips) if all(x in clips[s] for x in systems)
             and os.path.exists(os.path.join(a.refs, f"{s}.wav")) and (not a.stems or s in a.stems)]
    if not stems:
        raise SystemExit("no target has every system in --pair plus a reference: check --clips / --refs naming")
    rules = [tuple(r.split("=", 1)) for r in a.domain]
    out_audio = os.path.join(a.out, "audio")
    os.makedirs(out_audio, exist_ok=True)
    files, levels, used = {}, {}, set()

    def put(stem, role, sig, sr):
        name = f"{rng.integers(0, 2 ** 48):012x}.wav"
        while name in used:
            name = f"{rng.integers(0, 2 ** 48):012x}.wav"
        used.add(name)
        y, g, lim = normalize_loudness(sig, sr, a.lufs)
        _write(os.path.join(out_audio, name), y, sr)
        files[(stem, role)] = f"../audio/{name}"
        levels[f"{stem}/{role}"] = {"gain_db": round(float(g), 2), "limited": bool(lim)}

    check_system = a.check_system or a.pair[0][1]
    for s in stems:
        ref, sr = _read(os.path.join(a.refs, f"{s}.wav"))
        put(s, "__ref__", ref, sr)
        put(s, "__truth__", ref, sr)                     # the reference again, in its own anonymous file
        for x in systems:
            sig, sr2 = _read(clips[s][x])
            put(s, x, sig, sr2)
    base = [{"stem": s, "x": x, "y": y, "check": False} for x, y in a.pair for s in stems]
    key = {"protocol": "synth-jepa-4.3", "seed": a.seed, "lufs": a.lufs, "pairs": a.pair, "systems": systems,
           "stems": stems, "domains": {s: domain_of(s, rules) for s in stems}, "check_system": check_system,
           "levels": levels, "listeners": {}}
    for li in range(1, a.listeners + 1):
        lid = f"L{li:02d}"
        check_stems = list(rng.choice(stems, size=min(a.checks, len(stems)), replace=False))
        trials = base + [{"stem": s, "x": "__truth__", "y": check_system, "check": True} for s in check_stems]
        order = rng.permutation(len(trials))
        public, k = [], {}
        for n, i in enumerate(order):
            t = trials[i]
            tid = f"t{n + 1:03d}"
            A, B = (t["y"], t["x"]) if rng.integers(0, 2) else (t["x"], t["y"])
            public.append({"trial": tid, "ref": files[(t["stem"], "__ref__")], "A": files[(t["stem"], A)],
                           "B": files[(t["stem"], B)]})
            k[tid] = {"stem": t["stem"], "A": A, "B": B, "check": t["check"]}
        key["listeners"][lid] = k
        d = os.path.join(a.out, f"listener_{li:02d}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
            f.write(_html(public, lid))
    with open(os.path.join(a.out, "key.json"), "w") as f:
        json.dump(key, f, indent=1)
    n_lim = sum(v["limited"] for v in levels.values())
    print(f"{a.listeners} listener pages x {len(base) + min(a.checks, len(stems))} trials ({len(stems)} targets, "
          f"{len(a.pair)} pairs, {min(a.checks, len(stems))} checks) in {a.out}; {n_lim} files turned down to avoid "
          "clipping. key.json is the answer key: keep it away from listeners.")
    return key


def _html(public, lid):
    rows = "\n".join(
        f'<div class="t"><h3>{n + 1} / {len(public)}</h3>'
        f'<p>Reference <audio controls preload="none" src="{t["ref"]}"></audio></p>'
        f'<p>A <audio controls preload="none" src="{t["A"]}"></audio></p>'
        f'<p>B <audio controls preload="none" src="{t["B"]}"></audio></p>'
        f'<p>Closer to the reference: <label><input type="radio" name="{t["trial"]}" value="A"> A</label> '
        f'<label><input type="radio" name="{t["trial"]}" value="B"> B</label></p></div>' for n, t in enumerate(public))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Sound matching test</title>
<style>body{{font-family:sans-serif;max-width:760px;margin:2em auto;padding:0 16px}}.t{{border:1px solid #ccc;
padding:8px 12px;margin:12px 0;border-radius:6px}}audio{{vertical-align:middle;width:70%;max-width:100%}}</style></head><body>
<h1>Which attempt sounds closer to the reference?</h1>
<p>Use headphones or good speakers at a comfortable, fixed volume. For each trial, listen to the reference, then A and
B, and choose the one closer to the reference. Answers stay in this browser until you download them.</p>
{rows}
<p><button onclick="dl()">Download answers</button> <span id="n"></span></p>
<script>
const LID='{lid}';const K='lt5_'+LID+'_'+location.pathname;
const saved=JSON.parse(localStorage.getItem(K)||'{{}}');
document.querySelectorAll('input[type=radio]').forEach(r=>{{
  if(saved[r.name]===r.value) r.checked=true;
  r.addEventListener('change',()=>{{saved[r.name]=r.value;localStorage.setItem(K,JSON.stringify(saved));upd();}});}});
function upd(){{document.getElementById('n').textContent=Object.keys(saved).length+' / {len(public)} answered';}}
function dl(){{let s='trial,choice,listener\\n';for(const [t,c] of Object.entries(saved)) s+=t+','+c+','+LID+'\\n';
  const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([s],{{type:'text/csv'}}));
  a.download='answers_'+LID+'.csv';a.click();}}
upd();
</script></body></html>
"""


# --------------------------------------------------------------------------------------------------- analyze
def _boot_pref(W, N, n_boot, rng, two_way):
    """W, N: [listeners, targets] wins of x and trial counts. Preference CI by resampling listeners (and targets)."""
    L, S = W.shape
    ml = rng.multinomial(L, np.full(L, 1.0 / L), size=n_boot).astype(float)               # [B, L]
    ms = rng.multinomial(S, np.full(S, 1.0 / S), size=n_boot).astype(float) if two_way else np.ones((n_boot, S))
    num = np.einsum("bl,ls,bs->b", ml, W, ms)
    den = np.einsum("bl,ls,bs->b", ml, N, ms)
    p = num[den > 0] / den[den > 0]
    return float(np.quantile(p, 0.025)), float(np.quantile(p, 0.975))


def analyze_answers(key, answers, exclude_failed=True, focus=None, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    by_l = {}
    for r in answers:
        lid = r.get("listener", "")
        t = key["listeners"].get(lid, {}).get(r["trial"])
        if t is None or r["choice"] not in ("A", "B"):
            continue
        by_l.setdefault(lid, []).append((t, t[r["choice"]], r["choice"]))
    listeners, excluded = {}, []
    for lid, rows in sorted(by_l.items()):
        checks = [chosen == "__truth__" for t, chosen, _ in rows if t["check"]]
        info = {"n": len(rows), "checks_passed": int(sum(checks)), "checks_total": len(checks),
                "side_A": float(np.mean([c == "A" for _, _, c in rows]))}
        listeners[lid] = info
        if exclude_failed and info["checks_passed"] < info["checks_total"]:
            excluded.append(lid)
    keep = [l for l in sorted(by_l) if l not in excluded]
    stems = key["stems"]
    s_idx = {s: i for i, s in enumerate(stems)}
    domains = sorted(set(key["domains"].values()))
    out = {"listeners": listeners, "excluded": excluded, "n_listeners": len(keep), "pairs": [], "focus": None}

    def tally(pred):
        """pred(trial) -> +1 if the trial counts as a win of the subject, 0 as a loss, None to skip."""
        W = np.zeros((len(keep), len(stems)))
        N = np.zeros_like(W)
        for li, lid in enumerate(keep):
            for t, chosen, _ in by_l[lid]:
                if t["check"]:
                    continue
                v = pred(t, chosen)
                if v is not None:
                    W[li, s_idx[t["stem"]]] += v
                    N[li, s_idx[t["stem"]]] += 1
        return W, N

    def summary(W, N, cols=None):
        if cols is not None:
            W, N = W[:, cols], N[:, cols]
        if N.sum() == 0 or len(keep) == 0:
            return None
        res = {"pref": float(W.sum() / N.sum()), "n_trials": int(N.sum())}
        res["ci_participants"] = _boot_pref(W, N, n_boot, rng, two_way=False)
        res["ci_participants_targets"] = _boot_pref(W, N, n_boot, rng, two_way=True)
        per_t = W.sum(0)[N.sum(0) > 0] / N.sum(0)[N.sum(0) > 0]
        res["target_majorities"] = [int((per_t > 0.5).sum()), int((per_t < 0.5).sum())]
        return res

    for x, y in key["pairs"]:
        W, N = tally(lambda t, c, x=x, y=y: None if {t["A"], t["B"]} != {x, y} else float(c == x))
        rec = {"x": x, "y": y, "all": summary(W, N)}
        for d in domains:
            rec[d] = summary(W, N, [i for i, s in enumerate(stems) if key["domains"][s] == d])
        out["pairs"].append(rec)
    if focus:
        W, N = tally(lambda t, c: None if focus not in (t["A"], t["B"]) else float(c == focus))
        out["focus"] = {"system": focus, "all": summary(W, N),
                        **{d: summary(W, N, [i for i, s in enumerate(stems) if key["domains"][s] == d]) for d in domains}}
    return out


def _fmt(s):
    if not s:
        return "n/a"
    lo, hi = s["ci_participants"]
    lo2, hi2 = s["ci_participants_targets"]
    return (f"{100 * s['pref']:.1f}% of {s['n_trials']} trials, 95% CI participants [{100 * lo:.1f}, {100 * hi:.1f}], "
            f"participants x targets [{100 * lo2:.1f}, {100 * hi2:.1f}]; target majorities "
            f"{s['target_majorities'][0]}-{s['target_majorities'][1]}")


def analyze(a):
    with open(a.key) as f:
        key = json.load(f)
    answers = []
    for p in a.answers:
        with open(p, newline="") as f:
            answers += list(csv.DictReader(f))
    res = analyze_answers(key, answers, exclude_failed=not a.keep_failed, focus=a.focus, n_boot=a.n_boot)
    for lid, info in res["listeners"].items():
        print(f"{lid}: {info['n']} answers, checks {info['checks_passed']}/{info['checks_total']}, "
              f"chose A {100 * info['side_A']:.0f}%" + ("  EXCLUDED" if lid in res["excluded"] else ""))
    for p in res["pairs"]:
        print(f"{p['x']} vs {p['y']}: {p['x']} preferred in {_fmt(p['all'])}")
        for d in sorted(set(key["domains"].values())):
            if d in p and p[d] and len(set(key["domains"].values())) > 1:
                print(f"    {d}: {_fmt(p[d])}")
    if res["focus"]:
        print(f"{res['focus']['system']} overall: {_fmt(res['focus']['all'])}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--clips", required=True)
    m.add_argument("--refs", required=True)
    m.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SYSTEM_X", "SYSTEM_Y"))
    m.add_argument("--stems", nargs="*", default=None)
    m.add_argument("--domain", action="append", default=[], metavar="GLOB=NAME")
    m.add_argument("--listeners", type=int, default=1)
    m.add_argument("--checks", type=int, default=4)
    m.add_argument("--check_system", default=None, help="the foil in attention checks (default: the first pair's y)")
    m.add_argument("--lufs", type=float, default=-23.0)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    z = sub.add_parser("analyze")
    z.add_argument("--key", required=True)
    z.add_argument("--answers", nargs="+", required=True)
    z.add_argument("--focus", default=None)
    z.add_argument("--keep_failed", action="store_true", help="keep listeners who failed an attention check")
    z.add_argument("--n_boot", type=int, default=10000)
    z.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "make":
        make(a)
    else:
        analyze(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
