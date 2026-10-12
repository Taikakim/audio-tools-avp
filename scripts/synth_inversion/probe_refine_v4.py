# Owner: KUANG (created 2026-10-11)
"""Does the existing `refine_v3.Refiner` work with the v4 objectives, and what does it recover on a stand-in synth?

The Refiner looks up the module-level `objective(cand, target, env_weight)` at call time, so a different objective is a one-line
swap (`use_objective`). This runs the real Refiner (coordinate search with renders) against a stand-in renderer whose controls
are mapped onto the Refiner's axes, from a perturbed start, under three objectives:

  v3        refine_v3.objective itself (paper-style multi-scale log-mel + envelope cosine)
  logmel    objectives_v4.logmel_l1
  two_anchor objectives_v4.TwoAnchorObjective (onset-aligned attack + sustain + global), plus diagnostic variants:
            _nosnap = no per-candidate onset snapping (the snapped onset moves in 1 ms steps as controls change, which can make
            the landscape jittery; here the stand-in target and candidates share onset 0, so snapping is not needed),
            _111 = equal weights instead of the brief's (4, 2, 1)

Reported per objective (mean over seeds): mean absolute error of the recovered controls (in the [0, 1] axis units of the Refiner),
and the final distance under a HELD-OUT metric (mss_l1) so no objective is scored by its own yardstick.

Stand-in only: this shows the plumbing and a clean-setting behavior, not Surge or stem performance. No thresholds are set.
"""
import argparse
import sys
from contextlib import contextmanager

import numpy as np

import refine_v3
from objectives_v4 import TwoAnchorObjective, logmel_l1, mss_l1
from standin_synth_v4 import StandinPatch, render

AXES = ("cutoff", "feg_amount", "feg_decay", "aeg_decay", "aeg_sustain", "resonance", "aeg_release")   # feg_sustain has no stand-in control
TRUTH = dict(cutoff=0.5, feg_amount=0.6, feg_decay=0.4, aeg_decay=0.5, aeg_sustain=0.3, feg_sustain=0.5, resonance=0.5, aeg_release=0.3)


def standin_from_axes(p: dict, f0: float = 55.0):
    return StandinPatch(f0=f0, fc_hz=200.0 * 2 ** (4.0 * p["cutoff"]), fc_env_oct=3.0 * p["feg_amount"],
                        fc_decay_s=0.02 * 20 ** p["feg_decay"], q=0.7 * 40 ** p["resonance"],
                        amp_decay_s=0.05 * 16 ** p["aeg_decay"], amp_sustain=p["aeg_sustain"],
                        release_s=0.02 * 10 ** p["aeg_release"], phase_seed=1)


def render_axes(p: dict) -> np.ndarray:
    return render(standin_from_axes(p), dur=0.8, note_dur=0.5, max_harm=80)


@contextmanager
def use_objective(fn):
    old = refine_v3.objective
    refine_v3.objective = fn
    try:
        yield
    finally:
        refine_v3.objective = old


def objective_for(name: str):
    if name == "v3":
        return refine_v3.objective
    if name == "logmel":
        return lambda cand, target, env_weight=10.0: logmel_l1(cand, target)
    variants = {"two_anchor": dict(), "two_anchor_nosnap": dict(snap=False), "two_anchor_111": dict(weights=(1.0, 1.0, 1.0)),
                "two_anchor_nosnap_111": dict(snap=False, weights=(1.0, 1.0, 1.0))}
    if name in variants:
        ta = TwoAnchorObjective(**variants[name])
        return lambda cand, target, env_weight=10.0: ta(cand, target)
    raise ValueError(name)


def run(names, seeds, perturb, passes):
    target = render_axes(TRUTH)
    rows = {n: {"param_err": [], "heldout": [], "start_err": [], "renders": []} for n in names}
    for s in range(seeds):
        rng = np.random.default_rng(s)
        start = {k: float(np.clip(v + rng.uniform(-perturb, perturb), 0.0, 1.0)) for k, v in TRUTH.items()}
        for n in names:
            with use_objective(objective_for(n)):
                r = refine_v3.Refiner(render_axes, target)
                out, _ = r.refine(start, axes=[(k, 0.15) for k in AXES], passes=passes, n_coarse=5, n_fine=3)
            rows[n]["param_err"].append(float(np.mean([abs(out[k] - TRUTH[k]) for k in AXES])))
            rows[n]["start_err"].append(float(np.mean([abs(start[k] - TRUTH[k]) for k in AXES])))
            rows[n]["heldout"].append(float(mss_l1(render_axes(out), target)))
            rows[n]["renders"].append(r.n_renders)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--perturb", type=float, default=0.2)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--objectives", default="v3,logmel,two_anchor,two_anchor_nosnap,two_anchor_111,two_anchor_nosnap_111")
    ap.add_argument("--out", default="probe_refine_v4.md")
    a = ap.parse_args(argv)
    rows = run(a.objectives.split(","), a.seeds, a.perturb, a.passes)
    L = ["# Refiner probe (stand-in synth, NOT Surge)", "",
         f"{a.seeds} starts, each control perturbed by up to +-{a.perturb}; {a.passes} passes; mean over starts.", "",
         "| objective | start error | recovered error (mean abs, [0,1] units) | held-out mss_l1 | renders |", "|---|---|---|---|---|"]
    for n, r in rows.items():
        L.append(f"| {n} | {np.mean(r['start_err']):.3f} | {np.mean(r['param_err']):.3f} (per start: "
                 + ", ".join(f"{v:.3f}" for v in r["param_err"]) + f") | {np.mean(r['heldout']):.3f} | {int(np.mean(r['renders']))} |")
    L += ["", "Stand-in, clean, single note, one target: plumbing and behavior in a clean setting only."]
    rep = "\n".join(L)
    print(rep)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(rep + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
