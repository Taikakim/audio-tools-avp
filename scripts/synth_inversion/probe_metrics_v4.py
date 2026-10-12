# Owner: KUANG (created 2026-10-11)
"""Probe harness for candidate metrics and descriptors (research spec section 4). Prints and saves a table.

For each sweep, one control varies while the others stay fixed. Two kinds of rows:

  distance rows    Spearman between |sweep value - reference value| and the distance of that render to the reference
                   render (reference = the first value of the sweep). 1.0 = the metric grows monotonically with the
                   change. Mean and worst case over (patch, note) cells.
  descriptor rows  Spearman between the sweep value and a descriptor (log attack time, early decay slope,
                   resonance height, comb residual).

and a nuisance block: the distance caused by a time offset or a synthetic room, as a ratio to the MEAN distance caused by
three real parameter changes (amp decay x2, Q x3, filter envelope +1 octave). Lower is better; this is the
"signal-to-nuisance" row of the spec. The effect is averaged over several changes because a metric that is not built to
see one of them (the harmonic-envelope distance does not see amplitude decay) would otherwise get an inflated ratio.

Renderers (--render):
  standin   standin_synth_v4.py, CPU, runs anywhere. Results are LABELED "stand-in": they show a measure follows a control
            in a clean setting, nothing about Surge or real stems.
  surge     UNTESTED adapter for the training machine (surge_spec_v3.render_patch). Needs --base-patch <json>; sweeps are
            over the v3 patch keys. Not run here (no Surge). Check one cell by ear before trusting a table.

No pass/fail thresholds are set here. Kim decides after reading the table.
"""
import argparse
import json
import sys
import time

import numpy as np
from scipy.stats import spearmanr

from buzz_v4 import comb_residual
from objectives_v4 import TwoAnchorObjective, harmonic_envelope_distance, logmel_l1, mss_l1
from resonance_v4 import resonance_card
from transient_descriptors_v4 import attack_decay_descriptors

SR = 44100
NOTES = (28, 36, 44, 50)

# ---------------------------------------------------------------------------------------------------------------------
# stand-in renderer
# ---------------------------------------------------------------------------------------------------------------------
PATCHES = {   # stand-in controls; every sweep overrides one
    "pluck":  dict(fc_hz=700.0, fc_env_oct=2.0, fc_decay_s=0.08, q=2.0, attack_s=0.003, amp_decay_s=0.25, amp_sustain=0.2),
    "sustain": dict(fc_hz=1500.0, fc_env_oct=0.5, fc_decay_s=0.15, q=1.0, attack_s=0.005, amp_decay_s=0.5, amp_sustain=0.7),
}
SWEEPS = {   # name -> (control, values, descriptor name or None)
    "attack":   ("attack_s",    np.geomspace(0.002, 0.04, 7), "log_attack_time"),
    "decay":    ("amp_decay_s", np.geomspace(0.05, 0.8, 7), "early_decay_db_per_s"),
    "filt_env": ("fc_decay_s",  np.geomspace(0.02, 0.4, 7), None),
    "reso":     ("q",           np.array([0.7, 1.0, 2.0, 4.0, 8.0, 14.0, 20.0, 30.0]), "resonance_height"),
    "fm":       ("fm_depth",    np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5]), "comb_residual"),
}


def standin_render(patch_name, control, value, midi_note):   # controls: any StandinPatch field
    from standin_synth_v4 import StandinPatch, midi_to_hz, render
    kw = dict(PATCHES[patch_name])
    kw.update(f0=midi_to_hz(midi_note), phase_seed=1)
    if control == "fm_depth":
        kw.update(fm_ratio=1.37)
    kw[control] = float(value)
    return render(StandinPatch(**kw), dur=0.8, note_dur=0.5), kw["f0"]


def surge_render_factory(base_patch_path):
    """UNTESTED. Uses the calls refine_v3.py uses. v3 patch keys in [0, 1]; the sweeps map to aeg_decay, feg_decay,
    resonance and fm_depth. There is no attack control in the v3 vector, so the 'attack' sweep is skipped."""
    from refine_v3 import patch_from_json
    from surge_spec_v3 import DEFAULT_PLUGIN_PATH, init_synth, render_patch
    synth = init_synth(DEFAULT_PLUGIN_PATH, verify=False)
    base = patch_from_json(base_patch_path)
    keymap = {"amp_decay_s": "aeg_decay", "fc_decay_s": "feg_decay", "q": "resonance", "fm_depth": "fm_depth"}

    def fn(patch_name, control, value, midi_note):
        if control not in keymap:
            raise KeyError(control)
        p = dict(base)
        lo, hi = {"amp_decay_s": (0.05, 0.8), "fc_decay_s": (0.02, 0.4), "q": (0.7, 30.0), "fm_depth": (0.0, 1.5)}[control]
        p[keymap[control]] = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        from standin_synth_v4 import midi_to_hz
        return np.asarray(render_patch(synth, p, int(midi_note), 0.5, duration=0.8), np.float32), midi_to_hz(midi_note)
    return fn


# ---------------------------------------------------------------------------------------------------------------------
# metrics and descriptors
# ---------------------------------------------------------------------------------------------------------------------
_TA = TwoAnchorObjective()
METRICS = {
    "logmel_l1": lambda c, t, f0: logmel_l1(c, t),
    "mss_l1": lambda c, t, f0: mss_l1(c, t),
    "two_anchor": lambda c, t, f0: _TA(c, t),
    "harmonic_env": lambda c, t, f0: harmonic_envelope_distance(c, t, f0),
}


def descriptor(name, x, f0):
    if name == "resonance_height":
        return resonance_card(x, SR, f0).height_db
    if name == "comb_residual":
        return comb_residual(x, SR, f0)["residual"]
    return attack_decay_descriptors(x, SR, 0.0)[name]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


def run_sweeps(render_fn, sweeps, patches, notes):
    out = {}
    for sname in sweeps:
        control, values, desc = SWEEPS[sname]
        dist = {m: [] for m in METRICS}
        dsc = []
        for pname in patches:
            for note in notes:
                try:
                    renders = [render_fn(pname, control, v, note) for v in values]
                except KeyError:
                    out[sname] = None
                    break
                f0 = renders[0][1]
                ref = renders[0][0]
                for m, fn in METRICS.items():
                    d = [fn(r, ref, f0) for r, _ in renders]
                    dist[m].append(spearman(np.abs(values - values[0]), d))
                if desc:
                    dsc.append(spearman(values, [descriptor(desc, r, f0) for r, _ in renders]))
            else:
                continue
            break
        else:
            out[sname] = {"control": control, "distance": {m: (float(np.nanmean(v)), float(np.nanmin(v))) for m, v in dist.items()},
                          "descriptor": (desc, (float(np.nanmean(dsc)), float(np.nanmin(dsc)))) if desc else None}
    return out


def apply_room(x, rng, rt60=0.3, wet=0.15):
    n = int(rt60 * SR)
    ir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
    ir /= np.sqrt((ir ** 2).sum())
    w = np.convolve(x, ir)[:len(x)]
    return ((1 - wet) * x + wet * w * (np.std(x) / (np.std(w) + 1e-12))).astype(np.float32)


def run_nuisance(render_fn, patches, notes):
    rng = np.random.default_rng(0)
    rows = {m: {"shift10": [], "shift20": [], "room": []} for m in METRICS}
    for pname in patches:
        for note in notes:
            base = PATCHES[pname]
            clean, f0 = render_fn(pname, "amp_decay_s", base["amp_decay_s"], note)
            changes = [render_fn(pname, "amp_decay_s", 2 * base["amp_decay_s"], note)[0],
                       render_fn(pname, "q", min(3 * base["q"], 30.0), note)[0]]
            try:
                changes.append(render_fn(pname, "fc_env_oct", base["fc_env_oct"] + 1.0, note)[0])
            except KeyError:
                pass
            variants = {"shift10": np.concatenate([np.zeros(int(0.010 * SR), np.float32), clean])[:len(clean)],
                        "shift20": np.concatenate([np.zeros(int(0.020 * SR), np.float32), clean])[:len(clean)],
                        "room": apply_room(clean, rng)}
            for m, fn in METRICS.items():
                eff = float(np.mean([fn(c, clean, f0) for c in changes]))
                for k, v in variants.items():
                    rows[m][k].append(fn(v, clean, f0) / max(eff, 1e-9))
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in rows.items()}


def format_report(label, sweeps, nuis):
    L = [f"# Probe results ({label})", ""]
    L.append("Distance rows: Spearman(|sweep change|, distance to the first value), mean / worst over patch x note cells.")
    L.append("")
    L.append("| sweep | " + " | ".join(METRICS) + " | descriptor (Spearman mean / worst) |")
    L.append("|---|" + "---|" * (len(METRICS) + 1))
    for s, r in sweeps.items():
        if r is None:
            L.append(f"| {s} | skipped (control not available in this renderer) |" + " |" * len(METRICS))
            continue
        cells = " | ".join(f"{r['distance'][m][0]:.2f} / {r['distance'][m][1]:.2f}" for m in METRICS)
        d = r["descriptor"]
        L.append(f"| {s} ({r['control']}) | {cells} | " + (f"{d[0]}: {d[1][0]:.2f} / {d[1][1]:.2f}" if d else "-") + " |")
    L += ["", "Nuisance: distance caused by the nuisance divided by the mean distance of three real changes (amp decay x2, Q x3, filter envelope +1 octave); lower is better.", ""]
    L.append("| metric | shift 10 ms | shift 20 ms | room (RT60 0.3 s, 15% wet) |")
    L.append("|---|---|---|---|")
    for m, r in nuis.items():
        L.append(f"| {m} | {r['shift10']:.2f} | {r['shift20']:.2f} | {r['room']:.2f} |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--render", choices=("standin", "surge"), default="standin")
    ap.add_argument("--base-patch", default="", help="surge only: patch json (refine_v3.patch_from_json format)")
    ap.add_argument("--sweeps", default=",".join(SWEEPS))
    ap.add_argument("--patches", default=",".join(PATCHES))
    ap.add_argument("--notes", default=",".join(map(str, NOTES)))
    ap.add_argument("--out", default="probe_results_v4.md")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if a.render == "surge":
        if not a.base_patch:
            ap.error("--render surge needs --base-patch")
        fn, label = surge_render_factory(a.base_patch), "Surge (UNTESTED adapter)"
    else:
        fn, label = standin_render, "stand-in synth, NOT Surge"
    notes = [int(n) for n in a.notes.split(",")]
    t0 = time.time()
    sw = run_sweeps(fn, [x for x in a.sweeps.split(",") if x], a.patches.split(","), notes)
    nu = run_nuisance(fn, a.patches.split(","), notes)
    rep = format_report(label, sw, nu) + f"\n\n(patches {a.patches}; MIDI notes {a.notes}; {time.time() - t0:.0f} s)\n"
    print(rep)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(rep)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"label": label, "sweeps": {k: v for k, v in sw.items()}, "nuisance": nu}, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
