"""H4 ladders v2 (SAT venv, CPU): the scale-up for the knob-direction work (W/C split, 2026-10-06).

v1 (h4_render_ladders.py) had 4 knobs, one phrase rhythm and 8 rungs, and showed that which knobs are
readable in SAME depends on note length vs the knob's envelope times. v2 makes rhythm a factor:
  knobs    10: cutoff (full range), resonance, feg_amount, feg_decay, aeg_decay, aeg_sustain, aeg_release,
           shape, sub_mix, fm_depth. Each ladder sweeps ONE knob over its full CONT_BOUNDS range, with the
           context that makes it audible (e.g. FEG decay needs FEG amount up and FEG sustain 0).
  rhythm   r16 (off-beat 16th roll), e8 (8ths), q4 (quarters, gate 0.8), leg (legato quarters, gate 1.0),
           balanced per knob
  rungs    8 or 16, alternating
Anchors come from the realistic-bass prior's VAL split (--split train for the 105-preset generality set). Each ladder's knob value is stored exactly
(`knob_value`), and the swept column is `vecs[:, PARAM_INDEX[knob]]`.
Out: <out>/ladders_v2.npz (audio float16 mono, vecs, knob_id, ladder_id, rung, n_rungs, knob_value, rhythm_id,
midi_note) + run_meta.json. Encode/eval: SAO/eval/h4_same_directions.py --v2 (stage 2).
Run: sat-venv/bin/python scripts/synth_inversion/h4_render_ladders_v2.py [--per_knob 200] [--procs 6]
"""
import argparse
import json
import os
import sys
import time
from datetime import date
from multiprocessing import get_context

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = "/run/media/kim/Mantu/surge_200k_models/h4_gate_v2"
BPM, DUR = 145.0, 4.0
VERSION = 2
KNOBS = ("cutoff", "resonance", "feg_amount", "feg_decay", "aeg_decay", "aeg_sustain", "aeg_release",
         "shape", "sub_mix", "fm_depth")
RHYTHMS = ("r16", "e8", "q4", "leg")


def phrase_events(note, rhythm):
    import mido
    six = 60.0 / BPM / 4
    if rhythm == "r16":
        starts, gate = [b * 4 * six + k * six for b in range(int(DUR / (4 * six)) + 1) for k in (1, 2, 3)], 0.8 * six
    else:
        step = {"e8": 2 * six, "q4": 4 * six, "leg": 4 * six}[rhythm]
        starts, gate = [i * step for i in range(int(DUR / step) + 1)], (0.8 if rhythm != "leg" else 0.98) * step
    ev = []
    for on in starts:
        if on + gate <= DUR:
            ev.append(mido.Message("note_on", note=int(note), velocity=100, time=on))
            ev.append(mido.Message("note_off", note=int(note), velocity=0, time=on + gate))
    return sorted(ev, key=lambda m: m.time)


def make_ladder_v2(base, knob, n, rng):
    from surge_spec import CONT_BOUNDS, PARAM_INDEX as I, canonicalize_vector
    v = np.array(base, dtype=np.float32, copy=True)

    def setp(name, lo, hi):
        a, b = CONT_BOUNDS[name]
        v[I[name]] = float(np.clip(rng.uniform(lo, hi), a, b))

    if knob in ("feg_amount", "feg_decay"):
        setp("cutoff", 0.08, 0.35)
        v[I["feg_sustain"]] = 0.0
        if knob == "feg_decay":
            setp("feg_amount", 0.6, 0.9)
    elif knob == "aeg_decay":
        setp("aeg_sustain", 0.0, 0.2)
    elif knob == "aeg_release":
        setp("aeg_sustain", 0.3, 0.8)
    elif knob == "resonance":
        setp("cutoff", 0.15, 0.5)
    elif knob == "fm_depth":
        v[I["sync"]] = 0.0
    out = np.repeat(v[None], n, axis=0)
    out[:, I[knob]] = np.linspace(*CONT_BOUNDS[knob], n)
    return canonicalize_vector(out)


def render_knob(args):
    k, knob, per_knob, seed, split = args
    from eval_refine_phrase import make_renderer
    from realistic_bass_prior import MANIFOLD_PATH, RealisticBassPrior
    from surge_spec import DEFAULT_PLUGIN_PATH, PARAM_INDEX, init_synth, vector_to_patch
    synth = init_synth(DEFAULT_PLUGIN_PATH, verify=False)
    prior = RealisticBassPrior(MANIFOLD_PATH, split=split)
    rng = np.random.RandomState(seed + 1000 * k)
    cols = {c: [] for c in ("audio", "vecs", "ladder", "rung", "n_rungs", "rhythm", "note", "archetype")}
    for li in range(per_knob):
        peek = np.random.RandomState()
        peek.set_state(rng.get_state())
        archetype = peek.randint(0, prior.N)        # sample_vector's first draw picks the base preset
        _, base, note, _ = prior.sample_patch_and_midi(rng)
        rhythm, n = li % len(RHYTHMS), (8 if (li // len(RHYTHMS)) % 2 == 0 else 16)
        render = make_renderer(synth, phrase_events(note, RHYTHMS[rhythm]), DUR)
        for r, vec in enumerate(make_ladder_v2(base, knob, n, rng)):
            cols["audio"].append(render(vector_to_patch(vec)).astype(np.float16))
            cols["vecs"].append(vec)
            cols["ladder"].append(li)
            cols["rung"].append(r)
            cols["n_rungs"].append(n)
            cols["rhythm"].append(rhythm)
            cols["note"].append(int(note))
            cols["archetype"].append(int(archetype))
    print(f"{knob}: {len(cols['audio'])} renders", flush=True)
    out = {c: np.stack(v) if c in ("audio", "vecs") else np.array(v) for c, v in cols.items()}
    out["knob_value"] = out["vecs"][:, PARAM_INDEX[knob]]
    return k, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_knob", type=int, default=200)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--knobs", default=",".join(KNOBS), help="subset, e.g. cutoff (the v2c supplement)")
    ap.add_argument("--split", default="val", help="prior split: val (17 presets) or train (many more)")
    ap.add_argument("--name", default="ladders_v2", help="output file stem, e.g. ladders_v2c")
    a = ap.parse_args()
    from surge_spec import SAMPLE_RATE
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    with get_context("spawn").Pool(min(a.procs, len(KNOBS))) as pool:
        parts = sorted(pool.map(render_knob, [(k, kn, a.per_knob, a.seed, a.split) for k, kn in enumerate(KNOBS)
                                              if kn in a.knobs.split(",")]),
                       key=lambda p: p[0])
    cat = {c: np.concatenate([p[1][c] for p in parts]) for c in parts[0][1]}
    knob_id = np.concatenate([np.full(len(p[1]["rung"]), p[0]) for p in parts])
    ladder_id = knob_id * a.per_knob + cat["ladder"]
    np.savez(f"{a.out}/{a.name}.npz", audio=cat["audio"], vecs=cat["vecs"], knob_id=knob_id,
             ladder_id=ladder_id, rung=cat["rung"], n_rungs=cat["n_rungs"], knob_value=cat["knob_value"],
             rhythm_id=cat["rhythm"], midi_note=cat["note"], archetype_id=cat["archetype"], prior_split=a.split, sample_rate=SAMPLE_RATE, knobs=np.array(KNOBS),
             rhythms=np.array(RHYTHMS), version=VERSION)
    json.dump({"purpose": "H4 v2: one-knob ladders over 10 Surge knobs x 4 phrase rhythms x 8/16 rungs, for the "
                          "knob-direction baselines (W) and the g(z) linearising map (C).",
               "hypothesis": "Knob directions in SAME depend on note length vs envelope times; rhythm as a factor "
                             "separates knob geometry from phrase geometry.",
               "created": str(date.today()), "version": VERSION,
               "script": "stable-audio-tools/scripts/synth_inversion/h4_render_ladders_v2.py",
               "file": f"{a.name}.npz", "knobs_rendered": a.knobs.split(","), "knobs": list(KNOBS), "rhythms": list(RHYTHMS), "per_knob": a.per_knob, "seed": a.seed,
               "prior_split": a.split, "n_renders": int(len(knob_id)), "seconds": round(time.time() - t0, 1)},
              open(f"{a.out}/run_meta{'' if a.name == 'ladders_v2' else '_' + a.name}.json", "w"), indent=1)
    print(f"wrote {len(knob_id)} renders in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
