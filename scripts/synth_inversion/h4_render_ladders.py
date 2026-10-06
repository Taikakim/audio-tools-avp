"""H4 gate, stage 1 (SAT venv, CPU): render knob ladders as short bass phrases for SAME encoding.

Per knob axis (training_controls.AXES), N anchor patches from the realistic-bass prior; each anchor is swept
along that one knob over n_steps rungs (make_ladder: everything else fixed). Each rung is rendered as the
SAME 4 s phrase -- a 145 BPM off-beat 16th roll on the anchor's note -- because SAME's 10.77 Hz latent cannot
see a single short note well. Peak-normalised, as eval_refine_phrase renders are.

Out: <out>/ladders.npz  audio [n_axes*N*n_steps, samples] float16 mono @ SAMPLE_RATE, vecs, axis_ids,
anchor_ids, rung; plus run_meta.json. Stage 2 is SAO/eval/h4_same_directions.py (SA3 venv).
Run: sat-venv/bin/python scripts/synth_inversion/h4_render_ladders.py [--anchors 50] [--procs 3]
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
OUT = "/run/media/kim/Mantu/surge_200k_models/h4_gate"
BPM, DUR = 145.0, 4.0


def phrase_events(note, division=16):
    """division 16: the off-beat 16th roll. Any other division: one note per 1/division, gate 0.8."""
    import mido
    if division != 16:
        step = 60.0 / BPM * 4 / division
        ev, t = [], 0.0
        while t + 0.8 * step <= DUR:
            ev.append(mido.Message("note_on", note=int(note), velocity=100, time=t))
            ev.append(mido.Message("note_off", note=int(note), velocity=0, time=t + 0.8 * step))
            t += step
        return ev
    six = 60.0 / BPM / 4
    ev, t = [], 0.0
    while t + six <= DUR:
        for k in (1, 2, 3):                      # beat = kick slot; bass on the three off-16ths
            on = t + k * six
            if on + 0.8 * six <= DUR:
                ev.append(mido.Message("note_on", note=int(note), velocity=100, time=on))
                ev.append(mido.Message("note_off", note=int(note), velocity=0, time=on + 0.8 * six))
        t += 4 * six
    return sorted(ev, key=lambda m: m.time)


def render_axis(args):
    ax, axis, n_anchors, n_steps, seed, division = args
    from eval_refine_phrase import make_renderer
    from realistic_bass_prior import MANIFOLD_PATH, RealisticBassPrior
    from surge_spec import DEFAULT_PLUGIN_PATH, init_synth, vector_to_patch
    from training_controls import make_ladder
    synth = init_synth(DEFAULT_PLUGIN_PATH, verify=False)
    prior = RealisticBassPrior(MANIFOLD_PATH, split="val")
    rng = np.random.RandomState(seed + 1000 * ax)
    audio, vecs, anchors, rungs, notes = [], [], [], [], []
    for a in range(n_anchors):
        _, base, midi_note, _ = prior.sample_patch_and_midi(rng)
        render = make_renderer(synth, phrase_events(midi_note, division), DUR)
        for r, vec in enumerate(make_ladder(base, axis, n_steps, rng)):
            audio.append(render(vector_to_patch(vec)).astype(np.float16))
            vecs.append(vec)
            anchors.append(a)
            rungs.append(r)
            notes.append(int(midi_note))
    print(f"{axis}: {len(audio)} renders", flush=True)
    return ax, np.stack(audio), np.stack(vecs), np.array(anchors), np.array(rungs), np.array(notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=50)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--division", type=int, default=16, help="16 = off-beat 16th roll; 4 = quarter notes")
    a = ap.parse_args()
    from surge_spec import SAMPLE_RATE
    from training_controls import AXES
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    jobs = [(ax, axis, a.anchors, a.steps, a.seed, a.division) for ax, axis in enumerate(AXES)]
    with get_context("spawn").Pool(min(a.procs, len(jobs))) as pool:
        parts = sorted(pool.map(render_axis, jobs), key=lambda p: p[0])
    audio = np.concatenate([p[1] for p in parts])
    axis_ids = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
    np.savez(f"{a.out}/ladders.npz", audio=audio, vecs=np.concatenate([p[2] for p in parts]),
             axis_ids=axis_ids, anchor_ids=np.concatenate([p[3] for p in parts]),
             rung=np.concatenate([p[4] for p in parts]), midi_note=np.concatenate([p[5] for p in parts]),
             sample_rate=SAMPLE_RATE, axes=np.array(AXES))
    json.dump({"purpose": "H4 gate stage 1: one-knob ladders rendered as 4 s bass phrases, to test whether each Surge "
                          "knob has a consistent direction in SAME latent space (EXPERIMENTS.md H4).",
               "created": str(date.today()), "script": "stable-audio-tools/scripts/synth_inversion/h4_render_ladders.py",
               "next": "SAO/eval/h4_same_directions.py", "axes": list(AXES), "anchors_per_axis": a.anchors,
               "steps": a.steps, "seed": a.seed, "phrase": (f"{BPM:g} BPM off-beat 16th roll" if a.division == 16 else f"{BPM:g} BPM 1/{a.division} notes")
                         + f", {DUR:g} s, gate 0.8",
               "prior_split": "val", "seconds": round(time.time() - t0, 1)},
              open(f"{a.out}/run_meta.json", "w"), indent=1)
    print(f"wrote {len(audio)} renders in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
