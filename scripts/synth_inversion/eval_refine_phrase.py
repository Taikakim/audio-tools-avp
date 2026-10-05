"""Phrase-mode refinement with a held-out half (follow-up to eval_refine_real_stems.py).

Single-note refinement (refine/) improved the phrase on 17/24 stems but worsened the envelope
match: its target is one 16th note cut off with a 5 ms fade, so it fitted envelopes to an
artificial truncation. Here the refinement scores the FIRST HALF of the real phrase, played from
the MuScriptor MIDI through the candidate patch (real note lengths, overlaps, several notes), and
every version is judged on the SECOND HALF, which no refinement saw:
  baseline        saved inverted patch
  note_refined    refine/<id>_refined_patch.json (from eval_refine_real_stems.py)
  phrase_refined  this script
Outputs: real_stems_eval/refine_phrase/ (wavs: real | baseline | note_refined | phrase_refined on
the held-out half, patches, results.json, run_meta.json).
"""
import json
import os
import sys
import time
from datetime import date

import mido
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference_metrics as rm  # noqa: E402
from refine import patch_from_json, refine_patch_phrase  # noqa: E402
from surge_spec import DEFAULT_PLUGIN_PATH, SAMPLE_RATE, apply_patch, init_synth  # noqa: E402

E = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval"
OUT = f"{E}/refine_phrase"


def midi_events(path, dur):
    t, ev = 0.0, []
    for msg in mido.MidiFile(path):
        t += msg.time
        if t > dur:
            break
        if msg.type in ("note_on", "note_off"):
            ev.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity, time=t))
    return sorted(ev, key=lambda m: m.time)


def make_renderer(synth, events, dur):
    def render(patch):
        apply_patch(synth, patch)
        synth.reset()
        a = synth.process(events, duration=dur, sample_rate=SAMPLE_RATE, num_channels=2).mean(axis=0)
        return (a / (np.max(np.abs(a)) + 1e-7)).astype(np.float32)
    return render


def scores(ref, x):
    return {"mss": round(rm.mss(ref, x), 3), "wmfcc": round(rm.wmfcc(ref, x), 3),
            "env_cos": round(rm.rms_env_cos(ref, x), 4)}


def main():
    os.makedirs(OUT, exist_ok=True)
    synth = init_synth(DEFAULT_PLUGIN_PATH)
    catalog = {d["id"]: d for d in json.load(open(f"{E}/real_stems_summary.json"))}
    rows, versions = [], ("baseline", "note_refined", "phrase_refined")
    gap = np.zeros(int(0.5 * SAMPLE_RATE), np.float32)
    for sid in sorted(catalog):
        pj, mid = f"{E}/vstpresets/Inverted_{sid}.json", f"{E}/midi_muscriptor/{sid}_muscriptor.mid"
        npj = f"{E}/refine/{sid}_refined_patch.json"
        if not all(os.path.exists(p) for p in (pj, mid, npj)):
            continue
        y, _ = sf.read(f"{E}/audio/{sid}_real.wav")
        phrase = (y.mean(axis=1) if y.ndim > 1 else y).astype(np.float32)
        phrase /= np.max(np.abs(phrase)) + 1e-7
        dur = len(phrase) / SAMPLE_RATE
        render = make_renderer(synth, midi_events(mid, dur), dur)
        base = patch_from_json(pj)
        t0 = time.time()
        pref, j1, j0, n_r, trace = refine_patch_phrase(render, base, phrase, fit_fraction=0.5)
        dt = time.time() - t0
        json.dump({"patch": pref}, open(f"{OUT}/{sid}_phrase_refined_patch.json", "w"), indent=1)
        h = len(phrase) // 2
        audio = {"baseline": render(base), "note_refined": render(patch_from_json(npj)), "phrase_refined": render(pref)}
        held = {v: scores(phrase[h:], a[h:]) for v, a in audio.items()}
        rows.append({"stem_id": sid, "fit_objective": [round(j0, 3), round(j1, 3)], "renders": n_r,
                     "seconds": round(dt, 1), "heldout_half": held,
                     "moves": [(k, [round(a, 3), round(b, 3)]) for k, (a, b), _ in trace[1:]]})
        sf.write(f"{OUT}/{sid}_heldout_real-base-note-phrase.wav",
                 np.concatenate([phrase[h:], gap] + [x for v in versions for x in (audio[v][h:], gap)]), SAMPLE_RATE)
        print(f"{sid:34s} {dt:5.1f}s | held-out MSS " + " / ".join(f"{held[v]['mss']:6.2f}" for v in versions)
              + " | env " + " / ".join(f"{held[v]['env_cos']:.3f}" for v in versions), flush=True)

    summ = {}
    for m in ("mss", "wmfcc", "env_cos"):
        vals = {v: np.array([r["heldout_half"][v][m] for r in rows]) for v in versions}
        better = (lambda a, b: a > b) if m == "env_cos" else (lambda a, b: a < b)
        summ[m] = {v: round(float(vals[v].mean()), 3) for v in versions}
        summ[m]["phrase_beats_baseline"] = f"{int(better(vals['phrase_refined'], vals['baseline']).sum())}/{len(rows)}"
        summ[m]["phrase_beats_note"] = f"{int(better(vals['phrase_refined'], vals['note_refined']).sum())}/{len(rows)}"
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": rows}, open(f"{OUT}/results.json", "w"), indent=1)
    json.dump({
        "purpose": "Phrase-mode renderer-in-the-loop refinement, fitted on the first half of each real "
                   "phrase and judged on the held-out second half, vs baseline and single-note refinement.",
        "hypothesis": "Single-note refinement fits envelopes to an artificially truncated 16th; fitting on real "
                      "phrase context generalises better to unseen notes.",
        "created": str(date.today()),
        "scripts": ["stable-audio-tools/scripts/synth_inversion/refine.py",
                    "stable-audio-tools/scripts/synth_inversion/eval_refine_phrase.py"],
        "listen": "refine_phrase/<id>_heldout_real-base-note-phrase.wav = held-out half: real | baseline | "
                  "note-refined | phrase-refined",
        "result": summ,
    }, open(f"{OUT}/run_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
