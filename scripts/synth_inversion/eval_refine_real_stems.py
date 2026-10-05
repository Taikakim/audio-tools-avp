"""Does rendered refinement (refine.py) improve the inverted patches of the 24 real bass stems?

For each stem in real_stems_eval (built by invert_stem_collection.py):
  * target = the same single 16th note the inversion was run on (first onset of the phrase,
    LPF 3.5 kHz for Demucs stems, faded, padded to 0.8 s) -- reproduced from <id>_real.wav;
  * baseline = the saved inverted patch (vstpresets/Inverted_<id>.json);
  * refined  = refine.refine_patch(baseline, target).
Scored twice:
  NOTE   on the note it was tuned on (in-sample: it should improve by construction);
  PHRASE on the whole real phrase, played from the MuScriptor MIDI through each patch -- the
         honest check, because the other notes were never seen by the refinement.
Metrics: paper MSS and wMFCC (reference_metrics.py) and RMS-envelope cosine.

Outputs to real_stems_eval/refine/: per-stem A/B wavs (real | baseline | refined, note and
phrase), refined patch JSONs, results.json, run_meta.json.

Run (SAT venv): sat-venv/bin/python scripts/synth_inversion/eval_refine_real_stems.py
"""
import json
import os
import sys
import time
from datetime import date

import librosa
import mido
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference_metrics as rm  # noqa: E402
from refine import patch_from_json, refine_patch  # noqa: E402
from surge_spec import AUDIO_LEN, DEFAULT_PLUGIN_PATH, SAMPLE_RATE, apply_patch, init_synth  # noqa: E402

E = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval"
OUT = f"{E}/refine"


def target_note(phrase_mono, bpm, is_demucs):
    y = phrase_mono
    if is_demucs:
        y = sosfilt(butter(4, 3500.0, btype="lowpass", fs=SAMPLE_RATE, output="sos"), y)
    sixteenth = (60.0 / bpm) / 4.0
    on = librosa.onset.onset_detect(y=y.astype(np.float32), sr=SAMPLE_RATE, units="samples")
    start = on[0] if len(on) else 0
    chunk = y[start:start + int(SAMPLE_RATE * sixteenth)].astype(np.float32).copy()
    fade = int(SAMPLE_RATE * 0.005)
    if len(chunk) > fade:
        chunk[-fade:] *= np.linspace(1.0, 0.0, fade)
    out = np.zeros(AUDIO_LEN, np.float32)
    out[:len(chunk)] = chunk
    return out / (np.max(np.abs(out)) + 1e-7), sixteenth * 0.85


def render_phrase(synth, patch, midi_path, dur):
    t, ev = 0.0, []
    for msg in mido.MidiFile(midi_path):
        t += msg.time
        if t > dur:
            break
        if msg.type in ("note_on", "note_off"):
            ev.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity, time=t))
    apply_patch(synth, patch)
    synth.reset()
    a = synth.process(sorted(ev, key=lambda m: m.time), duration=dur, sample_rate=SAMPLE_RATE, num_channels=2)
    m = a.mean(axis=0).astype(np.float32)
    return m / (np.max(np.abs(m)) + 1e-7)


def scores(ref, x):
    return {"mss": round(rm.mss(ref, x), 3), "wmfcc": round(rm.wmfcc(ref, x), 3),
            "env_cos": round(rm.rms_env_cos(ref, x), 4)}


def main():
    os.makedirs(OUT, exist_ok=True)
    synth = init_synth(DEFAULT_PLUGIN_PATH)
    catalog = {d["id"]: d for d in json.load(open(f"{E}/real_stems_summary.json"))}
    rows = []
    gap = np.zeros(int(0.5 * SAMPLE_RATE), np.float32)
    for sid, d in sorted(catalog.items()):
        pj = f"{E}/vstpresets/Inverted_{sid}.json"
        mid = f"{E}/midi_muscriptor/{sid}_muscriptor.mid"
        if not (os.path.exists(pj) and os.path.exists(mid)):
            continue
        y, sr = sf.read(f"{E}/audio/{sid}_real.wav")
        phrase = (y.mean(axis=1) if y.ndim > 1 else y).astype(np.float32)
        phrase /= np.max(np.abs(phrase)) + 1e-7
        tgt, note_dur = target_note(phrase, d["bpm"], "Demucs" in d.get("style", ""))
        base = patch_from_json(pj)
        t0 = time.time()
        ref_patch, j_end, j_start, n_r, trace = refine_patch(synth, base, tgt, d["midi_note"], note_dur)
        dt = time.time() - t0

        from surge_spec import render_patch
        nb = render_patch(synth, base, d["midi_note"], note_dur, duration=0.8)
        nr = render_patch(synth, ref_patch, d["midi_note"], note_dur, duration=0.8)
        dur = len(phrase) / SAMPLE_RATE
        pb = render_phrase(synth, base, mid, dur)
        pr = render_phrase(synth, ref_patch, mid, dur)
        row = {"stem_id": sid, "objective": [round(j_start, 3), round(j_end, 3)], "renders": n_r,
               "seconds": round(dt, 1),
               "moves": [(k, [round(a, 3), round(b, 3)]) for k, ab, _ in trace[1:] for a, b in [ab]],
               "note": {"baseline": scores(tgt, nb), "refined": scores(tgt, nr)},
               "phrase": {"baseline": scores(phrase, pb), "refined": scores(phrase, pr)}}
        rows.append(row)
        json.dump({"patch": ref_patch}, open(f"{OUT}/{sid}_refined_patch.json", "w"), indent=1)
        sf.write(f"{OUT}/{sid}_note_real-base-refined.wav", np.concatenate([tgt, gap, nb, gap, nr]), SAMPLE_RATE)
        sf.write(f"{OUT}/{sid}_phrase_real-base-refined.wav", np.concatenate([phrase, gap, pb, gap, pr]), SAMPLE_RATE)
        nb_, nr_ = row["phrase"]["baseline"], row["phrase"]["refined"]
        print(f"{sid:34s} J {j_start:6.2f}->{j_end:6.2f} ({n_r} renders, {dt:4.1f}s) | phrase MSS "
              f"{nb_['mss']:6.2f}->{nr_['mss']:6.2f}  wMFCC {nb_['wmfcc']:6.2f}->{nr_['wmfcc']:6.2f}  "
              f"env {nb_['env_cos']:.3f}->{nr_['env_cos']:.3f}", flush=True)

    summ = {}
    for lvl in ("note", "phrase"):
        for m in ("mss", "wmfcc", "env_cos"):
            b = np.array([r[lvl]["baseline"][m] for r in rows])
            a = np.array([r[lvl]["refined"][m] for r in rows])
            better = (a > b) if m == "env_cos" else (a < b)
            summ[f"{lvl}_{m}"] = {"baseline": round(float(b.mean()), 3), "refined": round(float(a.mean()), 3),
                                  "improved_stems": f"{int(better.sum())}/{len(rows)}"}
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": rows}, open(f"{OUT}/results.json", "w"), indent=1)
    json.dump({
        "purpose": "Test renderer-in-the-loop refinement of salient axes (cutoff, filter/amp envelopes, "
                   "resonance) on the 24 real-stem inversions; judged on the WHOLE phrase, which the "
                   "refinement never saw (it tunes on one note).",
        "hypothesis": "Flow proposals are close but miss on axes that trade off on a single note; bracketing "
                      "those axes with real renders against a human-timbre-aligned objective moves the sound closer.",
        "created": str(date.today()),
        "scripts": ["stable-audio-tools/scripts/synth_inversion/refine.py",
                    "stable-audio-tools/scripts/synth_inversion/eval_refine_real_stems.py"],
        "inputs": {"baseline_patches": f"{E}/vstpresets/Inverted_<id>.json",
                   "phrase_midi": f"{E}/midi_muscriptor/<id>_muscriptor.mid",
                   "baseline_model": "/run/media/kim/Mantu/surge_200k_models/modular_shampoo_sf_b64/flow_latest.pt "
                                     "(via invert_stem_collection.py)"},
        "objective": "reference_metrics.mss + 10 * (1 - rms_env_cos); chosen by timbre_alignment_check.py",
        "listen": "refine/<id>_{note,phrase}_real-base-refined.wav = real | baseline | refined",
        "result": summ,
    }, open(f"{OUT}/run_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
