"""Do the bracket_refiner.py ideas improve refine.py? Held-out half-phrase protocol, 24 real stems.

Arms, all starting from the SAME reranked flow candidate (old model, 20 candidates):
  baseline      no refinement
  fixed         refine.py as shipped: fixed per-axis half-widths around the start value
  spread        first-pass bounds from the spread of the top-20 candidates
  spread_probe  spread + one probe 10% of the range beyond each end
Each refinement is fitted on the FIRST half of the real phrase (played from the MuScriptor MIDI) and
scored on the SECOND half (paper MSS / wMFCC / envelope cosine). Out: real_stems_eval/refine_bounds/.
Run (SAT venv, CPU): sat-venv/bin/python scripts/synth_inversion/eval_refine_bounds.py [--procs 3]
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
E = "/run/media/kim/Mantu/surge_200k_models/real_stems_eval"
OUT = f"{E}/refine_bounds"
FLOW = "/run/media/kim/Mantu/surge_200k_models/modular_shampoo_sf_b64/flow_latest.pt"
ARMS = ("baseline", "fixed", "spread", "spread_probe")


def run_stem(args):
    idx, sid, d = args
    import soundfile as sf
    import torch
    torch.set_num_threads(2)
    from audio_utils import make_mel_spec
    from eval_refine_phrase import make_renderer, midi_events, scores
    from eval_refine_real_stems import target_note
    from inference import predict_and_rerank_candidates
    from models import load_inverter
    from refine import Refiner, bounds_from_candidates
    from surge_spec import DEFAULT_PLUGIN_PATH, SAMPLE_RATE, init_synth

    synth = init_synth(DEFAULT_PLUGIN_PATH, verify=False)
    model = load_inverter(FLOW, device="cpu")
    y, _ = sf.read(f"{E}/audio/{sid}_real.wav")
    phrase = (y.mean(axis=1) if y.ndim > 1 else y).astype(np.float32)
    phrase /= np.max(np.abs(phrase)) + 1e-7
    tgt, note_dur = target_note(phrase, d["bpm"], "Demucs" in d.get("style", ""))
    mel = torch.from_numpy(make_mel_spec(tgt)).unsqueeze(0)
    best, _, cands = predict_and_rerank_candidates(model, synth, tgt, mel, d["midi_note"], note_dur=note_dur,
                                                   n_candidates=20, steps=25, seed=42 + idx)
    start = best["patch"]
    dur = len(phrase) / SAMPLE_RATE
    render = make_renderer(synth, midi_events(f"{E}/midi_muscriptor/{sid}_muscriptor.mid", dur), dur)
    h = len(phrase) // 2
    bounds = bounds_from_candidates([c["patch"] for c in cands])
    cfg = {"fixed": {}, "spread": {"cand_bounds": bounds}, "spread_probe": {"cand_bounds": bounds, "probe_out": 0.1}}
    res, renders, secs = {}, {}, {}
    res["baseline"] = scores(phrase[h:], render(start)[h:])
    for arm, kw in cfg.items():
        t0 = time.time()
        r = Refiner(render, phrase, score_slice=slice(0, h))
        patch, _ = r.refine(start, **kw)
        res[arm] = scores(phrase[h:], render(patch)[h:])
        renders[arm], secs[arm] = r.n_renders, round(time.time() - t0, 1)
    print(f"{sid:34s} held-out MSS " + " / ".join(f"{res[a]['mss']:6.2f}" for a in ARMS), flush=True)
    return {"stem_id": sid, "heldout_half": res, "renders": renders, "seconds": secs,
            "bounds_width": {k: round(hi - lo, 3) for k, (lo, hi) in bounds.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    catalog = json.load(open(f"{E}/real_stems_summary.json"))
    jobs = [(int(d["index"]) - 1, d["id"], d) for d in catalog
            if os.path.exists(f"{E}/midi_muscriptor/{d['id']}_muscriptor.mid")]
    with get_context("spawn").Pool(a.procs) as pool:
        rows = sorted(pool.map(run_stem, jobs), key=lambda r: r["stem_id"])
    summ = {}
    for m in ("mss", "wmfcc", "env_cos"):
        vals = {arm: np.array([r["heldout_half"][arm][m] for r in rows]) for arm in ARMS}
        better = (lambda x, y: x > y) if m == "env_cos" else (lambda x, y: x < y)
        summ[m] = {arm: round(float(vals[arm].mean()), 3) for arm in ARMS}
        for arm in ("spread", "spread_probe"):
            summ[m][f"{arm}_beats_fixed"] = f"{int(better(vals[arm], vals['fixed']).sum())}/{len(rows)}"
    summ["mean_renders"] = {arm: float(np.mean([r["renders"][arm] for r in rows])) for arm in ARMS[1:]}
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": rows}, open(f"{OUT}/results.json", "w"), indent=1)
    json.dump({"purpose": "Test two ideas from an untracked bracket_refiner.py (candidate-spread bounds, 10% outward "
                          "probe) as refine.py options, under the held-out half-phrase protocol on 24 real stems.",
               "hypothesis": "Bounds set by where the model's own candidates disagree search the right range per axis; "
                             "an outward probe catches a shared bias.",
               "created": str(date.today()), "script": "stable-audio-tools/scripts/synth_inversion/eval_refine_bounds.py",
               "model": "modular_shampoo_sf_b64/flow_latest.pt (pre-FX-fix run)", "result": summ},
              open(f"{OUT}/run_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
