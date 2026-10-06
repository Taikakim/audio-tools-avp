"""Audio-side envelope meter for H4 envelope ladders: does each knob move the part of the note it should?

Per render and per note (gate times from h4_render_ladders_v2.note_times):
  tail_ms     time from gate-OFF until the level falls 30 dB below its level at gate-off (capped at the next
              note-on): the RELEASE segment, which only exists in the gap after the gate
  decay_db    peak level after note-on minus the level at gate-off: how far the note fell BEFORE the gate (the
              decay toward sustain)
Per ladder: Spearman of the knob value against each, median over the ladder's notes. The question
(a release can sound like a longer decay): a release knob should move tail_ms and leave decay_db alone, and a
decay knob the reverse. Level = 2 ms-hop RMS in dB.
Out: <dir>/envelope_meter_<name>.json + .npz (per-render tail_ms, decay_db).
Run: sat-venv/bin/python scripts/synth_inversion/h4_gate_envelope_meter.py --name ladders_v2s
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h4_render_ladders_v2 import note_times  # noqa: E402

DIR = "/run/media/kim/Mantu/surge_200k_models/h4_gate_v2"


def level_db(x, sr, hop_ms=2.0, win_ms=6.0):
    hop, win = int(sr * hop_ms / 1000), int(sr * win_ms / 1000)
    n = 1 + max(0, (len(x) - win) // hop)
    frames = np.lib.stride_tricks.as_strided(x, (n, win), (x.strides[0] * hop, x.strides[0]))
    return 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-7), hop_ms / 1000


def measure(x, sr, notes):
    L, dt = level_db(x.astype(np.float32), sr)
    tails, decays = [], []
    for j, (on, off) in enumerate(notes):
        nxt = notes[j + 1][0] if j + 1 < len(notes) else len(x) / sr
        i_on, i_off, i_nxt = int(on / dt), int(off / dt), min(int(nxt / dt), len(L))
        if i_off - i_on < 3 or i_nxt - i_off < 3:
            continue
        l_off = L[max(i_on, i_off - 3):i_off].mean()
        below = np.where(L[i_off:i_nxt] < l_off - 30)[0]
        tails.append((below[0] if len(below) else i_nxt - i_off) * dt * 1000)
        decays.append(L[i_on:i_off].max() - l_off)
    return float(np.median(tails)), float(np.median(decays))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--name", default="ladders_v2s")
    a = ap.parse_args()
    d = np.load(f"{a.dir}/{a.name}.npz")
    audio, sr = d["audio"], int(d["sample_rate"])
    knobs, rhythms = [str(x) for x in d["knobs"]], [str(x) for x in d["rhythms"]]
    kid, lid, kv, rh = d["knob_id"], d["ladder_id"], d["knob_value"], d["rhythm_id"]
    tail, dec = np.zeros(len(audio)), np.zeros(len(audio))
    for i in range(len(audio)):
        tail[i], dec[i] = measure(audio[i], sr, note_times(rhythms[rh[i]]))
    np.savez(f"{a.dir}/envelope_meter_{a.name}.npz", tail_ms=tail, decay_db=dec)
    res = {"file": a.name, "per_knob": {}}
    for k in np.unique(kid):
        rho_t, rho_d, rng_t, rng_d = [], [], [], []
        for l in np.unique(lid[kid == k]):
            m = lid == l
            o = np.argsort(kv[m])
            t, dd = tail[m][o], dec[m][o]
            rho_t.append(spearmanr(np.arange(len(t)), t).statistic if np.ptp(t) > 0 else 0.0)
            rho_d.append(spearmanr(np.arange(len(dd)), dd).statistic if np.ptp(dd) > 0 else 0.0)
            rng_t.append(t[-1] - t[0])
            rng_d.append(dd[-1] - dd[0])
        res["per_knob"][knobs[k]] = {
            "n_ladders": len(rho_t),
            "tail_ms": {"spearman_median": round(float(np.nanmedian(rho_t)), 3),
                        "change_first_to_last_median_ms": round(float(np.median(rng_t)), 1)},
            "decay_db": {"spearman_median": round(float(np.nanmedian(rho_d)), 3),
                         "change_first_to_last_median_db": round(float(np.median(rng_d)), 2)}}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{a.dir}/envelope_meter_{a.name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
