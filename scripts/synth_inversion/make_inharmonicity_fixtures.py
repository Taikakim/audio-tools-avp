"""Generate reference fixtures from the INSTALLED Essentia (never from the spec, never from memory).

Run from scripts/synth_inversion, on the machine that trains (it needs Essentia):
    python make_inharmonicity_fixtures.py

Writes inharmonicity_reference_fixtures.json next to this file. The JSON records the Essentia version.
Do not edit the output by hand; regenerate it. Prints a table: paste the table into your report.
"""
import json
import os
import platform

import numpy as np
import essentia
import essentia.standard as es

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inharmonicity_reference_fixtures.json")
TOLERANCE = 0.2
MAX_HARMONICS = 20
F0 = 100.0


def _f32(x):
    return np.asarray(x, dtype=np.float32)


def run_harmonic_peaks(freqs, mags, pitch):
    try:
        hp = es.HarmonicPeaks(maxHarmonics=MAX_HARMONICS, tolerance=TOLERANCE)
        hf, hm = hp(_f32(freqs), _f32(mags), float(pitch))
        return {"ok": True, "hf": [float(v) for v in hf], "hm": [float(v) for v in hm]}
    except Exception as e:  # record whatever the installed version does; this is a fixture generator
        return {"ok": False, "exc_type": type(e).__name__, "exc_msg": str(e)}


def run_inharmonicity(freqs, mags):
    try:
        v = es.Inharmonicity()(_f32(freqs), _f32(mags))
        return {"ok": True, "value": float(v)}
    except Exception as e:
        return {"ok": False, "exc_type": type(e).__name__, "exc_msg": str(e)}


def harmonics(n, f0=F0):
    return [f0 * k for k in range(1, n + 1)]


def build_cases():
    c = []
    c.append(("complete_series_4", harmonics(4), [1.0, 0.5, 0.3, 0.2], F0))
    c.append(("missing_h3", [100.0, 200.0, 400.0], [1.0, 0.5, 0.2], F0))
    c.append(("unequal_amps_5", harmonics(5), [1.0, 0.1, 0.8, 0.05, 0.4], F0))
    for off in (-0.04, -0.01, 0.01, 0.04):  # small offsets of partial 3, as a fraction of f0
        fr = harmonics(4)
        fr[2] += off * F0
        c.append((f"offset_p3_{off:+.2f}", fr, [1.0, 0.5, 0.3, 0.2], F0))
    for d in (0.199, 0.2, 0.201):  # partial 2 displaced to just inside / on / just outside the tolerance
        c.append((f"boundary_p2_d{d}", [100.0, 200.0 + d * F0, 300.0], [1.0, 0.5, 0.3], F0))
    c.append(("hand_displaced_305", [100.0, 200.0, 305.0], [1.0, 0.5, 0.5], F0))
    c.append(("dropout_325", [100.0, 200.0, 325.0], [1.0, 0.5, 0.5], F0))
    c.append(("dropout_worked_example", [100.0, 200.0, 250.0, 350.0], [1.0, 0.5, 0.5, 0.5], F0))
    c.append(("single_sinusoid", [100.0], [1.0], F0))
    c.append(("missing_fundamental", [200.0, 300.0, 400.0], [0.5, 0.3, 0.2], F0))
    c.append(("two_unrelated_pitches", [100.0, 137.0, 211.0, 289.0, 353.0], [1.0] * 5, F0))
    # structurally bad inputs: the reference may raise, return empty, or return a sentinel; record which
    c.append(("empty", [], [], F0))
    c.append(("size_mismatch", [100.0, 200.0], [1.0], F0))
    c.append(("unsorted", [200.0, 100.0], [0.5, 1.0], F0))
    c.append(("duplicate", [100.0, 100.0, 200.0], [1.0, 1.0, 0.5], F0))
    c.append(("zero_frequency_peak", [0.0, 100.0, 200.0], [0.1, 1.0, 0.5], F0))
    c.append(("all_zero_magnitudes", [100.0, 200.0, 300.0], [0.0, 0.0, 0.0], F0))
    c.append(("pitch_zero", [100.0, 200.0], [1.0, 0.5], 0.0))
    c.append(("pitch_negative", [100.0, 200.0], [1.0, 0.5], -100.0))
    c.append(("nan_frequency", [100.0, float("nan"), 300.0], [1.0, 0.5, 0.3], F0))
    c.append(("inf_magnitude", [100.0, 200.0, 300.0], [1.0, float("inf"), 0.3], F0))
    return c


def record(name, freqs, mags, pitch):
    hp = run_harmonic_peaks(freqs, mags, pitch)
    rec = {
        "name": name,
        "freqs": [float(v) for v in freqs],
        "mags": [float(v) for v in mags],
        "pitch": float(pitch),
        "harmonic_peaks": hp,
        "inharmonicity_on_input": run_inharmonicity(freqs, mags),
    }
    if hp["ok"]:
        rec["inharmonicity_on_selected"] = run_inharmonicity(hp["hf"], hp["hm"])
    return rec


def _fmt(r):
    if r is None:
        return "-"
    return f"{r['value']:.9g}" if r["ok"] else f"RAISES {r['exc_type']}: {r['exc_msg'][:60]}"


def main():
    data = {
        "essentia_version": essentia.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "tolerance": TOLERANCE,
        "max_harmonics": MAX_HARMONICS,
        "cases": [record(*c) for c in build_cases()],
    }
    with open(OUT, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {OUT} | essentia {essentia.__version__} | {len(data['cases'])} cases")
    print(f"{'case':28s} {'harmonic_peaks':14s} {'inharm(selected)':44s} inharm(raw input)")
    for r in data["cases"]:
        hp = "ok" if r["harmonic_peaks"]["ok"] else "RAISES " + r["harmonic_peaks"]["exc_type"]
        print(f"{r['name']:28s} {hp:14s} {_fmt(r.get('inharmonicity_on_selected')):44s} {_fmt(r['inharmonicity_on_input'])}")


if __name__ == "__main__":
    main()
