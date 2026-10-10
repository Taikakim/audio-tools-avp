"""Measure what render_patch ACTUALLY plays. Needs Surge XT; run from scripts/synth_inversion:
    python measure_phrase_timeline.py [--manifold PATH] [--plugin PATH] [--n 8] [--seed 0]
    python measure_phrase_timeline.py --hash-out render_baseline_hashes.json     (step S0, BEFORE any change)
    python measure_phrase_timeline.py --hash-check render_baseline_hashes.json   (step S6/S8, AFTER the changes)

Prints, per sampled patch, the mean level (dB re the clip's loudest 10 ms) in six time windows, then the median
over patches. It writes nothing except --hash-out. It does NOT decide anything: paste the table into the report.

Why this exists: render_patch passes MIDI event times 0.0, 0.125, 0.125, 0.250 and its comments call the last two
"absolute 0.250 / 0.500". Pedalboard reads a message's `time` as ABSOLUTE seconds from the start of the buffer
(sampleIndex = time * sampleRate, no accumulation), which would put note 2 at 0.125-0.250 s. The comments assume a
delta reading (note 2 at 0.250-0.500 s). The envelope below is the empirical tie-breaker.
"""
import argparse
import hashlib
import json
import sys

import numpy as np

from realistic_bass_prior_v3 import MANIFOLD_PATH, RealisticBassPrior
from surge_spec_v3 import DEFAULT_PLUGIN_PATH, SAMPLE_RATE, init_synth, render_patch

HOP_S = 0.010
WINDOWS = [(0.100, 0.125), (0.125, 0.150), (0.200, 0.250), (0.250, 0.300), (0.450, 0.500), (0.500, 0.550)]


def digest(audio):
    return hashlib.sha256(np.ascontiguousarray(audio, dtype=np.float32).tobytes()).hexdigest()


def envelope_db(audio, sr, hop_s=HOP_S):
    hop = int(round(sr * hop_s))
    n = len(audio) // hop
    rms = np.sqrt(np.mean(np.asarray(audio[: n * hop], dtype=np.float64).reshape(n, hop) ** 2, axis=1) + 1e-20)
    return 20.0 * np.log10(rms / (rms.max() + 1e-20) + 1e-12)


def mean_db(env, lo, hi, hop_s=HOP_S):
    return float(np.mean(env[int(round(lo / hop_s)): int(round(hi / hop_s))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifold", default=MANIFOLD_PATH)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hash-out")
    ap.add_argument("--hash-check")
    args = ap.parse_args()

    synth = init_synth(args.plugin, verify=False)
    prior = RealisticBassPrior(args.manifold, split="train")
    rng = np.random.RandomState(args.seed)

    rows, hashes, deterministic = [], [], True
    for k in range(args.n):
        patch, vec, midi_note, note_dur = prior.sample_patch_and_midi(rng)
        a1 = render_patch(synth, patch, midi_note, note_dur)
        a2 = render_patch(synth, patch, midi_note, note_dur)
        h1, h2 = digest(a1), digest(a2)
        deterministic = deterministic and (h1 == h2)
        hashes.append(h1)
        env = envelope_db(a1, SAMPLE_RATE)
        rows.append((k, int(midi_note), [mean_db(env, lo, hi) for lo, hi in WINDOWS]))

    head = " ".join(f"{lo:.3f}-{hi:.3f}" for lo, hi in WINDOWS)
    print(f"mean level in dB re the loudest 10 ms of each clip; windows (s): {head}")
    for k, midi, vals in rows:
        print(f"patch {k:2d} midi {midi:3d} | " + "  ".join(f"{v:7.1f}" for v in vals))
    med = np.median(np.array([r[2] for r in rows]), axis=0)
    print("median            | " + "  ".join(f"{v:7.1f}" for v in med))
    print(f"render deterministic across two renders of the same patch: {deterministic}")
    print()
    print("Reading guide (REPORT THE TABLE; do not conclude, Kim decides):")
    print(" - code as written + Pedalboard source: note 1 on 0.000/off 0.125, note 2 (+12) on 0.125/off 0.250;")
    print("   expect level to FALL after about 0.250 s with no new onset.")
    print(" - the comment in render_patch: note 2 on 0.250/off 0.500; expect a NEW onset near 0.250 s.")

    if args.hash_out:
        with open(args.hash_out, "w") as f:
            json.dump({"seed": args.seed, "n": args.n, "deterministic": deterministic, "hashes": hashes}, f, indent=1)
            f.write("\n")
        print(f"wrote {args.hash_out}")
    if args.hash_check:
        with open(args.hash_check) as f:
            base = json.load(f)
        if (base["seed"], base["n"]) != (args.seed, args.n):
            print("hash-check needs the same --seed and --n as the baseline")
            sys.exit(2)
        if not base["deterministic"]:
            print("baseline render was NOT deterministic: bit-identity cannot be checked; rely on the events test")
            sys.exit(0)
        bad = [i for i, (a, b) in enumerate(zip(base["hashes"], hashes)) if a != b]
        print("render hashes identical to baseline" if not bad else f"RENDER CHANGED for patches {bad}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
