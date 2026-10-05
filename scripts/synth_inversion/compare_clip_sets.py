"""Score and A/B several invert_stem_collection.py outputs against the real phrases.

    compare_clip_sets.py OUT_DIR name1=dir1 name2=dir2 ...

Each dir holds audio/<id>_real.wav and audio/<id>_midi_playback.wav (same phrase slicing everywhere, so the
real clips are identical). Writes OUT_DIR/<id>_AB_real-<name1>-<name2>-....wav (0.5 s gaps),
OUT_DIR/comparison.json (paper MSS / wMFCC / RMS-envelope cosine per stem and mean) and run_meta.json.
"""
import glob
import json
import os
import sys
from datetime import date

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_metrics as rm  # noqa: E402


def mono(x):
    return (x.mean(axis=1) if x.ndim > 1 else x).astype(np.float32)


def main():
    out = sys.argv[1]
    sets = dict(a.split("=", 1) for a in sys.argv[2:])
    os.makedirs(out, exist_ok=True)
    names = list(sets)
    first = sets[names[0]]
    rows = []
    for real_path in sorted(glob.glob(f"{first}/audio/*_real.wav")):
        sid = os.path.basename(real_path)[:-len("_real.wav")]
        real, sr = sf.read(real_path)
        r = mono(real)
        clips, row = [], {"stem_id": sid}
        for n in names:
            p = f"{sets[n]}/audio/{sid}_midi_playback.wav"
            if not os.path.exists(p):
                continue
            x = mono(sf.read(p)[0])
            row[n] = {"mss": round(rm.mss(r, x), 3), "wmfcc": round(rm.wmfcc(r, x), 3),
                      "env_cos": round(rm.rms_env_cos(r, x), 4)}
            clips.append(x / (np.max(np.abs(x)) + 1e-7))
        rows.append(row)
        gap = np.zeros(int(0.5 * sr), np.float32)
        seq = [r / (np.max(np.abs(r)) + 1e-7), gap] + [c for x in clips for c in (x, gap)]
        sf.write(f"{out}/{sid}_AB_real-{'-'.join(names)}.wav", np.concatenate(seq), sr)
    summ = {n: {m: round(float(np.mean([row[n][m] for row in rows if n in row])), 3)
                for m in ("mss", "wmfcc", "env_cos")} for n in names}
    for n in names:
        summ[n]["best_mss_stems"] = sum(1 for row in rows if all(row.get(n, {}).get("mss", 1e9) <= row[o]["mss"]
                                                                 for o in names if o in row))
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "rows": rows}, open(f"{out}/comparison.json", "w"), indent=1)
    json.dump({"purpose": "A/B of inverted patches for the 24 real bass phrases, each model through the identical "
                          "pipeline (FX-fixed renderer, flow rerank + phrase refinement, MuScriptor playback)",
               "created": str(date.today()), "sets": sets, "result": summ,
               "listen": f"<id>_AB_real-{'-'.join(names)}.wav", "kim_feedback": None},
              open(f"{out}/run_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
