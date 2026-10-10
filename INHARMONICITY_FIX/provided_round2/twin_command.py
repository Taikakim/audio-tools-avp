#!/usr/bin/env python3
"""Print the TWIN command of a live synth-inversion run, built from the run's own run_meta.json, and show where that
run differs from the reference recipe (ROUND2_HANDOVER.md section 3).

    python twin_command.py <live out_dir or run_meta.json> --out-dir <twin out_dir> [--python PY] [--script PATH]

Decision D7: a twin of the live run. It keeps EVERY flag of the live run, so nothing is retyped from memory, except:
--out_dir (new), --inharm_weight (forced to 0), --inharm_config (dropped), --resume_from (dropped: the twin starts
fresh) and --purpose (rewritten). max_steps, hours and the LR settings are kept, so the schedule is the same.

Caveats: run_meta.json is written at the run's FRESH launch only; if the run was later resumed with other flags,
confirm against `ps -o args= -p <pid>`. The trainer has no --seed flag, so a twin is statistically matched, not
bit-identical. Standard library only.
"""
import argparse
import json
import math
import os
import shlex
import sys

# Reference recipe values: run_v3.sh sizes and LRs + the run_v3_clean_home.sh recipe (ROUND2_HANDOVER.md section 3).
RECIPE = {
    "batch_size": 64, "embed_dim": 768, "flow_dim": 768, "ff_dim": 2048,
    "optimizer": "modular", "whitening": "shampoo", "ns_poly": "cubic5", "radial_brake": 0.8,
    "var_dampening": 1.15, "sf_c_warmup": 1000, "sf_r": 1.0, "wd_jepa": 0.02, "wd_overtraining": True,
    "lr_jepa": 3.0e-4, "lr_flow": 2.0e-4, "lr_floor": 0.333, "decay_frac": 0.3,
    "random_patch_frac": 0.05, "num_workers": 8, "prefetch_factor": 4,
    "ladder_every": 4, "ladder_anchors": 4, "ladder_steps": 16, "ladder_workers": 3, "lambda_ord": 0.5,
    "ladder_val_anchors": 8, "ot_coupling": True, "ema_halflife_steps": 2000.0,
    "cond_noise": True, "cond_noise_clean_frac": 0.3, "flow_loss": "mae",
    "snr_gate": False,  # the recipe wants it OFF (run_modular_shampoo_b64.sh header: "avoiding artificial LR braking")
}
REPLACED = ("out_dir", "resume_from", "inharm_config", "inharm_weight", "purpose")


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def fmt(v):
    return repr(v) if isinstance(v, float) else str(v)


def load_meta(path):
    p = os.path.join(path, "run_meta.json") if os.path.isdir(path) else path
    with open(p) as f:
        meta = json.load(f)
    if "args" not in meta or not isinstance(meta["args"], dict):
        raise SystemExit(f"{p}: no 'args' dict: not a run_meta.json written by the v3 trainer")
    return meta


def recipe_table(args):
    rows = []
    for k, rv in RECIPE.items():
        lv = args.get(k, "<missing>")
        mark = "NOT IN META" if k not in args else ("" if same(lv, rv) else "DIFFERS")
        rows.append((k, lv, rv, mark))
    return rows


def twin_tokens(args, out_dir, purpose):
    toks = []
    for key, val in args.items():
        if key in REPLACED or val is None or val == "":
            continue
        if isinstance(val, bool):
            if val:
                toks.append("--" + key)
            continue
        toks += ["--" + key, fmt(val)]
    toks += ["--out_dir", out_dir, "--inharm_weight", "0", "--purpose", purpose]
    return toks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the live run's out_dir, or its run_meta.json")
    ap.add_argument("--out-dir", required=True, help="out_dir for the twin (must differ from the live run's)")
    ap.add_argument("--python", default="python3", help="interpreter to print in the command (default python3)")
    ap.add_argument("--script", default=None, help="trainer path to print (default: the one recorded in run_meta)")
    a = ap.parse_args(argv)

    meta = load_meta(a.run)
    args = meta["args"]
    live_out = args.get("out_dir")
    if live_out and os.path.normpath(a.out_dir) == os.path.normpath(live_out):
        raise SystemExit("--out-dir must differ from the live run's out_dir")
    script = a.script or meta.get("script") or "train_realistic_bass_v3.py"
    run_name = meta.get("run") or "live"
    purpose = f"twin of {run_name}: identical flags except --inharm_weight 0 (aux head off). Question: what the aux head adds."

    print(f"# live run: {run_name}  created {meta.get('created')}  status {meta.get('status')}")
    print(f"# live aux: inharm_weight={args.get('inharm_weight')}  inharm_config={args.get('inharm_config')}")
    print("# reference-recipe flags: live value vs recipe (only DIFFERS rows matter)")
    print(f"# {'flag':24s} {'live':14s} {'recipe':10s}")
    n_diff = 0
    for k, lv, rv, mark in recipe_table(args):
        n_diff += (mark == "DIFFERS")
        print(f"# {k:24s} {fmt(lv):14s} {fmt(rv):10s} {mark}")
    print(f"# {n_diff} of {len(RECIPE)} recipe flags differ from the live run")
    if "seed" not in args:
        print("# note: the trainer has no --seed flag: the twin is statistically matched, not bit-identical")
    print("# note: run_meta.json is written at the FRESH launch; confirm against `ps -o args=` if the run was resumed")
    print()
    toks = twin_tokens(args, a.out_dir, purpose)
    print(shlex.quote(a.python) + " " + shlex.quote(script) + " \\")
    i = 0
    lines = []
    while i < len(toks):
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            lines.append(shlex.quote(toks[i]) + " " + shlex.quote(toks[i + 1]))
            i += 2
        else:
            lines.append(shlex.quote(toks[i]))
            i += 1
    print("  " + " \\\n  ".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
