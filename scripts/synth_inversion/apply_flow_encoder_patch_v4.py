# Owner: KUANG (created 2026-10-11)
"""Adds --flow_encoder / --flow_film to train_realistic_bass_v3.py so a v4 flow can be trained (Stage 2). Stdlib only.

    python apply_flow_encoder_patch_v4.py --check     # report what would change, write nothing
    python apply_flow_encoder_patch_v4.py             # apply (edits train_realistic_bass_v3.py in this folder)

Why a script and not a .patch: the files in this folder use CRLF (rule R13). The script detects the file's line ending,
requires each anchor line to occur exactly once, writes the new lines with the SAME ending, and refuses (changing nothing)
if an anchor is missing, duplicated, or already patched. Defaults are unchanged: with `--flow_encoder conv` and no
`--flow_film` the trainer builds exactly the v3 flow.

Edits (two hunks):
  1. build_parser: two new arguments after --flow_dim;
  2. train(): the flow is built by models_v4.build_model_v4 when either new argument is used.

NOT run against the trainer here (no pedalboard, essentia, Surge or GPU on the laptop): only the text edit and a Python
syntax check were tested. After applying, smoke-test with a few steps, a validation, a checkpoint and a resume (ERRORS.md
Part D step 3) with `--flow_encoder temporal_asp`. A v3 flow checkpoint cannot be resumed into a v4 flow and vice versa:
use a new --out_dir.
"""
import argparse
import ast
import os
import sys

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_realistic_bass_v3.py")

ANCHOR_ARG = '    ap.add_argument("--flow_dim", type=int, default=512, help="Flow hidden dim")'
NEW_ARGS = [
    '    ap.add_argument("--flow_encoder", default="conv", choices=("conv", "temporal_asp", "temporal_segments", "temporal_queries"),',
    '                    help="flow audio encoder: conv = the v3 pooled encoder (default, unchanged); temporal_* = models_v4 (keeps the time axis)")',
    '    ap.add_argument("--flow_film", action="store_true", help="per-block FiLM from the audio embedding (models_v4)")',
]
ANCHOR_BUILD = '    flow = build_model(**FLOW_KWARGS, cond_noise=args.cond_noise).to(device)'
NEW_BUILD = [
    '    if args.flow_encoder == "conv" and not args.flow_film:',
    '        flow = build_model(**FLOW_KWARGS, cond_noise=args.cond_noise).to(device)',
    '    else:',
    '        from models_v4 import build_model_v4  # a v4 flow has different encoder keys: use a new --out_dir',
    '        flow = build_model_v4(**FLOW_KWARGS, cond_noise=args.cond_noise, encoder=args.flow_encoder, film=args.flow_film).to(device)',
]


def patch_text(text: str):
    """Returns (new_text, line_ending). Raises ValueError if the file cannot be patched safely."""
    nl = "\r\n" if text.count("\r\n") > text.count("\n") / 2 else "\n"
    lines = text.split(nl)
    if any("--flow_encoder" in ln for ln in lines):
        raise ValueError("already patched (found --flow_encoder)")
    out, n_arg, n_build = [], 0, 0
    for ln in lines:
        if ln == ANCHOR_ARG:
            out.append(ln)
            out.extend(NEW_ARGS)
            n_arg += 1
        elif ln == ANCHOR_BUILD:
            out.extend(NEW_BUILD)
            n_build += 1
        else:
            out.append(ln)
    if n_arg != 1 or n_build != 1:
        raise ValueError(f"anchors must each occur exactly once (--flow_dim line: {n_arg}, flow build line: {n_build}); nothing changed")
    return nl.join(out), nl


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report only; write nothing")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args(argv)
    with open(a.file, "rb") as f:
        raw = f.read()
    try:
        new, nl = patch_text(raw.decode("utf-8"))
        ast.parse(new)
    except (ValueError, SyntaxError) as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"ok: 2 hunks, line ending {'CRLF' if nl == chr(13) + chr(10) else 'LF'}, result parses")
    if a.check:
        return 0
    with open(a.file, "wb") as f:
        f.write(new.encode("utf-8"))
    print(f"patched {a.file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
