"""UNTESTED here (no Surge or surgepy on the laptop). Read-only discovery for the installed Surge build; run on the training
machine:    python enumerate_surge_v4.py --out surge_v4_enumeration.json

It answers, from the pinned build rather than from memory (research Q6; review B7, B8, B11):
  1. every modulation source name;
  2. every parameter whose name contains Sync, FM, Ring, Feedback or Filter Configuration;
  3. how many options "Filter Configuration" has and their display strings (v3 assumes 7, the registry 8, the brief 2);
  4. a routing smoke test: for each of the first matches of Sync and FM Depth, route "Filter EG" to it at depth 0 and 1 with
     `SurgeRenderer.set_mod_routing` and report whether the call succeeds and whether the rendered audio changes.

It only uses calls that `surge_synth.py` / `surge_v4_registry.py` already use (getControlGroup, getEntries, getParams,
getName, getParamMin/Max/Val/Display, setParamVal, getModSource, setModDepth01 through `set_mod_routing`, render_note).
If a call is missing in your surgepy build the failing step prints the exception and the script continues.
Archive the JSON with the Surge commit and surgepy version next to any dataset made with these routings.
"""
import argparse
import json
import sys

import numpy as np

KEYS = ("Sync", "FM", "Ring", "Feedback", "Filter Configuration")


def _safe(label, fn, errors):
    try:
        return fn()
    except Exception as e:                                           # noqa: BLE001 (discovery script: report, continue)
        errors.append(f"{label}: {type(e).__name__}: {e}")
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="surge_v4_enumeration.json")
    ap.add_argument("--note", type=int, default=36)
    a = ap.parse_args(argv)
    import surgepy
    from surge_synth import SurgeRenderer

    errors = []
    r = SurgeRenderer()
    res = {"surgepy_file": getattr(surgepy, "__file__", None), "mod_sources": sorted(r._mod_sources),
           "matching_params": {}, "filter_configuration": None, "routing_smoke": [], "errors": errors}
    names = sorted(r._param_map)
    for k in KEYS:
        res["matching_params"][k] = [n for n in names if k.lower() in n.lower()]

    fc = r._param_map.get("Filter Configuration")
    if fc is not None:
        lo, hi = _safe("getParamMin", lambda: r.synth.getParamMin(fc), errors), _safe("getParamMax", lambda: r.synth.getParamMax(fc), errors)
        shown = []
        if lo is not None and hi is not None:
            for v in range(int(round(lo)), int(round(hi)) + 1):
                _safe("setParamVal", lambda v=v: r.synth.setParamVal(fc, float(v)), errors)
                shown.append(_safe("getParamDisplay", lambda: str(r.synth.getParamDisplay(fc)), errors))
        res["filter_configuration"] = {"min": lo, "max": hi, "n_options": len(shown), "display": shown}
    else:
        errors.append("no parameter named exactly 'Filter Configuration' (see matching_params)")

    src = next((n for n in res["mod_sources"] if "filter eg" in n.lower()), None)
    for key in ("Sync", "FM Depth"):
        dest = next((n for n in names if key.lower() in n.lower()), None)
        entry = {"dest": dest, "source": src, "set_routing_ok": None, "audio_changes_with_depth": None}
        if dest and src:
            audio = {}
            for depth in (0.0, 1.0):
                entry["set_routing_ok"] = _safe("set_mod_routing", lambda d=depth: r.set_mod_routing(src, dest, d), errors)
                audio[depth] = _safe("render_note", lambda: r.render_note(midi_note=a.note, note_dur=0.4, total_dur=0.8), errors)
            if audio.get(0.0) is not None and audio.get(1.0) is not None:
                n = min(len(audio[0.0]), len(audio[1.0]))
                entry["audio_changes_with_depth"] = bool(np.max(np.abs(audio[0.0][:n] - audio[1.0][:n])) > 1e-4)
            _safe("reset routing", lambda: r.set_mod_routing(src, dest, 0.0), errors)
        res["routing_smoke"].append(entry)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({k: res[k] for k in ("filter_configuration", "routing_smoke", "errors")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
