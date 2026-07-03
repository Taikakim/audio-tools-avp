"""Eval the onset-density attribute head: does the OUTPUT's onset density follow the control?

The measurable proof the riffer never had. For a sweep of requested densities, generate
(fixed prompt+seed), decode, re-extract onset density with librosa, and check monotonicity.
Run on GPU (SA3 generate) — librosa onset detection is CPU.

    python sa3_control/onset_eval.py <onset_head.pt> [--gains 0,1,2,4] [--out DIR]
"""
import os, sys, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import librosa

from sa3_control.adapters import ControlContext, use_control_context
from sa3_control.audio_io import save_audio
from sa3_control.conditioner import ScalarAttributeEncoder
from sa3_control.inject import install_adapters
from sa3_control.generate import load_adapter_state

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--prompt", default="psytrance, 140 bpm")
ap.add_argument("--densities", default="2,4,6,8,10", help="raw onset_density values to request")
ap.add_argument("--gains", default="0,1,2,4")
ap.add_argument("--duration", type=float, default=20.0)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=7.0)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", default="/run/media/kim/Mantu/sa3_control_runs/onset_eval")
ap.add_argument("--notes", default="", help="human description of the run's logic/purpose; "
                "saved to run_meta.json so the eval GUI + inference UIs can show provenance")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
mean, std = ck.get("scalar_norm", [7.219, 1.424])
n_tokens = min(int(ck["args"].get("n_tokens", 16)), 16)
control_dim = int(ck["args"].get("control_dim", 768))
print(f"[eval] {os.path.basename(args.ckpt)}  field={ck.get('scalar_field')}  norm mean={mean:.3f} std={std:.3f}")

device = "cuda"
from stable_audio_3 import StableAudioModel
sam = StableAudioModel.from_pretrained("medium-base", device=device)
md = next(sam.model.model.parameters()).dtype
sr = sam.model.sample_rate
wrappers = install_adapters(sam, control_dim=control_dim)
enc = ScalarAttributeEncoder(control_dim=control_dim, n_tokens=n_tokens)
load_adapter_state(ck["state"], wrappers, enc)
for w in wrappers:
    w.adapter.to(device=device, dtype=md)
enc.to(device=device, dtype=md).eval()


def onset_density(audio_t):
    y = audio_t[0].float().cpu().numpy()
    if y.ndim > 1:
        y = y.mean(0)
    on = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    return len(on) / (len(y) / sr)


def gen(scalar_norm, gain):
    s = torch.tensor([scalar_norm], device=device, dtype=md)
    ctrl = enc(s)                                              # (1, n, d)
    cc = torch.cat([ctrl, torch.zeros_like(ctrl)], 0) if args.cfg != 1.0 else ctrl
    with use_control_context(ControlContext(cc, gain=gain)), torch.inference_mode():
        return sam.generate(prompt=args.prompt, duration=args.duration, steps=args.steps,
                            cfg_scale=args.cfg, seed=args.seed, sampler_type="euler")


densities = [float(x) for x in args.densities.split(",")]
gains = [float(x) for x in args.gains.split(",")]
print("\n  gain | requested_density -> MEASURED output onset density (onsets/sec)")
rows = []
for g in gains:
    line = []
    for draw in densities:
        dn = (draw - mean) / std
        ao = gen(dn, g)
        od = onset_density(ao[0])
        save_audio(f"{args.out}/onset_g{g:g}_d{draw:g}.wav", ao[0], sr)
        line.append(od); rows.append({"gain": g, "requested": draw, "measured": od})
    # correlation of measured vs requested at this gain (the key number)
    r = np.corrcoef(densities, line)[0, 1] if len(set(line)) > 1 else 0.0
    print(f"  {g:<4} | " + "  ".join(f"d{d:g}->{m:.1f}" for d, m in zip(densities, line)) + f"   | corr {r:+.2f}")

import json
json.dump(rows, open(f"{args.out}/onset_eval.json", "w"), indent=2)
print(f"\n[eval] wrote {args.out}/onset_eval.json")

# run_meta.json — provenance the eval GUI + inference UIs read alongside the clips (spec: MASTER §4)
_ta = ck.get("args", {}) or {}
run_meta = {
    "ckpt": os.path.abspath(args.ckpt),
    "ckpt_name": os.path.basename(args.ckpt),
    "scalar_field": ck.get("scalar_field"),
    "scalar_norm": {"mean": float(mean), "std": float(std)},
    "train": {k: _ta.get(k) for k in
              ("lr", "optimizer", "steps", "scalar_field", "crop_frames",
               "random_crop", "batch", "save_every", "warmup_steps", "encoded_dir")},
    "eval": {"prompt": args.prompt, "gains": args.gains, "densities": args.densities,
             "seed": args.seed, "cfg": args.cfg, "duration": args.duration, "steps": args.steps},
    "notes": args.notes,
}
json.dump(run_meta, open(f"{args.out}/run_meta.json", "w"), indent=2)
print(f"[eval] wrote {args.out}/run_meta.json")
print("SUCCESS criterion: at gain>0, MEASURED density rises with requested (corr → +1)")
