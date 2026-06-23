"""Multi-prompt / multi-seed onset-density eval grid for ONE checkpoint, model
loaded ONCE (vs onset_eval.py which is single-prompt/seed). Saves wavs +
manifest.json (file -> prompt, seed, gain, density, measured_density). GPU.

    python multi_eval.py <ckpt.pt> --prompts "a||b||c" --seeds 1234,777,42 \
        --gains 1,2,3 --densities 6.5,7,7.5,8,9,10 --out <dir>
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import torch

from sa3_control.adapters import ControlContext, use_control_context
from sa3_control.audio_io import save_audio
from sa3_control.conditioner import ScalarAttributeEncoder
from sa3_control.inject import install_adapters
from sa3_control.generate import load_adapter_state

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--prompts", required=True, help="'||'-separated prompts")
ap.add_argument("--seeds", default="1234")
ap.add_argument("--gains", default="1,2,3")
ap.add_argument("--densities", default="6.5,7,7.5,8,9,10")
ap.add_argument("--duration", type=float, default=20.0)
ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--cfg", type=float, default=7.0)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
mean, std = ck.get("scalar_norm", [7.219, 1.424])
n_tokens = min(int(ck["args"].get("n_tokens", 16)), 16)
control_dim = int(ck["args"].get("control_dim", 768))
print(f"[multi] {os.path.basename(args.ckpt)} norm mean={mean:.3f} std={std:.3f}", flush=True)

device = "cuda"
from stable_audio_3 import StableAudioModel
# Turn OFF the FlexAttention torch.compile for eval gens: the per-process max-autotune
# compile cost isn't amortized over a handful of clips, and with CK on ROCm 7.13 it barely
# speeds inference. Falls back to the standard math-equivalent masked-SDPA path. (Training
# imports the module separately and is unaffected.)
from stable_audio_3.models import transformer as _sa3_tf
_sa3_tf.flex_attention_available = False
_sa3_tf.flex_attention_compiled = None
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
    return len(librosa.onset.onset_detect(y=y, sr=sr, units="time")) / (len(y) / sr)


def gen(prompt, seed, scalar_norm, gain):
    s = torch.tensor([scalar_norm], device=device, dtype=md)
    ctrl = enc(s)
    cc = torch.cat([ctrl, torch.zeros_like(ctrl)], 0) if args.cfg != 1.0 else ctrl
    with use_control_context(ControlContext(cc, gain=gain)), torch.inference_mode():
        return sam.generate(prompt=prompt, duration=args.duration, steps=args.steps,
                            cfg_scale=args.cfg, seed=seed, sampler_type="euler")


prompts = args.prompts.split("||")
seeds = [int(s) for s in args.seeds.split(",")]
gains = [float(g) for g in args.gains.split(",")]
densities = [float(d) for d in args.densities.split(",")]
manifest = []
total = len(prompts) * len(seeds) * len(gains) * len(densities)
i = 0
for pi, prompt in enumerate(prompts):
    for seed in seeds:
        for g in gains:
            for draw in densities:
                dn = (draw - mean) / std
                ao = gen(prompt, seed, dn, g)
                od = onset_density(ao[0])
                fn = f"p{pi}_s{seed}_g{g:g}_d{draw:g}.wav"
                save_audio(f"{args.out}/{fn}", ao[0], sr)
                manifest.append({"file": fn, "prompt": prompt, "pi": pi, "seed": seed,
                                 "gain": g, "density": draw, "measured": round(od, 3)})
                i += 1
                if i % 18 == 0 or i == total:
                    print(f"  [{i}/{total}] {fn} meas={od:.2f}", flush=True)
json.dump(manifest, open(f"{args.out}/manifest.json", "w"), indent=1)
print(f"[multi] wrote {len(manifest)} -> {args.out}/manifest.json", flush=True)
