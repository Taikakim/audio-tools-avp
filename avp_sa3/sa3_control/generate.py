"""Generate a riff: medium-base + a trained audio-reference adapter, conditioned on a
reference track. The reference's SAME latent is encoded to control tokens that steer
generation toward its style/content while the text prompt + duration shape the rest.

Run with the SA3 .venv:
    PYTORCH_TUNABLEOP_ENABLED=0 .../python -m sa3_control.generate \
        --adapter <run>/riffer_final.pt --reference track.flac \
        --prompt "psychedelic goa trance" --duration 30 --out riff.wav
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchaudio

from sa3_control.adapters import ControlContext, use_control_context
from sa3_control.audio_io import save_audio
from sa3_control.conditioner import AudioRefEncoder
from sa3_control.inject import install_adapters


def load_adapter_state(state, wrappers, cond_enc):
    for i, w in enumerate(wrappers):
        pfx = f"adapter.{i}."
        sub = {k[len(pfx):]: v for k, v in state.items() if k.startswith(pfx)}
        if sub:
            w.adapter.load_state_dict(sub)
    csub = {k[len("conditioner."):]: v for k, v in state.items() if k.startswith("conditioner.")}
    if csub:
        cond_enc.load_state_dict(csub)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--adapter", required=True, help="trained adapter checkpoint (.pt)")
    ap.add_argument("--reference", "-r", required=True, help="audio file to riff on")
    ap.add_argument("--ref-seconds", type=float, default=30.0)
    ap.add_argument("--prompt", default="psychedelic goa trance")
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--gain", type=float, default=2.0,
                    help="control strength. BY EAR (2026-06-19): 1.0 already clean+matching on "
                         "out-of-dist refs (subtle), ~1.5-3 clean + clearly riffing, >4 glitchy/"
                         "experimental. (chroma-corr keeps rising past that but the metric ignores "
                         "the artifacts — trust ears.)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="riff.wav")
    args = ap.parse_args()

    ck = torch.load(args.adapter, map_location="cpu")
    cargs = ck.get("args", {})
    control_dim = int(cargs.get("control_dim", 768))
    n_tokens = int(cargs.get("n_tokens", 256))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from stable_audio_3 import StableAudioModel
    sam = StableAudioModel.from_pretrained(args.model, device=device)     # fp16 inference
    model_dtype = next(sam.model.model.parameters()).dtype

    wrappers = install_adapters(sam, control_dim=control_dim)
    cond_enc = AudioRefEncoder(256, control_dim, n_tokens).to(device)
    load_adapter_state(ck["state"], wrappers, cond_enc)
    for w in wrappers:
        w.adapter.to(device=device, dtype=model_dtype)
    cond_enc.to(device=device, dtype=model_dtype).eval()

    # reference track -> SAME latent -> control tokens
    wav, sr = torchaudio.load(args.reference)
    wav = wav[..., :int(args.ref_seconds * sr)]
    ds = sam.model.pretransform.downsampling_ratio
    ref_len = max(ds, (wav.shape[-1] // ds) * ds)
    with torch.inference_mode():
        ref_latent, _ = sam._encode_audio_input((sr, wav), ref_len, None)        # (1,256,T)
        ctrl = cond_enc(ref_latent.to(device=device, dtype=model_dtype))         # (1,n,d)
    # CFG batch-doubling: the DiT runs [cond, uncond]; give control to cond, zeros to uncond
    if args.cfg != 1.0:
        ctrl = torch.cat([ctrl, torch.zeros_like(ctrl)], dim=0)

    with use_control_context(ControlContext(ctrl, gain=args.gain)):
        audio = sam.generate(prompt=args.prompt, duration=args.duration, steps=args.steps,
                             cfg_scale=args.cfg, seed=args.seed, sampler_type="euler")

    save_audio(args.out, audio[0], sam.model.sample_rate)   # float32+peak-norm+int16 (no clip)
    print(f"riff -> {args.out}  (ref={os.path.basename(args.reference)}, "
          f"prompt={args.prompt!r}, cfg={args.cfg})")


if __name__ == "__main__":
    main()
