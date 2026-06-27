"""Steered long-form generation: wrap the longform ChunkGenerator with a per-window
onset-density control schedule. Single-prompt sweeps only (no prompt transitions)."""
import torch
from sa3_control.adapters import ControlContext, use_control_context


class SteeredGenerator:
    """Tracks output time from its own frame accounting and applies the scheduled
    onset scalar via use_control_context around each window's inner generate."""

    def __init__(self, inner, schedule, encoder, mean, std, gain, fps, cfg_scale, device, dtype):
        self.inner = inner
        self.sched = schedule
        self.enc = encoder
        self.mean = float(mean); self.std = float(std); self.gain = float(gain)
        self.fps = float(fps); self.cfg = float(cfg_scale)
        self.device = device; self.dtype = dtype
        self._frames_before = 0
        self.applied = []  # (t_sec, raw_density) per window

    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        t = self._frames_before / self.fps
        raw = self.sched.resolve(t)
        self.applied.append((t, raw))
        s = torch.tensor([(raw - self.mean) / self.std], device=self.device, dtype=self.dtype)
        ctrl = self.enc(s)
        cc = torch.cat([ctrl, torch.zeros_like(ctrl)], 0) if self.cfg != 1.0 else ctrl
        with use_control_context(ControlContext(cc, gain=self.gain)):
            out = self.inner.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)
        self._frames_before += (n_frames - prefix_frames)
        return out


def main():
    import os, json, argparse
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
    import torch
    import soundfile as sf
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import (
        LongFormRenderer, InpaintContinuationGenerator, PromptSchedule)
    from sa3_control.inject import install_adapters
    from sa3_control.conditioner import ScalarAttributeEncoder
    from sa3_control.generate import load_adapter_state
    from sa3_control.density_schedule import ControlSchedule

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--shape", required=True)
    ap.add_argument("--prompt", default="goa trance, psychedelic, driving, 145 bpm")
    ap.add_argument("--duration", type=float, default=360.0)
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--overlap-sec", type=float, default=5.0)
    ap.add_argument("--lo", type=float, default=2.0); ap.add_argument("--hi", type=float, default=14.0)
    ap.add_argument("--gain", type=float, default=6.0)
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=777); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mean, std = ck.get("scalar_norm", [7.219, 1.424])
    n_tokens = min(int(ck["args"].get("n_tokens", 16)), 16)
    control_dim = int(ck["args"].get("control_dim", 768))
    sam = StableAudioModel.from_pretrained("medium-base", device="cuda")
    md = next(sam.model.model.parameters()).dtype
    fps = sam.model.sample_rate / sam.model.pretransform.downsampling_ratio
    wrappers = install_adapters(sam, control_dim=control_dim)
    enc = ScalarAttributeEncoder(control_dim=control_dim, n_tokens=n_tokens)
    load_adapter_state(ck["state"], wrappers, enc)
    for w in wrappers: w.adapter.to(device="cuda", dtype=md)
    enc.to(device="cuda", dtype=md).eval()

    inner = InpaintContinuationGenerator(sam, steps=args.steps, cfg_scale=args.cfg)
    sched = ControlSchedule(args.shape, args.duration, args.lo, args.hi)
    steered = SteeredGenerator(inner, sched, enc, mean, std, args.gain, fps, args.cfg, "cuda", md)
    f = lambda s: int(round(s * fps))
    r = LongFormRenderer(steered, channels=sam.model.io_channels, fps=fps,
                         window_frames=f(args.window_sec), overlap_frames=f(args.overlap_sec))
    lat = r.render_latents(PromptSchedule(args.prompt), total_frames=f(args.duration), base_seed=args.seed)
    with torch.no_grad():
        pt_dtype = next(sam.model.pretransform.parameters()).dtype
        audio = sam.model.pretransform.decode(lat.to(pt_dtype), chunked=True).float().cpu()
    wav = audio[0] if audio.dim() == 3 else audio
    sf.write(args.out, wav.clamp(-1, 1).transpose(0, 1).numpy(), int(sam.model.sample_rate))
    json.dump({"shape": args.shape, "duration": args.duration, "gain": args.gain,
               "scalar_field": ck.get("scalar_field"), "fps": fps,
               "applied": steered.applied}, open(args.out + ".schedule.json", "w"), indent=1)
    print(f"[steered] wrote {args.out}  windows={len(steered.applied)}  "
          f"density {steered.applied[0][1]:.1f}..{min(r[1] for r in steered.applied):.1f}.."
          f"{steered.applied[-1][1]:.1f}")


if __name__ == "__main__":
    main()
