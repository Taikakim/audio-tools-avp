"""Steered long-form generation: wrap the longform ChunkGenerator with a per-window
onset-density control schedule. Single-prompt sweeps only (no prompt transitions)."""
import torch
from sa3_control.adapters import ControlContext, use_control_context
from sa3_control.density_schedule import ridge_gain


class SteeredGenerator:
    """Tracks output time from its own frame accounting and applies the scheduled
    onset scalar via use_control_context around each window's inner generate."""

    def __init__(self, inner, schedule, encoder, mean, std, gain, fps, cfg_scale, device, dtype, ridge=False):
        self.inner = inner
        self.sched = schedule
        self.enc = encoder
        self.mean = float(mean); self.std = float(std); self.gain = float(gain)
        self.fps = float(fps); self.cfg = float(cfg_scale)
        self.device = device; self.dtype = dtype
        self.ridge = bool(ridge)
        self._frames_before = 0
        self.applied = []  # (t_sec, raw_density) per window
        self.gains = []    # applied gain per window (ridge curve, or constant)

    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        t = self._frames_before / self.fps
        raw = self.sched.resolve(t)
        g = ridge_gain(raw) if self.ridge else self.gain
        self.applied.append((t, raw)); self.gains.append(g)
        s = torch.tensor([(raw - self.mean) / self.std], device=self.device, dtype=self.dtype)
        ctrl = self.enc(s)
        cc = torch.cat([ctrl, torch.zeros_like(ctrl)], 0) if self.cfg != 1.0 else ctrl
        with use_control_context(ControlContext(cc, gain=g)):
            out = self.inner.generate(prompt, prefix_latents, prefix_frames, n_frames, seed)
        self._frames_before += (n_frames - prefix_frames)
        return out


def _parse_prompt_arc(arg):
    """Parse the ``'0:A|t:B'`` arc format -> ``[(0.0,'A'),(t,'B')]``; a single
    prompt (no leading ``float:`` on every segment) -> the bare string.

    Reuses ``parse_schedule`` from ``stable-audio-3/scripts/longform_render.py``
    (the spec's canonical parser). If that script path is unavailable it falls back
    to the byte-identical ``sa3_control.chroma_guided_generator.parse_progression``.
    Colon-safe: ``'120bpm: deep techno'`` stays a single prompt.
    """
    try:
        import importlib.util
        p = "/home/kim/Projects/SAO/stable-audio-3/scripts/longform_render.py"
        spec = importlib.util.spec_from_file_location("_longform_render_arc", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.parse_schedule(arg)
    except Exception:
        from sa3_control.chroma_guided_generator import parse_progression
        return parse_progression(arg)


class BestOfNGenerator:
    """Best-of-N continuation selector — the *render side* of the cross-venv MERT hook.

    Wraps a fully-steered inner ``ChunkGenerator``. On each window it generates N
    candidate latents at distinct seeds, decodes each to a wav, and **shells out to
    the scorer in the mir venv**::

        <mert_python> -m sa3_control.mert_selector job.json scores.json

    with ``PYTHONPATH=<avp_sa3>:<mir/src>`` so the subprocess can import both
    ``sa3_control.mert_selector`` (MERT-v1-330M) AND ``timbral.audiobox_aesthetics``
    (Audiobox CE). The job/scores JSON contract is the seam (see the longform design
    spec, Part 4 §5.2). It keeps the argmax-reward candidate's LATENTS and threads its
    decoded wav forward as the ``reference`` for the next window; every candidate's
    scores are appended to ``self.log`` (the Part-5 bootstrapping corpus, dumped into
    the ``.schedule.json`` sidecar).

    GPU/mir-venv bound: real scoring needs the mir venv + GPU. The orchestration
    (N candidates, decode, job/scores threading, argmax) is the render-venv glue.
    """

    def __init__(self, inner, *, pretransform, sample_rate, fps, n_candidates=4,
                 seeds=None, weights=None,
                 mert_python="/home/kim/Projects/mir/mir/bin/python",
                 avp_sa3_path=None, mir_path="/home/kim/Projects/mir/src",
                 workdir=None):
        import os
        import tempfile
        self.inner = inner
        self.pretransform = pretransform
        self.sample_rate = int(sample_rate)
        self.fps = float(fps)
        self.n_candidates = int(n_candidates)
        self.seeds = list(seeds) if seeds else None
        self.weights = weights
        self.mert_python = mert_python
        if avp_sa3_path is None:
            from pathlib import Path
            avp_sa3_path = str(Path(__file__).resolve().parents[1])  # .../avp_sa3
        self.avp_sa3_path = avp_sa3_path
        self.mir_path = mir_path
        self.workdir = workdir or tempfile.mkdtemp(prefix="bestofn_")
        os.makedirs(self.workdir, exist_ok=True)
        self._k = 0
        self._prev_best_wav = None
        self.log = []

    def _decode(self, lat):
        with torch.no_grad():
            pt_dtype = next(self.pretransform.parameters()).dtype
            audio = self.pretransform.decode(lat.to(pt_dtype), chunked=True).float().cpu()
        return audio[0] if audio.dim() == 3 else audio

    def _run_scorer(self, job):
        import os
        import json
        import subprocess
        job_path = os.path.join(self.workdir, "job.json")
        out_path = os.path.join(self.workdir, "scores.json")
        with open(job_path, "w") as f:
            json.dump(job, f)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([self.avp_sa3_path, self.mir_path])
        subprocess.run([self.mert_python, "-m", "sa3_control.mert_selector",
                        job_path, out_path], check=True, env=env)
        with open(out_path) as f:
            return json.load(f)

    def generate(self, prompt, prefix_latents, prefix_frames, n_frames, seed):
        import os
        import shutil
        from sa3_control.audio_io import save_audio
        k = self._k
        self._k += 1
        lats, cand_paths = [], []
        for c in range(self.n_candidates):
            sc = self.seeds[c] if self.seeds else seed + 9173 * c
            lat_c = self.inner.generate(prompt, prefix_latents, prefix_frames, n_frames, sc)
            wav = self._decode(lat_c)
            p = os.path.join(self.workdir, f"win{k}_cand{c}.wav")
            save_audio(p, wav, self.sample_rate)
            lats.append(lat_c)
            cand_paths.append(p)
        w = self.weights
        job = {
            "candidates": cand_paths,
            "reference": self._prev_best_wav,
            "sample_rate": self.sample_rate,
            "weights": {
                "w_ce": w.w_ce, "w_rhythm": w.w_rhythm, "w_melody": w.w_melody,
                "band_center": w.band_center, "band_lo": w.band_lo, "band_hi": w.band_hi,
            },
            "mert_layers_mid": [3, 4, 5, 6],
            "mert_layers_upper": [23],
        }
        scores = self._run_scorer(job)
        best = max(range(len(scores)), key=lambda i: scores[i]["score"])
        best_path = os.path.join(self.workdir, f"win{k}_best.wav")
        shutil.copyfile(cand_paths[best], best_path)
        self._prev_best_wav = best_path
        self.log.append({"window": k, "best": best, "reference": job["reference"],
                         "candidates": cand_paths, "scores": scores})
        return lats[best]


def _reward_weights(mert_weights, melody_band):
    """('1,1,1', '0.55,0.80') -> RewardWeights with band_center=midpoint."""
    from sa3_control.mert_selector import RewardWeights
    wc, wr, wm = (float(x) for x in mert_weights.split(","))
    lo, hi = (float(x) for x in melody_band.split(","))
    return RewardWeights(w_ce=wc, w_rhythm=wr, w_melody=wm,
                         band_center=0.5 * (lo + hi), band_lo=lo, band_hi=hi)


def main():
    import os, json, argparse
    os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
    import torch
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.longform import (
        InpaintContinuationGenerator, PromptSchedule)
    from sa3_control.inject import install_adapters
    from sa3_control.conditioner import ScalarAttributeEncoder
    from sa3_control.generate import load_adapter_state
    from sa3_control.density_schedule import ControlSchedule
    from sa3_control.development_renderer import (
        DevelopmentRenderer, SDEditContinuationGenerator)

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--shape", required=True)
    ap.add_argument("--prompt", default="goa trance, psychedelic, driving, 145 bpm",
                    help="prompt, or a '0:A|t:B' arc schedule (parse_schedule format)")
    ap.add_argument("--duration", type=float, default=360.0)
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--overlap-sec", type=float, default=5.0)
    ap.add_argument("--lo", type=float, default=2.0); ap.add_argument("--hi", type=float, default=14.0)
    ap.add_argument("--gain", type=float, default=6.0)
    ap.add_argument("--ridge", action="store_true",
                    help="density-dependent ridge gain (low density ~2.75 -> high density ~1.0); overrides --gain")
    ap.add_argument("--steps", type=int, default=50); ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=777); ap.add_argument("--out", required=True)
    # -- Part 1: continuation mode -------------------------------------------------
    ap.add_argument("--continuation", choices=("clamp", "sdedit", "crossfade"),
                    default="clamp",
                    help="window continuation mode (default clamp = today's behaviour)")
    ap.add_argument("--crossfade-frac", type=float, default=1.0,
                    help="crossfade mode: fraction of the overlap to slerp every window")
    # -- Part 2: prompt arc --------------------------------------------------------
    ap.add_argument("--xfade-sec", type=float, default=4.0,
                    help="crossfade seconds on a prompt-arc transition")
    # -- Part 3: chroma (HPCP) guidance --------------------------------------------
    ap.add_argument("--chroma", "--chroma-progression", dest="chroma", default=None,
                    help="chroma chord-progression schedule '0:Am|32:F|64:C' (needs --chroma-head)")
    ap.add_argument("--chroma-head", default=None,
                    help="path to latch_sa3_hpcp_best.pt (12-ch HPCP guidance head, fp32)")
    ap.add_argument("--chroma-rho", type=float, default=64.0)
    ap.add_argument("--chroma-mu", type=float, default=64.0)
    ap.add_argument("--chroma-key", default=None,
                    help="key for roman-numeral degrees in the progression (e.g. 'A')")
    # -- Part 4: best-of-N MERT selector (cross-venv) ------------------------------
    ap.add_argument("--best-of-n", type=int, default=0,
                    help="N candidate windows/step, MERT+Audiobox re-ranked (0/1 = off)")
    ap.add_argument("--mert-weights", default="1,1,1",
                    help="w_ce,w_rhythm,w_melody for the best-of-N reward")
    ap.add_argument("--melody-band", default="0.55,0.80",
                    help="lo,hi melody-sim band (band_center = midpoint)")
    ap.add_argument("--mert-python", default="/home/kim/Projects/mir/mir/bin/python",
                    help="interpreter for the mir-venv MERT/Audiobox scorer subprocess")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mean, std = ck.get("scalar_norm", [7.219, 1.424])
    n_tokens = min(int(ck["args"].get("n_tokens", 16)), 16)
    control_dim = int(ck["args"].get("control_dim", 768))
    sam = StableAudioModel.from_pretrained("medium-base", device="cuda")
    md = next(sam.model.model.parameters()).dtype
    fps = sam.model.sample_rate / sam.model.pretransform.downsampling_ratio
    sr = int(sam.model.sample_rate)
    wrappers = install_adapters(sam, control_dim=control_dim)
    enc = ScalarAttributeEncoder(control_dim=control_dim, n_tokens=n_tokens)
    load_adapter_state(ck["state"], wrappers, enc)
    for w in wrappers: w.adapter.to(device="cuda", dtype=md)
    enc.to(device="cuda", dtype=md).eval()

    # -- inner generator: chroma-guided > sdedit > inpaint (clamp/crossfade) -------
    chroma_prog = None
    if args.chroma_head:
        from sa3_control.chroma_guided_generator import (
            ChromaGuidedGenerator, ChromaSchedule, parse_progression, load_chroma_head)
        head = load_chroma_head(args.chroma_head, device="cuda")  # fp32 (MASTER §5)
        prog = parse_progression(args.chroma or "")
        if isinstance(prog, str):
            prog = [(0.0, prog)] if prog else [(0.0, "C")]
        chroma_prog = prog
        chsched = ChromaSchedule(prog, fps=fps,
                                 head_metadata=getattr(head, "metadata", None),
                                 key=args.chroma_key)
        inner = ChromaGuidedGenerator(sam, head, chsched, steps=args.steps,
                                      cfg_scale=args.cfg, rho=args.chroma_rho,
                                      mu=args.chroma_mu)
    elif args.continuation == "sdedit":
        inner = SDEditContinuationGenerator(sam, steps=args.steps, cfg_scale=args.cfg)
    else:  # clamp + crossfade both use the inpaint generator (crossfade forces prefix=0)
        inner = InpaintContinuationGenerator(sam, steps=args.steps, cfg_scale=args.cfg)

    sched = ControlSchedule(args.shape, args.duration, args.lo, args.hi)
    steered = SteeredGenerator(inner, sched, enc, mean, std, args.gain, fps, args.cfg,
                               "cuda", md, ridge=args.ridge)

    # -- Part 4: optional best-of-N wrapper (sits OUTSIDE SteeredGenerator) --------
    gen = steered
    bestofn = None
    if args.best_of_n and args.best_of_n > 1:
        weights = _reward_weights(args.mert_weights, args.melody_band)
        bestofn = BestOfNGenerator(steered, pretransform=sam.model.pretransform,
                                   sample_rate=sr, fps=fps, n_candidates=args.best_of_n,
                                   weights=weights, mert_python=args.mert_python)
        gen = bestofn

    f = lambda s: int(round(s * fps))
    prompt_sched = PromptSchedule(_parse_prompt_arc(args.prompt), crossfade_sec=args.xfade_sec)
    r = DevelopmentRenderer(gen, channels=sam.model.io_channels, fps=fps,
                            window_frames=f(args.window_sec), overlap_frames=f(args.overlap_sec),
                            continuation_mode=args.continuation,
                            crossfade_overlap_frac=args.crossfade_frac)
    lat = r.render_latents(prompt_sched, total_frames=f(args.duration), base_seed=args.seed)
    with torch.no_grad():
        pt_dtype = next(sam.model.pretransform.parameters()).dtype
        audio = sam.model.pretransform.decode(lat.to(pt_dtype), chunked=True).float().cpu()
    wav = audio[0] if audio.dim() == 3 else audio
    from sa3_control.audio_io import save_audio
    save_audio(args.out, wav, sr)   # peak-normalize, never raw-clamp: SA3 output peaks >1.0 (MASTER §5)
    json.dump({"shape": args.shape, "duration": args.duration,
               "gain": ("ridge" if args.ridge else args.gain), "ridge": args.ridge,
               "scalar_field": ck.get("scalar_field"), "fps": fps,
               "continuation": args.continuation,
               "prompt_arc": _parse_prompt_arc(args.prompt),
               "chroma_progression": chroma_prog,
               "best_of_n": (args.best_of_n if bestofn else 0),
               "mert_weights": (args.mert_weights if bestofn else None),
               "bestof_log": (bestofn.log if bestofn else None),
               "applied": steered.applied, "gains": steered.gains},
              open(args.out + ".schedule.json", "w"), indent=1)
    dvals = [a[1] for a in steered.applied]
    print(f"[steered] wrote {args.out}  windows={len(steered.applied)}  "
          f"mode={args.continuation}  density {min(dvals):.1f}..{max(dvals):.1f}  "
          f"gain {min(steered.gains):.2f}..{max(steered.gains):.2f}")


if __name__ == "__main__":
    main()
