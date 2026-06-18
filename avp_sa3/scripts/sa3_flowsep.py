"""
FlowSep: inversion-free text-guided source separation on SA3 (FlowEdit / AUDEDIT).

This is the correct successor to sa3_zerosep_lite.py. The lite probe was plain
SDEdit (init_audio + init_noise_level -> single linear noise blend), which loses
all relation to the input at the noise levels needed for the prompt to take hold.

FlowEdit (Kulikov et al. 2024, arXiv:2412.08629; AUDEDIT arXiv:2606.15149 applies
it to Stable Audio 3) is inversion-FREE: it never visits pure noise. It keeps an
edited latent z_edit (initialized at the encoded mixture x0) and integrates a
DIFFERENCE velocity field along SA3's own rectified-flow schedule:

    for each step i (t descends 1 -> 0), skip while (T - i) > n_max:
        eps   ~ N(0, I)
        z_src = (1 - t)*x0 + t*eps            # noised SOURCE (the mixture)
        z_tar = z_edit + z_src - x0           # shared-noise TARGET proxy
        Vd    = v(z_tar, t, TARGET_prompt, cfg_tar) - v(z_src, t, SOURCE_prompt, cfg_src)
        z_edit = z_edit + (t_next - t)*Vd     # dt < 0
    decode(z_edit)

Because both branches share the same eps, the common (mixture) noise cancels and
z_edit stays anchored to the input by construction — no inversion error, no
attention hooks, no training. For SOURCE SEPARATION the source prompt describes
the full mixture and each target prompt names one isolated source.

We reuse SA3's plumbing by monkey-patching sample_discrete_euler for the duration
of one model.generate() call (same trick as sa3_steer_chroma.py), so conditioning,
varlen padding-mask, schedule (incl. dist_shift) and decode are all reused. The
patched sampler ignores the incoming fresh noise and starts from the encoded x0.

Use a -base checkpoint: cfg_scale is inert on post-trained models, and FlowEdit
needs the live, deterministic Euler velocity (post-trained defaults to stochastic
ping-pong). All velocity math runs in the model dtype; no autograd needed.

Examples:
    python sa3_flowsep.py -i mix.wav --start 400 --max-seconds 12 \
        --prompts "solo drum kit" "isolated bass synth" "lead synth melody alone" \
        --source-prompt "psychedelic goa trance full mix: drums, bass and acid lead" \
        --cfg-src 3.5 --cfg-tar 13.5 --n-max 33 --steps 50
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sa3_control.audio_io import save_audio  # noqa: E402  float32+peak-norm+int16 (no clip)

logger = logging.getLogger(__name__)


def build_cond_inputs(sam, prompt, duration, batch_size, latent_sample_size, device):
    """Replicate generate()'s conditioning assembly (model.py:252-318) for one
    prompt: conditioner -> add zero inpaint mask/input -> get_conditioning_inputs,
    cast to the diffusion model dtype. Returns the cross-attn/global/etc. dict."""
    import torch
    conditioning, _ = sam._build_conditioning_dicts(prompt, None, duration, batch_size)
    ct = sam.model.conditioner(conditioning, device)
    mask = torch.zeros((batch_size, 1, latent_sample_size), device=device)
    io_ch = sam.model.io_channels
    ct["inpaint_mask"] = [mask]
    ct["inpaint_masked_input"] = [
        torch.zeros((batch_size, io_ch, latent_sample_size), device=device)
    ]
    ci = sam.model.get_conditioning_inputs(ct)
    md = next(sam.model.model.parameters()).dtype
    return {k: (v.type(md) if torch.is_tensor(v) else v) for k, v in ci.items()}


def make_flowedit_euler(sam, wav, in_sr, source_prompt, duration, batch_size,
                        cfg_src, n_max, n_avg, seed, anchor_eta=0.0, tau=0.3):
    """Clone of sample_discrete_euler that runs the FlowEdit difference-ODE.

    The patched sampler receives the TARGET cond_inputs (+ cfg_tar via cfg_scale,
    batch_cfg/rescale_cfg/padding_mask/apg_scale) as **extra_args from generate().
    It builds the SOURCE cond_inputs itself, encodes the mixture to x0 at exactly
    the latent length generate() allocated, and integrates z_edit.

    Hybrid fidelity anchor (anchor_eta/tau): FlowEdit supplies a robust EDIT
    DIRECTION (the difference field); anchor_eta adds the RF-Solver-style fidelity
    dial on top, blending z_edit back toward the source mixture x0 each high-noise
    step (t >= tau): z_edit <- (1-anchor_eta)*z_edit + anchor_eta*x0. 0 = pure
    FlowEdit; higher = pulled toward the input. The per-step blend compounds, so
    useful values are small (~0.02-0.2). This is the FlowEdit+RF-Solver hybrid."""
    import torch

    def flowedit(model, x, sigmas, callback=None, disable_tqdm=False, **extra_args):
        device = x.device
        ds = sam.model.pretransform.downsampling_ratio
        latent_len = x.shape[-1]

        # Encode the mixture to x0 at exactly the latent grid generate() built.
        x0, _ = sam._encode_audio_input((in_sr, wav), latent_len * ds, None)
        x0 = x0.to(device=device, dtype=x.dtype)
        if x0.shape[0] != batch_size:
            x0 = x0.repeat(batch_size, 1, 1)

        # TARGET args = what generate() passed (cfg_scale already == cfg_tar).
        # SOURCE args = same control kwargs, source conditioning, cfg_src.
        src_cond = build_cond_inputs(sam, source_prompt, duration, batch_size,
                                     latent_len, device)
        tar_args = extra_args
        src_args = dict(extra_args)
        src_args.update(src_cond)
        src_args["cfg_scale"] = cfg_src

        t = sigmas.to(device)
        per_element = t.dim() == 2
        num_steps = t.shape[-1] - 1
        nmax = min(n_max, num_steps)
        edit_steps = sum(1 for i in range(num_steps) if num_steps - i <= nmax)
        logger.info(f"flowedit: {num_steps} steps, editing the last {edit_steps} "
                    f"(n_max={nmax}), n_avg={n_avg}, cfg_src={cfg_src}")

        gen = torch.Generator(device=device).manual_seed(seed)
        z_edit = x0.clone()
        for i in range(num_steps):
            if num_steps - i > nmax:
                continue  # skip the high-noise head (FlowEdit never visits ~pure noise)
            if per_element:
                t_i = t[:, i].to(x.dtype)
                t_im1 = t[:, i + 1].to(x.dtype)
            else:
                t_i = (t[i] * torch.ones(batch_size, device=device)).to(x.dtype)
                t_im1 = (t[i + 1] * torch.ones(batch_size, device=device)).to(x.dtype)
            t_i_b = t_i.view(-1, 1, 1)
            dt_b = (t_im1 - t_i).view(-1, 1, 1)

            Vd = torch.zeros_like(x0)
            for _ in range(n_avg):
                eps = torch.randn(x0.shape, generator=gen, device=device, dtype=x.dtype)
                z_src = (1 - t_i_b) * x0 + t_i_b * eps
                z_tar = z_edit + z_src - x0
                v_src = model(z_src, t_i, **src_args)
                v_tar = model(z_tar, t_i, **tar_args)
                Vd = Vd + (v_tar - v_src) / n_avg

            z_edit = z_edit + dt_b * Vd
            if anchor_eta > 0.0 and float(t_i_b.reshape(-1)[0]) >= tau:
                z_edit = (1.0 - anchor_eta) * z_edit + anchor_eta * x0
        return z_edit

    return flowedit


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--prompts", "-p", type=str, nargs="+",
                        default=["solo drum kit", "isolated bass synth",
                                 "lead synth melody alone"],
                        help="TARGET source prompts (one isolated source each)")
    parser.add_argument("--source-prompt", type=str,
                        default="full mix with drums, bass and lead",
                        help="prompt describing the whole input mixture")
    parser.add_argument("--prompt-prefix", type=str,
                        default="TrackType: Instrument,",
                        help="prefix for TARGET (isolated-source) prompts. Per the SA3 "
                             "guide, 'TrackType: Instrument' maximizes isolated-instrument "
                             "output. '' to disable.")
    parser.add_argument("--source-prefix", type=str,
                        default="TrackType: Music, VocalType: Instrumental,",
                        help="prefix for the SOURCE (full-mix) prompt")
    parser.add_argument("--bpm", type=float, default=None,
                        help="if set, append ', <bpm> BPM' to every prompt (known from "
                             "the stem folder names, e.g. 'Acid Alien 136')")
    parser.add_argument("--model", type=str, default="medium-base",
                        help="MUST be a -base checkpoint")
    parser.add_argument("--start", type=float, default=0.0,
                        help="window start in seconds (fixes the lite intro-only bug)")
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--n-max", type=int, default=33,
                        help="edit the last n_max of --steps steps (FlowEdit SD3: 33; "
                             "lower = gentler, stays closer to the mixture)")
    parser.add_argument("--n-avg", type=int, default=1,
                        help="velocity draws averaged per step (>1 reduces noise)")
    parser.add_argument("--cfg-src", type=float, default=3.5)
    parser.add_argument("--cfg-tar", type=float, default=13.5)
    parser.add_argument("--anchor-eta", type=float, nargs="+", default=[0.0],
                        help="hybrid fidelity dial: blend z_edit toward the source mix "
                             "by this weight each high-noise step (t>=tau). 0 = pure "
                             "FlowEdit; small values (0.02-0.2) pull toward the input. "
                             "Pass several to sweep.")
    parser.add_argument("--tau", type=float, default=0.3,
                        help="anchor active for t>=tau (high noise), released below")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_dir", type=str, default="flowsep_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "base" not in args.model:
        logger.warning(f"'{args.model}' is not a -base checkpoint - cfg is likely "
                       f"inert (post-trained models bake guidance in) and the "
                       f"default sampler is stochastic ping-pong, not Euler.")

    import torch
    import torchaudio
    from stable_audio_3 import StableAudioModel
    import stable_audio_3.inference.sampling as smp

    device = "cuda" if torch.cuda.is_available() else "cpu"

    wav, sr = torchaudio.load(args.input)                       # (C, T)
    s0 = int(args.start * sr)
    s1 = s0 + int(args.max_seconds * sr)
    wav = wav[..., s0:s1]
    duration = wav.shape[-1] / sr
    logger.info(f"input: {args.input} window [{args.start:.1f}, "
                f"{args.start + duration:.1f}]s -> {wav.shape} @ {sr}Hz")

    model = StableAudioModel.from_pretrained(args.model, device=device)
    out_sr = model.model.sample_rate

    out_dir = Path(args.out_dir) / f"{Path(args.input).stem}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_audio(out_dir / "original.wav", wav, sr, normalize=False)   # keep input level

    gen_kwargs = dict(duration=duration, steps=args.steps, cfg_scale=args.cfg_tar,
                      seed=args.seed, sampler_type="euler")
    bpm_sfx = f", {args.bpm:g} BPM" if args.bpm else ""
    pre_t = (args.prompt_prefix.rstrip() + " ") if args.prompt_prefix else ""
    pre_s = (args.source_prefix.rstrip() + " ") if args.source_prefix else ""
    src_prompt = pre_s + args.source_prompt + bpm_sfx

    results = []
    orig_euler = smp.sample_discrete_euler
    for prompt in args.prompts:
        base = "".join(c if c.isalnum() else "_" for c in prompt)[:40].strip("_")
        for aeta in args.anchor_eta:
            tag = f"{base}_aeta{aeta:g}"
            logger.info(f"--- separating: '{prompt}' (anchor_eta={aeta}, tau={args.tau}) ---")
            smp.sample_discrete_euler = make_flowedit_euler(
                model, wav, sr, src_prompt, duration, batch_size=1,
                cfg_src=args.cfg_src, n_max=args.n_max, n_avg=args.n_avg, seed=args.seed,
                anchor_eta=aeta, tau=args.tau)
            try:
                audio = model.generate(prompt=pre_t + prompt + bpm_sfx, **gen_kwargs)
            finally:
                smp.sample_discrete_euler = orig_euler
            path = out_dir / f"{tag}.wav"
            save_audio(path, audio[0], out_sr)
            results.append({"prompt": prompt, "anchor_eta": aeta, "file": path.name})

    (out_dir / "params.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    logger.info(f"done -> {out_dir}")
    logger.info("LISTEN: each <target>.wav should isolate that source FROM THE INPUT "
                "(same notes/timing as original.wav), not a generic re-synthesis. "
                "If outputs ignore the input, lower --n-max / --cfg-tar; if they barely "
                "change, raise --n-max / --cfg-tar.")


if __name__ == "__main__":
    main()
