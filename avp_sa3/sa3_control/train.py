"""Train the audio-reference (riffer) control adapter on pre-encoded SA3 latents.

Loads medium-base (fp32, model_half=False), installs decoupled cross-attn adapters
(base frozen), and trains the adapter + audio-ref conditioner with the rectified-flow
loss, conditioning each step on a DIFFERENT crop of the same track (the riffer pairing
from LatentControlDataset). Control tokens are injected via the ContextVar around our
own DiT forward (cfg_scale=1.0 -> no CFG batch-doubling).

Run with the SA3 .venv:
    PYTORCH_TUNABLEOP_ENABLED=0 /home/kim/Projects/SAO/stable-audio-3/.venv/bin/python \
        -m sa3_control.train --encoded_dir /run/media/kim/Lehto/latents_sa3 --smoke
"""

import argparse
import os
import subprocess
import sys
import time

# append (not insert(0)) so site-packages resolves first — a wandb run-data dir that
# wandb writes under the package root must not shadow the real `wandb` package on reimport.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sa3_control.adapters import ControlContext, use_control_context
from sa3_control.conditioner import AudioRefEncoder, ScalarAttributeEncoder, AttributeEncoder
from sa3_control.dataset import LatentControlDataset, CONTROL_FIELDS, CONTROL_DIMS
from sa3_control.inject import (adapter_state_dict, freeze_base_train_adapters,
                                install_adapters)


def collate(batch):
    out = {
        "latent": torch.stack([b["latent"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
    }
    if "ref_latent" in batch[0]:
        out["ref_latent"] = torch.stack([b["ref_latent"] for b in batch])
    if "scalar" in batch[0]:
        out["scalar"] = torch.stack([b["scalar"] for b in batch])
    if batch[0].get("controls"):                # time-varying attribute features {name: (C,T)}
        out["controls"] = {k: torch.stack([b["controls"][k] for b in batch]) for k in batch[0]["controls"]}
    return out


# The prompt is fixed per crop, so the frozen text encoder gives the same output every
# epoch — re-running it every step was the per-step bottleneck (GPU starved between DiT
# bursts). Encode each unique prompt ONCE and reuse. Cached output is on `device`.
_TEXT_COND_CACHE = {}   # prompt -> conditioner output dict


def _sample_t(sampler, B, device):
    """Diffusion timestep sampler (ported from underfit/SAT). 'logit_normal' is our
    original (= sigmoid(randn)); 'log_snr' samples Gaussian on logSNR (mean -1.2, std 2.0)
    -> t=sigmoid(-logsnr), biasing toward the informative band instead of pure mid-noise."""
    if sampler == "uniform":
        return torch.rand(B, device=device)
    if sampler == "logit_normal":
        return torch.sigmoid(torch.randn(B, device=device)).clamp(1e-4, 1 - 1e-4)
    if sampler == "log_snr":
        ls = torch.randn(B, device=device) * 2.0 - 1.2
        return torch.sigmoid(-ls).clamp(1e-4, 1 - 1e-4)
    if sampler == "log_snr_uniform":
        ls = torch.rand(B, device=device) * 11.0 - 6.0
        return torch.sigmoid(-ls).clamp(1e-4, 1 - 1e-4)
    raise ValueError(f"unknown timestep_sampler: {sampler}")


def _encode_text(sam, prompt, seconds, device):
    """Run the frozen text conditioner for ONE prompt. no_grad: the encoder is frozen,
    so no graph is built and the cached tensors act as constants in the train forward."""
    with torch.no_grad():
        return sam.model.conditioner([{"prompt": prompt, "seconds_total": float(seconds)}], device)


def preencode_text(sam, prompts, seconds, device):
    """Encode every unique prompt once into the cache, so no training step pays the
    text-encoder cost."""
    uniq = sorted(set(prompts))
    for i, p in enumerate(uniq):
        if p not in _TEXT_COND_CACHE:
            _TEXT_COND_CACHE[p] = _encode_text(sam, p, seconds, device)
        if (i + 1) % 200 == 0:
            print(f"[preencode] {i + 1}/{len(uniq)} prompts", flush=True)
    print(f"[preencode] cached {len(uniq)} unique prompts", flush=True)


def build_train_cond(sam, prompts, seconds, latent_T, device, dtype, use_cache=True):
    """cond_inputs for a training batch (cfg_scale=1.0 path), mirroring generate(). The
    frozen text-encoder output is cached per prompt (batch==1 path)."""
    B = len(prompts)
    io = sam.model.io_channels
    if use_cache and B == 1:
        ct = _TEXT_COND_CACHE.get(prompts[0])
        if ct is None:
            ct = _encode_text(sam, prompts[0], seconds, device)
            _TEXT_COND_CACHE[prompts[0]] = ct
        ct = dict(ct)                                   # shallow copy: don't mutate the cached dict
    else:
        conditioning = [{"prompt": p, "seconds_total": float(seconds)} for p in prompts]
        ct = dict(sam.model.conditioner(conditioning, device))
    ct["inpaint_mask"] = [torch.zeros((B, 1, latent_T), device=device)]
    ct["inpaint_masked_input"] = [torch.zeros((B, io, latent_T), device=device)]
    ci = sam.model.get_conditioning_inputs(ct)
    return {k: (v.type(dtype) if torch.is_tensor(v) else v) for k, v in ci.items()}


def export_control_onnx_on_finish(ckpt_path, save_dir, frames, field):
    """ADDITIVE, non-fatal: after the final checkpoint is saved, shell out to the SA3
    control-DiT ONNX exporter (forward-only adapter bake-in). Runs under sys.executable
    (the SA3 .venv the trainer already uses, which has the exporter's deps). Returns the
    subprocess returncode; never raises (check=False). `field` is informational only."""
    out_path = os.path.join(save_dir, f"dit_medium-base_L{frames}_ctrl.onnx")
    cmd = [sys.executable,
           "/home/kim/Projects/SAO/stable-audio-3/scripts/export_dit_control_onnx.py",
           "--ckpt", ckpt_path, "--model", "medium-base", "--frames", str(frames),
           "--text-seq", "128", "--fp16", "--out", out_path]
    env = {**os.environ, "FLASH_ATTENTION_TRITON_AMD_ENABLE": "FALSE"}
    print(f"[export] start: control-DiT ONNX -> {out_path} (field={field})", flush=True)
    rc = subprocess.run(cmd, env=env, check=False).returncode
    if rc == 0:
        print(f"[export] done: {out_path}", flush=True)
    else:
        print(f"[export] failed (returncode {rc}); training already saved, continuing", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--encoded_dir", default="/run/media/kim/Lehto/latents_sa3")
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--control-dim", type=int, default=768)
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--crop-frames", type=int, default=1024,
                    help="train on the first N latent frames (memory; full=4096)")
    ap.add_argument("--random-crop", action="store_true",
                    help="train on a RANDOM beat-aligned crop-frames window of each latent (vs always the "
                         "first N) — more crop variety per track so the trainer stops seeing only the first "
                         "~47s of each crop; snaps the offset to beat_activation_ts peaks.")
    ap.add_argument("--cfg-dropout", type=float, default=0.1,
                    help="per-item probability of dropping the control tokens")
    ap.add_argument("--save-dir", default="/run/media/kim/Lehto/sa3_control_runs/riffer")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--save-cooldown", type=float, default=60,
                    help="seconds to idle the GPU after each checkpoint save (thermal relief; 0 disables)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--subset-tracks", type=float, default=None,
                    help="train on a random fraction (0.2 = 20%% of tracks, all their crops; "
                         "preserves the riffer pairing) or count (>1) of tracks. None = all.")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--wandb-project", default="sa3-riffer")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--profile", action="store_true",
                    help="log a per-step timing breakdown (data/text/ref/fwd/bwd/opt)")
    ap.add_argument("--no-preencode-text", action="store_false", dest="preencode_text",
                    help="skip the upfront unique-prompt text pre-encode (cache lazily instead)")
    ap.add_argument("--no-checkpoint", action="store_false", dest="use_checkpointing",
                    help="disable DiT gradient checkpointing — much faster backward if VRAM fits "
                         "(only the 116M adapter trains, but checkpointing recomputes the whole "
                         "2.4B forward in backward; the bwd was ~90%% of step time)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock stop after this many hours (saves riffer_final.pt)")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="linear LR warmup over this many steps (0 = none). Best practice for "
                         "higher LRs — prevents the early gradient explosion seen at lr 1e-3.")
    ap.add_argument("--optimizer", choices=["adamw", "fusion", "sfadamw", "fusion_nm", "fusion_full"], default="adamw",
                    help="adamw (default); fusion (SF-NorMuon = ns5+normuon+sf); sfadamw (sf-only = "
                         "ScheduleFree-AdamW); fusion_nm (mona+ns5+normuon+sf — everything EXCEPT KL-Shampoo, "
                         "now viable on large adapters thanks to component-gated state alloc); fusion_full "
                         "(all 5 incl. Shampoo — heavy, may OOM on large adapters).")
    ap.add_argument("--resume", default="", help="warm-start: load adapter+conditioner weights from a "
                    "checkpoint .pt (optimizer restarts fresh; not an exact-state resume).")
    ap.add_argument("--resume-exact", default="", help="EXACT-state resume: restore adapter weights + "
                    "optimizer state + step counter from a checkpoint saved with the 'opt' key (continues "
                    "the run as if never interrupted). Errors on pre-opt-state checkpoints.")
    ap.add_argument("--timestep-sampler",
                    choices=["logit_normal", "log_snr", "log_snr_uniform", "uniform"], default="logit_normal",
                    help="diffusion t sampler (underfit borrow). logit_normal = original; "
                         "log_snr biases toward the informative sigma band (may de-noise the loss).")
    ap.add_argument("--control-mode", choices=["audio_ref", "scalar", "attribute"], default="audio_ref",
                    help="audio_ref = the riffer (opaque reference latent); scalar = a per-crop scalar "
                         "(e.g. onset_density); attribute = a TIME-VARYING per-frame feature "
                         "(dynamics/rhythm/melody curve) via the time-aligned AttributeEncoder.")
    ap.add_argument("--scalar-field", default="onset_density",
                    help="which per-crop .json scalar to condition on when --control-mode scalar")
    ap.add_argument("--control-feature", default="melody",
                    help="which .TIMESERIES.npz feature for --control-mode attribute: "
                         "dynamics(4) | rhythm(3) | melody(12, =hpcp). (chroma384 once that data exists.)")
    ap.add_argument("--attr-downsample", type=int, default=8,
                    help="time downsample for the AttributeEncoder (T -> T/this control tokens)")
    ap.add_argument("--scalar-from-timeseries", default="",
                    help="scalar mode: derive the scalar as the WINDOW-MEAN of this .TIMESERIES feature "
                         "(e.g. density_ts) over the trained crop [:T] instead of the full-crop .json value "
                         "— fixes the crop-length label mismatch (the .json scalar describes all 4096 frames).")
    ap.set_defaults(preencode_text=True, use_checkpointing=True)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 = base+adapters in bfloat16 (the supported ROCm path, ~2x "
                         "less memory); fp32 for max numerical stability")
    ap.add_argument("--smoke", action="store_true", help="3 steps, tiny, sanity only")
    ap.add_argument("--export-onnx-on-finish", action=argparse.BooleanOptionalAction, default=True,
                    help="after the final checkpoint is saved, shell out to the SA3 control-DiT ONNX "
                         "exporter to bake the trained adapter into a forward-only ONNX graph "
                         "(additive, non-fatal; --no-export-onnx-on-finish to disable)")
    ap.add_argument("--export-onnx-frames", type=int, default=256,
                    help="latent frame length (rung) for the on-finish ONNX export")
    args = ap.parse_args()

    if args.smoke:
        args.steps, args.batch, args.crop_frames, args.num_workers = 3, 1, 512, 0
        args.precision = "fp32"

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from stable_audio_3 import StableAudioModel
    print(f"[load] {args.model} ({args.precision})", flush=True)
    sam = StableAudioModel.from_pretrained(args.model, device=device, model_half=False)
    if dtype != torch.float32:
        sam.model.to(dtype)                                 # base in bf16
    dit = sam.model.model                                    # DiTWrapper -> DiffusionTransformer
    latent_rate = float(sam.model.sample_rate) / float(sam.model.pretransform.downsampling_ratio)
    crop_seconds = args.crop_frames / latent_rate

    # adapters + conditioner
    wrappers = install_adapters(sam, control_dim=args.control_dim)
    if args.control_mode == "scalar":
        cond_enc = ScalarAttributeEncoder(control_dim=args.control_dim,
                                          n_tokens=min(args.n_tokens, 16)).to(device=device, dtype=dtype)
        print(f"[control] scalar attribute '{args.scalar_field}' -> ScalarAttributeEncoder", flush=True)
    elif args.control_mode == "attribute":
        in_ch = CONTROL_DIMS[args.control_feature]
        if args.control_feature == "chroma384":                # chroma-aware (pitch-circular, 3 bands)
            from sa3_control.conditioner import ChromaAttributeEncoder
            cond_enc = ChromaAttributeEncoder(control_dim=args.control_dim,
                                              downsample=args.attr_downsample).to(device=device, dtype=dtype)
            print(f"[control] attribute 'chroma384' (3x128, pitch-circular, /{cond_enc.downsample}) "
                  f"-> ChromaAttributeEncoder", flush=True)
        else:
            cond_enc = AttributeEncoder(in_channels=in_ch, control_dim=args.control_dim,
                                        downsample=args.attr_downsample).to(device=device, dtype=dtype)
            print(f"[control] attribute '{args.control_feature}' ({in_ch}ch, /{cond_enc.downsample}) "
                  f"-> time-aligned AttributeEncoder", flush=True)
    else:
        cond_enc = AudioRefEncoder(latent_dim=256, control_dim=args.control_dim,
                                   n_tokens=args.n_tokens).to(device=device, dtype=dtype)
    if dtype != torch.float32:
        for w in wrappers:
            w.adapter.to(dtype)
    params = freeze_base_train_adapters(sam, wrappers, extra_trainable=[cond_enc])
    n_train = sum(p.numel() for p in params)
    n_base = sum(p.numel() for p in sam.model.parameters())
    print(f"[adapters] wrapped {len(wrappers)} cross-attn; trainable {n_train/1e6:.1f}M "
          f"of {n_base/1e6:.0f}M base ({100*n_train/n_base:.2f}%)", flush=True)
    if args.resume:                                          # warm-start (weights only; optimizer fresh)
        from sa3_control.generate import load_adapter_state
        load_adapter_state(torch.load(args.resume, map_location="cpu")["state"], wrappers, cond_enc)
        print(f"[resume] warm-started adapter+conditioner from {args.resume}", flush=True)

    if args.optimizer in ("fusion", "sfadamw", "fusion_nm", "fusion_full"):
        sys.path.append("/home/kim/Projects/SAO/stable-audio-tools")
        from stable_audio_tools.training.fusion_opt import FusionOpt
        from stable_audio_tools.training.fusion_groups import build_fusion_param_groups
        trainable_mod = torch.nn.ModuleList([w.adapter for w in wrappers] + [cond_enc])
        groups = build_fusion_param_groups(trainable_mod, spectral_wd=0.01, scalar_wd=0.0)
        comps = ({"sf"} if args.optimizer == "sfadamw"
                 else None if args.optimizer == "fusion_full"        # None = all 5 (mona+shampoo+ns5+normuon+sf)
                 else {"mona", "ns5", "normuon", "sf"} if args.optimizer == "fusion_nm"  # all but KL-Shampoo
                 else {"ns5", "normuon", "sf"})                      # fusion = SF-NorMuon
        opt = FusionOpt(groups, lr=args.lr, warmup_steps=args.warmup_steps, hot_dtype="bf16", components=comps)
        _sf = bool(getattr(opt, "uses_sf_averaging", False))
        if _sf:
            opt.train()
        print(f"[opt] {args.optimizer} (components={sorted(comps)}, SF={_sf})", flush=True)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        _sf = False

    start_step = 0
    if args.resume_exact:                                    # exact-state resume (weights + optimizer + step)
        from sa3_control.generate import load_adapter_state
        _ck = torch.load(args.resume_exact, map_location="cpu")
        if "opt" not in _ck:
            raise SystemExit(f"--resume-exact: {args.resume_exact} has no optimizer state "
                             "(saved before opt-state support); use --resume for warm-start instead.")
        load_adapter_state(_ck.get("model_train", _ck["state"]), wrappers, cond_enc)   # train-mode (y) iterate
        opt.load_state_dict(_ck["opt"])
        start_step = int(_ck.get("step", 0))
        if "torch_rng" in _ck:
            torch.set_rng_state(_ck["torch_rng"])            # main-process RNG (dataloader workers re-seed)
        print(f"[resume-exact] restored weights+optimizer+step from {args.resume_exact} "
              f"@ step {start_step} -> continues to {args.steps}", flush=True)

    if args.control_mode == "scalar" and args.scalar_from_timeseries:
        # scalar = window-mean of a timeseries feature over the trained crop [:T] (crop-correct).
        sf = args.scalar_from_timeseries
        ds = LatentControlDataset(args.encoded_dir, controls=(sf,), audio_ref=None,
                                  seed=args.seed, subset_tracks=args.subset_tracks)
        T0 = args.crop_frames
        samp = ds.paths[:: max(1, len(ds.paths) // 600)][:600]    # ~600-crop sample for stats
        wm = []
        for p in samp:
            try:
                z = np.load(p[:-4] + ".TIMESERIES.npz")
                arr = np.concatenate([z[f][None] if z[f].ndim == 1 else z[f].T for f in CONTROL_FIELDS[sf]], 0)
                wm.append(float(arr[:, :T0].mean()))
            except Exception:
                pass
        wm = np.array(wm, dtype=np.float32)
        ds.scalar_mean, ds.scalar_std = float(wm.mean()), float(wm.std() + 1e-8)
        print(f"[control] scalar from '{sf}' window-mean over [:{T0}]: mean {ds.scalar_mean:.4f} "
              f"std {ds.scalar_std:.4f} (n={len(wm)}); crop-correct", flush=True)
    elif args.control_mode == "scalar":
        ds = LatentControlDataset(args.encoded_dir, controls=(), audio_ref=None,
                                  seed=args.seed, subset_tracks=args.subset_tracks,
                                  scalar_field=args.scalar_field,
                                  random_crop_frames=(args.crop_frames if args.random_crop else None))
        vals = np.array([m[args.scalar_field] for m in ds.meta.values() if args.scalar_field in m], dtype=np.float32)
        ds.scalar_mean, ds.scalar_std = float(vals.mean()), float(vals.std())
        print(f"[control] {args.scalar_field}: mean {ds.scalar_mean:.3f} std {ds.scalar_std:.3f} "
              f"(n={len(vals)}); standardized at train time", flush=True)
    elif args.control_mode == "attribute":
        ds = LatentControlDataset(args.encoded_dir, controls=(args.control_feature,), audio_ref=None,
                                  seed=args.seed, subset_tracks=args.subset_tracks)
    else:
        ds = LatentControlDataset(args.encoded_dir, controls=(), audio_ref="same_track",
                                  seed=args.seed, subset_tracks=args.subset_tracks)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, collate_fn=collate,
                    # keep workers alive across epochs: respawning them every epoch
                    # stalled ~9 min/epoch (worker teardown/join), ~90 min/run wasted.
                    persistent_workers=(args.num_workers > 0))
    print(f"[data] {len(ds)} crops, {ds.track_stats()['tracks']} tracks; "
          f"crop {args.crop_frames}f ({crop_seconds:.1f}s)", flush=True)

    wb = None
    if args.wandb:
        try:
            import wandb as wb
            wb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
        except Exception as e:
            print(f"[wandb] disabled ({e})", flush=True)
            wb = None

    from sa3_control.telemetry import TrainTelemetry        # rich per-layer + trajectory + opt logging
    _telem_mod = torch.nn.ModuleList([w.adapter for w in wrappers] + [cond_enc])
    telem = TrainTelemetry(_telem_mod, wb, scalar_every=args.log_every,
                           traj_every=args.save_every, optimizer=opt)

    if args.preencode_text:
        preencode_text(sam, [ds.meta[p].get("prompt", "") for p in ds.paths], crop_seconds, device)

    os.makedirs(args.save_dir, exist_ok=True)
    prof = {"data": 0.0, "text": 0.0, "ref": 0.0, "fwd": 0.0, "bwd": 0.0, "opt": 0.0}

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    step = start_step
    t0 = time.time()
    t_iter = time.time()
    cond_enc.train()
    while step < args.steps:
        for b in dl:
            if args.profile:
                _sync(); _tm = time.time(); prof["data"] += _tm - t_iter
            T = args.crop_frames
            clean = b["latent"][:, :, :T].to(device=device, dtype=dtype)
            B = clean.shape[0]

            t = _sample_t(args.timestep_sampler, B, device)
            tb = t.view(B, 1, 1)
            noise = torch.randn_like(clean)
            noised = clean * (1 - tb) + noise * tb
            target = noise - clean                          # rectified-flow velocity

            if args.control_mode == "scalar" and args.scalar_from_timeseries:
                feat = b["controls"][args.scalar_from_timeseries][:, :, :T]    # (B, C, T)
                raw = feat.to(device=device, dtype=torch.float32).mean(dim=(1, 2))   # window-mean
                sc = ((raw - ds.scalar_mean) / ds.scalar_std).to(dtype)
                ctrl = cond_enc(sc)
            elif args.control_mode == "scalar":
                ctrl = cond_enc(b["scalar"].to(device=device, dtype=dtype))    # (B, n_tokens, control_dim)
            elif args.control_mode == "attribute":
                feat = b["controls"][args.control_feature][:, :, :T].to(device=device, dtype=dtype)  # (B,C,T)
                ctrl = cond_enc(feat)                                          # (B, T/ds, control_dim)
            else:
                ctrl = cond_enc(b["ref_latent"].to(device=device, dtype=dtype))
            if args.cfg_dropout > 0:                        # per-item control dropout
                drop = (torch.rand(B, device=device) < args.cfg_dropout).view(B, 1, 1)
                ctrl = ctrl.masked_fill(drop, 0.0)
            if args.profile:
                _sync(); _tr = time.time(); prof["ref"] += _tr - _tm

            cond_inputs = build_train_cond(sam, b["prompt"], crop_seconds, T, device, dtype)
            if args.profile:
                _sync(); _tt = time.time(); prof["text"] += _tt - _tr

            opt.zero_grad(set_to_none=True)
            # keep the control context active THROUGH backward: the DiT uses gradient
            # checkpointing, which re-runs the block forward during backward — the
            # adapter branch must see the same ContextVar on recompute or tensor counts mismatch.
            with use_control_context(ControlContext(ctrl)):
                v = dit(noised, t, **cond_inputs, cfg_scale=1.0, cfg_dropout_prob=0.0,
                        use_checkpointing=args.use_checkpointing)
                loss = torch.nn.functional.mse_loss(v.float(), target.float())
                if args.profile:
                    _sync(); _tf = time.time(); prof["fwd"] += _tf - _tt
                loss.backward()
            if args.profile:
                _sync(); _tb = time.time(); prof["bwd"] += _tb - _tf
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if args.warmup_steps > 0 and args.optimizer == "adamw":  # FusionOpt warms up internally
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * min(1.0, (step + 1) / args.warmup_steps)
            if hasattr(opt, "_telem_on"):       # FusionOpt: instrument the step we're about to log
                opt._telem_on = ((step + 1) % args.log_every == 0)
            opt.step()
            step += 1
            if args.profile:
                _sync(); t_iter = time.time(); prof["opt"] += t_iter - _tb
            else:
                t_iter = time.time()

            if step % args.log_every == 0 or args.smoke:
                rate = step / (time.time() - t0)
                print(f"[step {step}/{args.steps}] loss {loss.item():.4f} "
                      f"gnorm {float(gnorm):.3f} {rate:.2f} it/s", flush=True)
                if args.profile:
                    tot = sum(prof.values()) or 1.0
                    brk = "  ".join(f"{k}={v / args.log_every * 1000:.0f}ms/{100 * v / tot:.0f}%"
                                    for k, v in prof.items())
                    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
                    print(f"    [profile] per-step avg: {brk}  | peak VRAM {peak:.1f} GB", flush=True)
                    for k in prof:
                        prof[k] = 0.0
                telem.log(step, loss=loss.item(), gnorm=float(gnorm), it_s=rate,
                          lr=opt.param_groups[0]["lr"], epoch=step / max(1, len(ds)))
            if step % args.save_every == 0 and not args.smoke:
                p = os.path.join(args.save_dir, f"riffer_step{step}.pt")
                # capture exact-resume state in TRAIN mode (y iterate + opt state) BEFORE the SF eval swap
                _resume_state = {"model_train": adapter_state_dict(wrappers, cond_enc), "opt": opt.state_dict(),
                                 "step": step, "torch_rng": torch.get_rng_state()}
                if _sf:
                    opt.eval()                               # SF: save the averaged iterate
                torch.save({"state": adapter_state_dict(wrappers, cond_enc), "args": vars(args),
                            "control_mode": args.control_mode, "scalar_field": getattr(args, "scalar_field", None),
                            "control_feature": getattr(args, "control_feature", None),
                            "scalar_from_timeseries": getattr(args, "scalar_from_timeseries", ""),
                            "scalar_norm": [getattr(ds, "scalar_mean", 0.0), getattr(ds, "scalar_std", 1.0)],
                            **_resume_state}, p)
                if _sf:
                    opt.train()
                print(f"[save] {p}", flush=True)
                if args.save_cooldown and not args.smoke:
                    print(f"[cooldown] {args.save_cooldown:g}s GPU idle (thermal relief)", flush=True)
                    time.sleep(args.save_cooldown)
            if args.max_hours and (time.time() - t0) >= args.max_hours * 3600:
                print(f"[time] reached {args.max_hours}h limit at step {step}", flush=True)
                step = args.steps   # force the outer while to exit -> final save runs
                break
            if step >= args.steps:
                break

    if args.smoke:
        # confirm the adapters (not the base) received gradients
        g = [float(p.grad.norm()) for w in wrappers for p in w.adapter.parameters() if p.grad is not None]
        base_frozen = all(not p.requires_grad for w in wrappers for p in w.base_attention.parameters())
        base_no_grad = all(p.grad is None for w in wrappers for p in w.base_attention.parameters())
        print(f"[smoke OK] {len(g)} adapter tensors got grads; mean grad-norm {np.mean(g):.4e}; "
              f"base cross-attn frozen={base_frozen} got_no_grad={base_no_grad}", flush=True)
    else:
        _resume_state = {"model_train": adapter_state_dict(wrappers, cond_enc), "opt": opt.state_dict(),
                         "step": step, "torch_rng": torch.get_rng_state()}
        if _sf:
            opt.eval()                                       # SF: final save = averaged iterate
        torch.save({"state": adapter_state_dict(wrappers, cond_enc), "args": vars(args),
                            "control_mode": args.control_mode, "scalar_field": getattr(args, "scalar_field", None),
                            "control_feature": getattr(args, "control_feature", None),
                            "scalar_from_timeseries": getattr(args, "scalar_from_timeseries", ""),
                            "scalar_norm": [getattr(ds, "scalar_mean", 0.0), getattr(ds, "scalar_std", 1.0)],
                            **_resume_state},
                   os.path.join(args.save_dir, "riffer_final.pt"))
        if getattr(args, "export_onnx_on_finish", True) and not getattr(args, "smoke", False):
            try:
                export_control_onnx_on_finish(os.path.join(args.save_dir, "riffer_final.pt"),
                                              args.save_dir, args.export_onnx_frames,
                                              getattr(args, "scalar_field", None))
            except Exception as e:
                print(f"[export] hook failed ({e}); training already saved, continuing", flush=True)
    if wb:
        wb.finish()
    print(f"done -> {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
