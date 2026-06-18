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
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from sa3_control.adapters import ControlContext, use_control_context
from sa3_control.conditioner import AudioRefEncoder
from sa3_control.dataset import LatentControlDataset
from sa3_control.inject import (adapter_state_dict, freeze_base_train_adapters,
                                install_adapters)


def collate(batch):
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "ref_latent": torch.stack([b["ref_latent"] for b in batch]),
        "prompt": [b["prompt"] for b in batch],
    }


def build_train_cond(sam, prompts, seconds, latent_T, device, dtype):
    """cond_inputs for a training batch (cfg_scale=1.0 path), mirroring generate()."""
    conditioning = [{"prompt": p, "seconds_total": float(seconds)} for p in prompts]
    ct = sam.model.conditioner(conditioning, device)
    B = len(prompts)
    io = sam.model.io_channels
    ct["inpaint_mask"] = [torch.zeros((B, 1, latent_T), device=device)]
    ct["inpaint_masked_input"] = [torch.zeros((B, io, latent_T), device=device)]
    ci = sam.model.get_conditioning_inputs(ct)
    return {k: (v.type(dtype) if torch.is_tensor(v) else v) for k, v in ci.items()}


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
    ap.add_argument("--cfg-dropout", type=float, default=0.1,
                    help="per-item probability of dropping the control tokens")
    ap.add_argument("--save-dir", default="/run/media/kim/Lehto/sa3_control_runs/riffer")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                    help="bf16 = base+adapters in bfloat16 (the supported ROCm path, ~2x "
                         "less memory); fp32 for max numerical stability")
    ap.add_argument("--smoke", action="store_true", help="3 steps, tiny, sanity only")
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

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    ds = LatentControlDataset(args.encoded_dir, controls=(), audio_ref="same_track", seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=args.num_workers, collate_fn=collate)
    print(f"[data] {len(ds)} crops, {ds.track_stats()['tracks']} tracks; "
          f"crop {args.crop_frames}f ({crop_seconds:.1f}s)", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    step = 0
    t0 = time.time()
    cond_enc.train()
    while step < args.steps:
        for b in dl:
            T = args.crop_frames
            clean = b["latent"][:, :, :T].to(device=device, dtype=dtype)
            ref = b["ref_latent"].to(device=device, dtype=dtype)
            B = clean.shape[0]

            t = torch.sigmoid(torch.randn(B, device=device)).clamp(1e-4, 1 - 1e-4)
            tb = t.view(B, 1, 1)
            noise = torch.randn_like(clean)
            noised = clean * (1 - tb) + noise * tb
            target = noise - clean                          # rectified-flow velocity

            ctrl = cond_enc(ref)                            # (B, n_tokens, control_dim)
            if args.cfg_dropout > 0:                        # per-item control dropout
                drop = (torch.rand(B, device=device) < args.cfg_dropout).view(B, 1, 1)
                ctrl = ctrl.masked_fill(drop, 0.0)

            cond_inputs = build_train_cond(sam, b["prompt"], crop_seconds, T, device, dtype)

            opt.zero_grad(set_to_none=True)
            # keep the control context active THROUGH backward: the DiT uses gradient
            # checkpointing, which re-runs the block forward during backward — the
            # adapter branch must see the same ContextVar on recompute or tensor counts mismatch.
            with use_control_context(ControlContext(ctrl)):
                v = dit(noised, t, **cond_inputs, cfg_scale=1.0, cfg_dropout_prob=0.0)
                loss = torch.nn.functional.mse_loss(v.float(), target.float())
                loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1

            if step % args.log_every == 0 or args.smoke:
                rate = step / (time.time() - t0)
                print(f"[step {step}/{args.steps}] loss {loss.item():.4f} "
                      f"gnorm {float(gnorm):.3f} {rate:.2f} it/s", flush=True)
            if step % args.save_every == 0 and not args.smoke:
                p = os.path.join(args.save_dir, f"riffer_step{step}.pt")
                torch.save({"state": adapter_state_dict(wrappers, cond_enc), "args": vars(args)}, p)
                print(f"[save] {p}", flush=True)
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
        torch.save({"state": adapter_state_dict(wrappers, cond_enc), "args": vars(args)},
                   os.path.join(args.save_dir, "riffer_final.pt"))
    print(f"done -> {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
