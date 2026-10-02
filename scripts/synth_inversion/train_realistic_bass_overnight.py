"""Overnight Joint Training: Synth-JEPA & Deep Flow Inverter on Realistic Bass Manifold.

Trains both the Synth-JEPA joint embedding model and the Conditional Flow Matching
inverter concurrently on procedurally synthesized audio drawn directly from 430
real curated Surge XT bass presets.

GPU: AMD Radeon RX 9070 XT (cuda:0 / ROCm)
Target Duration: ~8 hours (~50,000 steps with batch_size=32 -> 1.6M samples)
Checkpoints saved to: /run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass/
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add local path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from audio_utils import ExactGpuMel
from models import build_model
import param_codec as codec
from realistic_bass_prior import draw_realistic_patch
from surge_spec import (
    CAT_INDICES,
    CONT_INDICES,
    DEFAULT_PLUGIN_PATH,
    NUM_PARAMS,
    PARAM_NAMES,
    domain_weights,
    init_synth,
    patch_to_vector,
    render_patch,
)
from synth_jepa_model import SynthJEPA
from synth_jepa_search import WelfordNormalizer

# Graceful termination handling
_STOP_REQUESTED = False
def _sig_handler(sig, frame):
    global _STOP_REQUESTED
    print(f"\n[!] Signal {sig} received. Saving checkpoint and terminating gracefully...")
    _STOP_REQUESTED = True

signal.signal(signal.SIGINT, _sig_handler)
signal.signal(signal.SIGTERM, _sig_handler)


class RealisticOnlineDataset(Dataset):
    """Procedurally renders realistic bass notes online using persistent per-worker Surge instances."""
    def __init__(self, length=2_000_000, plugin_path=None):
        self.length = length
        self.plugin_path = plugin_path or DEFAULT_PLUGIN_PATH
        self._synth = None

    def __len__(self):
        return self.length

    def _get_synth(self):
        if self._synth is None:
            self._synth = init_synth(self.plugin_path, verify=False)
        return self._synth

    def __getitem__(self, idx):
        patch, p_vec, midi_note, note_dur = draw_realistic_patch()
        synth = self._get_synth()
        audio = render_patch(synth, patch, midi_note, note_dur)
        return torch.from_numpy(audio), torch.from_numpy(p_vec)


def prepare_jepa_inputs(params: torch.Tensor):
    """[B, 23] stored params -> (continuous in [-1, 1] [B, 20], list of one-hot [B, k])."""
    cont = params[:, CONT_INDICES] * 2.0 - 1.0
    cat_onehots = [
        F.one_hot(torch.round(params[:, i] * (k - 1)).clamp(0, k - 1).long(), k).float()
        for i, k in CAT_INDICES
    ]
    return cont, cat_onehots


def compute_flow_loss(model, mel, params, w_enc):
    """Conditional Flow Matching loss."""
    x1 = codec.encode(params)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt, target = model.path(x0, x1, t)
    pred = model(xt, t, mel)
    return ((w_enc * (pred - target) ** 2).sum(dim=1) / w_enc.sum()).mean()


def main():
    parser = argparse.ArgumentParser(description="Overnight Realistic Bass Training")
    parser.add_argument("--out_dir", type=str, default="/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass")
    parser.add_argument("--hours", type=float, default=8.0, help="Training duration in hours")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_jepa", type=float, default=9e-5, help="30% of 3e-4")
    parser.add_argument("--lr_flow", type=float, default=6e-5, help="30% of 2e-4")
    parser.add_argument("--checkpoint_interval_steps", type=int, default=1000)
    parser.add_argument("--log_interval_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_file_path = os.path.join(args.out_dir, "training.log")
    log_file = open(log_file_path, "a", buffering=1)

    def log(msg: str):
        ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
        formatted = f"{ts} {msg}"
        print(formatted, flush=True)
        log_file.write(formatted + "\n")
        log_file.flush()

    device = torch.device(args.device)
    log("=" * 70)
    log(f"Starting Overnight Realistic Bass Training on {device}")
    log(f"Allocated time: {args.hours:.2f} hours (target finish: {time.ctime(time.time() + args.hours * 3600)})")
    log(f"Checkpoints directory: {args.out_dir}")
    log(f"Batch size: {args.batch_size}, Workers: {args.num_workers}")
    log("=" * 70)

    # 1. Initialize audio mel extractor (GPU native)
    mel_extractor = ExactGpuMel(device=device)

    # 2. Initialize Synth-JEPA model
    jepa_model = SynthJEPA(
        embed_dim=512,
        predictor_hidden=1024,
        num_audio_layers=8,
        num_param_layers=8,
        ff_dim=1024,
        in_frames=81,
    ).to(device)

    # 3. Initialize Deep Flow Inverter model
    flow_model = build_model(
        model_type="flow",
        hidden_dim=512,
        num_layers=8,
        encoding=codec.ENCODING_V2,
        time_scale=1000.0,
    ).to(device)

    # Optimizers
    opt_jepa = torch.optim.AdamW(jepa_model.parameters(), lr=args.lr_jepa, weight_decay=0.05)
    opt_flow = torch.optim.AdamW(flow_model.parameters(), lr=args.lr_flow, weight_decay=1e-4)

    # Flow loss encoding weights
    w23 = torch.from_numpy(domain_weights()).to(device)
    w_enc = codec.encoded_weights(w23)

    # 4. Create Dataset and DataLoader (workers will initialize their own clean synths)
    dataset = RealisticOnlineDataset(length=2_000_000)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    loader_iter = iter(loader)

    step = 0
    total_duration_s = args.hours * 3600.0
    start_time = time.time()
    end_time = start_time + total_duration_s

    # Check for existing latest checkpoint to resume
    latest_ckpt_path = os.path.join(args.out_dir, "checkpoint_latest.pt")
    if os.path.exists(latest_ckpt_path):
        log(f"Resuming from existing checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device)
        jepa_model.load_state_dict(ckpt["jepa_state_dict"])
        flow_model.load_state_dict(ckpt["flow_state_dict"])
        normalizer = WelfordNormalizer.from_state_dict(ckpt["normalizer"])
        step = ckpt.get("step", 0)
        elapsed_prev = ckpt.get("elapsed_s", 0.0)
        start_time = time.time() - elapsed_prev
        log(f"Resumed at step {step} (previous elapsed: {elapsed_prev/3600:.2f}h) with 30% LR!")
    else:
        # 5. Fit / Initialize Welford Normalizer from the first 8 batches of real synthesized mels
        log("Estimating Mel normalizer from empirical realistic bass batches...")
        pre_pass_batches = 8
        for _ in range(pre_pass_batches):
            aud_b, _ = next(loader_iter)
            mel_b = mel_extractor(aud_b.to(device)).cpu().numpy()
            normalizer.update(mel_b)
        normalizer.freeze()
        log("Mel normalizer initialized and frozen.")

    running_jepa_loss = 0.0
    running_flow_loss = 0.0
    running_r_a2p = 0.0
    running_r_p2a = 0.0
    best_jepa_loss = float("inf")
    best_flow_loss = float("inf")

    log("Entering main training loop...")
    loader_iter = iter(loader)

    while time.time() < end_time and not _STOP_REQUESTED:
        step_t0 = time.time()
        try:
            audio_raw, params_cpu = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            audio_raw, params_cpu = next(loader_iter)

        audio_raw = audio_raw.to(device, non_blocking=True)
        params = params_cpu.to(device, non_blocking=True)

        # Compute log-mel spectrogram on GPU
        mel_raw = mel_extractor(audio_raw)  # [B, 128, 81]
        mel_norm = normalizer.normalize(mel_raw)

        # --- A. Step Synth-JEPA ---
        opt_jepa.zero_grad(set_to_none=True)
        cont, cats = prepare_jepa_inputs(params)
        lp, la, sa, sp, za, zp = jepa_model(mel_norm, cont, cats)
        jepa_loss = lp + la + 1.0 * (sa + sp)
        jepa_loss.backward()
        nn.utils.clip_grad_norm_(jepa_model.parameters(), 1.0)
        opt_jepa.step()

        # Compute retrieval metric within batch
        with torch.no_grad():
            tgt = torch.arange(za.shape[0], device=device)
            r_a2p = (torch.cdist(jepa_model.f_a2p(za), zp).argmin(dim=1) == tgt).float().mean()
            r_p2a = (torch.cdist(jepa_model.f_p2a(zp), za).argmin(dim=1) == tgt).float().mean()

        # --- B. Step Deep Flow Inverter ---
        opt_flow.zero_grad(set_to_none=True)
        flow_l = compute_flow_loss(flow_model, mel_norm, params, w_enc)
        flow_l.backward()
        nn.utils.clip_grad_norm_(flow_model.parameters(), 1.0)
        opt_flow.step()

        # Metrics tracking
        step += 1
        running_jepa_loss += float(jepa_loss.item())
        running_flow_loss += float(flow_l.item())
        running_r_a2p += float(r_a2p.item())
        running_r_p2a += float(r_p2a.item())

        # Logging
        if step % args.log_interval_steps == 0:
            avg_jepa = running_jepa_loss / args.log_interval_steps
            avg_flow = running_flow_loss / args.log_interval_steps
            avg_a2p = running_r_a2p / args.log_interval_steps
            avg_p2a = running_r_p2a / args.log_interval_steps
            running_jepa_loss = 0.0
            running_flow_loss = 0.0
            running_r_a2p = 0.0
            running_r_p2a = 0.0

            elapsed = time.time() - start_time
            remaining = max(0.0, end_time - time.time())
            rate = step / elapsed

            log(
                f"Step {step:6d} | Elapsed: {elapsed/3600:5.2f}h | ETA: {remaining/3600:5.2f}h | "
                f"JEPA Loss: {avg_jepa:6.4f} | R(a->p): {avg_a2p*100:5.1f}% | R(p->a): {avg_p2a*100:5.1f}% | "
                f"Flow Loss: {avg_flow:6.4f} | Speed: {rate*args.batch_size:5.1f} smp/s"
            )

        # Checkpointing
        if step % args.checkpoint_interval_steps == 0 or _STOP_REQUESTED:
            ckpt_data = {
                "step": step,
                "elapsed_s": time.time() - start_time,
                "jepa_state_dict": jepa_model.state_dict(),
                "flow_state_dict": flow_model.state_dict(),
                "normalizer": normalizer.state_dict(),
                "model_kwargs_jepa": {
                    "embed_dim": 512, "predictor_hidden": 1024,
                    "num_audio_layers": 8, "num_param_layers": 8,
                    "ff_dim": 1024, "in_frames": 81,
                },
                "model_kwargs_flow": {
                    "model_type": "flow", "dim": 512, "layers": 8,
                    "time_scale": 1000.0, "param_encoding": "onehot_v2",
                }
            }
            # Save rolling checkpoint
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_step_{step:06d}.pt")
            torch.save(ckpt_data, ckpt_path)
            # Latest
            latest_path = os.path.join(args.out_dir, "checkpoint_latest.pt")
            torch.save(ckpt_data, latest_path)

            log(f"--> Saved checkpoint: {ckpt_path}")

    # Final wrap-up save
    final_path = os.path.join(args.out_dir, "checkpoint_final.pt")
    final_data = {
        "step": step,
        "elapsed_s": time.time() - start_time,
        "jepa_state_dict": jepa_model.state_dict(),
        "flow_state_dict": flow_model.state_dict(),
        "normalizer": normalizer.state_dict(),
    }
    torch.save(final_data, final_path)
    log("=" * 70)
    log(f"Overnight Training Completed at step {step} ({time.time() - start_time:.1f}s)")
    log(f"Final model saved to: {final_path}")
    log("=" * 70)
    log_file.close()


if __name__ == "__main__":
    main()
