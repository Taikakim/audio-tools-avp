#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Phase 2: Train Dynamic 10 Hz Parameter Inverter with Occam Curvature Regularization.

Trains DynamicParameterInverterV8 to simultaneously predict:
1. 59-parameter static patch core (Surge v6 specification).
2. 4 dynamic automation lanes (Cutoff, Resonance, Drive, FM Depth) at 10.76 Hz (43 frames).
3. Enforces second-order curvature penalty to prevent high-frequency chatter.
4. Manages GPU lock via gpu_guard.sh.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dynamic_dataset_v8 import DynamicSweepDataset
from dynamic_inverter_v8 import DynamicParameterInverterV8, dynamic_inverter_loss

GUARD_SCRIPT = "/home/kim/Projects/SAO/Misc/gpu_guard.sh"


def acquire_gpu_guard(wait: bool = False) -> bool:
    pid = str(os.getpid())
    env = os.environ.copy()
    env["TARGET_GPU"] = "0"
    while True:
        res = subprocess.run(["bash", GUARD_SCRIPT, "acquire", "ANTIGRAVITY", pid], env=env)
        if res.returncode == 0:
            return True
        if not wait:
            return False
        print("GPU busy. Waiting 15s before retry...")
        time.sleep(15)


def release_gpu_guard():
    subprocess.run(["bash", GUARD_SCRIPT, "release", "ANTIGRAVITY"])


def run_training(args):
    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("/home/kim/sao_runs/v8_dynamic_cache")
    train_cache = cache_dir / "dynamic_train.npz"
    val_cache = cache_dir / "dynamic_val.npz"

    if not train_cache.exists() or not val_cache.exists():
        print(f"Error: dynamic caches not found in {cache_dir}. Run dynamic_dataset_v8.py first.", file=sys.stderr)
        sys.exit(1)

    print("Loading dynamic sweep datasets...")
    train_ds = DynamicSweepDataset(train_cache)
    val_ds = DynamicSweepDataset(val_cache)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Datasets loaded: {len(train_ds)} train, {len(val_ds)} val samples.")

    # Model & Optimizer
    model = DynamicParameterInverterV8().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Starting training for {args.epochs} epochs with bfloat16 AMP...")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_start = time.time()
        train_loss = 0.0
        train_static_loss = 0.0
        train_dyn_loss = 0.0

        for batch in train_loader:
            mel = batch["mel"].to(device, non_blocking=True)
            t_static = batch["target_static"].to(device, non_blocking=True)
            t_dyn = batch["target_dynamic"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                p_static, p_dyn = model(mel)
                loss, metrics = dynamic_inverter_loss(
                    p_static, t_static, p_dyn, t_dyn,
                    lambda_smooth=args.lambda_smooth,
                    lambda_dynamic=args.lambda_dynamic,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += metrics["loss_total"]
            train_static_loss += metrics["loss_static"]
            train_dyn_loss += metrics["loss_dynamic"]

        scheduler.step()
        train_dur = time.time() - t_start
        n_train = len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_static_loss = 0.0
        val_dyn_loss = 0.0

        with torch.no_grad():
            for v_batch in val_loader:
                v_mel = v_batch["mel"].to(device)
                v_static = v_batch["target_static"].to(device)
                v_dyn = v_batch["target_dynamic"].to(device)

                with torch.amp.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
                    vp_static, vp_dyn = model(v_mel)
                    v_l, v_m = dynamic_inverter_loss(
                        vp_static, v_static, vp_dyn, v_dyn,
                        lambda_smooth=args.lambda_smooth,
                        lambda_dynamic=args.lambda_dynamic,
                    )
                val_loss += v_m["loss_total"]
                val_static_loss += v_m["loss_static"]
                val_dyn_loss += v_m["loss_dynamic"]

        n_val = len(val_loader)
        mean_val = val_loss / n_val
        mean_train = train_loss / n_train

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{train_dur:.1f}s] | "
            f"Train Loss: {mean_train:.4f} (Static: {train_static_loss/n_train:.4f}, Dyn: {train_dyn_loss/n_train:.4f}) | "
            f"Val Loss: {mean_val:.4f} (Static: {val_static_loss/n_val:.4f}, Dyn: {val_dyn_loss/n_val:.4f})"
        )

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_ckpt = out_dir / "dynamic_inverter_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": best_val_loss,
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"  * New best validation loss ({best_val_loss:.4f})! Saved {best_ckpt}")

    print("Phase 2 Dynamic Inverter Training Complete!")


def main():
    parser = argparse.ArgumentParser(description="Train V8 Dynamic Parameter Inverter")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lambda-smooth", type=float, default=0.5, help="Curvature smoothness loss weight")
    parser.add_argument("--lambda-dynamic", type=float, default=2.0, help="Dynamic lane loss weight")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="/home/kim/sao_runs/v8_phase2_inverter")
    parser.add_argument("--wait-gpu", action="store_true", help="Wait on GPU lock if busy")
    args = parser.parse_args()

    print("=" * 80)
    print("V8 Phase 2: Dynamic 10 Hz Parameter Inverter Training")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    if "cuda" in args.device:
        print("Checking GPU lock via gpu_guard.sh...")
        if not acquire_gpu_guard(wait=args.wait_gpu):
            print("GPU is currently busy. Exiting.")
            sys.exit(0)

    try:
        run_training(args)
    finally:
        if "cuda" in args.device:
            release_gpu_guard()


if __name__ == "__main__":
    main()
