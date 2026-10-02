"""Train Synth-JEPA on Surge XT synthesizer data.

Reference: Hayes et al. (arXiv:2609.31024, Sept 2026):
"Synth-JEPA: Joint Embedding Prediction for Renderer-Free Synthesizer Parameter Search"

Loss:
  L = MSE(z_hat_p, sg(z_p)) + MSE(z_hat_a, sg(z_a)) + lambda_sig * (SIGReg(z_a) + SIGReg(z_p))
"""
import argparse
import json
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from surge_spec import CAT_INDICES, CONT_INDICES, NUM_PARAMS
from synth_jepa_model import SynthJEPA
from synth_jepa_search import WelfordNormalizer


class SurgeH5Dataset(Dataset):
    def __init__(self, h5_path, split="train", val_ratio=0.05):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            total_len = len(f["mel"])
        val_len = int(total_len * val_ratio)
        train_len = total_len - val_len
        self.indices = range(0, train_len) if split == "train" else range(train_len, total_len)
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        real_idx = self.indices[idx]
        mel = torch.from_numpy(self.h5_file["mel"][real_idx])       # [128, 81]
        params = torch.from_numpy(self.h5_file["params"][real_idx]) # [23]
        return mel, params


def prepare_inputs(params: torch.Tensor, cat_sizes):
    # params: [B, 23] in [0, 1]
    # Continuous controls mapped to [-1, 1]
    cont_raw = params[:, CONT_INDICES]  # [B, 20] in [0, 1]
    cont = cont_raw * 2.0 - 1.0         # [-1, 1]

    # Categorical controls mapped to one-hot vectors
    cat_onehots = []
    for (i, k), cat_dim in zip(CAT_INDICES, cat_sizes):
        cls = torch.round(params[:, i] * (k - 1)).clamp(0, k - 1).long()
        onehot = F.one_hot(cls, k).float()
        cat_onehots.append(onehot)

    return cont, cat_onehots


def main():
    parser = argparse.ArgumentParser(description="Train Synth-JEPA model")
    parser.add_argument("--h5_path", type=str, default="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5")
    parser.add_argument("--out_dir", type=str, default="/run/media/kim/Mantu/surge_200k_models/synth_jepa_runs")
    parser.add_argument("--run_id", type=str, default="synth_jepa_v1_20ep")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lambda_sig", type=float, default=1.0)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--burnin_batches", type=int, default=8000, help="Batches to estimate Welford stats before freezing")
    args = parser.parse_args()

    save_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    print(f"=== Synth-JEPA Training on {device} ===")
    print(f"Dataset: {args.h5_path}")
    print(f"Checkpoints: {save_dir}")

    train_ds = SurgeH5Dataset(args.h5_path, split="train")
    val_ds = SurgeH5Dataset(args.h5_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    cat_sizes = [k for _, k in CAT_INDICES]

    model = SynthJEPA(
        embed_dim=args.embed_dim,
        predictor_hidden=1024,
        num_audio_layers=args.num_layers,
        num_param_layers=args.num_layers,
        num_slices_sigreg=64,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count / 1e6:.2f}M")

    # Welford Online Normalizer
    normalizer = WelfordNormalizer(shape=(128, 81))
    norm_path = os.path.join(save_dir, "welford_stats.pt")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay down to 0.1 * lr
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step = 0
    burnin_count = 0
    print(f"Estimating input log-mel statistics for first {args.burnin_batches} batches...")

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_lp = 0.0
        epoch_la = 0.0
        epoch_siga = 0.0
        epoch_sigp = 0.0
        t0 = time.time()

        for mel_batch, p_batch in train_loader:
            step += 1

            # Welford update during burn-in
            if not normalizer.frozen:
                normalizer.update(mel_batch.numpy())
                burnin_count += 1
                if burnin_count >= args.burnin_batches:
                    normalizer.freeze()
                    print(f"\n[Step {step}] Welford statistics frozen! Saving to {norm_path}...")
                    torch.save({"mean": normalizer.frozen_mean, "std": normalizer.frozen_std}, norm_path)

            mel_batch = mel_batch.to(device)
            p_batch = p_batch.to(device)

            norm_mel = normalizer.normalize(mel_batch)
            cont, cat_onehots = prepare_inputs(p_batch, cat_sizes)

            optimizer.zero_grad()
            l_pred_p, l_pred_a, l_sig_a, l_sig_p, za, zp = model(norm_mel, cont, cat_onehots)

            loss = (l_pred_p + l_pred_a) + args.lambda_sig * (l_sig_a + l_sig_p)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            epoch_lp += float(l_pred_p.item())
            epoch_la += float(l_pred_a.item())
            epoch_siga += float(l_sig_a.item())
            epoch_sigp += float(l_sig_p.item())

            if step % 200 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"Ep {epoch:02d} | Step {step:05d} | Loss: {loss.item():.4f} "
                      f"(Pred_P: {l_pred_p.item():.4f}, Pred_A: {l_pred_a.item():.4f}, "
                      f"SIG_A: {l_sig_a.item():.4f}, SIG_P: {l_sig_p.item():.4f}) | LR: {cur_lr:.2e}")

        # If still not frozen at end of epoch, freeze now
        if not normalizer.frozen:
            normalizer.freeze()
            print(f"\nFreezing Welford normalizer after epoch {epoch}...")
            torch.save({"mean": normalizer.frozen_mean, "std": normalizer.frozen_std}, norm_path)

        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_lp = epoch_lp / n_batches
        avg_la = epoch_la / n_batches
        avg_sa = epoch_siga / n_batches
        avg_sp = epoch_sigp / n_batches
        dt = time.time() - t0

        # Validation
        model.eval()
        v_loss, v_lp, v_la, v_sa, v_sp = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            v_count = 0
            for v_mel, v_p in val_loader:
                v_mel = v_mel.to(device)
                v_p = v_p.to(device)
                v_norm_mel = normalizer.normalize(v_mel)
                v_cont, v_cat = prepare_inputs(v_p, cat_sizes)

                lp, la, sa, sp, _, _ = model(v_norm_mel, v_cont, v_cat)
                tot = (lp + la) + args.lambda_sig * (sa + sp)

                v_loss += float(tot.item())
                v_lp += float(lp.item())
                v_la += float(la.item())
                v_sa += float(sa.item())
                v_sp += float(sp.item())
                v_count += 1
                if v_count >= 50: # evaluate 50 batches (3200 samples)
                    break

        v_loss /= v_count
        v_lp /= v_count
        v_la /= v_count
        v_sa /= v_count
        v_sp /= v_count

        print(f"--- Epoch {epoch:02d} ({dt:.1f}s) ---")
        print(f"  Train: Loss {avg_loss:.4f} | Lp: {avg_lp:.4f}, La: {avg_la:.4f}, Sa: {avg_sa:.4f}, Sp: {avg_sp:.4f}")
        print(f"  Val:   Loss {v_loss:.4f} | Lp: {v_lp:.4f}, La: {v_la:.4f}, Sa: {v_sa:.4f}, Sp: {v_sp:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "val_loss": v_loss,
            "val_lp": v_lp,
            "val_la": v_la,
            "args": vars(args),
        }
        latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
        torch.save(ckpt, latest_path)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_path = os.path.join(save_dir, "checkpoint_best.pt")
            torch.save(ckpt, best_path)
            print(f"  [+] New best val loss: {best_val_loss:.4f} -> saved to checkpoint_best.pt")

        if epoch % 5 == 0 or epoch == args.epochs:
            ep_path = os.path.join(save_dir, f"checkpoint_ep{epoch:02d}.pt")
            torch.save(ckpt, ep_path)
            print(f"  [*] Periodic checkpoint saved: {ep_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
