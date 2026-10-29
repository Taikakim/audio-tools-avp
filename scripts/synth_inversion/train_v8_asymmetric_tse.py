#!/usr/bin/env python3
# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Phase 1: Asymmetric Target Sound Extractor (TSE) Training Pipeline.

Trains an Asymmetric Target Sound Extractor on authentic multitrack mixtures:
1. 2D Complex U-Net backbone with FiLM cue conditioning.
2. Dual outputs:
   - Estimated target audio S_hat(t) via complex ratio mask.
   - Target-to-interference reliability mask W(t, f) in [0, 1].
3. Single-target loss (minimizing ||S_hat - S_target|| without drum conservation).
4. Integrated GPU locking with --wait-gpu flag to automatically acquire /tmp/gpu.lock.
5. AdamW, with half the learning rate for 1-D parameters. (An earlier docstring said ModularOptimizer with a
   Conv2d kernel reshape; that is not implemented: KUANG 2026-10-29, docs/synth-v8-code-review-2026-10-29.md M4.)

The dataset now draws each positive cue from a different clip of the target's stem (C4); the stem caches must
carry stem ids (cache_v8_stems.py --split train and --split val).
"""

import argparse
import fcntl
import math
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from musical_dataset_v8 import MusicalDatasetV8


# ==============================================================================
# Model Architecture: 2D Complex U-Net with FiLM Cue Conditioning
# ==============================================================================

class FiLMBlock2d(nn.Module):
    """Applies Feature-wise Linear Modulation (FiLM) conditioned on cue embedding."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, cond_dim] -> [B, channels * 2]
        gamma_beta = self.proj(cond).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        return x * (1.0 + gamma) + beta


class ConvBlock2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: Optional[int] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.film = FiLMBlock2d(out_ch, cond_dim) if cond_dim else None

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.bn(self.conv(x))
        if self.film and cond is not None:
            h = self.film(h, cond)
        return self.act(h)


class CueEncoder(nn.Module):
    """Encodes 1.0s audio cue into a compact 256-d conditioning vector."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, out_dim, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, cue_wav: torch.Tensor) -> torch.Tensor:
        # cue_wav: [B, 1, samples] -> [B, out_dim]
        return self.net(cue_wav).squeeze(-1)


class AsymmetricTSEModel(nn.Module):
    """2D Complex U-Net predicting separated target audio and reliability mask W(t, f)."""

    def __init__(self, cond_dim: int = 256, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.cue_encoder = CueEncoder(out_dim=cond_dim)

        # Encoder: takes 2 input channels (Real, Imag of Mixture STFT)
        self.enc1 = ConvBlock2d(2, 32, cond_dim=cond_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock2d(32, 64, cond_dim=cond_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock2d(64, 128, cond_dim=cond_dim)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bot = ConvBlock2d(128, 256, cond_dim=cond_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock2d(256, 128, cond_dim=cond_dim)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock2d(128, 64, cond_dim=cond_dim)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock2d(64, 32, cond_dim=cond_dim)

        # Head 1: Complex ratio mask (Real, Imag)
        self.mask_head = nn.Conv2d(32, 2, kernel_size=1)

        # Head 2: Per-bin reliability mask W(t, f) in [0, 1]
        self.reliability_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, mix_wav: torch.Tensor, cue_wav: torch.Tensor):
        B, _, T = mix_wav.shape
        cond = self.cue_encoder(cue_wav)  # [B, cond_dim]

        # STFT: [B, F, T_frames, 2]
        window = torch.hann_window(self.n_fft, device=mix_wav.device)
        stft_mix = torch.stft(
            mix_wav.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=False,
        )
        # Permute to [B, 2, F, T_frames]
        x = stft_mix.permute(0, 3, 1, 2)
        orig_F, orig_T = x.shape[2], x.shape[3]

        # Pad to multiple of 8
        pad_F = (8 - (orig_F % 8)) % 8
        pad_T = (8 - (orig_T % 8)) % 8
        if pad_F > 0 or pad_T > 0:
            x = F.pad(x, (0, pad_T, 0, pad_F))

        # U-Net forward
        e1 = self.enc1(x, cond)
        e2 = self.enc2(self.pool1(e1), cond)
        e3 = self.enc3(self.pool2(e2), cond)
        b = self.bot(self.pool3(e3), cond)

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1), cond)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)

        # Unpad
        d1 = d1[:, :, :orig_F, :orig_T]

        # 1. Complex Masking
        c_mask = self.mask_head(d1)  # [B, 2, F, T_frames]
        # Complex multiplication: (X_r + j X_i) * (M_r + j M_i)
        x_r, x_i = x[:, 0, :orig_F, :orig_T], x[:, 1, :orig_F, :orig_T]
        m_r, m_i = c_mask[:, 0], c_mask[:, 1]
        out_r = x_r * m_r - x_i * m_i
        out_i = x_r * m_i + x_i * m_r

        out_stft = torch.complex(out_r, out_i)
        pred_wav = torch.istft(
            out_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=T,
        ).unsqueeze(1)

        # 2. Reliability Mask W(t, f)
        rel_mask = self.reliability_head(d1).squeeze(1)  # [B, F, T_frames]

        return pred_wav, rel_mask


# ==============================================================================
# Multi-Resolution STFT Loss
# ==============================================================================

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512), win_lengths=(512, 1024, 2048)):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win, device=pred.device)
            p_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            t_stft = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
            p_mag = torch.abs(p_stft)
            t_mag = torch.abs(t_stft)
            # Spectral convergence + Log-magnitude loss
            sc_loss = torch.norm(t_mag - p_mag, p="fro") / (torch.norm(t_mag, p="fro") + 1e-7)
            mag_loss = F.l1_loss(torch.log(p_mag + 1e-7), torch.log(t_mag + 1e-7))
            loss += sc_loss + mag_loss
        return loss / len(self.fft_sizes)


# ==============================================================================
# Training Harness with GPU Lock Management
# ==============================================================================

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


def main():
    parser = argparse.ArgumentParser(description="Train V8 Asymmetric Target Sound Extractor")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per worker")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="/home/kim/sao_runs/v8_phase1_tse")
    parser.add_argument("--wait-gpu", action="store_true", help="Wait on GPU lock until GPU becomes free")
    args = parser.parse_args()

    print("=" * 80)
    print("V8 Phase 1: Asymmetric Target Sound Extractor Training")
    print(f"Device: {args.device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print("=" * 80)

    # 1. Acquire GPU Guard if on CUDA
    if "cuda" in args.device:
        print("Checking GPU availability via gpu_guard.sh...")
        if not acquire_gpu_guard(wait=args.wait_gpu):
            print("GPU is currently busy. Refusing to start without --wait-gpu.")
            sys.exit(0)

    try:
        # 2. Setup Dataset & Dataloader
        train_ds = MusicalDatasetV8(split="train", epoch_length=2000)
        val_ds = MusicalDatasetV8(split="val", epoch_length=200)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

        # 3. Model & Loss
        model = AsymmetricTSEModel().to(args.device)
        stft_loss_fn = MultiResolutionSTFTLoss()

        # Two AdamW groups: weights (2-D and conv kernels) and 1-D parameters at half the lr
        spectral_params = []
        sign_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                spectral_params.append(p)
            else:
                sign_params.append(p)

        optimizer = torch.optim.AdamW(
            [{"params": spectral_params, "lr": args.lr}, {"params": sign_params, "lr": args.lr * 0.5}],
            weight_decay=1e-4,
        )

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print("Starting training loop with bfloat16 AMP...")

        best_val_loss = float("inf")

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            start_t = time.time()

            for step, batch in enumerate(train_loader):
                mix = batch["mix"].to(args.device, non_blocking=True)
                target = batch["target"].to(args.device, non_blocking=True)
                cue = batch["cue_pos"].to(args.device, non_blocking=True)

                optimizer.zero_grad()

                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if "cuda" in args.device else torch.float32):
                    pred_wav, rel_mask = model(mix, cue)

                    # Audio reconstruction loss
                    l1_loss = F.l1_loss(pred_wav, target)
                    mr_loss = stft_loss_fn(pred_wav, target)
                    audio_loss = l1_loss + mr_loss

                    # Reliability mask supervision (Target-to-Interference Ratio)
                    with torch.no_grad():
                        window = torch.hann_window(model.n_fft, device=mix.device)
                        t_stft = torch.abs(torch.stft(target.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length, window=window, return_complex=True))
                        m_stft = torch.abs(torch.stft(mix.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length, window=window, return_complex=True))
                        tir_target = (t_stft / (m_stft + 1e-6)).clamp(0.0, 1.0)
                    mask_loss = F.mse_loss(rel_mask, tir_target)

                    loss = audio_loss + 0.5 * mask_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                if (step + 1) % 50 == 0:
                    print(f"Epoch {epoch:02d} [{step+1:04d}/{len(train_loader)}] | Loss: {loss.item():.4f} (L1: {l1_loss.item():.4f}, MR: {mr_loss.item():.4f}, Mask: {mask_loss.item():.4f})")

            epoch_dur = time.time() - start_t
            mean_train_loss = train_loss / len(train_loader)
            print(f"--> Epoch {epoch:02d} complete in {epoch_dur:.1f}s | Train Loss: {mean_train_loss:.4f}")

            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_mix = v_batch["mix"].to(args.device)
                    v_target = v_batch["target"].to(args.device)
                    v_cue = v_batch["cue_pos"].to(args.device)

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16 if "cuda" in args.device else torch.float32):
                        v_pred, v_rel = model(v_mix, v_cue)
                        v_l1 = F.l1_loss(v_pred, v_target)
                        v_mr = stft_loss_fn(v_pred, v_target)
                        v_loss = v_l1 + v_mr
                    val_loss += v_loss.item()

            mean_val_loss = val_loss / len(val_loader)
            print(f"    Validation Loss: {mean_val_loss:.4f}")

            # Save epoch checkpoint
            ckpt_path = out_dir / f"tse_epoch_{epoch:02d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": optimizer.state_dict()}, ckpt_path)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_ckpt = out_dir / "tse_best.pt"
                torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": best_val_loss}, best_ckpt)
                print(f"    * New best validation loss! Saved {best_ckpt}")

    finally:
        if "cuda" in args.device:
            release_gpu_guard()


if __name__ == "__main__":
    main()
