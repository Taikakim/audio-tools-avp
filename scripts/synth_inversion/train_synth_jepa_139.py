"""Synth-JEPA 139-Parameter Training Script.

Reference: Hayes, Tian, Lattner, "Synth-JEPA: Joint Embedding Prediction for Renderer-Free
Synthesizer Parameter Search", arXiv:2609.31024 (Sept 2026).

Trains the full 139-parameter model on 3.0-second stereo procedural audio rendered
online by CPU workers in real-time, with audio effects strictly disabled.
"""

import argparse
import json
import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from audio_utils import ExactGpuMel
from surge_139_spec import (
    CAT_INDICES,
    CONT_INDICES,
    DURATION_S,
    NUM_PARAMS,
    SAMPLE_RATE,
    draw_patch_139,
    init_synth_139,
    patch_to_vector_139,
    render_patch_139,
)
from synth_jepa_model import (
    AudioTransformerEncoder,
    CrossDomainPredictor,
    PerceiverParameterEncoder,
    SIGRegLoss,
)


class WelfordNormalizer139:
    """Channel-wise (per mel band across stereo channels) running mean/variance."""
    def __init__(self, n_channels: int = 256, eps: float = 1e-5):
        self.count = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.M2 = np.zeros(n_channels, dtype=np.float64)
        self.eps = eps
        self.frozen = False
        self.frozen_mean = None
        self.frozen_std = None

    def update(self, mel_batch: np.ndarray):
        if self.frozen:
            raise RuntimeError("WelfordNormalizer is frozen")
        # mel_batch: [B, 256, T]
        B, C, T = mel_batch.shape
        x_flat = mel_batch.transpose(0, 2, 1).reshape(-1, C)
        n = x_flat.shape[0]
        batch_mean = np.mean(x_flat, axis=0)
        batch_m2 = np.sum((x_flat - batch_mean) ** 2, axis=0)

        new_count = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / new_count)
        self.M2 = self.M2 + batch_m2 + delta ** 2 * (self.count * n / new_count)
        self.count = new_count

    def freeze(self):
        if self.count < 2:
            raise RuntimeError("need at least 2 samples to freeze")
        var = self.M2 / (self.count - 1)
        self.frozen_mean = torch.from_numpy(self.mean.astype(np.float32)).view(1, -1, 1)
        self.frozen_std = torch.from_numpy(np.sqrt(var + self.eps).astype(np.float32)).view(1, -1, 1)
        self.frozen = True

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.frozen:
            raise RuntimeError("Cannot normalize with unfrozen Welford stats")
        mean = self.frozen_mean.to(mel.device)
        std = self.frozen_std.to(mel.device)
        return (mel - mean) / std

    def state_dict(self):
        return {"mean": self.frozen_mean, "std": self.frozen_std, "count": self.count}

    def load_state_dict(self, d):
        self.frozen_mean = d["mean"]
        self.frozen_std = d["std"]
        self.count = d["count"]
        self.frozen = True


class SurgeOnlineDataset139(Dataset):
    """Infinite online procedural dataset rendering 3.0s stereo audio on CPU workers."""
    def __init__(self, length: int = 160000, min_seed: int = 1_000_000):
        self.length = length
        self.min_seed = min_seed
        self._synth = None

    def __len__(self):
        return self.length

    def _get_synth(self):
        if self._synth is None:
            self._synth = init_synth_139()
        return self._synth

    def __getitem__(self, idx: int):
        seed = self.min_seed + idx
        patch = draw_patch_139(seed)
        synth = self._get_synth()
        audio = render_patch_139(synth, patch, patch["midi_note"], patch["note_dur"])
        p_vec = patch_to_vector_139(patch)
        return torch.from_numpy(audio), torch.from_numpy(p_vec)


class SynthJEPA139(nn.Module):
    """Full 139-Parameter Synth-JEPA model matching Hayes et al. (arXiv:2609.31024)."""
    def __init__(
        self,
        embed_dim: int = 512,
        predictor_hidden: int = 1024,
        num_audio_layers: int = 8,
        num_param_layers: int = 8,
        num_slices_sigreg: int = 64,
        ff_dim: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Stereo mel: 2 channels * 128 = 256 in_mels, 301 frames -> 150 conv tokens + [CLS]
        self.audio_encoder = AudioTransformerEncoder(
            in_mels=256,
            embed_dim=embed_dim,
            num_layers=num_audio_layers,
            num_heads=8,
            ff_dim=ff_dim,
            in_frames=301,
            token_stride=2,
        )
        cat_sizes = [k for _, k in CAT_INDICES]
        self.param_encoder = PerceiverParameterEncoder(
            num_continuous=len(CONT_INDICES),
            cat_sizes=cat_sizes,
            embed_dim=embed_dim,
            num_latents=32,
            num_blocks=num_param_layers,
            num_heads=8,
            ff_dim=ff_dim,
        )
        self.p_to_a = CrossDomainPredictor(embed_dim, predictor_hidden)
        self.a_to_p = CrossDomainPredictor(embed_dim, predictor_hidden)
        self.sigreg = SIGRegLoss(num_slices=num_slices_sigreg)

    def forward(self, mel: torch.Tensor, cont_params: torch.Tensor, cat_params: List[torch.Tensor]):
        za = self.audio_encoder(mel)
        zp = self.param_encoder(cont_params, cat_params)
        z_hat_a = self.p_to_a(zp)
        z_hat_p = self.a_to_p(za)
        return za, zp, z_hat_a, z_hat_p


def prepare_inputs_139(params: torch.Tensor):
    cont = params[:, CONT_INDICES] * 2.0 - 1.0
    cat_onehots = [
        F.one_hot(torch.round(params[:, i] * (k - 1)).clamp(0, k - 1).long(), k).float()
        for i, k in CAT_INDICES
    ]
    return cont, cat_onehots


def wsd_lambda(total_steps, warmup_frac=0.05, decay_frac=0.20):
    warmup = max(1, int(warmup_frac * total_steps))
    decay = max(1, int(decay_frac * total_steps))
    decay_start = total_steps - decay

    def f(step):
        if step < warmup:
            return step / warmup
        if step < decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / decay)
    return f


@torch.no_grad()
def evaluate_retrieval(model, val_loader, mel_extractor, normalizer, device, num_batches=20):
    model.eval()
    all_za, all_zp = [], []
    for i, (audio, p_vec) in enumerate(val_loader):
        if i >= num_batches:
            break
        audio = audio.to(device)
        mel_l = mel_extractor(audio[:, 0])
        mel_r = mel_extractor(audio[:, 1])
        stereo_mel = normalizer(torch.cat([mel_l, mel_r], dim=1))
        cont, cat = prepare_inputs_139(p_vec.to(device))

        za = model.audio_encoder(stereo_mel)
        zp = model.param_encoder(cont, cat)
        all_za.append(za)
        all_zp.append(zp)

    za_all = torch.cat(all_za, dim=0)  # [N, 512]
    zp_all = torch.cat(all_zp, dim=0)  # [N, 512]
    sim = torch.mm(F.normalize(za_all, dim=-1), F.normalize(zp_all, dim=-1).t())  # [N, N]
    targets = torch.arange(sim.shape[0], device=device)

    a_to_p_acc = (sim.argmax(dim=1) == targets).float().mean().item() * 100.0
    p_to_a_acc = (sim.argmax(dim=0) == targets).float().mean().item() * 100.0
    chance = (1.0 / sim.shape[0]) * 100.0
    return a_to_p_acc, p_to_a_acc, chance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="/run/media/kim/Mantu/surge_200k_models/synth_jepa_139_runs")
    parser.add_argument("--run_id", default="jepa_139_online_v1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--norm_spectrograms", type=int, default=8000)
    parser.add_argument("--samples_per_epoch", type=int, default=100000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    save_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"=== Full 139-Parameter Synth-JEPA Training on {device} ===")
    print(f"Checkpoints: {save_dir}")

    mel_extractor = ExactGpuMel(sample_rate=44100, n_fft=1102, hop_length=441, n_mels=128, device=device)

    train_ds = SurgeOnlineDataset139(length=args.samples_per_epoch, min_seed=2_000_000)
    val_ds = SurgeOnlineDataset139(length=2000, min_seed=1_000_000)

    mp_ctx = torch.multiprocessing.get_context("spawn") if args.num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True, multiprocessing_context=mp_ctx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(4, args.num_workers), pin_memory=True, drop_last=True, multiprocessing_context=mp_ctx)

    print(f"Calibrating channel-wise Welford normalizer on {args.norm_spectrograms} spectrograms...")
    normalizer = WelfordNormalizer139(n_channels=256)
    calib_ds = SurgeOnlineDataset139(length=args.norm_spectrograms, min_seed=500_000)
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False, num_workers=0)
    n_calib = 0
    with torch.no_grad():
        for audio, _ in calib_loader:
            audio = audio.to(device)
            m_l = mel_extractor(audio[:, 0])
            m_r = mel_extractor(audio[:, 1])
            stereo_mel = torch.cat([m_l, m_r], dim=1).cpu().numpy()
            normalizer.update(stereo_mel)
            n_calib += stereo_mel.shape[0]
            if n_calib >= args.norm_spectrograms:
                break
    normalizer.freeze()
    torch.save(normalizer.state_dict(), os.path.join(save_dir, "welford_stats.pt"))
    print("Welford statistics calibrated and frozen successfully.")

    model = SynthJEPA139(
        embed_dim=512,
        predictor_hidden=1024,
        num_audio_layers=8,
        num_param_layers=8,
        num_slices_sigreg=64,
        ff_dim=1024,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Instantiated SynthJEPA139: {n_params:.1f}M parameters (Hayes paper target: ~53M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lambda(total_steps))

    best_retrieval = 0.0
    step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "lp": 0.0, "la": 0.0, "sa": 0.0, "sp": 0.0}
        n_b = 0
        t0 = time.time()

        for audio, p_vec in train_loader:
            step += 1
            audio = audio.to(device, non_blocking=True)
            p_vec = p_vec.to(device, non_blocking=True)

            with torch.no_grad():
                mel_l = mel_extractor(audio[:, 0])
                mel_r = mel_extractor(audio[:, 1])
                stereo_mel = normalizer(torch.cat([mel_l, mel_r], dim=1))

            cont, cat = prepare_inputs_139(p_vec)

            za, zp, z_hat_a, z_hat_p = model(stereo_mel, cont, cat)

            loss_p = F.mse_loss(z_hat_a, za.detach())
            loss_a = F.mse_loss(z_hat_p, zp.detach())
            sigreg_a = model.sigreg(za)
            sigreg_p = model.sigreg(zp)

            loss = loss_p + loss_a + 0.1 * (sigreg_a + sigreg_p)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            sums["loss"] += loss.item()
            sums["lp"] += loss_p.item()
            sums["la"] += loss_a.item()
            sums["sa"] += sigreg_a.item()
            sums["sp"] += sigreg_p.item()
            n_b += 1

            if step % 200 == 0:
                dt = time.time() - t0
                speed = 200 * args.batch_size / dt
                print(f"Step {step:6d}/{total_steps} | Loss: {loss.item():.4f} (Lp={loss_p.item():.4f}, La={loss_a.item():.4f}, SigA={sigreg_a.item():.3f}) | {speed:.1f} samp/s")
                t0 = time.time()

        # End of epoch validation
        a_acc, p_acc, chance = evaluate_retrieval(model, val_loader, mel_extractor, normalizer, device)
        avg_retrieval = (a_acc + p_acc) / 2.0
        print(f"\n=== Epoch {epoch}/{args.epochs} Complete ===")
        print(f"Cross-Modal Retrieval: Audio->Param = {a_acc:.1f}%, Param->Audio = {p_acc:.1f}% (Chance = {chance:.2f}%)")

        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),
            "model_kwargs": {
                "embed_dim": 512,
                "predictor_hidden": 1024,
                "num_audio_layers": 8,
                "num_param_layers": 8,
                "num_slices_sigreg": 64,
                "ff_dim": 1024,
            },
            "welford_stats": normalizer.state_dict(),
            "retrieval_avg": avg_retrieval,
        }
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pt"))
        if avg_retrieval > best_retrieval:
            best_retrieval = avg_retrieval
            torch.save(ckpt, os.path.join(save_dir, "checkpoint_best.pt"))
            print(f"New best checkpoint saved! (Retrieval: {best_retrieval:.1f}%)")


if __name__ == "__main__":
    main()
