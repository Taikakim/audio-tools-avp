# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Training script for Cue-Gated Temporal ASP on GPU (Step 1 Cheap Variant).

Trains CueGatedTemporalAudioEncoder on synthetic mixtures (bank_v6 patches + Aavepyora drum stems)
to align directly with the frozen Synth-JEPA teacher embeddings of clean Surge renders.
"""

import argparse
import glob
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import soundfile as sf
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_v4 import load_inverter_v4
from audio_utils import ExactGpuMel
from cue_gated_temporal_asp import CueGatedTemporalAudioEncoder


class SynthMixtureMelDataset(Dataset):
    """Dataset yielding pairs of clean Surge mel and drum-mixed mel."""

    def __init__(
        self,
        bank_dir: str = "/home/kim/sao_runs/bank_v6",
        drum_dir: str = "/run/media/kim/Mantu/avp-stems-original-classified",
        epoch_length: int = 40000,
        n_patches: int = 4096,
    ):
        super().__init__()
        self.epoch_length = epoch_length
        self.n_patches = n_patches

        # Load clean patch audio (mmap)
        audio_path = os.path.join(bank_dir, "main_00000_audio.npy")
        self.audio_raw = np.load(audio_path, mmap_mode="r")

        # Load drum stems
        self.drum_files = []
        if os.path.exists(drum_dir):
            for song in os.listdir(drum_dir):
                sdir = os.path.join(drum_dir, song)
                if os.path.isdir(sdir):
                    for sub in ["drums", "kick"]:
                        p = os.path.join(sdir, sub)
                        if os.path.exists(p):
                            self.drum_files.extend(glob.glob(os.path.join(p, "*.flac")) + glob.glob(os.path.join(p, "*.wav")))
        if not self.drum_files:
            print("Warning: no drum files found, will use noise as backing")

        # Pre-cache diverse drum slices to memory using fast soundfile
        self.cached_drums = []
        sampled_files = random.sample(self.drum_files, min(100, len(self.drum_files))) if self.drum_files else []
        for df in sampled_files:
            try:
                with sf.SoundFile(df) as s:
                    if s.samplerate == 44100 and s.frames > 35280:
                        for _ in range(3):
                            start = random.randint(44100 * 5, s.frames - 35280)
                            s.seek(start)
                            data = s.read(35280, dtype="float32")
                            if data.ndim == 2:
                                data = data.mean(axis=1)
                            if abs(data).max() > 0.05:
                                self.cached_drums.append(torch.from_numpy(data))
            except Exception:
                pass
        if not self.cached_drums:
            self.cached_drums = [torch.randn(35280) * 0.1]
        print(f"Cached {len(self.cached_drums)} diverse drum slices.")

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Pick a patch
        p_idx = random.randint(0, min(self.n_patches, self.audio_raw.shape[0]) - 1)
        clean_audio = torch.from_numpy(self.audio_raw[p_idx].astype(np.float32) / 32768.0)

        # Pick two different drum backings for paired mixtures
        drum_a = random.choice(self.cached_drums)
        drum_b = random.choice(self.cached_drums)

        # SNR scaling (0 dB to +6 dB synth advantage)
        synth_rms = clean_audio.pow(2).mean().sqrt() + 1e-6
        d_rms_a = drum_a.pow(2).mean().sqrt() + 1e-6
        d_rms_b = drum_b.pow(2).mean().sqrt() + 1e-6

        snr_gain_a = 10 ** (random.uniform(-0.3, 0.3))
        snr_gain_b = 10 ** (random.uniform(-0.3, 0.3))

        mix_a = clean_audio + drum_a * (synth_rms / d_rms_a) * snr_gain_a
        mix_b = clean_audio + drum_b * (synth_rms / d_rms_b) * snr_gain_b

        # Normalize to prevent clipping
        max_a = max(1.0, mix_a.abs().max().item())
        max_b = max(1.0, mix_b.abs().max().item())

        # Construct realistic positive cue:
        # 50% separated stem simulation (clean + residual drum leakage)
        # 30% clean + mild additive noise
        # 20% pure clean
        mode = random.random()
        if mode < 0.5:
            leakage_drum = random.choice(self.cached_drums)
            leak_gain = random.uniform(0.1, 0.4)
            cue_audio = clean_audio + leakage_drum * (synth_rms / (leakage_drum.pow(2).mean().sqrt() + 1e-6)) * leak_gain
        elif mode < 0.8:
            cue_audio = clean_audio + torch.randn_like(clean_audio) * (synth_rms * random.uniform(0.05, 0.2))
        else:
            cue_audio = clean_audio

        max_cue = max(1.0, cue_audio.abs().max().item())

        return {
            "clean": clean_audio / max_a,
            "mix_a": mix_a / max_a,
            "mix_b": mix_b / max_b,
            "pos_cue_audio": cue_audio / max_cue,
            "neg_cue_audio": drum_a,
        }


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Training Step 1 Cue-Gated ASP on {device} ===")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load frozen V6 C teacher model
    print(f"Loading V6 C flow model from {args.v6c_ckpt}...")
    flow = load_inverter_v4(args.v6c_ckpt, device=str(device)).eval()
    for p in flow.parameters():
        p.requires_grad = False

    base_encoder = flow.encoder
    mel_fn = ExactGpuMel(device=str(device))

    # 2. Build CueGatedTemporalAudioEncoder
    gated_encoder = CueGatedTemporalAudioEncoder(
        base_encoder=base_encoder,
        cue_dim=512,
        proj_dim=128,
        hidden_dim=512,
        freeze_base=True,
    ).to(device)

    trainable_params = [p for p in gated_encoder.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters in Cue-Gated ASP: {n_params:,}")

    # 3. Dataset & DataLoader
    dataset = SynthMixtureMelDataset(
        bank_dir=args.bank_dir,
        drum_dir=args.drum_dir,
        epoch_length=args.steps * args.batch_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

    gated_encoder.train()
    step = 0
    start_time = time.time()
    best_loss = float("inf")

    print(f"Launching training loop for {args.steps} steps (batch_size={args.batch_size})...")

    for batch in dataloader:
        if step >= args.steps:
            break

        curr_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        clean_wav = batch["clean"].to(device)
        mix_a_wav = batch["mix_a"].to(device)
        mix_b_wav = batch["mix_b"].to(device)
        pos_cue_wav = batch["pos_cue_audio"].to(device)
        neg_wav = batch["neg_cue_audio"].to(device)

        with torch.no_grad():
            mel_clean = mel_fn(clean_wav)      # [B, 128, 81]
            mel_mix_a = mel_fn(mix_a_wav)
            mel_mix_b = mel_fn(mix_b_wav)
            mel_pos_cue = mel_fn(pos_cue_wav)
            mel_neg = mel_fn(neg_wav)

            # Frozen teacher target: true clean embedding
            t_clean_packed = base_encoder(mel_clean)
            t_clean = t_clean_packed[:, :512]  # [B, 512]

            # Positive cue embedding (simulated separated / reference stem)
            pos_packed = base_encoder(mel_pos_cue)
            cue_pos = pos_packed[:, :512]

            # Negative cue (drum embedding)
            neg_packed = base_encoder(mel_neg)
            cue_neg = neg_packed[:, :512]

        # Forward passes
        out_a = gated_encoder(mel_mix_a, cue_pos)
        z_a = out_a[:, :512]

        out_b = gated_encoder(mel_mix_b, cue_pos)
        z_b = out_b[:, :512]

        out_neg = gated_encoder(mel_mix_a, cue_neg)
        z_neg = out_neg[:, :512]

        # 1. Alignment loss to true clean teacher
        loss_algn = F.mse_loss(z_a, t_clean) + (1.0 - F.cosine_similarity(z_a, t_clean).mean())

        # 2. Content invariance across different drum backings
        z_a_norm = F.normalize(z_a, p=2, dim=-1)
        z_b_norm = F.normalize(z_b, p=2, dim=-1)
        logits_inv = torch.matmul(z_a_norm, z_b_norm.T) / 0.1
        labels = torch.arange(args.batch_size, device=device)
        loss_inv = F.cross_entropy(logits_inv, labels)

        # 3. Selectivity hinge loss
        t_clean_norm = F.normalize(t_clean, p=2, dim=-1)
        z_neg_norm = F.normalize(z_neg, p=2, dim=-1)
        loss_sel = torch.relu((z_neg_norm * t_clean_norm).sum(dim=-1)).mean()

        total_loss = loss_algn + args.lambda_inv * loss_inv + args.lambda_sel * loss_sel

        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()

        step += 1

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(1, elapsed)
            print(
                f"Step {step:05d}/{args.steps} | Loss: {total_loss.item():.4f} "
                f"(Algn: {loss_algn.item():.4f}, Inv: {loss_inv.item():.4f}, Sel: {loss_sel.item():.4f}) "
                f"| LR: {curr_lr:.2e} | Speed: {steps_per_sec:.1f} step/s"
            )

        if step % args.save_interval == 0 or step == args.steps:
            ckpt_path = os.path.join(args.output_dir, f"cue_gated_asp_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model_state": gated_encoder.state_dict(),
                    "args": vars(args),
                    "loss": total_loss.item(),
                },
                ckpt_path,
            )
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_path = os.path.join(args.output_dir, "cue_gated_asp_best.pt")
                torch.save(
                    {
                        "step": step,
                        "model_state": gated_encoder.state_dict(),
                        "args": vars(args),
                        "loss": best_loss,
                    },
                    best_path,
                )
                print(f"Saved best checkpoint to {best_path} (loss: {best_loss:.4f})")

    print(f"Training completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Step 1 Cue-Gated ASP on GPU")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Min learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--lambda-inv", type=float, default=0.5, help="Weight for L_inv")
    parser.add_argument("--lambda-sel", type=float, default=0.25, help="Weight for L_sel")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--log-interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--output-dir", type=str, default="/home/kim/sao_runs/cue_gated_asp_v7", help="Output directory")
    parser.add_argument("--v6c-ckpt", type=str, default="/home/kim/sao_runs/v6_C/flow_best.pt", help="V6 C checkpoint")
    parser.add_argument("--bank-dir", type=str, default="/home/kim/sao_runs/bank_v6", help="Bank v6 directory")
    parser.add_argument("--drum-dir", type=str, default="/run/media/kim/Mantu/avp-stems-original-classified", help="Classified stems directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")

    args = parser.parse_args()
    train(args)
