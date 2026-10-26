# Owner: ANTIGRAVITY <antigravity@aavepyora.online>
"""Training script for Stage 1 CGSP Extractor with Invariance and Selectivity Losses.

Implements the multi-loss objective from Tian et al. (arXiv:2610.19445v1, Section 3.2):
  - L_inv:   Paired-mixture InfoNCE for content invariance
  - L_algn:  Target alignment InfoNCE for audio and text cues
  - L_sel:   Selectivity hinge loss against mismatched cues
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cgsp_extractor import PointAndPlayExtractor
from mixture_dataset import PairedMixtureDataset


def info_nce_loss(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE contrastive loss."""
    # Ensure L2-normalized
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)

    logits_ab = torch.matmul(a, b.T) / temperature
    logits_ba = logits_ab.T

    B = a.shape[0]
    labels = torch.arange(B, device=a.device, dtype=torch.long)

    loss_ab = F.cross_entropy(logits_ab, labels)
    loss_ba = F.cross_entropy(logits_ba, labels)
    return 0.5 * (loss_ab + loss_ba)


def selectivity_hinge_loss(z_neg: torch.Tensor, target: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """Hinge loss pushing mismatched cue extraction away from target."""
    z_neg = F.normalize(z_neg, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)
    cosine_sim = (z_neg * target).sum(dim=-1)
    hinge = torch.relu(cosine_sim - margin)
    return hinge.mean()


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Initializing Stage 1 CGSP Extractor training on {device}...")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Extractor
    extractor = PointAndPlayExtractor(
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        device=str(device),
    ).to(device)

    print(f"Adapted {extractor.adapted_count} layers with LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}).")

    # Dataset & DataLoader
    dataset = PairedMixtureDataset(
        multitrack_dir=args.multitrack_dir,
        bank_v6_dir=args.bank_v6_dir,
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

    # Only optimize LoRA weights and CGSP parameters
    trainable_params = [p for p in extractor.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * float(step) / float(max(1, args.warmup_steps))
        progress = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))

    extractor.train()
    step = 0
    start_time = time.time()
    best_loss = float("inf")

    print(f"Starting training for {args.steps} steps...")

    for batch in dataloader:
        if step >= args.steps:
            break

        # Adjust LR
        curr_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

        mix_a = batch["mix_a"].to(device)
        mix_b = batch["mix_b"].to(device)
        target_clean = batch["target_clean"].to(device)
        cue_audio_pos = batch["cue_audio_pos"].to(device)
        cue_audio_neg = batch["cue_audio_neg"].to(device)
        text_pos = list(batch["text_pos"])

        # 1. Teacher embedding of clean target
        with torch.no_grad():
            t = extractor.encode_audio_cue(target_clean)
            q_audio_pos = extractor.encode_audio_cue(cue_audio_pos)
            q_audio_neg = extractor.encode_audio_cue(cue_audio_neg)
            q_text_pos = extractor.encode_text_cue(text_pos)

        # 2. Extract from Mixture A and Mixture B
        # Audio cue extraction on Mix A
        z_a_audio, g_a = extractor(mix_a, q_audio_pos)
        # Text cue extraction on Mix A
        z_a_text, _ = extractor(mix_a, q_text_pos)
        # Audio cue extraction on Mix B (paired content)
        _, g_b = extractor(mix_b, q_audio_pos)
        # Negative audio cue extraction on Mix A (for selectivity)
        z_a_neg, _ = extractor(mix_a, q_audio_neg)

        # 3. Compute loss terms
        loss_inv = info_nce_loss(g_a, g_b, temperature=args.temperature)
        loss_algn_a = info_nce_loss(z_a_audio, t, temperature=args.temperature)
        loss_algn_t = info_nce_loss(z_a_text, t, temperature=args.temperature)
        loss_sel = selectivity_hinge_loss(z_a_neg, t, margin=args.sel_margin)

        total_loss = (
            args.lambda_inv * loss_inv
            + args.lambda_algn_a * loss_algn_a
            + args.lambda_algn_t * loss_algn_t
            + args.lambda_sel * loss_sel
        )

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
                f"(Inv: {loss_inv.item():.4f}, Algn_A: {loss_algn_a.item():.4f}, Algn_T: {loss_algn_t.item():.4f}, Sel: {loss_sel.item():.4f}) "
                f"| LR: {curr_lr:.2e} | Speed: {steps_per_sec:.2f} step/s"
            )

        if step % args.save_interval == 0 or step == args.steps:
            ckpt_path = os.path.join(args.output_dir, f"cgsp_extractor_step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "lora_state": {k: v for k, v in extractor.state_dict().items() if "lora_" in k},
                    "cgsp_state": extractor.cgsp.state_dict(),
                    "args": vars(args),
                    "loss": total_loss.item(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_path = os.path.join(args.output_dir, "cgsp_extractor_best.pt")
                torch.save(
                    {
                        "step": step,
                        "lora_state": {k: v for k, v in extractor.state_dict().items() if "lora_" in k},
                        "cgsp_state": extractor.cgsp.state_dict(),
                        "args": vars(args),
                        "loss": best_loss,
                    },
                    best_path,
                )
                print(f"Saved best checkpoint to {best_path} (loss: {best_loss:.4f})")

    print(f"Training completed in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 CGSP Timbre Extractor")
    parser.add_argument("--steps", type=int, default=18000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature")
    parser.add_argument("--sel-margin", type=float, default=0.0, help="Selectivity hinge margin")
    parser.add_argument("--lambda-inv", type=float, default=1.0, help="Weight for L_inv")
    parser.add_argument("--lambda-algn-a", type=float, default=1.0, help="Weight for L_algn (audio)")
    parser.add_argument("--lambda-algn-t", type=float, default=1.0, help="Weight for L_algn (text)")
    parser.add_argument("--lambda-sel", type=float, default=0.25, help="Weight for L_sel")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--log-interval", type=int, default=50, help="Logging step interval")
    parser.add_argument("--save-interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--output-dir", type=str, default="/home/kim/sao_runs/cgsp_extractor_v7", help="Output directory")
    parser.add_argument("--multitrack-dir", type=str, default="/run/media/kim/Mantu/avp-stems-original-classified", help="Classified stems directory")
    parser.add_argument("--bank-v6-dir", type=str, default="/home/kim/sao_runs/bank_v6", help="Surge patch bank directory")
    parser.add_argument("--device", type=str, default="cuda", help="Target device")

    args = parser.parse_args()
    train(args)
