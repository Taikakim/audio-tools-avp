#!/usr/bin/env python3
"""
Stage 10.2: Decoder Adaptation vs. Frozen Decoder Benchmark (Robust FP32 Master Adapters)
========================================================================================
Tests whether adapting the decoder (via LoRA or Residual Adapter) with a dual-batch
retention loss improves separation without breaking clean autoencoder reconstruction.

Evaluates 3 Configurations:
  Config A: Frozen Decoder (Baseline SAME Decoder)
  Config B: Decoder LoRA (r=8 on upper transformer blocks)
  Config C: Residual Adapter (512 -> 64 -> 512 on high-resolution latent features)

Dual-Batch Loss:
  L = L_sep + lambda_retain * L_retain
  L_sep = ||D_psi(z_hat_b) - b||_1 + 0.5 * L_MRSTFT(D_psi(z_hat_b), b)
  L_retain = ||D_psi(z_b) - b||_1 + 0.5 * L_MRSTFT(D_psi(z_b), b)

Decision Rule:
  If adapter gains > 1.0 dB on separation with < 0.5 dB penalty on clean reconstruction,
  decoder adaptation is viable. Otherwise, keep the decoder frozen.
"""

import os
import sys
import copy
import json
import time
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-3")
from stable_audio_3 import StableAudioModel

sys.path.insert(0, "/home/kim/evals_aac/same_latent_research/batch9_phase9_residual_diagnostics")
from build_phase9_residual_dataset import render_mixture_components

CACHE_DIR = Path("/home/kim/evals_aac/same_latent_research/cache")
REPORTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase10_decision_funnel/reports")
CHECKPOINTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase10_decision_funnel/checkpoints")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8

def scale_dependent_sdr(reference, estimate, eps=EPS):
    ref = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else np.array(reference)
    est = estimate.detach().cpu().numpy() if isinstance(estimate, torch.Tensor) else np.array(estimate)
    min_len = min(ref.shape[-1], est.shape[-1])
    ref = ref[..., :min_len]
    est = est[..., :min_len]
    
    p_target = np.sum(ref ** 2)
    p_noise = np.sum((est - ref) ** 2)
    if p_target < eps:
        return -100.0
    if p_noise < eps:
        return 50.0
    return float(10.0 * np.log10(p_target / (p_noise + eps)))

# ---------------------------------------------------------------------------
# Multi-Resolution STFT Loss with Safe Clamp (Avoids FP16 Overflow)
# ---------------------------------------------------------------------------

class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x_pred, x_true):
        loss = 0.0
        x_pred_mono = x_pred.mean(dim=1)
        x_true_mono = x_true.mean(dim=1)

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length, device=x_pred.device)
            stft_pred = torch.stft(x_pred_mono, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
            stft_true = torch.stft(x_true_mono, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)

            mag_pred = torch.clamp(torch.abs(stft_pred), min=1e-3)
            mag_true = torch.clamp(torch.abs(stft_true), min=1e-3)

            sc_loss = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + 1e-3)
            log_loss = F.l1_loss(torch.log(mag_true), torch.log(mag_pred))
            loss += sc_loss + log_loss

        return loss / len(self.fft_sizes)

# ---------------------------------------------------------------------------
# Architecture: DirectSeparator
# ---------------------------------------------------------------------------

class DirectSeparator(nn.Module):
    def __init__(self, num_classes=32, latent_dim=256, embed_dim=128, hidden_dim=256, num_layers=4):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.in_proj = nn.Conv1d(latent_dim + embed_dim, hidden_dim, 1)

        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
            ])
        self.trunk = nn.Sequential(*layers)
        self.out_head = nn.Conv1d(hidden_dim, latent_dim, 1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, z_m, q):
        e = self.class_embed(q).unsqueeze(2).repeat(1, 1, z_m.shape[2])
        x = torch.cat([z_m, e], dim=1)
        h = self.in_proj(x)
        h = h + self.trunk(h)
        delta = self.out_head(h)
        return z_m + delta

# ---------------------------------------------------------------------------
# Robust Float32 Master Weight Adapters
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, orig_linear, r=8, alpha=16.0):
        super().__init__()
        self.orig_linear = orig_linear
        self.r = r
        self.scaling = alpha / r
        in_features = orig_linear.in_features
        out_features = orig_linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=orig_linear.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=orig_linear.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.orig_linear.weight.requires_grad = False
        if self.orig_linear.bias is not None:
            self.orig_linear.bias.requires_grad = False

    def forward(self, x):
        orig_out = self.orig_linear(x)
        lora_out = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
        return orig_out + lora_out.to(x.dtype)

class DecoderResidualAdapter(nn.Module):
    def __init__(self, in_channels=512, bottleneck=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(bottleneck, in_channels, 3, padding=1)
        ).float()
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x.float()).to(x.dtype)

class AdaptedSAMEDecoder(nn.Module):
    def __init__(self, orig_decoder, adapter):
        super().__init__()
        self.orig_decoder = orig_decoder
        self.adapter = adapter

    def forward(self, x, **kwargs):
        h = self.orig_decoder(x, **kwargs)
        return self.adapter(h)

# ---------------------------------------------------------------------------
# Training Orchestrator for Decoder Adaptation
# ---------------------------------------------------------------------------

def train_adaptation(adaptation_type, sa_model, separator, train_loader, val_loader, device, epochs=15, lr=2e-4):
    print(f"\n--- Training Decoder Adaptation: {adaptation_type} ({epochs} epochs, lr={lr}) ---", flush=True)
    same_ae = sa_model.model.pretransform
    same_ae.enable_grad = True
    same_ae.model.pretransform.enable_grad = True

    # Freeze separator
    for p in separator.parameters():
        p.requires_grad = False
    separator.eval()

    # Freeze base decoder
    for p in same_ae.model.decoder.parameters():
        p.requires_grad = False

    adapter_params = []
    if adaptation_type == "LoRA":
        decoder = same_ae.model.decoder
        for i in range(6, 12):
            t_block = decoder.layers[3].transformers[i]
            t_block.self_attn.to_qkv = LoRALinear(t_block.self_attn.to_qkv, r=8)
            t_block.self_attn.to_out = LoRALinear(t_block.self_attn.to_out, r=8)
            adapter_params.extend([
                t_block.self_attn.to_qkv.lora_A, t_block.self_attn.to_qkv.lora_B,
                t_block.self_attn.to_out.lora_A, t_block.self_attn.to_out.lora_B
            ])
    elif adaptation_type == "ResidualAdapter":
        adapter = DecoderResidualAdapter(512, 64).to(device)
        same_ae.model.decoder = AdaptedSAMEDecoder(same_ae.model.decoder, adapter)
        adapter_params = list(adapter.parameters())
    else:
        raise ValueError(f"Unknown adaptation type: {adaptation_type}")

    print(f"Trainable adaptation parameters: {sum(p.numel() for p in adapter_params):,}", flush=True)

    optimizer = torch.optim.AdamW(adapter_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mr_loss_fn = MultiResolutionSTFTLoss().to(device)

    best_val_loss = float("inf")
    best_weights = None
    t0 = time.time()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            z_m = batch["z_m"].to(device)
            z_b = batch["z_b"].to(device)
            q = batch["q"].to(device)
            bg_wav = batch["bg_wav"].to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                z_hat_b = separator(z_m, q)

            # 1. Separation branch
            b_hat_wav = same_ae.decode(z_hat_b.half()).float()
            l_sep = F.l1_loss(b_hat_wav, bg_wav) + 0.5 * mr_loss_fn(b_hat_wav, bg_wav)

            # 2. Clean retention branch (z_b -> bg_wav)
            b_rec_wav = same_ae.decode(z_b.half()).float()
            l_retain = F.l1_loss(b_rec_wav, bg_wav) + 0.5 * mr_loss_fn(b_rec_wav, bg_wav)

            loss = l_sep + 1.0 * l_retain
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / n_batches

        # Validation
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                z_m = batch["z_m"].to(device)
                z_b = batch["z_b"].to(device)
                q = batch["q"].to(device)
                bg_wav = batch["bg_wav"].to(device)

                z_hat_b = separator(z_m, q)
                b_hat_wav = same_ae.decode(z_hat_b.half()).float()
                l_sep = F.l1_loss(b_hat_wav, bg_wav) + 0.5 * mr_loss_fn(b_hat_wav, bg_wav)

                b_rec_wav = same_ae.decode(z_b.half()).float()
                l_retain = F.l1_loss(b_rec_wav, bg_wav) + 0.5 * mr_loss_fn(b_rec_wav, bg_wav)

                val_loss += (l_sep + 1.0 * l_retain).item()
                n_val += 1

        val_loss /= n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if adaptation_type == "ResidualAdapter":
                best_weights = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            elif adaptation_type == "LoRA":
                best_weights = [p.cpu().clone() for p in adapter_params]

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] - Train: {train_loss:.4f}, Val: {val_loss:.4f} ({time.time() - t0:.1f}s)", flush=True)

    ckpt_path = CHECKPOINTS_DIR / f"stage10_2_{adaptation_type}.pt"
    torch.save(best_weights, ckpt_path)
    print(f"  Saved best {adaptation_type} checkpoint to {ckpt_path}", flush=True)
    return best_weights

def main():
    print("=====================================================================", flush=True)
    print("   STAGE 10.2: DECODER ADAPTATION VS FROZEN DECODER BENCHMARK", flush=True)
    print("=====================================================================", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    cache_file = CACHE_DIR / "phase9_residual_dataset.pt"
    dataset = torch.load(cache_file, map_location="cpu", weights_only=False)
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]

    all_fams = sorted(list(set(item["foreground_family"] for item in train_data + test_data)))
    fam_to_idx = {fam: idx for idx, fam in enumerate(all_fams)}
    num_classes = len(all_fams)

    sep_ckpt_path = CHECKPOINTS_DIR / "stage10_1_Obj_A_Latent_L1.pt"
    print(f"Loading Stage 10.1 winner checkpoint: {sep_ckpt_path}", flush=True)
    separator = DirectSeparator(num_classes=num_classes).to(device)
    separator.load_state_dict(torch.load(sep_ckpt_path, map_location=device))
    separator.eval()

    def prepare_batch_cache(data_split, max_items=250):
        cache = []
        for item in data_split:
            mix_w, bg_w, _ = render_mixture_components(
                item["background_path"],
                item["foreground_path"],
                item["alpha_gain"],
                item["tau_offset_samples"]
            )
            if torch.sum(bg_w ** 2) > 1e-4:
                cache.append({
                    "z_m": item["z_m"],
                    "z_b": item["z_b"],
                    "q": torch.tensor(fam_to_idx[item["foreground_family"]], dtype=torch.long),
                    "bg_wav": bg_w,
                    "mix_wav": mix_w,
                    "foreground_family": item["foreground_family"]
                })
            if len(cache) >= max_items:
                break
        return cache

    print("Pre-rendering audio tensors for training split...", flush=True)
    train_cache = prepare_batch_cache(train_data, max_items=250)
    val_cache = prepare_batch_cache(val_data, max_items=80)
    print(f"Loaded {len(train_cache)} training items and {len(val_cache)} validation items.", flush=True)

    def collate_fn(batch_list):
        return {
            "z_m": torch.stack([x["z_m"] for x in batch_list]),
            "z_b": torch.stack([x["z_b"] for x in batch_list]),
            "q": torch.stack([x["q"] for x in batch_list]),
            "bg_wav": torch.stack([x["bg_wav"] for x in batch_list])
        }

    train_loader = torch.utils.data.DataLoader(train_cache, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_cache, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Filter active test items
    active_test_items = []
    for item in test_data:
        mix_w, bg_w, _ = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )
        if torch.sum(bg_w ** 2) > 1e-4:
            active_test_items.append({
                "z_m": item["z_m"],
                "z_b": item["z_b"],
                "q": fam_to_idx[item["foreground_family"]],
                "bg_wav": bg_w,
                "mix_wav": mix_w,
                "family": item["foreground_family"]
            })
    print(f"Active test items: {len(active_test_items)}", flush=True)

    # -----------------------------------------------------------------------
    # Step 1: Base SAME Model Initialization & Frozen Eval
    # -----------------------------------------------------------------------
    print("\nInitializing base StableAudioModel...", flush=True)
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform
    same_ae.enable_grad = False

    print("\n--- Evaluating Config A: Baseline Frozen Decoder ---", flush=True)
    frozen_sep_sdrs = []
    frozen_clean_sdrs = []
    mix_sdrs = []
    by_fam_frozen_sep = {fam: [] for fam in all_fams}
    by_fam_frozen_clean = {fam: [] for fam in all_fams}

    t0 = time.time()
    for idx, item in enumerate(active_test_items):
        z_m = item["z_m"].unsqueeze(0).to(device)
        z_b = item["z_b"].unsqueeze(0).to(device)
        q = torch.tensor([item["q"]], dtype=torch.long, device=device)
        bg_wav = item["bg_wav"].to(device)
        mix_wav = item["mix_wav"].to(device)
        fam = item["family"]

        sdr_mix = scale_dependent_sdr(bg_wav, mix_wav)
        mix_sdrs.append(sdr_mix)

        with torch.no_grad():
            z_hat_b = separator(z_m, q)
            b_hat = same_ae.decode(z_hat_b.half()).float()
            sdr_sep = scale_dependent_sdr(bg_wav, b_hat)
            frozen_sep_sdrs.append(sdr_sep)
            by_fam_frozen_sep[fam].append(sdr_sep)

            b_rec = same_ae.decode(z_b.half()).float()
            sdr_clean = scale_dependent_sdr(bg_wav, b_rec)
            frozen_clean_sdrs.append(sdr_clean)
            by_fam_frozen_clean[fam].append(sdr_clean)

        if (idx + 1) % 25 == 0 or idx == len(active_test_items) - 1:
            print(f"  Frozen Evaluated [{idx+1:3d}/{len(active_test_items)}] ({time.time() - t0:.1f}s)", flush=True)

    mean_mix = float(np.mean(mix_sdrs))
    median_mix = float(np.median(mix_sdrs))
    frozen_sep_mean = float(np.mean(frozen_sep_sdrs))
    frozen_sep_med = float(np.median(frozen_sep_sdrs))
    frozen_clean_mean = float(np.mean(frozen_clean_sdrs))
    print(f"Frozen Decoder Baseline -> Sep Mean: {frozen_sep_mean:+.2f} dB, Clean AE Mean: {frozen_clean_mean:+.2f} dB", flush=True)

    # -----------------------------------------------------------------------
    # Step 2: Train & Evaluate Config B: Decoder LoRA
    # -----------------------------------------------------------------------
    del sa_model
    torch.cuda.empty_cache()
    sa_model_lora = StableAudioModel.from_pretrained("medium-base", device=device)
    train_adaptation("LoRA", sa_model_lora, separator, train_loader, val_loader, device, epochs=15, lr=1e-4)

    lora_sep_sdrs = []
    lora_clean_sdrs = []
    by_fam_lora_sep = {fam: [] for fam in all_fams}
    by_fam_lora_clean = {fam: [] for fam in all_fams}

    t0 = time.time()
    for idx, item in enumerate(active_test_items):
        z_m = item["z_m"].unsqueeze(0).to(device)
        z_b = item["z_b"].unsqueeze(0).to(device)
        q = torch.tensor([item["q"]], dtype=torch.long, device=device)
        bg_wav = item["bg_wav"].to(device)
        fam = item["family"]

        with torch.no_grad():
            z_hat_b = separator(z_m, q)
            b_hat = sa_model_lora.model.pretransform.decode(z_hat_b.half()).float()
            sdr_sep = scale_dependent_sdr(bg_wav, b_hat)
            lora_sep_sdrs.append(sdr_sep)
            by_fam_lora_sep[fam].append(sdr_sep)

            b_rec = sa_model_lora.model.pretransform.decode(z_b.half()).float()
            sdr_clean = scale_dependent_sdr(bg_wav, b_rec)
            lora_clean_sdrs.append(sdr_clean)
            by_fam_lora_clean[fam].append(sdr_clean)

        if (idx + 1) % 25 == 0 or idx == len(active_test_items) - 1:
            print(f"  LoRA Evaluated [{idx+1:3d}/{len(active_test_items)}] ({time.time() - t0:.1f}s)", flush=True)

    lora_sep_mean = float(np.mean(lora_sep_sdrs))
    lora_sep_med = float(np.median(lora_sep_sdrs))
    lora_clean_mean = float(np.mean(lora_clean_sdrs))
    print(f"LoRA Decoder -> Sep Mean: {lora_sep_mean:+.2f} dB, Clean AE Mean: {lora_clean_mean:+.2f} dB", flush=True)

    # -----------------------------------------------------------------------
    # Step 3: Train & Evaluate Config C: Residual Adapter
    # -----------------------------------------------------------------------
    del sa_model_lora
    torch.cuda.empty_cache()
    sa_model_adapt = StableAudioModel.from_pretrained("medium-base", device=device)
    train_adaptation("ResidualAdapter", sa_model_adapt, separator, train_loader, val_loader, device, epochs=15, lr=5e-4)

    adapt_sep_sdrs = []
    adapt_clean_sdrs = []
    by_fam_adapt_sep = {fam: [] for fam in all_fams}
    by_fam_adapt_clean = {fam: [] for fam in all_fams}

    t0 = time.time()
    for idx, item in enumerate(active_test_items):
        z_m = item["z_m"].unsqueeze(0).to(device)
        z_b = item["z_b"].unsqueeze(0).to(device)
        q = torch.tensor([item["q"]], dtype=torch.long, device=device)
        bg_wav = item["bg_wav"].to(device)
        fam = item["family"]

        with torch.no_grad():
            z_hat_b = separator(z_m, q)
            b_hat = sa_model_adapt.model.pretransform.decode(z_hat_b.half()).float()
            sdr_sep = scale_dependent_sdr(bg_wav, b_hat)
            adapt_sep_sdrs.append(sdr_sep)
            by_fam_adapt_sep[fam].append(sdr_sep)

            b_rec = sa_model_adapt.model.pretransform.decode(z_b.half()).float()
            sdr_clean = scale_dependent_sdr(bg_wav, b_rec)
            adapt_clean_sdrs.append(sdr_clean)
            by_fam_adapt_clean[fam].append(sdr_clean)

        if (idx + 1) % 25 == 0 or idx == len(active_test_items) - 1:
            print(f"  Residual Adapter Evaluated [{idx+1:3d}/{len(active_test_items)}] ({time.time() - t0:.1f}s)", flush=True)

    adapt_sep_mean = float(np.mean(adapt_sep_sdrs))
    adapt_sep_med = float(np.median(adapt_sep_sdrs))
    adapt_clean_mean = float(np.mean(adapt_clean_sdrs))
    print(f"Residual Adapter -> Sep Mean: {adapt_sep_mean:+.2f} dB, Clean AE Mean: {adapt_clean_mean:+.2f} dB", flush=True)

    # -----------------------------------------------------------------------
    # Step 4: Summary and Decision Gate
    # -----------------------------------------------------------------------
    print("\n" + "=" * 96, flush=True)
    print(f"{'Configuration':<26} | {'Sep Mean SDR':<12} | {'Sep Median':<10} | {'Clean AE Mean':<13} | {'Clean Penalty':<13} | {'Delta Sep'}", flush=True)
    print("-" * 96, flush=True)
    print(f"{'Untouched Mixture':<26} | {mean_mix:>8.2f} dB   | {median_mix:>6.2f} dB  | {'—':<13} | {'—':<13} | {'0.00 dB'}", flush=True)

    configs_summary = {
        "Config_A_Frozen_Decoder": {
            "sep_mean_sdr": frozen_sep_mean,
            "sep_median_sdr": frozen_sep_med,
            "clean_ae_mean_sdr": frozen_clean_mean,
            "clean_ae_penalty_vs_frozen": 0.0,
            "delta_sep_vs_frozen": 0.0,
            "by_fam_sep": {f: float(np.mean(v)) for f, v in by_fam_frozen_sep.items()},
            "by_fam_clean": {f: float(np.mean(v)) for f, v in by_fam_frozen_clean.items()}
        },
        "Config_B_Decoder_LoRA": {
            "sep_mean_sdr": lora_sep_mean,
            "sep_median_sdr": lora_sep_med,
            "clean_ae_mean_sdr": lora_clean_mean,
            "clean_ae_penalty_vs_frozen": frozen_clean_mean - lora_clean_mean,
            "delta_sep_vs_frozen": lora_sep_mean - frozen_sep_mean,
            "by_fam_sep": {f: float(np.mean(v)) for f, v in by_fam_lora_sep.items()},
            "by_fam_clean": {f: float(np.mean(v)) for f, v in by_fam_lora_clean.items()}
        },
        "Config_C_Residual_Adapter": {
            "sep_mean_sdr": adapt_sep_mean,
            "sep_median_sdr": adapt_sep_med,
            "clean_ae_mean_sdr": adapt_clean_mean,
            "clean_ae_penalty_vs_frozen": frozen_clean_mean - adapt_clean_mean,
            "delta_sep_vs_frozen": adapt_sep_mean - frozen_sep_mean,
            "by_fam_sep": {f: float(np.mean(v)) for f, v in by_fam_adapt_sep.items()},
            "by_fam_clean": {f: float(np.mean(v)) for f, v in by_fam_adapt_clean.items()}
        }
    }

    for name, c in configs_summary.items():
        print(f"{name:<26} | {c['sep_mean_sdr']:>8.2f} dB   | {c['sep_median_sdr']:>6.2f} dB  | {c['clean_ae_mean_sdr']:>9.2f} dB   | {c['clean_ae_penalty_vs_frozen']:>9.2f} dB   | {c['delta_sep_vs_frozen']:>+6.2f} dB", flush=True)

    print("=" * 96, flush=True)

    print("\n--- STAGE 10.2 DECISION GATE ---", flush=True)
    best_candidate = None
    best_gain = 0.0

    for candidate in ["Config_B_Decoder_LoRA", "Config_C_Residual_Adapter"]:
        gain = configs_summary[candidate]["delta_sep_vs_frozen"]
        penalty = configs_summary[candidate]["clean_ae_penalty_vs_frozen"]
        print(f"{candidate}: Delta Sep = {gain:+.2f} dB, Clean AE Penalty = {penalty:+.2f} dB", flush=True)
        if gain > 1.0 and penalty < 0.5:
            if gain > best_gain:
                best_gain = gain
                best_candidate = candidate

    summary_out = {
        "mix_mean_sdr": mean_mix,
        "mix_median_sdr": median_mix,
        "configurations": configs_summary
    }

    if best_candidate is not None:
        decision = f"ADAPT_DECODER: {best_candidate} achieved +{best_gain:.2f} dB gain with <0.5 dB clean penalty."
        summary_out["gate_decision"] = "ADAPT_DECODER"
        summary_out["selected_adaptation"] = best_candidate
    else:
        decision = "KEEP_FROZEN: Neither adapter met the threshold (>1.0 dB sep gain with <0.5 dB clean penalty). Proceeding with frozen decoder."
        summary_out["gate_decision"] = "KEEP_FROZEN"
        summary_out["selected_adaptation"] = "Config_A_Frozen_Decoder"

    print(f"\nDECISION RESULT: {decision}", flush=True)

    report_path = REPORTS_DIR / "stage_10_2_decoder_adaptation_results.json"
    with open(report_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved Stage 10.2 results to {report_path}", flush=True)

if __name__ == "__main__":
    main()
