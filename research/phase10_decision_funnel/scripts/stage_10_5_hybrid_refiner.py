#!/usr/bin/env python3
"""
Stage 10.5: Hybrid Waveform Refiner Benchmark
=============================================
Tests whether a hybrid system (SAME latent separator draft + waveform refiner)
provides complementary gains over both:
  (A) Practical latent separator alone: b_draft = D(z_hat_b)
  (B) Practical hybrid separator: b_hat = b_draft + Delta_phi(b_draft, x_m, q)
  (C) Capacity-matched waveform-only baseline: b_hat = f_phi(x_m, q)

Evaluates:
  SDR, SI-SDR, and Multi-Resolution STFT distance across all 6 families.

Decision Rule:
  If hybrid (B) beats latent-only (A) by >= 1.0 dB AND beats waveform-only (C) by >= 0.5 dB,
  the hybrid architecture is validated as the recommended practical system.
  If (C) matches or beats (B), SAME latents provide no complementary value for waveform refinement.
"""

import os
import sys
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

def scale_invariant_sdr(reference, estimate, eps=EPS):
    ref = reference.detach().cpu().numpy() if isinstance(reference, torch.Tensor) else np.array(reference)
    est = estimate.detach().cpu().numpy() if isinstance(estimate, torch.Tensor) else np.array(estimate)
    min_len = min(ref.shape[-1], est.shape[-1])
    ref = ref[..., :min_len]
    est = est[..., :min_len]
    
    ref = ref - np.mean(ref, axis=-1, keepdims=True)
    est = est - np.mean(est, axis=-1, keepdims=True)
    
    dot = np.sum(ref * est)
    ref_energy = np.sum(ref ** 2) + eps
    alpha = dot / ref_energy
    
    e_target = alpha * ref
    e_res = est - e_target
    
    p_target = np.sum(e_target ** 2)
    p_res = np.sum(e_res ** 2) + eps
    if p_target < eps:
        return -100.0
    return float(10.0 * np.log10(p_target / p_res))

# ---------------------------------------------------------------------------
# Multi-Resolution STFT Metric & Loss
# ---------------------------------------------------------------------------

class MultiResolutionSTFT(nn.Module):
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

            mag_pred = torch.abs(stft_pred) + 1e-7
            mag_true = torch.abs(stft_true) + 1e-7

            sc_loss = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + 1e-7)
            log_loss = F.l1_loss(torch.log(mag_true), torch.log(mag_pred))
            loss += sc_loss + log_loss

        return loss / len(self.fft_sizes)

# ---------------------------------------------------------------------------
# Architecture: Waveform Refiner / Baseline
# ---------------------------------------------------------------------------

class WaveformDilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(4, channels)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))

class WaveformNetwork(nn.Module):
    def __init__(self, num_classes=32, in_audio_channels=4, embed_dim=16, hidden_dim=64, dilations=[1, 2, 4, 8, 16, 32, 1, 2]):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.in_proj = nn.Conv1d(in_audio_channels + embed_dim, hidden_dim, kernel_size=7, padding=3)
        self.blocks = nn.Sequential(*[WaveformDilatedBlock(hidden_dim, d) for d in dilations])
        self.out_head = nn.Conv1d(hidden_dim, 2, kernel_size=7, padding=3)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, audio_inputs, q):
        # audio_inputs: [B, 4, L]
        B, _, L = audio_inputs.shape
        e = self.class_embed(q).unsqueeze(2).expand(B, -1, L)
        x = torch.cat([audio_inputs, e], dim=1)
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out_head(h)

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
# Training Orchestrator for Waveform Models
# ---------------------------------------------------------------------------

def train_waveform_model(model_name, model, train_loader, val_loader, is_hybrid, device, epochs=20, lr=5e-4):
    print(f"\n--- Training Waveform Model: {model_name} (Hybrid={is_hybrid}, {epochs} epochs, lr={lr}) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mr_loss_fn = MultiResolutionSTFT().to(device)

    best_val_loss = float("inf")
    best_weights = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            mix_wav = batch["mix_wav"].to(device)
            bg_wav = batch["bg_wav"].to(device)
            b_draft = batch["b_draft"].to(device)
            q = batch["q"].to(device)

            if is_hybrid:
                audio_in = torch.cat([b_draft, mix_wav], dim=1) # [B, 4, L]
                delta = model(audio_in, q)
                b_hat = b_draft + delta
            else:
                zeros = torch.zeros_like(mix_wav)
                audio_in = torch.cat([zeros, mix_wav], dim=1) # [B, 4, L] capacity-matched
                b_hat = model(audio_in, q)

            loss = F.l1_loss(b_hat, bg_wav) + 0.5 * mr_loss_fn(b_hat, bg_wav)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                mix_wav = batch["mix_wav"].to(device)
                bg_wav = batch["bg_wav"].to(device)
                b_draft = batch["b_draft"].to(device)
                q = batch["q"].to(device)

                if is_hybrid:
                    audio_in = torch.cat([b_draft, mix_wav], dim=1)
                    delta = model(audio_in, q)
                    b_hat = b_draft + delta
                else:
                    zeros = torch.zeros_like(mix_wav)
                    audio_in = torch.cat([zeros, mix_wav], dim=1)
                    b_hat = model(audio_in, q)

                val_loss += (F.l1_loss(b_hat, bg_wav) + 0.5 * mr_loss_fn(b_hat, bg_wav)).item()
                n_val += 1

        val_loss /= n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] - Train: {train_loss:.4f}, Val: {val_loss:.4f} ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_weights)
    ckpt_path = CHECKPOINTS_DIR / f"stage10_5_{model_name}.pt"
    torch.save(best_weights, ckpt_path)
    print(f"  Saved best checkpoint to {ckpt_path}")
    return model

def main():
    print("=====================================================================")
    print("   STAGE 10.5: HYBRID WAVEFORM REFINER BENCHMARK")
    print("=====================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cache_file = CACHE_DIR / "phase9_residual_dataset.pt"
    dataset = torch.load(cache_file, map_location="cpu", weights_only=False)
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]

    all_fams = sorted(list(set(item["foreground_family"] for item in train_data + test_data)))
    fam_to_idx = {fam: idx for idx, fam in enumerate(all_fams)}
    num_classes = len(all_fams)

    # 1. Load SAME model and Stage 10.1 separator
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform
    same_ae.enable_grad = False

    sep_ckpt_path = CHECKPOINTS_DIR / "stage10_1_Obj_A_Latent_L1.pt"
    print(f"Loading Stage 10.1 winner checkpoint: {sep_ckpt_path}")
    latent_separator = DirectSeparator(num_classes=num_classes).to(device)
    latent_separator.load_state_dict(torch.load(sep_ckpt_path, map_location=device))
    latent_separator.eval()

    def build_cache_with_drafts(data_split, max_items=250):
        cache = []
        for item in data_split:
            mix_w, bg_w, _ = render_mixture_components(
                item["background_path"],
                item["foreground_path"],
                item["alpha_gain"],
                item["tau_offset_samples"]
            )
            if torch.sum(bg_w ** 2) > 1e-4:
                z_m = item["z_m"].unsqueeze(0).to(device)
                q = torch.tensor([fam_to_idx[item["foreground_family"]]], dtype=torch.long, device=device)
                with torch.no_grad():
                    z_hat_b = latent_separator(z_m, q)
                    b_draft = same_ae.decode(z_hat_b.half()).float().squeeze(0).cpu()
                
                # Align lengths
                min_l = min(b_draft.shape[-1], bg_w.shape[-1], mix_w.shape[-1])
                cache.append({
                    "mix_wav": mix_w[:, :min_l],
                    "bg_wav": bg_w[:, :min_l],
                    "b_draft": b_draft[:, :min_l],
                    "q": fam_to_idx[item["foreground_family"]],
                    "family": item["foreground_family"]
                })
            if len(cache) >= max_items:
                break
        return cache

    print("Pre-rendering audio tensors and latent drafts for training split...")
    train_cache = build_cache_with_drafts(train_data, max_items=250)
    val_cache = build_cache_with_drafts(val_data, max_items=80)
    print(f"Loaded {len(train_cache)} training items and {len(val_cache)} validation items.")

    def collate_fn(batch_list):
        return {
            "mix_wav": torch.stack([x["mix_wav"] for x in batch_list]),
            "bg_wav": torch.stack([x["bg_wav"] for x in batch_list]),
            "b_draft": torch.stack([x["b_draft"] for x in batch_list]),
            "q": torch.tensor([x["q"] for x in batch_list], dtype=torch.long)
        }

    train_loader = torch.utils.data.DataLoader(train_cache, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_cache, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # 2. Train Hybrid Refiner
    hybrid_refiner = WaveformNetwork(num_classes=num_classes).to(device)
    hybrid_refiner = train_waveform_model("Hybrid_Refiner", hybrid_refiner, train_loader, val_loader, is_hybrid=True, device=device, epochs=20, lr=5e-4)

    # 3. Train Capacity-Matched Waveform Baseline
    waveform_baseline = WaveformNetwork(num_classes=num_classes).to(device)
    waveform_baseline = train_waveform_model("Waveform_Baseline", waveform_baseline, train_loader, val_loader, is_hybrid=False, device=device, epochs=20, lr=5e-4)

    # -----------------------------------------------------------------------
    # Step 4: Comprehensive Test Evaluation
    # -----------------------------------------------------------------------
    print("\n=====================================================================")
    print("   STAGE 10.5 TEST EVALUATION (100 ACTIVE TEST MIXTURES)")
    print("=====================================================================")
    mrstft_metric = MultiResolutionSTFT().to(device)

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
                "q": fam_to_idx[item["foreground_family"]],
                "bg_wav": bg_w,
                "mix_wav": mix_w,
                "family": item["foreground_family"]
            })

    systems = [
        "Model_A_Latent_Only",
        "Model_B_Hybrid_Refiner",
        "Model_C_Waveform_Baseline"
    ]

    eval_sdrs = {s: [] for s in systems}
    eval_sisdrs = {s: [] for s in systems}
    eval_mrstft = {s: [] for s in systems}
    by_fam_sdr = {s: {f: [] for f in all_fams} for s in systems}
    mix_sdrs = []
    mix_sisdrs = []

    t_eval0 = time.time()
    for idx, item in enumerate(active_test_items):
        z_m = item["z_m"].unsqueeze(0).to(device)
        q = torch.tensor([item["q"]], dtype=torch.long, device=device)
        bg_wav = item["bg_wav"].unsqueeze(0).to(device)
        mix_wav = item["mix_wav"].unsqueeze(0).to(device)
        fam = item["family"]

        sdr_mix = scale_dependent_sdr(bg_wav[0], mix_wav[0])
        sisdr_mix = scale_invariant_sdr(bg_wav[0], mix_wav[0])
        mix_sdrs.append(sdr_mix)
        mix_sisdrs.append(sisdr_mix)

        with torch.no_grad():
            # A: Latent only
            z_hat_b = latent_separator(z_m, q)
            b_draft = same_ae.decode(z_hat_b.half()).float()
            min_l = min(b_draft.shape[-1], bg_wav.shape[-1], mix_wav.shape[-1])
            b_draft = b_draft[..., :min_l]
            bg_eval = bg_wav[..., :min_l]
            mix_eval = mix_wav[..., :min_l]

            # B: Hybrid Refiner
            audio_in_hybrid = torch.cat([b_draft, mix_eval], dim=1)
            delta = hybrid_refiner(audio_in_hybrid, q)
            b_hybrid = b_draft + delta

            # C: Waveform Baseline
            zeros = torch.zeros_like(mix_eval)
            audio_in_wave = torch.cat([zeros, mix_eval], dim=1)
            b_wave = waveform_baseline(audio_in_wave, q)

            candidates = {
                "Model_A_Latent_Only": b_draft,
                "Model_B_Hybrid_Refiner": b_hybrid,
                "Model_C_Waveform_Baseline": b_wave
            }

            for s_name, b_est in candidates.items():
                sdr = scale_dependent_sdr(bg_eval[0], b_est[0])
                sisdr = scale_invariant_sdr(bg_eval[0], b_est[0])
                mr_val = mrstft_metric(b_est, bg_eval).item()

                eval_sdrs[s_name].append(sdr)
                eval_sisdrs[s_name].append(sisdr)
                eval_mrstft[s_name].append(mr_val)
                by_fam_sdr[s_name][fam].append(sdr)

        if (idx + 1) % 25 == 0 or idx == len(active_test_items) - 1:
            print(f"  Evaluated [{idx+1:3d}/{len(active_test_items)}] test items ({time.time() - t_eval0:.1f}s)")

    mean_mix = float(np.mean(mix_sdrs))
    median_mix = float(np.median(mix_sdrs))
    sisdr_mix_mean = float(np.mean(mix_sisdrs))

    print("\n" + "=" * 92)
    print(f"{'System':<27} | {'Mean SDR':<10} | {'Median SDR':<11} | {'SI-SDR':<10} | {'MR-STFT Distance':<16} | {'Delta vs Mix'}")
    print("-" * 92)
    print(f"{'Untouched Mixture':<27} | {mean_mix:>6.2f} dB  | {median_mix:>7.2f} dB   | {sisdr_mix_mean:>6.2f} dB  | {'—':<16} | 0.00 dB")

    summary_out = {
        "mix_mean_sdr": mean_mix,
        "mix_median_sdr": median_mix,
        "mix_mean_sisdr": sisdr_mix_mean,
        "systems": {}
    }

    for s_name in systems:
        mean_sdr = float(np.mean(eval_sdrs[s_name]))
        median_sdr = float(np.median(eval_sdrs[s_name]))
        mean_sisdr = float(np.mean(eval_sisdrs[s_name]))
        mean_mr = float(np.mean(eval_mrstft[s_name]))
        delta_mix = mean_sdr - mean_mix

        print(f"{s_name:<27} | {mean_sdr:>6.2f} dB  | {median_sdr:>7.2f} dB   | {mean_sisdr:>6.2f} dB  | {mean_mr:>12.4f}     | {delta_mix:>+6.2f} dB")

        summary_out["systems"][s_name] = {
            "mean_sdr": mean_sdr,
            "median_sdr": median_sdr,
            "mean_sisdr": mean_sisdr,
            "mean_mrstft_dist": mean_mr,
            "delta_vs_mix": delta_mix,
            "by_fam": {f: float(np.mean(v)) for f, v in by_fam_sdr[s_name].items()}
        }

    print("=" * 92)

    # Decision Rule Check
    hybrid_sdr = summary_out["systems"]["Model_B_Hybrid_Refiner"]["mean_sdr"]
    latent_sdr = summary_out["systems"]["Model_A_Latent_Only"]["mean_sdr"]
    wave_sdr = summary_out["systems"]["Model_C_Waveform_Baseline"]["mean_sdr"]

    gain_over_latent = hybrid_sdr - latent_sdr
    gain_over_wave = hybrid_sdr - wave_sdr

    print("\n--- STAGE 10.5 DECISION GATE ---")
    print(f"Hybrid gain over Latent-only (A):  {gain_over_latent:+.2f} dB")
    print(f"Hybrid gain over Waveform-only (C): {gain_over_wave:+.2f} dB")

    if gain_over_latent >= 1.0 and gain_over_wave >= 0.5:
        decision = "VALIDATE_HYBRID: Hybrid refiner beats latent-only by >=1.0 dB AND beats waveform-only by >=0.5 dB. Hybrid architecture is validated."
        summary_out["gate_decision"] = "VALIDATE_HYBRID"
    elif wave_sdr >= hybrid_sdr:
        decision = "HYBRID_REJECTED: Waveform-only baseline matches or beats hybrid. SAME latents provide no complementary value for refinement."
        summary_out["gate_decision"] = "HYBRID_REJECTED"
    else:
        decision = f"HYBRID_PARTIAL: Hybrid improves over latent by {gain_over_latent:+.2f} dB and over wave by {gain_over_wave:+.2f} dB (modest synergy)."
        summary_out["gate_decision"] = "HYBRID_PARTIAL"

    print(f"\nDECISION RESULT: {decision}")
    summary_out["gain_over_latent"] = gain_over_latent
    summary_out["gain_over_waveform"] = gain_over_wave

    report_path = REPORTS_DIR / "stage_10_5_hybrid_refiner_results.json"
    with open(report_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved Stage 10.5 results to {report_path}")

if __name__ == "__main__":
    main()
