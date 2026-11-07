#!/usr/bin/env python3
"""
Stage 10.1: Decoder-Aware Direct Separation Benchmark
=====================================================
Tests whether backpropagating through the frozen SAME decoder (with enable_grad=True)
allows waveform and spectral objectives to overcome latent-coordinate loss mismatch.

Evaluates 4 Objectives:
  A. Phase 9 Control: L_A = ||z_hat_b - z_b||_1
  B. Latent + Waveform L1: L_B = L_z + lambda_w * ||b_hat - b||_1
  C. Latent + Multi-Resolution STFT (Mag + SC): L_C = L_z + lambda_MR * L_MRSTFT
  D. Complex STFT (Combined): L_D = lambda_z * L_z + lambda_w * L_wave + lambda_s * L_MRSTFT + lambda_c * L_complex

Evaluates on all test mixtures and stratifies across 6 families:
kick, snare, hi-hat, percussion, bass, synth lead.
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

# ---------------------------------------------------------------------------
# Loss Modules
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

            mag_pred = torch.abs(stft_pred) + 1e-7
            mag_true = torch.abs(stft_true) + 1e-7

            sc_loss = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + 1e-7)
            log_loss = F.l1_loss(torch.log(mag_true), torch.log(mag_pred))
            loss += sc_loss + log_loss

        return loss / len(self.fft_sizes)

class ComplexSTFTLoss(nn.Module):
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

            diff = stft_pred - stft_true
            loss += (torch.abs(diff.real).mean() + torch.abs(diff.imag).mean())

        return loss / len(self.fft_sizes)

# ---------------------------------------------------------------------------
# Architecture: DirectSeparator (Matching Phase 9 ~1.25M params)
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
        # Initialize output to zero
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, z_m, q):
        e = self.class_embed(q).unsqueeze(2).repeat(1, 1, z_m.shape[2])
        x = torch.cat([z_m, e], dim=1)
        h = self.in_proj(x)
        h = h + self.trunk(h)
        delta = self.out_head(h)
        return z_m + delta # Residual parameterization around z_m

# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------

def train_objective(objective_name, model, train_loader, val_loader, same_ae, device, epochs=25, lr=5e-4):
    print(f"\n--- Training Objective: {objective_name} ({epochs} epochs, lr={lr}) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    mr_loss_fn = MultiResolutionSTFTLoss().to(device)
    complex_loss_fn = ComplexSTFTLoss().to(device)

    best_val_loss = float("inf")
    best_weights = None
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            z_m = batch["z_m"].to(device)
            z_b = batch["z_b"].to(device)
            q = batch["q"].to(device)
            bg_wav = batch["bg_wav"].to(device) # [B, 2, 131072]

            optimizer.zero_grad()
            z_hat_b = model(z_m, q)

            # Latent L1 Loss
            l_z = F.l1_loss(z_hat_b, z_b)

            if objective_name == "Obj_A_Latent_L1":
                loss = l_z
            else:
                # Decode through frozen decoder with gradients
                b_hat_wav = same_ae.decode(z_hat_b.half()).float()

                if objective_name == "Obj_B_Latent_WaveL1":
                    l_wave = F.l1_loss(b_hat_wav, bg_wav)
                    loss = l_z + 1.0 * l_wave
                elif objective_name == "Obj_C_Latent_MRSTFT":
                    l_mr = mr_loss_fn(b_hat_wav, bg_wav)
                    loss = l_z + 0.5 * l_mr
                elif objective_name == "Obj_D_Complex_Combined":
                    l_wave = F.l1_loss(b_hat_wav, bg_wav)
                    l_mr = mr_loss_fn(b_hat_wav, bg_wav)
                    l_c = complex_loss_fn(b_hat_wav, bg_wav)
                    loss = l_z + 0.5 * l_wave + 0.5 * l_mr + 0.5 * l_c

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
                z_m = batch["z_m"].to(device)
                z_b = batch["z_b"].to(device)
                q = batch["q"].to(device)
                bg_wav = batch["bg_wav"].to(device)

                z_hat_b = model(z_m, q)
                l_z = F.l1_loss(z_hat_b, z_b)
                if objective_name == "Obj_A_Latent_L1":
                    v_loss = l_z
                else:
                    b_hat_wav = same_ae.decode(z_hat_b.half()).float()
                    v_loss = l_z + mr_loss_fn(b_hat_wav, bg_wav)

                val_loss += v_loss.item()
                n_val += 1

        val_loss /= n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] - Train: {train_loss:.4f}, Val: {val_loss:.4f} ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_weights)
    ckpt_path = CHECKPOINTS_DIR / f"stage10_1_{objective_name}.pt"
    torch.save(best_weights, ckpt_path)
    print(f"  Saved best checkpoint to {ckpt_path}")
    return model

def main():
    print("=====================================================================")
    print("   STAGE 10.1: DECODER-AWARE DIRECT SEPARATION (4 OBJECTIVES)")
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

    print("\nInitializing SAME Autoencoder with pretransform.enable_grad = True...")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform
    same_ae.enable_grad = True
    same_ae.model.pretransform.enable_grad = True

    # Freeze decoder parameters
    for p in same_ae.model.decoder.parameters():
        p.requires_grad_(False)
    print("Decoder parameters frozen successfully. Gradient tracking enabled through decode path.")

    # Cache waveforms for training efficiency (first 250 active items each)
    def prepare_batch_cache(split_items, max_items=250):
        cache = []
        for item in split_items:
            mix_w, bg_w, fg_w = render_mixture_components(
                item["background_path"],
                item["foreground_path"],
                item["alpha_gain"],
                item["tau_offset_samples"]
            )
            # Filter active backgrounds for training
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

    print("Pre-rendering audio tensors for training split...")
    train_cache = prepare_batch_cache(train_data, max_items=250)
    val_cache = prepare_batch_cache(val_data, max_items=80)
    print(f"Loaded {len(train_cache)} training items and {len(val_cache)} validation items.")

    def collate_fn(batch_list):
        return {
            "z_m": torch.stack([x["z_m"] for x in batch_list]),
            "z_b": torch.stack([x["z_b"] for x in batch_list]),
            "q": torch.stack([x["q"] for x in batch_list]),
            "bg_wav": torch.stack([x["bg_wav"] for x in batch_list])
        }

    train_loader = torch.utils.data.DataLoader(train_cache, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_cache, batch_size=4, shuffle=False, collate_fn=collate_fn)

    objectives = [
        "Obj_A_Latent_L1",
        "Obj_B_Latent_WaveL1",
        "Obj_C_Latent_MRSTFT",
        "Obj_D_Complex_Combined"
    ]

    trained_models = {}
    for obj_name in objectives:
        torch.manual_seed(42)
        model = DirectSeparator(num_classes=num_classes).to(device)
        trained_models[obj_name] = train_objective(obj_name, model, train_loader, val_loader, same_ae, device, epochs=20, lr=5e-4)

    # ---------------------------------------------------------------------------
    # Evaluation on 100 Active Test Items & 6 Acoustic Families
    # ---------------------------------------------------------------------------
    print("\n=====================================================================")
    print("   STAGE 10.1 EVALUATION ON UNSEEN TEST MIXTURES")
    print("=====================================================================")

    # Load Stage 10.0 audit to select active test items
    with open(REPORTS_DIR / "stage_10_0_metric_audit_report.json") as f:
        audit_info = json.load(f)

    active_indices = [r["index"] for r in audit_info["records"] if not r["is_silent_bg"]]
    active_test_items = [test_data[i] for i in active_indices]
    print(f"Evaluating {len(active_test_items)} verified active test mixtures across 4 models...")

    results_by_model = {obj: {"sdrs": [], "by_fam": {f: [] for f in all_fams}} for obj in objectives}
    baseline_mix_sdrs = []
    baseline_mix_by_fam = {f: [] for f in all_fams}
    ae_ceiling_sdrs = []

    t0 = time.time()
    for idx, item in enumerate(active_test_items):
        mix_w, bg_w, target_w = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )

        fam = item["foreground_family"]
        sdr_mix = scale_dependent_sdr(bg_w, mix_w)
        baseline_mix_sdrs.append(sdr_mix)
        baseline_mix_by_fam[fam].append(sdr_mix)

        zm_gpu = item["z_m"].unsqueeze(0).to(device)
        zb_gpu = item["z_b"].unsqueeze(0).to(device)
        q_gpu = torch.tensor([fam_to_idx[fam]], device=device)

        with torch.no_grad():
            clean_ae = same_ae.decode(zb_gpu.half()).float()[0].cpu()
            ae_ceiling_sdrs.append(scale_dependent_sdr(bg_w, clean_ae))

            for obj in objectives:
                model = trained_models[obj]
                z_hat = model(zm_gpu, q_gpu)
                b_hat = same_ae.decode(z_hat.half()).float()[0].cpu()

                sdr_hat = scale_dependent_sdr(bg_w, b_hat)
                results_by_model[obj]["sdrs"].append(sdr_hat)
                results_by_model[obj]["by_fam"][fam].append(sdr_hat)

        if (idx + 1) % 25 == 0 or idx == len(active_test_items) - 1:
            print(f"  Evaluated [{idx+1:3d}/{len(active_test_items):3d}] test mixtures ({time.time() - t0:.1f}s)")

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"{'Objective / Model':<28} | {'Mean SDR':<10} | {'Median SDR':<11} | {'Delta vs Mix':<13} | {'Delta vs Obj A'}")
    print("-" * 90)
    mix_mean = np.mean(baseline_mix_sdrs)
    mix_med = np.median(baseline_mix_sdrs)
    print(f"{'Untouched Mixture':<28} | {mix_mean:+7.2f} dB | {mix_med:+7.2f} dB  | {'0.00 dB':<13} | {'—'}")

    obj_a_mean = np.mean(results_by_model["Obj_A_Latent_L1"]["sdrs"])
    for obj in objectives:
        m_sdr = np.mean(results_by_model[obj]["sdrs"])
        med_sdr = np.median(results_by_model[obj]["sdrs"])
        delta_mix = m_sdr - mix_mean
        delta_a = m_sdr - obj_a_mean
        print(f"{obj:<28} | {m_sdr:+7.2f} dB | {med_sdr:+7.2f} dB  | {delta_mix:+7.2f} dB   | {delta_a:+7.2f} dB")

    ceil_mean = np.mean(ae_ceiling_sdrs)
    print(f"{'Clean AE Reference D(E(b))':<28} | {ceil_mean:+7.2f} dB | {np.median(ae_ceiling_sdrs):+7.2f} dB  | {ceil_mean - mix_mean:+7.2f} dB   | {ceil_mean - obj_a_mean:+7.2f} dB")
    print("=" * 90)

    # Print Per-Family Breakdown
    print("\n--- Per-Family SDR Breakdown (Active Test Set) ---")
    print(f"{'Family':<15} | {'Mix SDR':<9} | {'Obj A (Latent)':<15} | {'Obj C (MRSTFT)':<15} | {'Obj D (Complex)':<15} | {'Delta D vs A'}")
    print("-" * 85)
    for fam in all_fams:
        f_mix = np.mean(baseline_mix_by_fam[fam])
        f_a = np.mean(results_by_model["Obj_A_Latent_L1"]["by_fam"][fam])
        f_c = np.mean(results_by_model["Obj_C_Latent_MRSTFT"]["by_fam"][fam])
        f_d = np.mean(results_by_model["Obj_D_Complex_Combined"]["by_fam"][fam])
        print(f"{fam:<15} | {f_mix:+6.2f} dB | {f_a:+6.2f} dB        | {f_c:+6.2f} dB        | {f_d:+6.2f} dB        | {f_d - f_a:+6.2f} dB")

    # Save Results
    output_json = REPORTS_DIR / "stage_10_1_decoder_aware_results.json"
    with open(output_json, "w") as f:
        json.dump({
            "mix_mean_sdr": float(mix_mean),
            "mix_median_sdr": float(mix_med),
            "clean_ae_ceiling_mean_sdr": float(ceil_mean),
            "objectives": {
                obj: {
                    "mean_sdr": float(np.mean(results_by_model[obj]["sdrs"])),
                    "median_sdr": float(np.median(results_by_model[obj]["sdrs"])),
                    "delta_vs_mix": float(np.mean(results_by_model[obj]["sdrs"]) - mix_mean),
                    "delta_vs_obj_a": float(np.mean(results_by_model[obj]["sdrs"]) - obj_a_mean),
                    "by_fam": {fam: float(np.mean(results_by_model[obj]["by_fam"][fam])) for fam in all_fams}
                } for obj in objectives
            }
        }, f, indent=2)
    print(f"\nSaved Stage 10.1 results to {output_json}")

if __name__ == "__main__":
    main()
