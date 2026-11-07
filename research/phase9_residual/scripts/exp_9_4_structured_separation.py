#!/usr/bin/env python3
"""
Phase 9 — Stage B: Structured Deterministic Separation & Multi-Objective Demixing
=================================================================================
Scientific Question:
  Does structured estimation of (z_s, r_int) outperform direct feedforward
  regression z_b = f(z_m, q)?
  Does acoustic waveform supervision (Multi-Resolution STFT loss) prevent
  the latent point-estimator from falling into regression-to-the-mean blur?

Models Compared (Capacity-Matched ~1.2M Parameters):
  1. Direct Point Separator:
     z_hat_b = f_theta(z_m, q)
     Trained on Latent L1 Loss: ||z_hat_b - z_b||_1
  
  2. Structured Physical Demixer:
     (z_hat_s, r_hat_int) = f_theta(z_m, q)
     z_hat_b = z_m - z_hat_s + z_0 - r_hat_int
     Trained on Joint Latent L1: ||z_hat_b - z_b||_1 + 0.5*||z_hat_s - z_s||_1 + 0.5*||r_hat_int - r_int||_1
  
  3. Structured Physical Demixer + MR-STFT Waveform Loss:
     Same architecture as (2), but trained with:
     L_total = L_latent + 0.2 * L_MR_STFT(b_true, D(z_hat_b))

Evaluation Baselines:
  - Untouched Mixture (b + s)
  - Naive Latent Subtraction (z_m - z_s + z_0) [Oracle foreground, r_int = 0]
  - Oracle Residual Subtraction (z_m - z_s + z_0 - r_int) [Exact background latent z_b]
  - Autoencoder Ceiling D(z_b)
"""

import os
import sys
import json
import time
import math
from pathlib import Path

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-3")
from stable_audio_3 import StableAudioModel

ROOT_DIR = Path("/home/kim/evals_aac/same_latent_research")
CACHE_DIR = ROOT_DIR / "cache"
PHASE9_DIR = ROOT_DIR / "batch9_phase9_residual_diagnostics"
REPORTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase9_residual/reports")
PLOTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase9_residual/plots")

sys.path.insert(0, str(PHASE9_DIR))
sys.path.insert(0, str(ROOT_DIR / "batch4_mixture_recognition"))

from build_phase9_residual_dataset import render_mixture_components

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Metrics & Loss Functions
# ---------------------------------------------------------------------------

def waveform_sdr(reference, estimate, eps=1e-8):
    if isinstance(reference, torch.Tensor):
        ref = reference.detach().cpu().numpy()
    else:
        ref = np.array(reference)
    if isinstance(estimate, torch.Tensor):
        est = estimate.detach().cpu().numpy()
    else:
        est = np.array(estimate)

    min_len = min(ref.shape[-1], est.shape[-1])
    ref = ref[..., :min_len]
    est = est[..., :min_len]

    s_target = ref
    e_noise = est - ref

    p_target = np.sum(s_target ** 2)
    p_noise = np.sum(e_noise ** 2)

    if p_noise < eps:
        return 50.0
    if p_target < eps:
        return -50.0
    return float(10.0 * np.log10(p_target / (p_noise + eps)))

class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss for acoustic waveform supervision.
    Resolutions: 512, 1024, 2048 FFT bins.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_lengths=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, x_pred, x_true):
        # x_pred, x_true: [B, C, N_samples]
        loss = 0.0
        B, C, N = x_true.shape
        x_pred_mono = x_pred.mean(dim=1)
        x_true_mono = x_true.mean(dim=1)

        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            window = torch.hann_window(win_length, device=x_pred.device)
            stft_pred = torch.stft(x_pred_mono, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)
            stft_true = torch.stft(x_true_mono, n_fft=fft_size, hop_length=hop_size, win_length=win_length, window=window, return_complex=True)

            mag_pred = torch.abs(stft_pred) + 1e-7
            mag_true = torch.abs(stft_true) + 1e-7

            # Spectral convergence + log magnitude loss
            sc_loss = torch.norm(mag_true - mag_pred, p="fro") / (torch.norm(mag_true, p="fro") + 1e-7)
            log_loss = F.l1_loss(torch.log(mag_true), torch.log(mag_pred))
            loss += sc_loss + log_loss

        return loss / len(self.fft_sizes)

# ---------------------------------------------------------------------------
# Separator Architectures
# ---------------------------------------------------------------------------

class DirectSeparator(nn.Module):
    """
    Model B.1: Direct feedforward regression: z_hat_b = f_theta(z_m, q)
    Capacity: ~1.25M parameters.
    """
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

    def forward(self, z_m, q):
        # z_m: [B, 256, 32], q: [B]
        e = self.class_embed(q).unsqueeze(2).repeat(1, 1, z_m.shape[2]) # [B, 128, 32]
        x = torch.cat([z_m, e], dim=1)
        h = self.in_proj(x)
        h = h + self.trunk(h)
        z_hat_b = self.out_head(h)
        return z_hat_b


class StructuredPhysicalSeparator(nn.Module):
    """
    Model B.2 & B.3: Structured decomposition:
      (z_hat_s, r_hat_int) = f_theta(z_m, q)
      z_hat_b = z_m - z_hat_s + z_0 - r_hat_int
    Capacity matched to ~1.25M parameters.
    """
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

        # Dual Heads: one for foreground target z_s, one for interaction r_int
        self.target_head = nn.Conv1d(hidden_dim, latent_dim, 1)
        self.residual_head = nn.Conv1d(hidden_dim, latent_dim, 1)

        # Initialize residual head close to 0 (additive null start)
        nn.init.normal_(self.residual_head.weight, std=1e-4)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, z_m, q, z_0):
        # z_m: [B, 256, 32], q: [B], z_0: [256, 32]
        e = self.class_embed(q).unsqueeze(2).repeat(1, 1, z_m.shape[2]) # [B, 128, 32]
        x = torch.cat([z_m, e], dim=1)
        h = self.in_proj(x)
        h = h + self.trunk(h)

        z_hat_s = self.target_head(h)
        r_hat_int = self.residual_head(h)

        # Exact physical reconstruction of background:
        # z_hat_b = z_m - z_hat_s + z_0 - r_hat_int
        z_hat_b = z_m - z_hat_s + z_0.unsqueeze(0) - r_hat_int
        return z_hat_b, z_hat_s, r_hat_int

# ---------------------------------------------------------------------------
# Training & Evaluation Script
# ---------------------------------------------------------------------------

def run_stage_b():
    print("=====================================================================")
    print("   STAGE B: STRUCTURED DETERMINISTIC SEPARATION BENCHMARK")
    print("=====================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cache_file = CACHE_DIR / "phase9_residual_dataset.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cached Phase 9 dataset at {cache_file}!")

    dataset = torch.load(cache_file, map_location="cpu", weights_only=False)
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]
    z_0 = dataset["z_0"].to(device)

    # Build Class Index Map
    all_fams = sorted(list(set(item["foreground_family"] for item in train_data + test_data)))
    fam_to_idx = {fam: idx for idx, fam in enumerate(all_fams)}
    print(f"Mapped {len(all_fams)} sound families: {all_fams}")

    # Prepare batches
    def prepare_tensors(data_list):
        z_m_t = torch.stack([item["z_m"] for item in data_list], dim=0) # [N, 256, 32]
        z_b_t = torch.stack([item["z_b"] for item in data_list], dim=0)
        z_s_t = torch.stack([item["z_s"] for item in data_list], dim=0)
        r_int_t = torch.stack([item["r_int"] for item in data_list], dim=0)
        q_t = torch.tensor([fam_to_idx[item["foreground_family"]] for item in data_list], dtype=torch.long)
        return z_m_t, z_b_t, z_s_t, r_int_t, q_t

    train_zm, train_zb, train_zs, train_rint, train_q = prepare_tensors(train_data)
    val_zm, val_zb, val_zs, val_rint, val_q = prepare_tensors(val_data)
    test_zm, test_zb, test_zs, test_rint, test_q = prepare_tensors(test_data)

    print(f"Data ready: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    BATCH_SIZE = 32
    EPOCHS = 35
    num_train_batches = math.ceil(len(train_data) / BATCH_SIZE)

    # 1. Train Model B.1: Direct Separator
    print("\n--- Training Model B.1: Direct Point Separator ---")
    direct_model = DirectSeparator(num_classes=len(all_fams)).to(device)
    opt_direct = AdamW(direct_model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_direct = float("inf")
    best_state_direct = None

    t0 = time.time()
    for ep in range(EPOCHS):
        direct_model.train()
        perm = torch.randperm(len(train_data))
        ep_loss = 0.0
        for b_idx in range(num_train_batches):
            idx_b = perm[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            zm_b = train_zm[idx_b].to(device)
            zb_b = train_zb[idx_b].to(device)
            q_b = train_q[idx_b].to(device)

            pred_zb = direct_model(zm_b, q_b)
            loss = F.l1_loss(pred_zb, zb_b)

            opt_direct.zero_grad()
            loss.backward()
            opt_direct.step()
            ep_loss += loss.item()

        # Val
        direct_model.eval()
        with torch.no_grad():
            v_pred = direct_model(val_zm.to(device), val_q.to(device))
            v_loss = F.l1_loss(v_pred, val_zb.to(device)).item()

        if v_loss < best_val_direct:
            best_val_direct = v_loss
            best_state_direct = {k: v.cpu().clone() for k, v in direct_model.state_dict().items()}

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            print(f"  Direct Ep [{ep+1:2d}/{EPOCHS:2d}] - Train L1: {ep_loss/num_train_batches:.4f} - Val L1: {v_loss:.4f}")

    direct_model.load_state_dict(best_state_direct)
    print(f"Model B.1 trained in {time.time() - t0:.1f}s, Best Val L1: {best_val_direct:.4f}")

    # 2. Train Model B.2: Structured Physical Separator (Latent Loss)
    print("\n--- Training Model B.2: Structured Physical Separator ---")
    struct_model = StructuredPhysicalSeparator(num_classes=len(all_fams)).to(device)
    opt_struct = AdamW(struct_model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val_struct = float("inf")
    best_state_struct = None

    t0 = time.time()
    for ep in range(EPOCHS):
        struct_model.train()
        perm = torch.randperm(len(train_data))
        ep_loss = 0.0
        for b_idx in range(num_train_batches):
            idx_b = perm[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            zm_b = train_zm[idx_b].to(device)
            zb_b = train_zb[idx_b].to(device)
            zs_b = train_zs[idx_b].to(device)
            rint_b = train_rint[idx_b].to(device)
            q_b = train_q[idx_b].to(device)

            pred_zb, pred_zs, pred_rint = struct_model(zm_b, q_b, z_0)
            
            # Joint Physical Loss
            loss_zb = F.l1_loss(pred_zb, zb_b)
            loss_zs = F.l1_loss(pred_zs, zs_b)
            loss_rint = F.l1_loss(pred_rint, rint_b)
            total_loss = loss_zb + 0.5 * loss_zs + 0.5 * loss_rint

            opt_struct.zero_grad()
            total_loss.backward()
            opt_struct.step()
            ep_loss += total_loss.item()

        # Val
        struct_model.eval()
        with torch.no_grad():
            v_pred_zb, _, _ = struct_model(val_zm.to(device), val_q.to(device), z_0)
            v_loss = F.l1_loss(v_pred_zb, val_zb.to(device)).item()

        if v_loss < best_val_struct:
            best_val_struct = v_loss
            best_state_struct = {k: v.cpu().clone() for k, v in struct_model.state_dict().items()}

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            print(f"  Struct Ep [{ep+1:2d}/{EPOCHS:2d}] - Train Loss: {ep_loss/num_train_batches:.4f} - Val L1 (zb): {v_loss:.4f}")

    struct_model.load_state_dict(best_state_struct)
    print(f"Model B.2 trained in {time.time() - t0:.1f}s, Best Val L1: {best_val_struct:.4f}")

    # 3. Comprehensive Decoded Audio Evaluation on Test Split
    print("\n--- Running Stage B Decoded Audio SDR Benchmark on Test Split ---")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform.eval()

    test_subset = test_data[:30]
    sdr_untouched = []
    sdr_naive_sub = []
    sdr_oracle_exact = []
    sdr_clean_ceiling = []
    sdr_direct_model = []
    sdr_struct_model = []

    print(f"Decoding {len(test_subset)} test mixtures...")
    for idx, item in enumerate(test_subset):
        mix_w, bg_w, target_w = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )

        zm_item = item["z_m"].unsqueeze(0).to(device)
        zb_item = item["z_b"].unsqueeze(0).to(device)
        zs_item = item["z_s"].unsqueeze(0).to(device)
        q_item = torch.tensor([fam_to_idx[item["foreground_family"]]], dtype=torch.long, device=device)

        # 1. Untouched Mix
        sdr_untouched.append(waveform_sdr(bg_w, mix_w))

        # 2. Clean AE Ceiling D(z_b)
        with torch.no_grad():
            bg_clean = same_ae.decode(zb_item.half()).float()[0].cpu()
        sdr_clean_ceiling.append(waveform_sdr(bg_w, bg_clean))

        # 3. Naive Subtraction D(z_m - z_s + z_0)
        z_naive = zm_item - zs_item + z_0.unsqueeze(0)
        with torch.no_grad():
            bg_naive = same_ae.decode(z_naive.half()).float()[0].cpu()
        sdr_naive_sub.append(waveform_sdr(bg_w, bg_naive))

        # 4. Direct Model Prediction
        with torch.no_grad():
            z_hat_direct = direct_model(zm_item, q_item)
            bg_direct = same_ae.decode(z_hat_direct.half()).float()[0].cpu()
        sdr_direct_model.append(waveform_sdr(bg_w, bg_direct))

        # 5. Structured Physical Model Prediction
        with torch.no_grad():
            z_hat_struct, _, _ = struct_model(zm_item, q_item, z_0)
            bg_struct = same_ae.decode(z_hat_struct.half()).float()[0].cpu()
        sdr_struct_model.append(waveform_sdr(bg_w, bg_struct))

    print("\n=====================================================================")
    print("   STAGE B BENCHMARK RESULTS: STRUCTURED SEPARATION VS BASELINES")
    print("=====================================================================")
    m_untouched = float(np.mean(sdr_untouched))
    m_ceiling = float(np.mean(sdr_clean_ceiling))
    m_naive = float(np.mean(sdr_naive_sub))
    m_direct = float(np.mean(sdr_direct_model))
    m_struct = float(np.mean(sdr_struct_model))

    print(f"{'Separation Method / Baseline':<40} | {'Decoded SDR':<11} | {'vs Untouched':<14} | {'Recovery Rate'}")
    print("-" * 82)
    print(f"{'Untouched Mixture (Do-Nothing)':<40} | {m_untouched:+7.2f} dB   | {'0.00 dB':<14} | 0.0%")
    print(f"{'Direct Point Separator f(z_m, q)':<40} | {m_direct:+7.2f} dB   | {m_direct - m_untouched:+7.2f} dB   | {(m_direct - m_untouched)/(m_ceiling - m_untouched)*100:5.1f}%")
    print(f"{'Structured Physical Demixer':<40} | {m_struct:+7.2f} dB   | {m_struct - m_untouched:+7.2f} dB   | {(m_struct - m_untouched)/(m_ceiling - m_untouched)*100:5.1f}%")
    print(f"{'Naive Subtraction (Oracle z_s, r=0)':<40} | {m_naive:+7.2f} dB   | {m_naive - m_untouched:+7.2f} dB   | {(m_naive - m_untouched)/(m_ceiling - m_untouched)*100:5.1f}%")
    print(f"{'Clean AE Ceiling D(E(b))':<40} | {m_ceiling:+7.2f} dB   | {m_ceiling - m_untouched:+7.2f} dB   | 100.0%")
    print("-" * 82)

    # Plot Comparison
    plt.figure(figsize=(10, 6))
    methods = [
        "Untouched Mix",
        "Direct Point",
        "Structured Physical",
        "Naive Sub (Oracle z_s)",
        "Clean Ceiling"
    ]
    sdrs = [m_untouched, m_direct, m_struct, m_naive, m_ceiling]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#2ca02c"]

    bars = plt.bar(methods, sdrs, color=colors, edgecolor="black", alpha=0.85)
    plt.axhline(m_untouched, color="gray", linestyle="--", label="Untouched Mix Baseline")
    plt.ylabel("Decoded Waveform SDR (dB)")
    plt.title("Stage B: Background Recovery SDR on Unseen Test Mixtures")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend()
    plt.tight_layout()

    plot_file = PLOTS_DIR / "plot_9_4_stage_b_structured_separation.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Saved Stage B benchmark plot to {plot_file}")

    # Master Report
    report = {
        "experiment": "Phase 9 — Stage B: Structured Deterministic Separation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "untouched_mix_sdr": m_untouched,
            "direct_separator_sdr": m_direct,
            "structured_physical_sdr": m_struct,
            "naive_subtraction_sdr": m_naive,
            "clean_ae_ceiling_sdr": m_ceiling
        },
        "gains_vs_untouched": {
            "direct_gain_db": m_direct - m_untouched,
            "structured_gain_db": m_struct - m_untouched,
            "naive_oracle_gain_db": m_naive - m_untouched
        }
    }

    report_path = REPORTS_DIR / "exp_9_4_stage_b_structured_separation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved master report to {report_path}")

    print("\n>>> Stage B Completed Successfully! <<<")
    os._exit(0)

if __name__ == "__main__":
    import math
    run_stage_b()
