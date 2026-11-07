#!/usr/bin/env python3
"""
Phase 9 — Experiment A.1: Oracle Interaction Residual Predictability
====================================================================
Scientific Question:
  Given the two isolated encoded components (z_b, z_s), is their interaction
  residual r_int = z_m - z_b - z_s + z_0 approximately deterministic?

Models Evaluated:
  1. Zero Baseline: r_hat = 0 (additive null hypothesis)
  2. Global Mean: r_hat = mean(r_train)
  3. Class Mean: r_hat = mean(r_train | class)
  4. Bilinear Model: r_hat,t = W_o [(W_b z_b,t) * (W_s z_s,t)]
  5. Local Temporal CNN: 3 layers of Conv1D (receptive field w=3 frames)
  6. Contextual Sequence Model: Conv1D + Multi-Head Self-Attention (receptive field = full clip)

Evaluation Metrics:
  - Standardized MSE (L_int) weighted by training set per-channel variance
  - Unstandardized NMSE_r
  - R^2 Variance Explained
  - Decoded Audio SDR (b vs D(z_m - z_s + z_0 - r_hat)) vs:
      * Naive Subtraction (r_hat = 0)
      * Untouched Mixture (b + s)
      * Autoencoder Ceiling (D(E(b)))
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

ROOT_DIR = Path("/home/kim/evals_aac/same_latent_research")
CACHE_DIR = ROOT_DIR / "cache"
PHASE9_DIR = ROOT_DIR / "batch9_phase9_residual_diagnostics"
REPORTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase9_residual/reports")
PLOTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase9_residual/plots")

sys.path.insert(0, str(PHASE9_DIR))
sys.path.insert(0, str(ROOT_DIR / "batch4_mixture_recognition"))

from stable_audio_3 import StableAudioModel

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Metric Functions
# ---------------------------------------------------------------------------

def calculate_nmse(r_true, r_pred, eps=1e-8):
    err = (r_pred - r_true).pow(2).sum()
    ref = r_true.pow(2).sum()
    return float(err / (ref + eps))

def calculate_r2(r_true, r_pred, r_mean, eps=1e-8):
    ss_res = (r_pred - r_true).pow(2).sum()
    ss_tot = (r_true - r_mean).pow(2).sum()
    return float(1.0 - ss_res / (ss_tot + eps))

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

# ---------------------------------------------------------------------------
# Model Architectures
# ---------------------------------------------------------------------------

class BilinearInteractionModel(nn.Module):
    """
    Exp A.1.4: Multiplicative interaction between z_b and z_s.
    r_hat,t = W_o [(W_b z_b,t) * (W_s z_s,t)]
    """
    def __init__(self, dim: int = 256, rank: int = 128):
        super().__init__()
        self.w_b = nn.Linear(dim, rank)
        self.w_s = nn.Linear(dim, rank)
        self.w_o = nn.Linear(rank, dim)
        # Initialize near zero (additive null hypothesis)
        nn.init.normal_(self.w_o.weight, std=1e-4)
        nn.init.zeros_(self.w_o.bias)

    def forward(self, z_b, z_s):
        # Input: [B, C, T] -> transpose to [B, T, C]
        b_t = z_b.transpose(1, 2)
        s_t = z_s.transpose(1, 2)
        h = self.w_b(b_t) * self.w_s(s_t)
        out = self.w_o(h)
        return out.transpose(1, 2) # [B, C, T]


class LocalConvInteractionModel(nn.Module):
    """
    Exp A.1.5: 3-layer 1D CNN with local receptive field (w=3 frames = ~278 ms).
    Captures transient collisions and spectral masking.
    """
    def __init__(self, dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.in_conv = nn.Conv1d(dim * 2, hidden_dim, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )
        self.out_conv = nn.Conv1d(hidden_dim, dim, kernel_size=3, padding=1)
        # Initialize output layer near zero
        nn.init.normal_(self.out_conv.weight, std=1e-4)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_b, z_s):
        x = torch.cat([z_b, z_s], dim=1) # [B, 512, T]
        h = self.in_conv(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.out_conv(h)


class ContextualSequenceModel(nn.Module):
    """
    Exp A.1.6: Sequence Model with Conv1D + Self-Attention.
    Receptive field covers the entire 32-frame clip (2.97 seconds).
    """
    def __init__(self, dim: int = 256, hidden_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.in_proj = nn.Conv1d(dim * 2, hidden_dim, 1)
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU()
        )
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Conv1d(hidden_dim, dim, 1)
        nn.init.normal_(self.out_proj.weight, std=1e-4)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_b, z_s):
        x = torch.cat([z_b, z_s], dim=1)
        h = self.in_proj(x)
        h = self.conv_block(h)
        # Self-attention over temporal axis
        h_t = h.transpose(1, 2) # [B, T, H]
        attn_out, _ = self.attn(h_t, h_t, h_t)
        h_t = self.ln(h_t + attn_out)
        out = self.out_proj(h_t.transpose(1, 2))
        return out

# ---------------------------------------------------------------------------
# Training & Evaluation Loop
# ---------------------------------------------------------------------------

def train_and_eval_oracle_model(name: str, model: nn.Module, train_data, val_data, test_data, channel_var, r_train_mean, epochs=25, lr=1e-3, device="cuda"):
    print(f"\nTraining Oracle Model: {name}...")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.to(device)

    # Standardized MSE weight: 1 / (var + eps), clamped to prevent outlier spikes
    clamped_var = torch.clamp(channel_var.to(device), min=1e-3)
    inv_var = (1.0 / clamped_var).view(1, 256, 1)

    best_val_loss = float("inf")
    best_state = None

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for item in train_data:
            z_b = item["z_b"].unsqueeze(0).to(device)
            z_s = item["z_s"].unsqueeze(0).to(device)
            r_int = item["r_int"].unsqueeze(0).to(device)

            pred_r = model(z_b, z_s)

            # Standardized MSE loss
            loss = ((pred_r - r_int).pow(2) * inv_var).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in val_data:
                z_b = item["z_b"].unsqueeze(0).to(device)
                z_s = item["z_s"].unsqueeze(0).to(device)
                r_int = item["r_int"].unsqueeze(0).to(device)
                pred_r = model(z_b, z_s)
                v_loss = ((pred_r - r_int).pow(2) * inv_var).mean()
                val_loss += v_loss.item()

        val_loss /= len(val_data)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"  Epoch [{ep+1:2d}/{epochs:2d}] - Train Loss: {total_loss/len(train_data):.4f} - Val Loss: {val_loss:.4f}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    # Test Evaluation
    test_nmse_list = []
    test_r2_list = []
    test_std_mse_list = []
    test_preds = []

    with torch.no_grad():
        for item in test_data:
            z_b = item["z_b"].unsqueeze(0).to(device)
            z_s = item["z_s"].unsqueeze(0).to(device)
            r_int = item["r_int"].to(device)

            pred_r = model(z_b, z_s)[0]
            test_preds.append(pred_r.cpu())

            nmse = calculate_nmse(r_int, pred_r)
            r2 = calculate_r2(r_int, pred_r, r_train_mean.to(device))
            std_mse = ((pred_r - r_int).pow(2) * inv_var[0]).mean().item()

            test_nmse_list.append(nmse)
            test_r2_list.append(r2)
            test_std_mse_list.append(std_mse)

    return {
        "model_name": name,
        "train_time_sec": train_time,
        "test_std_mse": float(np.mean(test_std_mse_list)),
        "test_nmse": float(np.mean(test_nmse_list)),
        "test_r2": float(np.mean(test_r2_list)),
        "predictions": test_preds
    }

# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    print("=====================================================================")
    print("   STAGE A — EXPERIMENT A.1: ORACLE INTERACTION RESIDUAL PREDICTABILITY")
    print("=====================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cache_file = CACHE_DIR / "phase9_residual_dataset.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cached Phase 9 dataset at {cache_file}!")

    print(f"Loading cached Phase 9 dataset from {cache_file}...")
    dataset = torch.load(cache_file, map_location="cpu", weights_only=False)
    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]
    z_0 = dataset["z_0"].to(device)

    print(f"Dataset splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")

    # Compute training set statistics for standardization
    all_train_r = torch.stack([item["r_int"] for item in train_data], dim=0) # [600, 256, 32]
    r_train_mean = all_train_r.mean(dim=(0, 2), keepdim=True)[0] # [256, 1]
    r_train_var = all_train_r.var(dim=(0, 2), keepdim=False) # [256]
    print(f"Computed training residual stats: mean norm {r_train_mean.norm():.3f}, mean channel std {r_train_var.sqrt().mean():.4f}")

    # 1. Zero Baseline (Additive null hypothesis)
    zero_nmse = [calculate_nmse(item["r_int"], torch.zeros_like(item["r_int"])) for item in test_data]
    zero_r2 = [calculate_r2(item["r_int"], torch.zeros_like(item["r_int"]), r_train_mean) for item in test_data]
    inv_var_cpu = (1.0 / (r_train_var + 1e-6)).view(256, 1)
    zero_std_mse = [((torch.zeros_like(item["r_int"]) - item["r_int"]).pow(2) * inv_var_cpu).mean().item() for item in test_data]
    
    results = {
        "zero_baseline": {
            "model_name": "Zero Baseline (Additive Null)",
            "test_std_mse": float(np.mean(zero_std_mse)),
            "test_nmse": float(np.mean(zero_nmse)),
            "test_r2": float(np.mean(zero_r2)),
            "predictions": [torch.zeros_like(item["r_int"]) for item in test_data]
        }
    }
    print(f"Zero Baseline -> NMSE: {results['zero_baseline']['test_nmse']:.4f}, R2: {results['zero_baseline']['test_r2']:.4f}")

    # 2. Global Mean Baseline
    mean_nmse = [calculate_nmse(item["r_int"], r_train_mean.expand_as(item["r_int"])) for item in test_data]
    mean_r2 = [calculate_r2(item["r_int"], r_train_mean.expand_as(item["r_int"]), r_train_mean) for item in test_data]
    mean_std_mse = [((r_train_mean.expand_as(item["r_int"]) - item["r_int"]).pow(2) * inv_var_cpu).mean().item() for item in test_data]
    results["global_mean"] = {
        "model_name": "Global Mean Baseline",
        "test_std_mse": float(np.mean(mean_std_mse)),
        "test_nmse": float(np.mean(mean_nmse)),
        "test_r2": float(np.mean(mean_r2)),
        "predictions": [r_train_mean.expand_as(item["r_int"]) for item in test_data]
    }
    print(f"Global Mean   -> NMSE: {results['global_mean']['test_nmse']:.4f}, R2: {results['global_mean']['test_r2']:.4f}")

    # 3. Class-Conditional Mean Baseline
    class_means = {}
    for fam in set(item["foreground_family"] for item in train_data):
        fam_items = [item["r_int"] for item in train_data if item["foreground_family"] == fam]
        class_means[fam] = torch.stack(fam_items, dim=0).mean(dim=(0, 2), keepdim=True)[0]
    
    class_nmse = [calculate_nmse(item["r_int"], class_means[item["foreground_family"]].expand_as(item["r_int"])) for item in test_data]
    class_r2 = [calculate_r2(item["r_int"], class_means[item["foreground_family"]].expand_as(item["r_int"]), r_train_mean) for item in test_data]
    class_std_mse = [((class_means[item["foreground_family"]].expand_as(item["r_int"]) - item["r_int"]).pow(2) * inv_var_cpu).mean().item() for item in test_data]
    results["class_mean"] = {
        "model_name": "Class Mean Baseline",
        "test_std_mse": float(np.mean(class_std_mse)),
        "test_nmse": float(np.mean(class_nmse)),
        "test_r2": float(np.mean(class_r2)),
        "predictions": [class_means[item["foreground_family"]].expand_as(item["r_int"]) for item in test_data]
    }
    print(f"Class Mean    -> NMSE: {results['class_mean']['test_nmse']:.4f}, R2: {results['class_mean']['test_r2']:.4f}")

    # Train Learned Models
    torch.manual_seed(42)
    np.random.seed(42)

    # 4. Bilinear Interaction Model
    bilinear_model = BilinearInteractionModel(dim=256, rank=128)
    res_bilinear = train_and_eval_oracle_model("Bilinear Interaction Model", bilinear_model, train_data, val_data, test_data, r_train_var, r_train_mean, epochs=25, lr=1e-3, device=device)
    results["bilinear"] = res_bilinear
    print(f"Bilinear      -> NMSE: {res_bilinear['test_nmse']:.4f}, R2: {res_bilinear['test_r2']:.4f}")

    # 5. Local Temporal CNN
    local_conv_model = LocalConvInteractionModel(dim=256, hidden_dim=256)
    res_local = train_and_eval_oracle_model("Local Temporal CNN (w=3)", local_conv_model, train_data, val_data, test_data, r_train_var, r_train_mean, epochs=25, lr=1e-3, device=device)
    results["local_conv"] = res_local
    print(f"Local Conv    -> NMSE: {res_local['test_nmse']:.4f}, R2: {res_local['test_r2']:.4f}")

    # 6. Contextual Sequence Model
    context_model = ContextualSequenceModel(dim=256, hidden_dim=256, n_heads=4)
    res_context = train_and_eval_oracle_model("Contextual Sequence Model (Conv+Attn)", context_model, train_data, val_data, test_data, r_train_var, r_train_mean, epochs=25, lr=1e-3, device=device)
    results["contextual"] = res_context
    print(f"Contextual    -> NMSE: {res_context['test_nmse']:.4f}, R2: {res_context['test_r2']:.4f}")

    # ---------------------------------------------------------------------------
    # Acoustic Waveform Decoded Validation (The Crucial Ground-Truth SDR Test)
    # ---------------------------------------------------------------------------
    print("\n--- Running Acoustic Waveform Decoded SDR Benchmark ---")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform.eval()

    from build_phase9_residual_dataset import render_mixture_components
    TOTAL_SAMPLES = 131072

    # Evaluate SDR on 30 test mixtures across all models
    eval_subset = test_data[:30]
    sdr_by_model = {k: [] for k in results.keys()}
    sdr_untouched_mix = []
    sdr_ae_ceiling = []

    print(f"Decoding and evaluating audio for {len(eval_subset)} test mixtures...")
    for idx, item in enumerate(eval_subset):
        # Render true audio
        mix_w, bg_w, target_w = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )

        # Baseline: Untouched mixture
        sdr_untouched_mix.append(waveform_sdr(bg_w, mix_w))

        # Baseline: Clean AE ceiling
        with torch.no_grad():
            bg_reconstructed = same_ae.decode(item["z_b"].unsqueeze(0).to(device).half()).float()[0].cpu()
        sdr_ae_ceiling.append(waveform_sdr(bg_w, bg_reconstructed))

        z_m = item["z_m"].to(device)
        z_s = item["z_s"].to(device)

        # Evaluate each residual model's reconstructed background:
        # z_hat_b = z_m - z_s + z_0 - r_hat
        for m_key in results.keys():
            r_hat = results[m_key]["predictions"][idx].to(device)
            z_hat_b = z_m - z_s + z_0 - r_hat
            with torch.no_grad():
                b_hat_audio = same_ae.decode(z_hat_b.unsqueeze(0).half()).float()[0].cpu()
            sdr_by_model[m_key].append(waveform_sdr(bg_w, b_hat_audio))

    print("\n=====================================================================")
    print("   STAGE A.1 FINAL RESULTS: RESIDUAL PREDICTABILITY & DECODED SDR")
    print("=====================================================================")
    print(f"{'Model Architecture':<32} | {'NMSE_r':<8} | {'R^2':<8} | {'Decoded SDR':<11} | {'vs. Untouched':<14}")
    print("-" * 80)
    mean_untouched_sdr = float(np.mean(sdr_untouched_mix))
    mean_ceiling_sdr = float(np.mean(sdr_ae_ceiling))
    print(f"{'Untouched Mixture (Do-Nothing)':<32} | {'—':<8} | {'—':<8} | {mean_untouched_sdr:+7.2f} dB   | {'0.00 dB':<14}")
    print(f"{'Clean AE Ceiling D(E(b))':<32} | {'—':<8} | {'—':<8} | {mean_ceiling_sdr:+7.2f} dB   | {mean_ceiling_sdr - mean_untouched_sdr:+7.2f} dB")
    print("-" * 80)

    summary_table = {}
    for m_key in results.keys():
        nmse = results[m_key]["test_nmse"]
        r2 = results[m_key]["test_r2"]
        sdr = float(np.mean(sdr_by_model[m_key]))
        delta_sdr = sdr - mean_untouched_sdr
        summary_table[m_key] = {
            "name": results[m_key]["model_name"],
            "nmse": nmse,
            "r2": r2,
            "mean_sdr": sdr,
            "delta_sdr_vs_mix": delta_sdr
        }
        print(f"{results[m_key]['model_name']:<32} | {nmse:8.4f} | {r2:8.4f} | {sdr:+7.2f} dB   | {delta_sdr:+7.2f} dB")

    # Generate Publication Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = [summary_table[k]["name"].replace("Baseline ", "").replace(" (Additive Null)", "") for k in summary_table.keys()]
    nmse_vals = [summary_table[k]["nmse"] for k in summary_table.keys()]
    sdr_vals = [summary_table[k]["mean_sdr"] for k in summary_table.keys()]

    # Bar 1: NMSE
    axes[0].bar(model_names, nmse_vals, color=["#d62728", "#ff7f0e", "#bcbd22", "#1f77b4", "#2ca02c", "#9467bd"])
    axes[0].set_ylabel("Interaction Residual NMSE (Lower is Better)")
    axes[0].set_title("Oracle Residual Predictability from (z_b, z_s)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")

    # Bar 2: Audio SDR
    axes[1].axhline(mean_untouched_sdr, color="red", linestyle="--", label=f"Untouched Mix ({mean_untouched_sdr:.1f} dB)")
    axes[1].axhline(mean_ceiling_sdr, color="green", linestyle=":", label=f"Clean AE Ceiling ({mean_ceiling_sdr:.1f} dB)")
    axes[1].bar(model_names, sdr_vals, color=["#d62728", "#ff7f0e", "#bcbd22", "#1f77b4", "#2ca02c", "#9467bd"])
    axes[1].set_ylabel("Decoded Background Audio SDR (dB)")
    axes[1].set_title("Waveform SDR of Decoded Background D(z_hat_b)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plot_file = PLOTS_DIR / "plot_9_1_oracle_residual_predictability.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"\nSaved diagnostic figure to {plot_file}")

    # Clean results for JSON report
    report_data = {
        "title": "Phase 9 — Experiment A.1: Oracle Interaction Residual Predictability",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "reference_baselines": {
            "untouched_mixture_sdr": mean_untouched_sdr,
            "autoencoder_ceiling_sdr": mean_ceiling_sdr
        },
        "models": summary_table
    }

    report_file = REPORTS_DIR / "exp_9_1_oracle_predictability_report.json"
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved master report to {report_file}")
    print("\n>>> Experiment A.1 Completed Successfully! <<<")
    os._exit(0)

if __name__ == "__main__":
    main()
