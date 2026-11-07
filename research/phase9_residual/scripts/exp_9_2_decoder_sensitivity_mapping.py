#!/usr/bin/env python3
"""
Phase 9 — Experiment A.2: Directional Decoder Sensitivity Mapping
================================================================
Scientific Question:
  Is the SAME decoder unusually sensitive along the non-linear interaction
  manifold, or does it behave isotropically? Does a displacement along r_int
  degrade audio quality significantly more than an isotropic random displacement
  or a displacement along data principal components?

Tested Perturbation Directions u (||u||_2 = 1):
  1. isotropic_random: Random Gaussian vector in R^{256 x 32}
  2. pca_top_1: Leading principal component of latent variation
  3. pca_bottom_1: Smallest variance principal component
  4. interaction_residual: True interaction residual direction u = r_int / ||r_int||_2
  5. removal_displacement: True removal displacement direction u = r_remove / ||r_remove||_2

Perturbation Magnitudes eps:
  [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0]

Metrics:
  - Q(eps, u) = SDR(b, D(z_b + eps * u))
  - Initial Sensitivity Slope dQ/deps | eps->0 (dB per unit latent norm)
  - Directional Sensitivity Ratio S(u) = (dQ/deps)_u / (dQ/deps)_random
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

def compute_pca_directions(all_z_b):
    """
    Computes top-1 and bottom-1 principal directions across background latents.
    all_z_b: [N, 256, 32]
    """
    N, C, T = all_z_b.shape
    # Compute covariance across channel dimensions [C, N*T]
    flat_channels = all_z_b.permute(1, 0, 2).reshape(C, -1) # [256, N*T]
    mean_ch = flat_channels.mean(dim=1, keepdim=True)
    centered = flat_channels - mean_ch
    
    # SVD
    U, S, V = torch.linalg.svd(centered, full_matrices=False)
    
    # Leading eigenvector in R^{256}
    u_top = U[:, 0] # [256]
    # Smallest eigenvector in R^{256}
    u_bottom = U[:, -1] # [256]
    
    # Broadcast across time T=32 and normalize to unit Frobenius norm in R^{256 x 32}
    u_pca_top = u_top.unsqueeze(1).repeat(1, T)
    u_pca_top = u_pca_top / u_pca_top.norm()
    
    u_pca_bottom = u_bottom.unsqueeze(1).repeat(1, T)
    u_pca_bottom = u_pca_bottom / u_pca_bottom.norm()
    
    return u_pca_top, u_pca_bottom

def main():
    print("=====================================================================")
    print("   STAGE A — EXPERIMENT A.2: DIRECTIONAL DECODER SENSITIVITY MAPPING")
    print("=====================================================================")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    cache_file = CACHE_DIR / "phase9_residual_dataset.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"Missing cached Phase 9 dataset at {cache_file}!")

    print(f"Loading cached Phase 9 dataset from {cache_file}...")
    dataset = torch.load(cache_file, map_location="cpu", weights_only=False)
    train_data = dataset["train"]
    test_data = dataset["test"]

    print("Computing PCA basis vectors over training background latents...")
    train_z_b = torch.stack([item["z_b"] for item in train_data], dim=0) # [600, 256, 32]
    u_pca_top, u_pca_bottom = compute_pca_directions(train_z_b)

    print("Loading SAME Autoencoder model...")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform.eval()

    # Perturbation magnitudes
    eps_values = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 20.0]

    # Directions to evaluate
    directions = [
        "isotropic_random",
        "pca_top_1",
        "pca_bottom_1",
        "interaction_residual",
        "removal_displacement"
    ]

    # Evaluate on a curated subset of 30 test mixtures
    test_subset = test_data[:30]
    print(f"Evaluating {len(test_subset)} test mixtures across {len(directions)} directions and {len(eps_values)} scales...")

    # Data structure: results_by_dir[dir_name][eps] = list of SDRs
    sdr_results = {d: {eps: [] for eps in eps_values} for d in directions}
    actual_norms_rint = []
    actual_norms_rremove = []
    ceiling_sdrs = []

    for idx, item in enumerate(test_subset):
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  Processing item [{idx+1:2d}/{len(test_subset):2d}]...")

        mix_w, bg_w, target_w = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )

        z_b = item["z_b"].to(device)
        r_int = item["r_int"].to(device)
        r_remove = item["r_remove"].to(device)

        norm_rint = r_int.norm().item()
        norm_rremove = r_remove.norm().item()
        actual_norms_rint.append(norm_rint)
        actual_norms_rremove.append(norm_rremove)

        # Build normalized unit directions for this item
        torch.manual_seed(idx * 1000 + 42)
        rand_vec = torch.randn_like(z_b)
        u_rand = rand_vec / rand_vec.norm()

        u_dict = {
            "isotropic_random": u_rand,
            "pca_top_1": u_pca_top.to(device),
            "pca_bottom_1": u_pca_bottom.to(device),
            "interaction_residual": r_int / (r_int.norm() + 1e-8),
            "removal_displacement": r_remove / (r_remove.norm() + 1e-8)
        }

        # Clean ceiling SDR (eps = 0)
        with torch.no_grad():
            b_clean = same_ae.decode(z_b.unsqueeze(0).half()).float()[0].cpu()
        clean_sdr = waveform_sdr(bg_w, b_clean)
        ceiling_sdrs.append(clean_sdr)

        for d_name, u_vec in u_dict.items():
            for eps in eps_values:
                if eps == 0.0:
                    sdr_results[d_name][eps].append(clean_sdr)
                else:
                    z_pert = z_b + eps * u_vec
                    with torch.no_grad():
                        b_pert = same_ae.decode(z_pert.unsqueeze(0).half()).float()[0].cpu()
                    sdr = waveform_sdr(bg_w, b_pert)
                    sdr_results[d_name][eps].append(sdr)

    # Compute statistics
    mean_ceiling = float(np.mean(ceiling_sdrs))
    mean_norm_rint = float(np.mean(actual_norms_rint))
    mean_norm_rremove = float(np.mean(actual_norms_rremove))

    print("\n=====================================================================")
    print("   STAGE A.2 DECODER SENSITIVITY PROFILE (Q(eps, u) in dB SDR)")
    print("=====================================================================")
    header = f"{'Epsilon (Norm)':<14} | " + " | ".join([f"{d[:15]:<15}" for d in directions])
    print(header)
    print("-" * len(header))

    mean_curves = {d: [] for d in directions}
    for eps in eps_values:
        row_str = f"{eps:<14.2f} | "
        for d in directions:
            m_sdr = float(np.mean(sdr_results[d][eps]))
            mean_curves[d].append(m_sdr)
            row_str += f"{m_sdr:+7.2f} dB        | "
        print(row_str)

    # Compute Initial Slope dQ/deps around eps in [0.0, 0.5]
    slopes = {}
    eps_arr = np.array(eps_values)
    idx_slope = [i for i, e in enumerate(eps_values) if 0.0 <= e <= 1.0]

    for d in directions:
        y = np.array(mean_curves[d])[idx_slope]
        x = eps_arr[idx_slope]
        # Linear regression slope: delta_y / delta_x
        slope, _ = np.polyfit(x, y, 1)
        slopes[d] = float(slope)

    ref_slope = abs(slopes["isotropic_random"])
    sensitivity_ratios = {d: abs(slopes[d]) / (ref_slope + 1e-8) for d in directions}

    print("\n--- Sensitivity Slopes & Directional Ratios ---")
    print(f"{'Direction':<25} | {'Slope dQ/deps (dB/unit)':<25} | {'Ratio S(u) vs Random':<22}")
    print("-" * 78)
    for d in directions:
        print(f"{d:<25} | {slopes[d]:+8.3f} dB/unit               | {sensitivity_ratios[d]:8.3f}x")

    print(f"\nAverage actual ||r_int||_2:    {mean_norm_rint:.3f}")
    print(f"Average actual ||r_remove||_2: {mean_norm_rremove:.3f}")
    print(f"Clean Autoencoder Ceiling SDR: {mean_ceiling:.2f} dB")

    # Plot Sensitivity Curves
    plt.figure(figsize=(10, 6))
    colors = {
        "isotropic_random": "#7f7f7f",
        "pca_top_1": "#1f77b4",
        "pca_bottom_1": "#2ca02c",
        "interaction_residual": "#d62728",
        "removal_displacement": "#9467bd"
    }
    styles = {
        "isotropic_random": "--",
        "pca_top_1": "-",
        "pca_bottom_1": ":",
        "interaction_residual": "-",
        "removal_displacement": "-."
    }

    for d in directions:
        plt.plot(eps_values, mean_curves[d], marker="o", linewidth=2.0, color=colors[d], linestyle=styles[d], label=f"{d} (Slope: {slopes[d]:.2f} dB/u)")

    plt.axvline(mean_norm_rint, color="red", linestyle=":", alpha=0.7, label=f"Mean ||r_int|| ({mean_norm_rint:.2f})")
    plt.axvline(mean_norm_rremove, color="purple", linestyle=":", alpha=0.7, label=f"Mean ||r_remove|| ({mean_norm_rremove:.2f})")
    plt.axhline(0.0, color="black", linestyle="-", alpha=0.3)

    plt.xlabel("Latent Perturbation Magnitude $\epsilon = \|\|\Delta z\|\|_F$")
    plt.ylabel("Decoded Background SDR (dB)")
    plt.title("SAME Decoder Directional Sensitivity: $Q(\epsilon, u) = \mathrm{SDR}(b, D(z_b + \epsilon u))$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "plot_9_2_decoder_sensitivity_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved sensitivity curves plot to {plot_path}")

    # Save JSON Report
    report = {
        "experiment": "Experiment A.2: Directional Decoder Sensitivity Mapping",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "clean_ceiling_sdr": mean_ceiling,
        "mean_norm_rint": mean_norm_rint,
        "mean_norm_rremove": mean_norm_rremove,
        "eps_values": eps_values,
        "mean_curves": mean_curves,
        "slopes_db_per_unit": slopes,
        "sensitivity_ratios": sensitivity_ratios
    }

    report_path = REPORTS_DIR / "exp_9_2_decoder_sensitivity_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved JSON report to {report_path}")

    print("\n>>> Experiment A.2 Completed Successfully! <<<")
    os._exit(0)

if __name__ == "__main__":
    main()
