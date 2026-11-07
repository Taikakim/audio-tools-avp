#!/usr/bin/env python3
"""
Phase 9 — Experiment A.3: Non-Additive Residual Decomposition & Forensic Analysis
================================================================================
Scientific Questions:
  1. Where does the non-additive residual r_int concentrate?
     - Across acoustic sound classes (kick, snare, bass, synth, etc.)?
     - Temporally: Is it strictly localized to the target sound's duration,
       or does encoder non-causality spread it across adjacent frames?
     - Channel-wise: Which of the 256 latent channels carry the non-additivity?
  2. How does the residual norm scale with foreground target amplitude alpha?
  3. What is the acoustic nature of the residual error when decoded to waveform?
     (Spectrogram difference between D(z_m - z_s + z_0) and clean D(z_b)).
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
import scipy.signal
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

def main():
    print("=====================================================================")
    print("   STAGE A — EXPERIMENT A.3: RESIDUAL DECOMPOSITION & SPECTROGRAMS")
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
    all_data = train_data + test_data
    z_0 = dataset["z_0"].cpu()

    print(f"Analyzing {len(all_data)} mixture items...")

    # 1. Breakdown by Sound Class / Family
    family_stats = {}
    for item in all_data:
        fam = item["foreground_family"]
        if fam not in family_stats:
            family_stats[fam] = {
                "norm_r_int": [],
                "norm_target_disp": [], # ||z_s - z_0||
                "relative_ratio": [],   # ||r_int|| / ||z_s - z_0||
                "alpha_gains": []
            }
        
        r_int = item["r_int"]
        z_s = item["z_s"]
        norm_r = float(r_int.norm())
        norm_s = float((z_s - z_0).norm())
        ratio = norm_r / (norm_s + 1e-8)

        family_stats[fam]["norm_r_int"].append(norm_r)
        family_stats[fam]["norm_target_disp"].append(norm_s)
        family_stats[fam]["relative_ratio"].append(ratio)
        family_stats[fam]["alpha_gains"].append(float(item["alpha_gain"]))

    print("\n--- Sound Family Non-Additivity Breakdown ---")
    print(f"{'Sound Family':<20} | {'Count':<6} | {'||r_int||_F':<12} | {'||z_s - z_0||_F':<15} | {'Ratio r/s':<10}")
    print("-" * 72)
    summary_by_fam = {}
    for fam, d in sorted(family_stats.items()):
        cnt = len(d["norm_r_int"])
        m_r = float(np.mean(d["norm_r_int"]))
        m_s = float(np.mean(d["norm_target_disp"]))
        m_ratio = float(np.mean(d["relative_ratio"]))
        summary_by_fam[fam] = {
            "count": cnt,
            "mean_norm_rint": m_r,
            "mean_norm_target_disp": m_s,
            "mean_relative_ratio": m_ratio
        }
        print(f"{fam:<20} | {cnt:<6d} | {m_r:<12.3f} | {m_s:<15.3f} | {m_ratio:<10.3f}")

    # 2. Temporal Localization Relative to Onset Frame
    # SAME hop is 4096 samples (92.88 ms at 44.1 kHz)
    # Target starts at tau_offset_samples -> frame = int(round(tau / 4096))
    HOP_SAMPLES = 4096
    temporal_profiles = [] # aligned relative to onset frame [-8, +16]

    for item in all_data:
        tau = item["tau_offset_samples"]
        onset_frame = int(round(tau / HOP_SAMPLES))
        r_int = item["r_int"] # [256, 32]
        frame_norms = r_int.norm(dim=0).numpy() # [32]

        # Extract window around onset frame
        profile = np.zeros(25) # relative frames -8 to +16
        for rel_idx, rel_f in enumerate(range(-8, 17)):
            abs_f = onset_frame + rel_f
            if 0 <= abs_f < 32:
                profile[rel_idx] = frame_norms[abs_f]
            else:
                profile[rel_idx] = np.nan
        temporal_profiles.append(profile)

    temporal_profiles = np.array(temporal_profiles)
    mean_temporal_impulse = np.nanmean(temporal_profiles, axis=0)

    # 3. Channel-wise Residual Variance Distribution
    all_r_int = torch.stack([item["r_int"] for item in all_data], dim=0) # [N, 256, 32]
    channel_residual_var = all_r_int.var(dim=(0, 2)).numpy() # [256]
    top_10_channels = np.argsort(channel_residual_var)[::-1][:10].tolist()

    # 4. Decoded Waveform Spectrogram Comparison on a Representative Item
    print("\nSynthesizing high-resolution spectrogram comparison...")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform.eval()

    sample_item = test_data[0]
    mix_w, bg_w, target_w = render_mixture_components(
        sample_item["background_path"],
        sample_item["foreground_path"],
        sample_item["alpha_gain"],
        sample_item["tau_offset_samples"]
    )

    z_m = sample_item["z_m"].to(device)
    z_b = sample_item["z_b"].to(device)
    z_s = sample_item["z_s"].to(device)
    z_0_dev = z_0.to(device)

    # Naive Subtraction Latent: z_naive = z_m - z_s + z_0
    z_naive = z_m - z_s + z_0_dev

    with torch.no_grad():
        bg_ae_clean = same_ae.decode(z_b.unsqueeze(0).half()).float()[0].cpu()
        bg_naive_sub = same_ae.decode(z_naive.unsqueeze(0).half()).float()[0].cpu()

    # Plot Multi-Panel Figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

    # Plot 1: Family Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    fams = list(summary_by_fam.keys())
    r_ratios = [summary_by_fam[f]["mean_relative_ratio"] for f in fams]
    ax1.bar(fams, r_ratios, color="#1f77b4", edgecolor="black", alpha=0.8)
    ax1.set_ylabel(r"Non-Additivity Ratio: $||r_{\mathrm{int}}|| / ||z_s - z_0||$")
    ax1.set_title("Non-Additivity Severity by Target Sound Class")
    ax1.tick_params(axis="x", rotation=30)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Plot 2: Temporal Impulse Response of Residual
    ax2 = fig.add_subplot(gs[0, 1])
    rel_time_ms = np.arange(-8, 17) * (HOP_SAMPLES / 44100.0) * 1000.0
    ax2.plot(rel_time_ms, mean_temporal_impulse, marker="o", color="#d62728", linewidth=2.0)
    ax2.axvline(0.0, color="black", linestyle="--", label="Target Onset Time")
    ax2.set_xlabel("Relative Time from Target Onset (ms)")
    ax2.set_ylabel(r"Residual Frame Norm $||r_{\mathrm{int}}(t)||_2$")
    ax2.set_title("Temporal Dispersion of Interaction Residual (Encoder Receptive Field Leak)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # Plot 3: Channel-wise Residual Variance
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(channel_residual_var, color="#2ca02c", linewidth=1.5)
    ax3.scatter(top_10_channels, channel_residual_var[top_10_channels], color="red", zorder=5, label=f"Top-10 Channels: {top_10_channels[:5]}...")
    ax3.set_xlabel("SAME Latent Channel Index (0 to 255)")
    ax3.set_ylabel("Residual Variance Across Dataset")
    ax3.set_title("Channel Localization of Interaction Residual")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # Plot 4: Amplitude Scaling
    ax4 = fig.add_subplot(gs[1, 1])
    all_alphas = [float(item["alpha_gain"]) for item in all_data]
    all_norms = [float(item["r_int"].norm()) for item in all_data]
    ax4.scatter(all_alphas, all_norms, alpha=0.3, color="#9467bd", edgecolors="none")
    # Linear fit
    fit_m, fit_c = np.polyfit(all_alphas, all_norms, 1)
    ax4.plot(np.sort(all_alphas), fit_m * np.sort(all_alphas) + fit_c, color="darkred", linewidth=2.0, label=f"Fit: ||r_int|| = {fit_m:.1f}*alpha + {fit_c:.1f}")
    ax4.set_xlabel(r"Foreground Target Amplitude $\alpha$ (Linear Scale)")
    ax4.set_ylabel(r"Total Residual Norm $||r_{\mathrm{int}}||_F$")
    ax4.set_title("Residual Norm vs Target Signal Amplitude")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    # Plot 5 & 6: Spectrogram Comparison of Error
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Convert audio to mono for spectrogram
    ref_audio = bg_ae_clean[0].numpy()
    naive_audio = bg_naive_sub[0].numpy()
    error_audio = naive_audio - ref_audio

    f_ref, t_ref, Sxx_ref = scipy.signal.spectrogram(ref_audio, fs=44100, nperseg=1024, noverlap=512)
    f_err, t_err, Sxx_err = scipy.signal.spectrogram(error_audio, fs=44100, nperseg=1024, noverlap=512)

    im1 = ax5.pcolormesh(t_ref, f_ref, 10 * np.log10(Sxx_ref + 1e-10), shading="gouraud", cmap="magma", vmin=-80, vmax=0)
    ax5.set_title(f"Clean Reconstructed Background D(z_b) - {sample_item['foreground_family']}")
    ax5.set_ylabel("Frequency (Hz)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylim(0, 15000)
    plt.colorbar(im1, ax=ax5, label="Power (dB)")

    im2 = ax6.pcolormesh(t_err, f_err, 10 * np.log10(Sxx_err + 1e-10), shading="gouraud", cmap="inferno", vmin=-80, vmax=0)
    ax6.set_title("Residual Waveform Artifact: D(z_m - z_s + z_0) - D(z_b)")
    ax6.set_ylabel("Frequency (Hz)")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylim(0, 15000)
    plt.colorbar(im2, ax=ax6, label="Artifact Power (dB)")

    plt.tight_layout()
    plot_file = PLOTS_DIR / "plot_9_3_residual_decomposition.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Saved residual decomposition figure to {plot_file}")

    # Save JSON Report
    report = {
        "experiment": "Experiment A.3: Non-Additive Residual Decomposition",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "family_summary": summary_by_fam,
        "temporal_profile_ms": rel_time_ms.tolist(),
        "temporal_impulse_norms": [float(x) for x in mean_temporal_impulse],
        "top_10_residual_channels": top_10_channels,
        "amplitude_scaling_slope": float(fit_m),
        "amplitude_scaling_intercept": float(fit_c)
    }

    report_file = REPORTS_DIR / "exp_9_3_residual_decomposition_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved master report to {report_file}")

    print("\n>>> Experiment A.3 Completed Successfully! <<<")
    os._exit(0)

if __name__ == "__main__":
    main()
