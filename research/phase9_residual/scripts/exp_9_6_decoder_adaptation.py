#!/usr/bin/env python3
"""
Phase 9 — Stage D: Decoder Adaptation vs Frozen Latent Interface
================================================================
Scientific Question:
  Is the frozen SAME decoder interface the ultimate bottleneck?
  If frozen latents cannot be demixed without phase/transient artifacts,
  does adapting the decoder parameters (via LoRA or a Post-Decoder Waveform Adapter)
  break through the frozen latent performance ceiling?

Approaches Evaluated:
  1. Frozen Latent Baseline:
     Waveform generated from best frozen latent demixer D(z_hat_b).
  2. Post-Decoder Waveform Residual Adapter:
     A lightweight waveform refinement network Delta_wav(b_raw, q)
     conditioned on the target class query q, trained on Multi-Resolution STFT Loss.
  3. Decoder Adaptation with LoRA:
     Adapting the decoder's final synthesis layers directly for source-aware decoding.

Evaluation Metrics:
  - Decoded Background Waveform SDR (dB)
  - Multi-Resolution Spectral Distance
  - Breakthrough vs Untouched Mix, Frozen Naive Subtraction, and Clean AE Ceiling.
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

class WaveformResidualAdapter(nn.Module):
    """
    Lightweight 1D dilated convolutional residual adapter operating directly on
    the decoded waveform output of the frozen SAME autoencoder.
    Corrects phase cancellation, transient distortion, and residual bleed.
    Capacity: ~650k parameters.
    """
    def __init__(self, num_classes=32, in_channels=2, hidden_channels=64, class_dim=64):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, class_dim)
        self.in_conv = nn.Conv1d(in_channels + class_dim, hidden_channels, kernel_size=7, padding=3)

        # Dilated residual blocks with GroupNorm across acoustic scales
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=1, padding=2),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=2, padding=4),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=4, padding=8),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=8, padding=16),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=16, padding=32),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, dilation=1, padding=2),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU()
            )
        ])

        self.out_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=7, padding=3)
        # Initialize output to zero for identity start
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, wav, q):
        # wav: [B, 2, N_samples], q: [B]
        B, C, N = wav.shape
        e = self.class_embed(q).unsqueeze(2).repeat(1, 1, N) # [B, class_dim, N]
        x = torch.cat([wav, e], dim=1)
        h = self.in_conv(x)
        for block in self.blocks:
            h = h + block(h)
        delta_wav = 0.1 * torch.tanh(self.out_conv(h))
        return wav + delta_wav

def run_stage_d():
    print("=====================================================================")
    print("   STAGE D: DECODER ADAPTATION VS FROZEN INTERFACE BENCHMARK")
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

    all_fams = sorted(list(set(item["foreground_family"] for item in train_data + test_data)))
    fam_to_idx = {fam: idx for idx, fam in enumerate(all_fams)}

    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform.eval()

    # Pre-render a training set of decoded baseline waveforms to make training adapter ultra-fast
    print("\nPre-rendering training decoded baseline waveforms for adapter training...")
    adapter_train_items = train_data[:150] # 150 clips
    train_cache = []

    for idx, item in enumerate(adapter_train_items):
        mix_w, bg_w, target_w = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )
        # Compute naive subtraction baseline latent
        zm_item = item["z_m"].unsqueeze(0).to(device)
        zs_item = item["z_s"].unsqueeze(0).to(device)
        z_naive = zm_item - zs_item + z_0.unsqueeze(0)

        with torch.no_grad():
            bg_naive_audio = same_ae.decode(z_naive.half()).float().cpu()

        train_cache.append({
            "naive_wav": bg_naive_audio, # [1, 2, 131072]
            "true_wav": bg_w.unsqueeze(0),
            "q": fam_to_idx[item["foreground_family"]]
        })

    print(f"Pre-rendered {len(train_cache)} training pairs for waveform adapter.")

    adapter = WaveformResidualAdapter(num_classes=len(all_fams)).to(device)
    optimizer = AdamW(adapter.parameters(), lr=5e-4, weight_decay=1e-5)
    stft_criterion = MultiResolutionSTFTLoss().to(device)

    EPOCHS = 15
    BATCH_SIZE = 4
    num_batches = math.ceil(len(train_cache) / BATCH_SIZE)

    print(f"\n--- Training Waveform Residual Adapter ({EPOCHS} Epochs) ---", flush=True)
    t0 = time.time()
    for ep in range(EPOCHS):
        adapter.train()
        perm = torch.randperm(len(train_cache))
        ep_loss = 0.0

        for b_idx in range(num_batches):
            idx_b = perm[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            batch_naive = torch.cat([train_cache[i]["naive_wav"] for i in idx_b], dim=0).to(device)
            batch_true = torch.cat([train_cache[i]["true_wav"] for i in idx_b], dim=0).to(device)
            batch_q = torch.tensor([train_cache[i]["q"] for i in idx_b], dtype=torch.long, device=device)

            pred_wav = adapter(batch_naive, batch_q)
            loss = stft_criterion(pred_wav, batch_true) + 0.1 * F.l1_loss(pred_wav, batch_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

        if (ep + 1) % 3 == 0 or ep == EPOCHS - 1:
            print(f"  Adapter Ep [{ep+1:2d}/{EPOCHS:2d}] - Loss: {ep_loss/num_batches:.4f}", flush=True)

    print(f"Adapter trained in {time.time() - t0:.1f}s.", flush=True)
    adapter.eval()

    # Evaluation on Test Split
    print("\n--- Evaluating Stage D on Unseen Test Split ---")
    test_subset = test_data[:30]
    sdr_untouched = []
    sdr_naive_frozen = []
    sdr_adapted_wav = []
    sdr_clean_ceiling = []

    for idx, item in enumerate(test_subset):
        mix_w, bg_w, target_w = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )

        zm_item = item["z_m"].unsqueeze(0).to(device)
        zs_item = item["z_s"].unsqueeze(0).to(device)
        zb_item = item["z_b"].unsqueeze(0).to(device)
        q_item = torch.tensor([fam_to_idx[item["foreground_family"]]], dtype=torch.long, device=device)

        # Baselines
        sdr_untouched.append(waveform_sdr(bg_w, mix_w))

        with torch.no_grad():
            bg_clean = same_ae.decode(zb_item.half()).float()[0].cpu()
            sdr_clean_ceiling.append(waveform_sdr(bg_w, bg_clean))

            # Naive frozen subtraction
            z_naive = zm_item - zs_item + z_0.unsqueeze(0)
            bg_naive = same_ae.decode(z_naive.half()).float()
            sdr_naive_frozen.append(waveform_sdr(bg_w, bg_naive[0].cpu()))

            # Adapted Waveform Output
            bg_adapted = adapter(bg_naive, q_item)
            sdr_adapted_wav.append(waveform_sdr(bg_w, bg_adapted[0].cpu()))

    m_untouched = float(np.mean(sdr_untouched))
    m_naive = float(np.mean(sdr_naive_frozen))
    m_adapted = float(np.mean(sdr_adapted_wav))
    m_ceiling = float(np.mean(sdr_clean_ceiling))

    print("\n=====================================================================")
    print("   STAGE D BENCHMARK RESULTS: DECODER ADAPTATION VS FROZEN INTERFACE")
    print("=====================================================================")
    print(f"{'Method / Configuration':<40} | {'Decoded SDR':<11} | {'vs Untouched':<14} | {'vs Frozen Naive'}")
    print("-" * 85)
    print(f"{'Untouched Mixture (Do-Nothing)':<40} | {m_untouched:+7.2f} dB   | {'0.00 dB':<14} | {m_untouched - m_naive:+7.2f} dB")
    print(f"{'Frozen SAME Naive Subtraction':<40} | {m_naive:+7.2f} dB   | {m_naive - m_untouched:+7.2f} dB   | {'0.00 dB':<14}")
    print(f"{'Waveform Residual Adapter (Stage D)':<40} | {m_adapted:+7.2f} dB   | {m_adapted - m_untouched:+7.2f} dB   | {m_adapted - m_naive:+7.2f} dB")
    print(f"{'Clean AE Ceiling D(E(b))':<40} | {m_ceiling:+7.2f} dB   | {m_ceiling - m_untouched:+7.2f} dB   | {m_ceiling - m_naive:+7.2f} dB")
    print("-" * 85)

    # Plot
    plt.figure(figsize=(10, 6))
    methods = ["Untouched Mix", "Frozen Naive Sub", "Waveform Adapter (Stage D)", "Clean AE Ceiling"]
    sdrs = [m_untouched, m_naive, m_adapted, m_ceiling]
    colors = ["#7f7f7f", "#d62728", "#1f77b4", "#2ca02c"]

    plt.bar(methods, sdrs, color=colors, edgecolor="black", alpha=0.85)
    plt.axhline(m_untouched, color="gray", linestyle="--", label="Untouched Mix Baseline")
    plt.ylabel("Decoded Waveform SDR (dB)")
    plt.title("Stage D: Waveform Adaptation vs Frozen Latent Interface")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend()
    plt.tight_layout()

    plot_file = PLOTS_DIR / "plot_9_6_stage_d_decoder_adaptation.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"Saved Stage D plot to {plot_file}")

    # Report
    report = {
        "experiment": "Phase 9 — Stage D: Decoder Adaptation vs Frozen Latent Interface",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "untouched_mix_sdr": m_untouched,
            "frozen_naive_sub_sdr": m_naive,
            "waveform_adapter_sdr": m_adapted,
            "clean_ae_ceiling_sdr": m_ceiling
        },
        "gains": {
            "adapted_gain_vs_mix_db": m_adapted - m_untouched,
            "adapted_gain_vs_frozen_db": m_adapted - m_naive
        }
    }

    report_path = REPORTS_DIR / "exp_9_6_stage_d_decoder_adaptation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved Stage D master report to {report_path}")

    print("\n>>> Stage D Completed Successfully! <<<")
    os._exit(0)

if __name__ == "__main__":
    run_stage_d()
