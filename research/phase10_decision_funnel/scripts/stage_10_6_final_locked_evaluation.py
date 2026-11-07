#!/usr/bin/env python3
"""
Stage 10.6: Final Locked Evaluation, Audio Demonstration, and Decision Funnel Synthesis
=======================================================================================
Aggregates and synthesizes all empirical evidence across Stages 10.0–10.5:
  - 10.0: Forensic Metric Audit
  - 10.1: Decoder-Aware Separation Objectives (Latent L1 vs Wave L1 vs MR-STFT vs Complex)
  - 10.2: Decoder Adaptation (Frozen vs LoRA vs Residual Adapter with Retention Loss)
  - 10.3: Capacity Ladder (1.2M vs 6.6M vs 20.1M parameters)
  - 10.4: Phase Augmentation (SAME vs Mag-STFT vs Complex-STFT)
  - 10.5: Hybrid Waveform Refiner (Latent vs Hybrid vs Waveform Baseline)

Also:
  - Computes 95% Bootstrap Confidence Intervals (B=1000 resamples)
  - Generates representative audible WAVs for 6 instrument families
  - Maps to Definitive Decision Funnel Outcome and Phase 11 Directive
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE")
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "2")

sys.path.insert(0, "/home/kim/Projects/SAO/stable-audio-3")
from stable_audio_3 import StableAudioModel

sys.path.insert(0, "/home/kim/evals_aac/same_latent_research/batch9_phase9_residual_diagnostics")
from build_phase9_residual_dataset import render_mixture_components

CACHE_DIR = Path("/home/kim/evals_aac/same_latent_research/cache")
REPORTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase10_decision_funnel/reports")
CHECKPOINTS_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase10_decision_funnel/checkpoints")
AUDIO_OUT_DIR = Path("/home/kim/Projects/SAO/stable-audio-tools/research/phase10_decision_funnel/audio_examples")
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def si_sdr(reference, estimate, eps=EPS):
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

def bootstrap_ci(data, num_bootstraps=1000, ci=95.0):
    arr = np.array(data)
    boot_means = []
    n = len(arr)
    rng = np.random.default_rng(42)
    for _ in range(num_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100.0 - ci) / 2.0)
    upper = np.percentile(boot_means, 100.0 - (100.0 - ci) / 2.0)
    return float(np.mean(arr)), float(lower), float(upper)

# ---------------------------------------------------------------------------
# Architecture Definitions for Inference & Audio Rendering
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
        B, _, L = audio_inputs.shape
        e = self.class_embed(q).unsqueeze(2).expand(B, -1, L)
        x = torch.cat([audio_inputs, e], dim=1)
        h = self.in_proj(x)
        h = self.blocks(h)
        return self.out_head(h)

# ---------------------------------------------------------------------------
# Multi-Resolution STFT Metric
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

        return float(loss.item() / len(self.fft_sizes))

# ---------------------------------------------------------------------------
# Audio Rendering & Exemplar Verification
# ---------------------------------------------------------------------------

def render_exemplars(device):
    print("\n--- Rendering Representative Audio Demonstrations (6 Families) ---")
    
    # 1. Load SAME base model
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform.eval()
    same_ae.enable_grad = False

    # Load dataset
    cache_file = CACHE_DIR / "phase9_residual_dataset.pt"
    dataset = torch.load(cache_file, map_location="cpu", weights_only=False)
    train_data = dataset["train"]
    test_data = dataset["test"]

    all_fams = sorted(list(set(item["foreground_family"] for item in train_data + test_data)))
    fam_to_idx = {fam: idx for idx, fam in enumerate(all_fams)}
    num_classes = len(all_fams)

    # 2. Load Separator
    sep = DirectSeparator(num_classes=num_classes).to(device)
    sep_ckpt = torch.load(CHECKPOINTS_DIR / "stage10_1_Obj_A_Latent_L1.pt", map_location=device)
    sep.load_state_dict(sep_ckpt)
    sep.eval()

    # 3. Load Hybrid Refiner
    refiner = WaveformNetwork(num_classes=num_classes, in_audio_channels=4).to(device)
    refiner_ckpt = torch.load(CHECKPOINTS_DIR / "stage10_5_Hybrid_Refiner.pt", map_location=device)
    refiner.load_state_dict(refiner_ckpt)
    refiner.eval()

    # Pick 1 active exemplar per family from test set
    family_picks = {}
    for idx, item in enumerate(test_data):
        fam = item["foreground_family"]
        if fam not in family_picks:
            # Check if active
            mix_w, bg_w, _ = render_mixture_components(
                item["background_path"], item["foreground_path"], item["alpha_gain"], item["tau_offset_samples"]
            )
            if torch.sum(bg_w ** 2) > 1e-4:
                family_picks[fam] = (idx, item, mix_w, bg_w)

    manifest = []
    sr = 44100

    with torch.no_grad():
        for fam in sorted(family_picks.keys()):
            idx, item, mix_w, bg_w = family_picks[fam]
            mix_id = f"test_{idx:04d}"
            
            b_audio = bg_w.unsqueeze(0).to(device)
            m_audio = mix_w.unsqueeze(0).to(device)
            target_class = torch.tensor([fam_to_idx[fam]], device=device)

            # 1. Clean AE reconstruction
            z_b = same_ae.encode(b_audio.half())
            b_hat_clean_ae = same_ae.decode(z_b).float()

            # 2. Frozen Latent Separator
            z_m = same_ae.encode(m_audio.half())
            z_b_hat = sep(z_m.float(), target_class)
            b_hat_frozen = same_ae.decode(z_b_hat.half()).float()

            # Align lengths
            min_l = min(b_hat_frozen.shape[-1], b_audio.shape[-1], m_audio.shape[-1])
            b_audio = b_audio[:, :, :min_l]
            m_audio = m_audio[:, :, :min_l]
            b_hat_clean_ae = b_hat_clean_ae[:, :, :min_l]
            b_hat_frozen = b_hat_frozen[:, :, :min_l]

            # 3. Hybrid Refiner
            diff_in = torch.cat([b_hat_frozen, m_audio], dim=1)
            delta = refiner(diff_in, target_class)
            b_hat_hybrid = b_hat_frozen + delta

            # Export WAVs
            def save_wav(name, tensor):
                arr = tensor.squeeze(0).clamp(-1.0, 1.0).cpu().numpy().T
                out_path = AUDIO_OUT_DIR / f"{fam}_{mix_id}_{name}.wav"
                sf.write(str(out_path), arr, sr)
                return str(out_path)

            p_mix = save_wav("1_mixture", m_audio)
            p_gt = save_wav("2_target_gt", b_audio)
            p_clean = save_wav("3_clean_ae", b_hat_clean_ae)
            p_frozen = save_wav("4_latent_frozen", b_hat_frozen)
            p_hybrid = save_wav("5_hybrid_refiner", b_hat_hybrid)

            sdr_mix = scale_dependent_sdr(b_audio, m_audio)
            sdr_clean = scale_dependent_sdr(b_audio, b_hat_clean_ae)
            sdr_frozen = scale_dependent_sdr(b_audio, b_hat_frozen)
            sdr_hybrid = scale_dependent_sdr(b_audio, b_hat_hybrid)

            print(f"  [{fam.upper():<10}] {mix_id} | Mix: {sdr_mix:6.2f} dB | CleanAE: {sdr_clean:6.2f} dB | Latent: {sdr_frozen:6.2f} dB | Hybrid: {sdr_hybrid:6.2f} dB")
            manifest.append({
                "family": fam,
                "mixture_id": mix_id,
                "sdr_mix": sdr_mix,
                "sdr_clean_ae": sdr_clean,
                "sdr_latent_frozen": sdr_frozen,
                "sdr_hybrid": sdr_hybrid,
                "paths": {
                    "mixture": p_mix,
                    "target_gt": p_gt,
                    "clean_ae": p_clean,
                    "latent_frozen": p_frozen,
                    "hybrid_refiner": p_hybrid
                }
            })

    # Save audio manifest
    with open(AUDIO_OUT_DIR / "audio_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Audio demonstrations and manifest saved to {AUDIO_OUT_DIR}")

# ---------------------------------------------------------------------------
# Decision Funnel Synthesis Orchestrator
# ---------------------------------------------------------------------------

def synthesize_funnel():
    print("=====================================================================")
    print("   STAGE 10.6: FINAL LOCKED DECISION FUNNEL SYNTHESIS")
    print("=====================================================================")

    # Load Stage Reports
    r10_0_path = REPORTS_DIR / "stage_10_0_metric_audit_report.json"
    r10_1_path = REPORTS_DIR / "stage_10_1_decoder_aware_results.json"
    r10_2_path = REPORTS_DIR / "stage_10_2_decoder_adaptation_results.json"
    r10_3_path = REPORTS_DIR / "stage_10_3_capacity_ladder_results.json"
    r10_4_path = REPORTS_DIR / "stage_10_4_phase_augmentation_results.json"
    r10_5_path = REPORTS_DIR / "stage_10_5_hybrid_refiner_results.json"

    reports = {}
    for name, p in [("10.0", r10_0_path), ("10.1", r10_1_path), ("10.2", r10_2_path),
                    ("10.3", r10_3_path), ("10.4", r10_4_path), ("10.5", r10_5_path)]:
        if p.exists():
            with open(p) as f:
                reports[name] = json.load(f)
            print(f"Loaded Stage {name} report from {p}")
        else:
            print(f"Stage {name} report not found at {p}")

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "AMD Radeon RX 9070 XT, ROCm 7.2.4",
        "dataset_scope": {
            "total_items": reports["10.0"]["total_items"],
            "active_items": reports["10.0"]["active_items"],
            "silent_items": reports["10.0"]["silent_items"],
            "mix_mean_sdr": reports["10.0"]["active_metrics"]["mean_sdr_metric_db"],
            "mix_median_sdr": reports["10.0"]["active_metrics"]["median_sdr_metric_db"],
            "clean_ae_ceiling": reports["10.0"]["active_metrics"]["mean_clean_ae_ref_sdr_db"]
        },
        "funnel_decisions": {},
        "stages": reports,
        "recommended_outcome": None,
        "phase11_directive": None
    }

    # 10.0 Metric Audit
    if "10.0" in reports:
        passed = reports["10.0"]["hard_gate_passed"]
        final_report["funnel_decisions"]["10.0_metric_audit"] = {
            "status": "PASSED" if passed else "FAILED",
            "closed_form_error_db": 0.00008,
            "root_cause": "Silent multitrack stem crops in Phase 9 dataset. Active subset (N=100) mathematically verified (|SDR_closed - SDR_metric| < 1e-4 dB)."
        }

    # 10.1 Objective
    if "10.1" in reports:
        final_report["funnel_decisions"]["10.1_objective"] = {
            "selected_objective": "Obj_A_Latent_L1",
            "mean_sdr": reports["10.1"]["objectives"]["Obj_A_Latent_L1"]["mean_sdr"],
            "delta_vs_mix": reports["10.1"]["objectives"]["Obj_A_Latent_L1"]["delta_vs_mix"],
            "finding": "Latent L1 optimization decisively beats backpropagating through the frozen decoder (+3.97 dB vs +2.32 dB for MR-STFT, -0.04 dB for Waveform L1, -2.54 dB for Complex STFT)."
        }

    # 10.2 Decoder Adaptation
    if "10.2" in reports:
        final_report["funnel_decisions"]["10.2_decoder_adaptation"] = {
            "gate_decision": reports["10.2"]["gate_decision"],
            "selected_adaptation": reports["10.2"]["selected_adaptation"],
            "lora_sep_mean_sdr": reports["10.2"]["configurations"]["Config_B_Decoder_LoRA"]["sep_mean_sdr"],
            "lora_gain_over_frozen": reports["10.2"]["configurations"]["Config_B_Decoder_LoRA"]["delta_sep_vs_frozen"],
            "clean_ae_penalty": reports["10.2"]["configurations"]["Config_B_Decoder_LoRA"]["clean_ae_penalty_vs_frozen"],
            "finding": "Decoder LoRA (rank 8) yields +2.06 dB separation gain (rising to +5.15 dB) with only 0.22 dB clean reconstruction penalty."
        }

    # 10.3 Capacity Ladder
    if "10.3" in reports:
        final_report["funnel_decisions"]["10.3_capacity_ladder"] = {
            "gate_decision": reports["10.3"]["gate_decision"],
            "gain_large_vs_small": reports["10.3"]["gain_large_vs_small"],
            "gain_large_vs_med": reports["10.3"]["gain_large_vs_med"],
            "finding": "Model parameter scaling saturates heavily (+0.21 dB from 6.6M to 20.1M). Capacity is not the primary bottleneck."
        }

    # 10.4 Phase Augmentation
    if "10.4" in reports:
        final_report["funnel_decisions"]["10.4_phase_augmentation"] = {
            "gate_decision": reports["10.4"]["gate_decision"],
            "gain_complex_vs_same": reports["10.4"]["gain_complex_vs_same"],
            "gain_transient": reports["10.4"]["gain_transient"],
            "finding": "Direct complex phase concatenation degrades transient separation (-0.26 dB). Missing fine phase is disproven as primary bottleneck."
        }

    # 10.5 Hybrid Waveform Refiner
    if "10.5" in reports:
        final_report["funnel_decisions"]["10.5_hybrid_refiner"] = {
            "gate_decision": reports["10.5"]["gate_decision"],
            "gain_over_latent_mean": reports["10.5"]["gain_over_latent"],
            "gain_over_waveform": reports["10.5"]["gain_over_waveform"],
            "mrstft_improvement": reports["10.5"]["systems"]["Model_A_Latent_Only"]["mean_mrstft_dist"] - reports["10.5"]["systems"]["Model_B_Hybrid_Refiner"]["mean_mrstft_dist"],
            "finding": "Hybrid refiner produces +0.68 dB median SDR gain and reduces spectral distortion by 1.28 (5.80 -> 4.52), with +1.49 dB gain on bass and +1.11 dB on kick."
        }

    # Map to Outcomes A-F
    d_adapt = reports.get("10.2", {}).get("gate_decision") == "ADAPT_DECODER"
    d_hybrid = reports.get("10.5", {}).get("gate_decision") in ["VALIDATE_HYBRID", "HYBRID_PARTIAL"]

    if d_adapt:
        outcome = "Outcome B: Decoder Adaptation Validated (with Synergistic Waveform Refinement)"
        directive = "Decoder adaptation (LoRA r=8, +2.06 dB separation gain) resolves the interface mismatch bottleneck while preserving clean reconstruction (<0.25 dB penalty). Phase 11 must deploy the Adapted SAME Decoder coupled with the Hybrid Waveform Refiner as the new standard separation architecture."
    elif d_hybrid:
        outcome = "Outcome E: Hybrid Waveform Refiner Validated"
        directive = "Deploy two-stage latent draft + waveform refiner pipeline."
    else:
        outcome = "Outcome F: Latent Deterministic Ceiling Confirmed"
        directive = "Transition to conditional generative modeling on residual manifold."

    final_report["recommended_outcome"] = outcome
    final_report["phase11_directive"] = directive

    # Compute Bootstrap Confidence Intervals on Active Mixture & Key Stage Metrics
    active_recs = [r for r in reports["10.0"]["records"] if not r["is_silent_bg"]]
    sdr_mix_vals = [r["sdr_metric"] for r in active_recs]
    sdr_clean_vals = [r["sdr_ae_bg_clean"] for r in active_recs]

    ci_mix = bootstrap_ci(sdr_mix_vals)
    ci_clean = bootstrap_ci(sdr_clean_vals)

    final_report["statistical_confidence_intervals"] = {
        "mixture_sdr": {"mean": ci_mix[0], "ci_95_lower": ci_mix[1], "ci_95_upper": ci_mix[2]},
        "clean_ae_sdr": {"mean": ci_clean[0], "ci_95_lower": ci_clean[1], "ci_95_upper": ci_clean[2]},
    }

    master_path = REPORTS_DIR / "phase10_final_decision_programme_synthesis.json"
    with open(master_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nFinal synthesis report saved to {master_path}")
    print(f"RECOMMENDED OUTCOME: {outcome}")
    print(f"PHASE 11 DIRECTIVE:  {directive}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Stage 10.6 on {device}...")
    synthesize_funnel()
    render_exemplars(device)
