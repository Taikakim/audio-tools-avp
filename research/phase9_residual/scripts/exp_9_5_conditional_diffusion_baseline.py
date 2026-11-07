#!/usr/bin/env python3
"""
Phase 9 — Stage C: Conditional Flow-Matching Generative Demixing
================================================================
Scientific Question:
  Does replacing point estimation (L1/L2 regression) with conditional generative
  modelling (Continuous Flow Matching) eliminate regression-to-the-mean blur
  and recover sharp, coherent audio backgrounds?
  Does conditional ambiguity exist (i.e. does the model sample multiple valid
  trajectories for the same mixture)?

Model Architecture:
  Conditional Latent Flow Demixer (Continuous Flow Matching):
    dz/dt = v_theta(z_t, t, z_m, q)
    z_t = (1 - t) * eps + t * z_b, where eps ~ N(0, I)
    Loss: E_{t, eps, z_b} || v_theta(z_t, t, z_m, q) - (z_b - eps) ||_2^2

Sampling:
  Euler integration from t=0 (Gaussian noise) to t=1 in 20 steps.
  Evaluated with K=1 single sample and K=3 diversity test.

Evaluation Metrics:
  - Decoded Background Waveform SDR (dB)
  - Sample Diversity Variance (measuring conditional ambiguity)
  - Comparison against Direct Point Model, Structured Demixer, Untouched Mixture, and AE Ceiling.
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

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] in [0, 1]
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class FlowMatchingResBlock(nn.Module):
    def __init__(self, dim=256, cond_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.gn2 = nn.GroupNorm(8, dim)
        self.cond_proj = nn.Linear(cond_dim, dim * 2) # AdaLN scale & shift

    def forward(self, x, cond):
        h = F.gelu(self.gn1(self.conv1(x)))
        scale_shift = self.cond_proj(cond).unsqueeze(2) # [B, 2*dim, 1]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        h = self.conv2(F.gelu(self.gn2(h)))
        return x + h

class LatentFlowDemixer(nn.Module):
    """
    Continuous Flow Matching Demixer for SAME Latents (R^{256 x 32}).
    Trained to predict the vector field v(z_t, t, z_m, q).
    Capacity: ~1.4M parameters.
    """
    def __init__(self, num_classes=32, latent_dim=256, hidden_dim=256, time_dim=128, class_dim=128, num_blocks=4):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.class_embed = nn.Embedding(num_classes, class_dim)

        self.cond_mlp = nn.Sequential(
            nn.Linear(time_dim + class_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # In-projection: takes z_t and condition z_m concatenated -> 512 channels
        self.in_conv = nn.Conv1d(latent_dim * 2, hidden_dim, 1)

        self.blocks = nn.ModuleList([
            FlowMatchingResBlock(hidden_dim, cond_dim=hidden_dim) for _ in range(num_blocks)
        ])

        self.out_norm = nn.GroupNorm(8, hidden_dim)
        self.out_conv = nn.Conv1d(hidden_dim, latent_dim, 1)

        # Initialize output layer to zero
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, z_t, t, z_m, q):
        # z_t: [B, 256, 32], t: [B] in [0, 1], z_m: [B, 256, 32], q: [B]
        t_emb = self.time_embed(t)
        q_emb = self.class_embed(q)
        cond = self.cond_mlp(torch.cat([t_emb, q_emb], dim=-1)) # [B, hidden_dim]

        x = torch.cat([z_t, z_m], dim=1) # [B, 512, 32]
        h = self.in_conv(x)

        for block in self.blocks:
            h = block(h, cond)

        v_pred = self.out_conv(F.gelu(self.out_norm(h)))
        return v_pred

    @torch.no_grad()
    def sample(self, z_m, q, steps=20):
        # Generate background estimate from noise via Euler integration
        B, C, T = z_m.shape
        z = torch.randn_like(z_m) # z_0 ~ N(0, I)
        dt = 1.0 / steps

        for step in range(steps):
            t_val = step / steps
            t_tensor = torch.full((B,), t_val, device=z_m.device, dtype=torch.float32)
            v = self.forward(z, t_tensor, z_m, q)
            z = z + v * dt

        return z

def run_stage_c():
    print("=====================================================================")
    print("   STAGE C: CONDITIONAL FLOW-MATCHING GENERATIVE DEMIXER")
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

    all_fams = sorted(list(set(item["foreground_family"] for item in train_data + test_data)))
    fam_to_idx = {fam: idx for idx, fam in enumerate(all_fams)}

    def prepare_tensors(data_list):
        zm_t = torch.stack([item["z_m"] for item in data_list], dim=0)
        zb_t = torch.stack([item["z_b"] for item in data_list], dim=0)
        q_t = torch.tensor([fam_to_idx[item["foreground_family"]] for item in data_list], dtype=torch.long)
        return zm_t, zb_t, q_t

    train_zm, train_zb, train_q = prepare_tensors(train_data)
    val_zm, val_zb, val_q = prepare_tensors(val_data)
    test_zm, test_zb, test_q = prepare_tensors(test_data)

    model = LatentFlowDemixer(num_classes=len(all_fams)).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    BATCH_SIZE = 32
    EPOCHS = 35
    num_train_batches = math.ceil(len(train_data) / BATCH_SIZE)

    print(f"\n--- Training Latent Flow Matching Demixer ({EPOCHS} Epochs) ---")
    t0 = time.time()
    best_val_loss = float("inf")
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(train_data))
        ep_loss = 0.0

        for b_idx in range(num_train_batches):
            idx_b = perm[b_idx * BATCH_SIZE : (b_idx + 1) * BATCH_SIZE]
            zm_b = train_zm[idx_b].to(device)
            zb_b = train_zb[idx_b].to(device)
            q_b = train_q[idx_b].to(device)
            B = zm_b.shape[0]

            # Sample random time t ~ Uniform(0, 1)
            t = torch.rand(B, device=device)

            # Sample standard Gaussian noise eps ~ N(0, I)
            eps = torch.randn_like(zb_b)

            # Linear interpolation trajectory: z_t = (1 - t) * eps + t * zb
            t_expand = t.view(B, 1, 1)
            z_t = (1.0 - t_expand) * eps + t_expand * zb_b

            # Target vector field: v_true = zb - eps
            v_true = zb_b - eps

            # Predict vector field
            v_pred = model(z_t, t, zm_b, q_b)

            loss = F.mse_loss(v_pred, v_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        # Val
        model.eval()
        val_loss = 0.0
        val_batches = math.ceil(len(val_data) / BATCH_SIZE)
        with torch.no_grad():
            for vb_idx in range(val_batches):
                v_zm = val_zm[vb_idx * BATCH_SIZE : (vb_idx + 1) * BATCH_SIZE].to(device)
                v_zb = val_zb[vb_idx * BATCH_SIZE : (vb_idx + 1) * BATCH_SIZE].to(device)
                v_q = val_q[vb_idx * BATCH_SIZE : (vb_idx + 1) * BATCH_SIZE].to(device)
                VB = v_zm.shape[0]

                v_t = torch.rand(VB, device=device)
                v_eps = torch.randn_like(v_zb)
                v_zt = (1.0 - v_t.view(VB, 1, 1)) * v_eps + v_t.view(VB, 1, 1) * v_zb
                v_vtrue = v_zb - v_eps

                v_pred = model(v_zt, v_t, v_zm, v_q)
                val_loss += F.mse_loss(v_pred, v_vtrue).item()

        val_loss /= val_batches
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            print(f"  Flow Ep [{ep+1:2d}/{EPOCHS:2d}] - Train CFM Loss: {ep_loss/num_train_batches:.4f} - Val CFM Loss: {val_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"Flow Matching Demixer trained in {time.time() - t0:.1f}s, Best Val Loss: {best_val_loss:.4f}")

    # Audio Benchmark on Test Subset
    print("\n--- Evaluating Generative Sampling & Audio Waveform SDR ---")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform.eval()

    test_subset = test_data[:30]
    sdr_flow_single = []
    sdr_untouched = []
    sdr_ceiling = []
    diversity_variances = []

    for idx, item in enumerate(test_subset):
        mix_w, bg_w, target_w = render_mixture_components(
            item["background_path"],
            item["foreground_path"],
            item["alpha_gain"],
            item["tau_offset_samples"]
        )

        zm_item = item["z_m"].unsqueeze(0).to(device)
        zb_item = item["z_b"].unsqueeze(0).to(device)
        q_item = torch.tensor([fam_to_idx[item["foreground_family"]]], dtype=torch.long, device=device)

        # Baseline SDRs
        sdr_untouched.append(waveform_sdr(bg_w, mix_w))
        with torch.no_grad():
            bg_clean = same_ae.decode(zb_item.half()).float()[0].cpu()
        sdr_ceiling.append(waveform_sdr(bg_w, bg_clean))

        # Flow Matching Sampling (K=1)
        z_sample_1 = model.sample(zm_item, q_item, steps=20)
        with torch.no_grad():
            bg_gen = same_ae.decode(z_sample_1.half()).float()[0].cpu()
        sdr_flow_single.append(waveform_sdr(bg_w, bg_gen))

        # Multi-Sample Diversity Test (K=3)
        z_sample_2 = model.sample(zm_item, q_item, steps=20)
        z_sample_3 = model.sample(zm_item, q_item, steps=20)
        samples_stack = torch.stack([z_sample_1, z_sample_2, z_sample_3], dim=0) # [3, 1, 256, 32]
        sample_var = float(samples_stack.var(dim=0).mean().cpu())
        diversity_variances.append(sample_var)

    m_untouched = float(np.mean(sdr_untouched))
    m_ceiling = float(np.mean(sdr_ceiling))
    m_flow = float(np.mean(sdr_flow_single))
    m_div_var = float(np.mean(diversity_variances))

    print("\n=====================================================================")
    print("   STAGE C BENCHMARK RESULTS: FLOW MATCHING GENERATIVE DEMIXING")
    print("=====================================================================")
    print(f"{'Method / Model':<35} | {'Decoded SDR':<11} | {'vs Untouched':<14}")
    print("-" * 68)
    print(f"{'Untouched Mixture (Do-Nothing)':<35} | {m_untouched:+7.2f} dB   | {'0.00 dB':<14}")
    print(f"{'Conditional Flow Demixer (20 steps)':<35} | {m_flow:+7.2f} dB   | {m_flow - m_untouched:+7.2f} dB")
    print(f"{'Clean AE Ceiling D(E(b))':<35} | {m_ceiling:+7.2f} dB   | {m_ceiling - m_untouched:+7.2f} dB")
    print("-" * 68)
    print(f"Sample Diversity Variance across K=3 trajectories: {m_div_var:.5f}")

    # Report
    report = {
        "experiment": "Phase 9 — Stage C: Conditional Flow-Matching Generative Demixing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "untouched_mix_sdr": m_untouched,
            "flow_matching_sdr": m_flow,
            "clean_ae_ceiling_sdr": m_ceiling,
            "sample_diversity_variance": m_div_var
        },
        "gain_vs_untouched_db": m_flow - m_untouched
    }

    report_path = REPORTS_DIR / "exp_9_5_stage_c_flow_demixer_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved Stage C master report to {report_path}")

    print("\n>>> Stage C Completed Successfully! <<<")
    os._exit(0)

if __name__ == "__main__":
    run_stage_c()
