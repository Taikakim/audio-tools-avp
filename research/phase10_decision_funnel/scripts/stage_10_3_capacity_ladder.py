#!/usr/bin/env python3
"""
Stage 10.3: Capacity and Context Ladder Benchmark
=================================================
Tests whether scaling separator model capacity (from ~1.2M to ~6.6M to ~20.1M)
unlocks higher separation SDR under the winning objective (Latent L1).

Evaluates 3 Scales:
  Model A: Small (~1.22M params, 2 ConvNeXt blocks, dim 256)
  Model B: Medium (~6.65M params, 3 ConvNeXt blocks, dim 512)
  Model C: Large (~20.13M params, 6 ConvNeXt blocks, dim 640)

All models parameterized with residual formulation:
  z_hat_b = z_m + g_theta(z_m, q)

Decision Rule:
  If the 15-25M model achieves >= 1.5 dB SDR gain over the 1.25M model with non-saturating curves,
  pursue a scaled deterministic latent separator before considering diffusion.
  If gains saturate (< 0.5 dB improvement from 6.6M to 20.1M), the bottleneck is not model capacity.
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
# Architecture: Scalable ConvNeXt-1D Separator
# ---------------------------------------------------------------------------

class Conv1DBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.GroupNorm(1, dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.transpose(1, 2)
        return res + x

class CapacitySeparator(nn.Module):
    def __init__(self, num_classes=32, latent_dim=256, embed_dim=128, hidden_dim=512, num_blocks=4):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        self.in_proj = nn.Conv1d(latent_dim + embed_dim, hidden_dim, 1)
        self.blocks = nn.Sequential(*[Conv1DBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_head = nn.Conv1d(hidden_dim, latent_dim, 1)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, z_m, q):
        e = self.class_embed(q).unsqueeze(2).repeat(1, 1, z_m.shape[2])
        x = torch.cat([z_m, e], dim=1)
        h = self.in_proj(x)
        h = self.blocks(h)
        delta = self.out_head(h)
        return z_m + delta

# ---------------------------------------------------------------------------
# Training Loop (Latent L1 Loss)
# ---------------------------------------------------------------------------

def train_ladder_model(model_name, model, train_loader, val_loader, device, epochs=30, lr=5e-4):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- Training {model_name} ({n_params/1e6:.2f}M params, {epochs} epochs, lr={lr}) ---")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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

            optimizer.zero_grad()
            z_hat_b = model(z_m, q)
            loss = F.l1_loss(z_hat_b, z_b)
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
                z_hat_b = model(z_m, q)
                val_loss += F.l1_loss(z_hat_b, z_b).item()
                n_val += 1

        val_loss /= n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] - Train: {train_loss:.4f}, Val: {val_loss:.4f} ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_weights)
    ckpt_path = CHECKPOINTS_DIR / f"stage10_3_{model_name}.pt"
    torch.save(best_weights, ckpt_path)
    print(f"  Saved best checkpoint to {ckpt_path}")
    return model

def main():
    print("=====================================================================")
    print("   STAGE 10.3: CAPACITY AND CONTEXT LADDER (1.2M vs 6.6M vs 20.1M)")
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

    # Filter active stems
    def extract_active_cache(data_split):
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
                    "family": item["foreground_family"]
                })
        return cache

    print("Extracting active items for training and test splits...")
    train_cache = extract_active_cache(train_data)
    val_cache = extract_active_cache(val_data)
    active_test_items = extract_active_cache(test_data)
    print(f"Active counts - Train: {len(train_cache)}, Val: {len(val_cache)}, Test: {len(active_test_items)}")

    def collate_fn(batch_list):
        return {
            "z_m": torch.stack([x["z_m"] for x in batch_list]),
            "z_b": torch.stack([x["z_b"] for x in batch_list]),
            "q": torch.stack([x["q"] for x in batch_list])
        }

    train_loader = torch.utils.data.DataLoader(train_cache, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_cache, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Initialize Ladder Models
    ladder_models = {
        "Model_A_Small_1.2M": CapacitySeparator(num_classes=num_classes, hidden_dim=256, num_blocks=2).to(device),
        "Model_B_Med_6.6M": CapacitySeparator(num_classes=num_classes, hidden_dim=512, num_blocks=3).to(device),
        "Model_C_Large_20.1M": CapacitySeparator(num_classes=num_classes, hidden_dim=640, num_blocks=6).to(device)
    }

    trained_ladder = {}
    for name, model in ladder_models.items():
        trained_ladder[name] = train_ladder_model(name, model, train_loader, val_loader, device, epochs=30, lr=5e-4)

    # -----------------------------------------------------------------------
    # Test Evaluation with SAME Decoder
    # -----------------------------------------------------------------------
    print("\n=====================================================================")
    print("   STAGE 10.3 TEST EVALUATION (100 ACTIVE TEST MIXTURES)")
    print("=====================================================================")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform
    same_ae.enable_grad = False

    eval_sdrs = {name: [] for name in trained_ladder}
    by_fam_sdrs = {name: {f: [] for f in all_fams} for name in trained_ladder}
    mix_sdrs = []

    t_eval0 = time.time()
    for idx, item in enumerate(active_test_items):
        z_m = item["z_m"].unsqueeze(0).to(device)
        q = item["q"].unsqueeze(0).to(device)
        bg_wav = item["bg_wav"].to(device)
        mix_wav = item["mix_wav"].to(device)
        fam = item["family"]

        sdr_mix = scale_dependent_sdr(bg_wav, mix_wav)
        mix_sdrs.append(sdr_mix)

        with torch.no_grad():
            for name, model in trained_ladder.items():
                z_hat_b = model(z_m, q)
                b_hat = same_ae.decode(z_hat_b.half()).float()
                sdr = scale_dependent_sdr(bg_wav, b_hat)
                eval_sdrs[name].append(sdr)
                by_fam_sdrs[name][fam].append(sdr)

        if (idx + 1) % 25 == 0 or idx == len(active_test_items) - 1:
            print(f"  Evaluated [{idx+1:3d}/{len(active_test_items)}] test items ({time.time() - t_eval0:.1f}s)")

    mean_mix = float(np.mean(mix_sdrs))
    median_mix = float(np.median(mix_sdrs))

    print("\n" + "=" * 80)
    print(f"{'Model / Capacity':<24} | {'Params (M)':<11} | {'Mean SDR':<10} | {'Median SDR':<11} | {'Delta vs Mix'}")
    print("-" * 80)
    print(f"{'Untouched Mixture':<24} | {'—':<11} | {mean_mix:>6.2f} dB  | {median_mix:>7.2f} dB   | 0.00 dB")

    summary_out = {
        "mix_mean_sdr": mean_mix,
        "mix_median_sdr": median_mix,
        "models": {}
    }

    for name, model in trained_ladder.items():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        mean_sdr = float(np.mean(eval_sdrs[name]))
        median_sdr = float(np.median(eval_sdrs[name]))
        delta_mix = mean_sdr - mean_mix
        print(f"{name:<24} | {n_params:>7.2f} M   | {mean_sdr:>6.2f} dB  | {median_sdr:>7.2f} dB   | {delta_mix:>+6.2f} dB")

        summary_out["models"][name] = {
            "params_m": n_params,
            "mean_sdr": mean_sdr,
            "median_sdr": median_sdr,
            "delta_vs_mix": delta_mix,
            "by_fam": {f: float(np.mean(v)) for f, v in by_fam_sdrs[name].items()}
        }

    print("=" * 80)

    # Decision Rule Check
    gain_a_to_c = summary_out["models"]["Model_C_Large_20.1M"]["mean_sdr"] - summary_out["models"]["Model_A_Small_1.2M"]["mean_sdr"]
    gain_b_to_c = summary_out["models"]["Model_C_Large_20.1M"]["mean_sdr"] - summary_out["models"]["Model_B_Med_6.6M"]["mean_sdr"]

    print("\n--- STAGE 10.3 DECISION GATE ---")
    print(f"Gain Large (20.1M) vs Small (1.2M): {gain_a_to_c:+.2f} dB")
    print(f"Gain Large (20.1M) vs Medium (6.6M): {gain_b_to_c:+.2f} dB")

    if gain_a_to_c >= 1.5 and gain_b_to_c >= 0.5:
        decision = "SCALE_CAPACITY: Scaling model capacity yields strong non-saturating gains (>=1.5 dB over 1.2M). Pursue scaled deterministic latent separator."
        summary_out["gate_decision"] = "SCALE_CAPACITY"
    else:
        decision = f"CAPACITY_SATURATION: Scaling from 6.6M to 20.1M gained only {gain_b_to_c:+.2f} dB (<0.5 dB). Bottleneck is NOT model capacity."
        summary_out["gate_decision"] = "CAPACITY_SATURATION"

    print(f"\nDECISION RESULT: {decision}")
    summary_out["gain_large_vs_small"] = gain_a_to_c
    summary_out["gain_large_vs_med"] = gain_b_to_c

    report_path = REPORTS_DIR / "stage_10_3_capacity_ladder_results.json"
    with open(report_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved Stage 10.3 results to {report_path}")

if __name__ == "__main__":
    main()
