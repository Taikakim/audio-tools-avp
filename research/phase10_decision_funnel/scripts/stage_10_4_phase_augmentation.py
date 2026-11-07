#!/usr/bin/env python3
"""
Stage 10.4: Phase-Information Augmentation Benchmark
===================================================
Directly probes the hypothesis that SAME latents lack fine phase information
required for complete separation, particularly on transients (hi-hats, snares).

Evaluates 3 Conditioning Modes:
  Model A: SAME latent only: z_hat_b = z_m + g_theta(z_m, q)
  Model B: SAME latent + Magnitude STFT: z_hat_b = z_m + g_theta(z_m, |STFT(x_m)|, q)
  Model C: SAME latent + Complex STFT: z_hat_b = z_m + g_theta(z_m, Real(STFT(x_m)), Imag(STFT(x_m)), q)

Decision Rule:
  If complex STFT features provide >= 1.0 dB gain over SAME-only, particularly on transients
  (hi-hats, snares), phase information is the missing ingredient.
  If complex STFT features provide < 0.3 dB gain, the phase hypothesis is disproven as a primary bottleneck.
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
# Architecture: Phase-Augmented Separator
# ---------------------------------------------------------------------------

class PhaseAugmentedSeparator(nn.Module):
    def __init__(self, mode="same_only", num_classes=32, latent_dim=256, embed_dim=128, stft_dim=64, hidden_dim=256, num_layers=4):
        super().__init__()
        self.mode = mode
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        in_channels = latent_dim + embed_dim
        if mode == "mag_stft":
            self.stft_proj = nn.Conv1d(513, stft_dim, 1)
            in_channels += stft_dim
        elif mode == "complex_stft":
            self.stft_proj = nn.Conv1d(1026, stft_dim, 1)
            in_channels += stft_dim

        self.in_proj = nn.Conv1d(in_channels, hidden_dim, 1)
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

    def extract_stft_features(self, mix_wav, target_len):
        # mix_wav: [B, 2, L]
        x_mono = mix_wav.mean(dim=1)
        n_fft = 1024
        hop_length = 512
        window = torch.hann_window(n_fft, device=mix_wav.device)
        stft = torch.stft(x_mono, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, return_complex=True)

        if self.mode == "mag_stft":
            mag = torch.abs(stft) + 1e-7 # [B, 513, T_stft]
            mag_pooled = F.adaptive_avg_pool1d(mag, target_len)
            return self.stft_proj(mag_pooled)
        elif self.mode == "complex_stft":
            comp = torch.cat([stft.real, stft.imag], dim=1) # [B, 1026, T_stft]
            comp_pooled = F.adaptive_avg_pool1d(comp, target_len)
            return self.stft_proj(comp_pooled)
        return None

    def forward(self, z_m, q, mix_wav=None):
        e = self.class_embed(q).unsqueeze(2).repeat(1, 1, z_m.shape[2])
        if self.mode == "same_only":
            x = torch.cat([z_m, e], dim=1)
        else:
            assert mix_wav is not None, "mix_wav required for STFT augmented modes"
            stft_feat = self.extract_stft_features(mix_wav, z_m.shape[2])
            x = torch.cat([z_m, e, stft_feat], dim=1)

        h = self.in_proj(x)
        h = h + self.trunk(h)
        delta = self.out_head(h)
        return z_m + delta

# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------

def train_phase_model(mode_name, model, train_loader, val_loader, device, epochs=25, lr=5e-4):
    print(f"\n--- Training Phase-Augmented Model: {mode_name} ({epochs} epochs, lr={lr}) ---")
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
            mix_wav = batch["mix_wav"].to(device)

            optimizer.zero_grad()
            z_hat_b = model(z_m, q, mix_wav)
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
                mix_wav = batch["mix_wav"].to(device)

                z_hat_b = model(z_m, q, mix_wav)
                val_loss += F.l1_loss(z_hat_b, z_b).item()
                n_val += 1

        val_loss /= n_val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] - Train: {train_loss:.4f}, Val: {val_loss:.4f} ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_weights)
    ckpt_path = CHECKPOINTS_DIR / f"stage10_4_{mode_name}.pt"
    torch.save(best_weights, ckpt_path)
    print(f"  Saved best checkpoint to {ckpt_path}")
    return model

def main():
    print("=====================================================================")
    print("   STAGE 10.4: PHASE-INFORMATION AUGMENTATION BENCHMARK")
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

    models = {
        "SAME_Only": PhaseAugmentedSeparator(mode="same_only", num_classes=num_classes).to(device),
        "SAME_plus_MagSTFT": PhaseAugmentedSeparator(mode="mag_stft", num_classes=num_classes).to(device),
        "SAME_plus_ComplexSTFT": PhaseAugmentedSeparator(mode="complex_stft", num_classes=num_classes).to(device)
    }

    # Check if all checkpoints already exist
    all_ckpts_exist = all((CHECKPOINTS_DIR / f"stage10_4_{name}.pt").exists() for name in models)

    trained_models = {}
    if all_ckpts_exist:
        print("Found existing trained checkpoints for all 3 models. Loading directly...")
        for name, m in models.items():
            ckpt_path = CHECKPOINTS_DIR / f"stage10_4_{name}.pt"
            m.load_state_dict(torch.load(ckpt_path, map_location=device))
            m.eval()
            trained_models[name] = m
    else:
        print("Extracting active items for training splits...")
        train_cache = extract_active_cache(train_data)
        val_cache = extract_active_cache(val_data)
        print(f"Active counts - Train: {len(train_cache)}, Val: {len(val_cache)}")

        def collate_fn(batch_list):
            return {
                "z_m": torch.stack([x["z_m"] for x in batch_list]),
                "z_b": torch.stack([x["z_b"] for x in batch_list]),
                "q": torch.stack([x["q"] for x in batch_list]),
                "mix_wav": torch.stack([x["mix_wav"] for x in batch_list])
            }

        train_loader = torch.utils.data.DataLoader(train_cache, batch_size=8, shuffle=True, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_cache, batch_size=8, shuffle=False, collate_fn=collate_fn)

        for name, m in models.items():
            trained_models[name] = train_phase_model(name, m, train_loader, val_loader, device, epochs=25, lr=5e-4)

    print("Extracting active test items for evaluation...")
    active_test_items = extract_active_cache(test_data)
    print(f"Active test items: {len(active_test_items)}")

    # -----------------------------------------------------------------------
    # Test Evaluation with SAME Decoder
    # -----------------------------------------------------------------------
    print("\n=====================================================================")
    print("   STAGE 10.4 TEST EVALUATION (100 ACTIVE TEST MIXTURES)")
    print("=====================================================================")
    sa_model = StableAudioModel.from_pretrained("medium-base", device=device)
    same_ae = sa_model.model.pretransform
    same_ae.enable_grad = False

    eval_sdrs = {name: [] for name in trained_models}
    by_fam_sdrs = {name: {f: [] for f in all_fams} for name in trained_models}
    mix_sdrs = []

    t_eval0 = time.time()
    for idx, item in enumerate(active_test_items):
        z_m = item["z_m"].unsqueeze(0).to(device)
        q = item["q"].unsqueeze(0).to(device)
        bg_wav = item["bg_wav"].to(device)
        mix_wav = item["mix_wav"].unsqueeze(0).to(device)
        fam = item["family"]

        sdr_mix = scale_dependent_sdr(bg_wav, mix_wav[0])
        mix_sdrs.append(sdr_mix)

        with torch.no_grad():
            for name, model in trained_models.items():
                z_hat_b = model(z_m, q, mix_wav)
                b_hat = same_ae.decode(z_hat_b.half()).float()
                sdr = scale_dependent_sdr(bg_wav, b_hat[0])
                eval_sdrs[name].append(sdr)
                by_fam_sdrs[name][fam].append(sdr)

        if (idx + 1) % 25 == 0 or idx == len(active_test_items) - 1:
            print(f"  Evaluated [{idx+1:3d}/{len(active_test_items)}] test items ({time.time() - t_eval0:.1f}s)")

    mean_mix = float(np.mean(mix_sdrs))
    median_mix = float(np.median(mix_sdrs))

    print("\n" + "=" * 88)
    print(f"{'Conditioning Mode':<24} | {'Mean SDR':<10} | {'Median SDR':<11} | {'Delta vs Mix':<13} | {'Delta vs SAME'}")
    print("-" * 88)
    print(f"{'Untouched Mixture':<24} | {mean_mix:>6.2f} dB  | {median_mix:>7.2f} dB   | 0.00 dB       | —")

    same_mean = float(np.mean(eval_sdrs["SAME_Only"]))

    summary_out = {
        "mix_mean_sdr": mean_mix,
        "mix_median_sdr": median_mix,
        "models": {}
    }

    for name in trained_models:
        mean_sdr = float(np.mean(eval_sdrs[name]))
        median_sdr = float(np.median(eval_sdrs[name]))
        delta_mix = mean_sdr - mean_mix
        delta_same = mean_sdr - same_mean
        print(f"{name:<24} | {mean_sdr:>6.2f} dB  | {median_sdr:>7.2f} dB   | {delta_mix:>+7.2f} dB     | {delta_same:>+6.2f} dB")

        summary_out["models"][name] = {
            "mean_sdr": mean_sdr,
            "median_sdr": median_sdr,
            "delta_vs_mix": delta_mix,
            "delta_vs_same_only": delta_same,
            "by_fam": {f: float(np.mean(v)) for f, v in by_fam_sdrs[name].items()}
        }

    print("=" * 88)

    # Breakdown for transient families (hihat, snare)
    print("\n--- Transient Families Breakdown (Hi-Hat & Snare) ---")
    print(f"{'Mode':<24} | {'Hi-Hat SDR':<12} | {'Snare SDR':<12} | {'Transient Mean'}")
    print("-" * 64)
    for name in trained_models:
        hihat_sdr = summary_out["models"][name]["by_fam"]["hihat"]
        snare_sdr = summary_out["models"][name]["by_fam"]["snare"]
        trans_mean = (hihat_sdr + snare_sdr) / 2.0
        print(f"{name:<24} | {hihat_sdr:>8.2f} dB   | {snare_sdr:>8.2f} dB   | {trans_mean:>8.2f} dB")

    # Decision Rule Check
    gain_complex_vs_same = summary_out["models"]["SAME_plus_ComplexSTFT"]["delta_vs_same_only"]
    transient_same = (summary_out["models"]["SAME_Only"]["by_fam"]["hihat"] + summary_out["models"]["SAME_Only"]["by_fam"]["snare"]) / 2.0
    transient_complex = (summary_out["models"]["SAME_plus_ComplexSTFT"]["by_fam"]["hihat"] + summary_out["models"]["SAME_plus_ComplexSTFT"]["by_fam"]["snare"]) / 2.0
    gain_transient = transient_complex - transient_same

    print("\n--- STAGE 10.4 DECISION GATE ---")
    print(f"Overall Complex STFT Gain vs SAME: {gain_complex_vs_same:+.2f} dB")
    print(f"Transient (Hi-Hat + Snare) Gain:    {gain_transient:+.2f} dB")

    if gain_complex_vs_same >= 1.0 or gain_transient >= 1.0:
        decision = "PHASE_CRITICAL: Complex STFT features provide >=1.0 dB gain. Missing phase is a primary bottleneck."
        summary_out["gate_decision"] = "PHASE_CRITICAL"
    elif gain_complex_vs_same < 0.3 and gain_transient < 0.3:
        decision = f"PHASE_DISPROVEN: Complex STFT gained only {gain_complex_vs_same:+.2f} dB (<0.3 dB). Phase is NOT the primary bottleneck."
        summary_out["gate_decision"] = "PHASE_DISPROVEN"
    else:
        decision = f"PHASE_MARGINAL: Complex STFT gained {gain_complex_vs_same:+.2f} dB (marginal/inconclusive)."
        summary_out["gate_decision"] = "PHASE_MARGINAL"

    print(f"\nDECISION RESULT: {decision}")
    summary_out["gain_complex_vs_same"] = gain_complex_vs_same
    summary_out["gain_transient"] = gain_transient

    report_path = REPORTS_DIR / "stage_10_4_phase_augmentation_results.json"
    with open(report_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nSaved Stage 10.4 results to {report_path}")

if __name__ == "__main__":
    main()
