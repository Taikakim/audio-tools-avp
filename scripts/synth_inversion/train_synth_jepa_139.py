"""Synth-JEPA 139-Parameter Training Script.

Reference: Hayes, Tian, Lattner, "Synth-JEPA: Joint Embedding Prediction for Renderer-Free
Synthesizer Parameter Search", arXiv:2609.31024 (Sept 2026).

Trains the full 139-parameter model on 3.0-second stereo procedural audio rendered
online by CPU workers, with audio effects strictly disabled.

v2 (2026-10-03 review):
  * Enum classes come from the plugin (surge_139_spec.calibrate_enums), and the parameter names
    are verified to exist; see that module's docstring for what v1 got wrong.
  * Online data is new every epoch: seeds advance by samples_per_epoch per epoch (v1 used
    min_seed + idx, so every epoch re-rendered the same 100k patches).
  * Retrieval goes through the predictors, as in the paper's objective: audio->param ranks
    candidates by ||a_to_p(z_a) - z_p||. v1 took the cosine between z_a and z_p directly; the two
    encoders live in different spaces, so that number measured nothing, and the "best"
    checkpoint was chosen by it. Best is now chosen by validation JEPA loss.
  * Silent renders (every oscillator muted) are redrawn.
  * lambda_sig is an argument (v1 hard-coded 0.1; train_synth_jepa.py defaults to 1.0; the paper
    does not give it). Gradient clipping at 1.0; checkpoints are written atomically and carry
    the enum tables and model kwargs.
"""

import argparse
import json
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from audio_utils import ExactGpuMel
from surge_139_spec import (CONT_INDICES, DEFAULT_PLUGIN_PATH, calibrate_enums, cat_indices, draw_patch_139,
                            init_synth_139, patch_to_vector_139, render_patch_139)
from synth_jepa_model import (
    AudioTransformerEncoder,
    CrossDomainPredictor,
    PerceiverParameterEncoder,
    SIGRegLoss,
)

SEED_RETRY_STRIDE = 1_000_000_000  # redraw offset for a silent render; far outside every seed range


class WelfordNormalizer139:
    """Channel-wise (per mel band across stereo channels) running mean/variance."""
    def __init__(self, n_channels: int = 256, eps: float = 1e-5):
        self.count = 0
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.M2 = np.zeros(n_channels, dtype=np.float64)
        self.eps = eps
        self.frozen = False
        self.frozen_mean = None
        self.frozen_std = None

    def update(self, mel_batch: np.ndarray):
        if self.frozen:
            raise RuntimeError("WelfordNormalizer is frozen")
        # mel_batch: [B, 256, T]
        B, C, T = mel_batch.shape
        x_flat = mel_batch.transpose(0, 2, 1).reshape(-1, C)
        n = x_flat.shape[0]
        batch_mean = np.mean(x_flat, axis=0)
        batch_m2 = np.sum((x_flat - batch_mean) ** 2, axis=0)

        new_count = self.count + n
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (n / new_count)
        self.M2 = self.M2 + batch_m2 + delta ** 2 * (self.count * n / new_count)
        self.count = new_count

    def freeze(self):
        if self.count < 2:
            raise RuntimeError("need at least 2 samples to freeze")
        var = self.M2 / (self.count - 1)
        self.frozen_mean = torch.from_numpy(self.mean.astype(np.float32)).view(1, -1, 1)
        self.frozen_std = torch.from_numpy(np.sqrt(var + self.eps).astype(np.float32)).view(1, -1, 1)
        self.frozen = True

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.frozen:
            raise RuntimeError("Cannot normalize with unfrozen Welford stats")
        mean = self.frozen_mean.to(mel.device)
        std = self.frozen_std.to(mel.device)
        return (mel - mean) / std

    def state_dict(self):
        return {"mean": self.frozen_mean, "std": self.frozen_std, "count": self.count}

    def load_state_dict(self, d):
        self.frozen_mean = d["mean"]
        self.frozen_std = d["std"]
        self.count = d["count"]
        self.frozen = True


class SurgeOnlineDataset139(Dataset):
    """Online procedural dataset rendering 3.0 s stereo audio on CPU workers.
    Sample idx of epoch e uses seed min_seed + e * length + idx (set .epoch before iterating)."""
    def __init__(self, tables, length: int = 160000, min_seed: int = 1_000_000,
                 plugin_path: str = DEFAULT_PLUGIN_PATH):
        self.tables = tables
        self.length = length
        self.min_seed = min_seed
        self.plugin_path = plugin_path
        self.epoch = 0
        self._synth = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        if self._synth is None:
            self._synth = init_synth_139(self.plugin_path, verify=False)  # names verified in the main process
        base = self.min_seed + self.epoch * self.length + idx
        for attempt in range(16):
            patch = draw_patch_139(base + attempt * SEED_RETRY_STRIDE, self.tables)
            audio = render_patch_139(self._synth, patch, patch["midi_note"], patch["note_dur"], self.tables)
            if audio is not None:
                return torch.from_numpy(audio), torch.from_numpy(patch_to_vector_139(patch))
        raise RuntimeError(f"16 consecutive silent renders from seed {base}; check the renderer")


class SynthJEPA139(nn.Module):
    """Full 139-Parameter Synth-JEPA model matching Hayes et al. (arXiv:2609.31024)."""
    def __init__(
        self,
        cat_sizes: List[int],
        embed_dim: int = 512,
        predictor_hidden: int = 1024,
        num_audio_layers: int = 8,
        num_param_layers: int = 8,
        num_slices_sigreg: int = 64,
        ff_dim: int = 1024,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # Stereo mel: 2 channels * 128 = 256 in_mels, 301 frames -> 150 conv tokens + [CLS]
        self.audio_encoder = AudioTransformerEncoder(
            in_mels=256,
            embed_dim=embed_dim,
            num_layers=num_audio_layers,
            num_heads=8,
            ff_dim=ff_dim,
            in_frames=301,
            token_stride=2,
        )
        self.param_encoder = PerceiverParameterEncoder(
            num_continuous=len(CONT_INDICES),
            cat_sizes=list(cat_sizes),
            embed_dim=embed_dim,
            num_latents=32,
            num_blocks=num_param_layers,
            num_heads=8,
            ff_dim=ff_dim,
        )
        self.p_to_a = CrossDomainPredictor(embed_dim, predictor_hidden)
        self.a_to_p = CrossDomainPredictor(embed_dim, predictor_hidden)
        self.sigreg = SIGRegLoss(num_slices=num_slices_sigreg)

    def forward(self, mel: torch.Tensor, cont_params: torch.Tensor, cat_params: List[torch.Tensor]):
        za = self.audio_encoder(mel)
        zp = self.param_encoder(cont_params, cat_params)
        z_hat_a = self.p_to_a(zp)
        z_hat_p = self.a_to_p(za)
        return za, zp, z_hat_a, z_hat_p


def prepare_inputs_139(params: torch.Tensor, cat_idx):
    cont = params[:, CONT_INDICES] * 2.0 - 1.0
    cat_onehots = [
        F.one_hot(torch.round(params[:, i] * (k - 1)).clamp(0, k - 1).long(), k).float()
        for i, k in cat_idx
    ]
    return cont, cat_onehots


def jepa_losses(model, za, zp, z_hat_a, z_hat_p, lambda_sig, generator=None):
    loss_p = F.mse_loss(z_hat_a, za.detach())
    loss_a = F.mse_loss(z_hat_p, zp.detach())
    sig_a = model.sigreg(za, generator=generator)
    sig_p = model.sigreg(zp, generator=generator)
    return loss_p + loss_a + lambda_sig * (sig_a + sig_p), loss_p, loss_a, sig_a, sig_p


def wsd_lambda(total_steps, warmup_frac=0.05, decay_frac=0.20):
    warmup = max(1, int(warmup_frac * total_steps))
    decay = max(1, int(decay_frac * total_steps))
    decay_start = total_steps - decay

    def f(step):
        if step < warmup:
            return step / warmup
        if step < decay_start:
            return 1.0
        return max(0.0, (total_steps - step) / decay)
    return f


def stereo_mel(mel_extractor, audio):
    return torch.cat([mel_extractor(audio[:, 0]), mel_extractor(audio[:, 1])], dim=1)


@torch.no_grad()
def validate(model, val_loader, mel_extractor, normalizer, cat_idx, lambda_sig, device):
    """Validation JEPA loss (fixed SIGReg slices) and retrieval through the predictors over the
    whole validation set (chance = 1/N)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    all_za, all_zp, loss_sum, n = [], [], 0.0, 0
    for audio, p_vec in val_loader:
        audio = audio.to(device)
        cont, cat = prepare_inputs_139(p_vec.to(device), cat_idx)
        za, zp, z_hat_a, z_hat_p = model(normalizer(stereo_mel(mel_extractor, audio)), cont, cat)
        loss = jepa_losses(model, za, zp, z_hat_a, z_hat_p, lambda_sig, generator=gen)[0]
        loss_sum += float(loss) * za.shape[0]
        n += za.shape[0]
        all_za.append(za)
        all_zp.append(zp)
    za_all, zp_all = torch.cat(all_za), torch.cat(all_zp)
    targets = torch.arange(len(za_all), device=device)
    a_to_p_acc = (torch.cdist(model.a_to_p(za_all), zp_all).argmin(dim=1) == targets).float().mean().item() * 100.0
    p_to_a_acc = (torch.cdist(model.p_to_a(zp_all), za_all).argmin(dim=1) == targets).float().mean().item() * 100.0
    model.train()
    return loss_sum / n, a_to_p_acc, p_to_a_acc, 100.0 / len(za_all)


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="/run/media/kim/Mantu/surge_200k_models/synth_jepa_139_runs")
    parser.add_argument("--run_id", default="jepa_139_online_v2")
    parser.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_sig", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--norm_spectrograms", type=int, default=8000)
    parser.add_argument("--samples_per_epoch", type=int, default=100000)
    parser.add_argument("--val_samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def train(args, plugin_factory=None):
    save_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"=== Full 139-Parameter Synth-JEPA Training on {device} ===")
    print(f"Checkpoints: {save_dir}")

    plugin = (plugin_factory or (lambda: init_synth_139(args.plugin, verify=True)))()
    tables = calibrate_enums(plugin)
    del plugin
    cat_idx = cat_indices(tables)
    print("Enum classes from the plugin: " + ", ".join(f"{n}={len(t)}" for n, t in tables.items()))
    with open(os.path.join(save_dir, "enum_tables.json"), "w") as f:
        json.dump(tables, f, indent=1)

    mel_extractor = ExactGpuMel(sample_rate=44100, n_fft=1102, hop_length=441, n_mels=128, device=device)

    # Seed ranges: calibration 500k.., validation 1M.. (fixed), training 2M + epoch * samples_per_epoch
    train_ds = SurgeOnlineDataset139(tables, length=args.samples_per_epoch, min_seed=2_000_000,
                                     plugin_path=args.plugin)
    val_ds = SurgeOnlineDataset139(tables, length=args.val_samples, min_seed=1_000_000, plugin_path=args.plugin)
    assert 1_000_000 + args.val_samples <= 2_000_000 and 500_000 + args.norm_spectrograms <= 1_000_000

    mp_ctx = torch.multiprocessing.get_context("spawn") if args.num_workers > 0 else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
                              multiprocessing_context=mp_ctx)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(4, args.num_workers), pin_memory=device.type == "cuda",
                            multiprocessing_context=mp_ctx if args.num_workers > 0 else None)

    print(f"Calibrating channel-wise Welford normalizer on {args.norm_spectrograms} spectrograms...")
    normalizer = WelfordNormalizer139(n_channels=256)
    calib_ds = SurgeOnlineDataset139(tables, length=args.norm_spectrograms, min_seed=500_000, plugin_path=args.plugin)
    calib_loader = DataLoader(calib_ds, batch_size=64, shuffle=False, num_workers=args.num_workers,
                              multiprocessing_context=mp_ctx)
    with torch.no_grad():
        for audio, _ in calib_loader:
            normalizer.update(stereo_mel(mel_extractor, audio.to(device)).cpu().numpy())
    normalizer.freeze()
    torch.save(normalizer.state_dict(), os.path.join(save_dir, "welford_stats.pt"))
    print("Welford statistics calibrated and frozen successfully.")

    model_kwargs = dict(cat_sizes=[k for _, k in cat_idx], embed_dim=512, predictor_hidden=1024,
                        num_audio_layers=8, num_param_layers=8, num_slices_sigreg=64, ff_dim=1024)
    model = SynthJEPA139(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Instantiated SynthJEPA139: {n_params:.1f}M parameters (Hayes paper target: ~53M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lambda(total_steps))

    best_val = float("inf")
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ds.epoch = epoch - 1  # new seeds every epoch (workers are re-created per epoch)
        sums = {"loss": 0.0, "lp": 0.0, "la": 0.0, "sa": 0.0, "sp": 0.0}
        n_b = 0
        t0 = time.time()

        for audio, p_vec in train_loader:
            step += 1
            audio = audio.to(device, non_blocking=True)
            p_vec = p_vec.to(device, non_blocking=True)
            with torch.no_grad():
                mel = normalizer(stereo_mel(mel_extractor, audio))
            cont, cat = prepare_inputs_139(p_vec, cat_idx)
            za, zp, z_hat_a, z_hat_p = model(mel, cont, cat)
            loss, loss_p, loss_a, sigreg_a, sigreg_p = jepa_losses(model, za, zp, z_hat_a, z_hat_p, args.lambda_sig)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            for k, v in zip(sums, (loss, loss_p, loss_a, sigreg_a, sigreg_p)):
                sums[k] += float(v.detach())
            n_b += 1

            if step % 200 == 0:
                dt = time.time() - t0
                print(f"Step {step:6d}/{total_steps} | Loss: {float(loss):.4f} (Lp={float(loss_p):.4f}, "
                      f"La={float(loss_a):.4f}, SigA={float(sigreg_a):.3f}) | {200 * args.batch_size / dt:.1f} samp/s")
                t0 = time.time()

        val_loss, a_acc, p_acc, chance = validate(model, val_loader, mel_extractor, normalizer, cat_idx,
                                                  args.lambda_sig, device)
        print(f"\n=== Epoch {epoch}/{args.epochs} Complete ===")
        print(f"Train loss {sums['loss'] / max(1, n_b):.4f} | Val loss {val_loss:.4f} | Retrieval (via predictors, "
              f"{args.val_samples} candidates): Audio->Param {a_acc:.1f}%, Param->Audio {p_acc:.1f}% (chance {chance:.3f}%)")
        with open(os.path.join(save_dir, "val.jsonl"), "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, "train_loss": sums["loss"] / max(1, n_b),
                                "val_loss": val_loss, "r_a2p": a_acc, "r_p2a": p_acc, "chance": chance}) + "\n")

        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "enum_tables": tables,
            "welford_stats": normalizer.state_dict(),
            "val_loss": val_loss,
            "retrieval_avg": (a_acc + p_acc) / 2.0,
            "args": vars(args),
        }
        atomic_save(ckpt, os.path.join(save_dir, "checkpoint_latest.pt"))
        if val_loss < best_val:
            best_val = val_loss
            atomic_save(ckpt, os.path.join(save_dir, "checkpoint_best.pt"))
            print(f"New best checkpoint saved (val loss {best_val:.4f})")


if __name__ == "__main__":
    train(build_parser().parse_args())
