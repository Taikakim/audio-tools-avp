"""Train Synth-JEPA on Surge XT synthesizer data.

Reference: Hayes, Tian, Lattner, "Synth-JEPA: Joint Embedding Prediction for Renderer-Free
Synthesizer Parameter Search", arXiv:2609.31024 (Sept 2026).

Loss (Eq. 3 + SIGReg, Sec. 2.1):
  L = MSE(z_hat_p, sg(z_p)) + MSE(z_hat_a, sg(z_a)) + lambda_sig * (SIGReg(z_a) + SIGReg(z_p))

Paper recipe (Sec. 3.1-3.2) vs this script:
  * AdamW, lr 3e-4, weight decay 0.05, batch 64, warmup-stable-decay schedule  -> same
    (warmup/decay fractions are not given in the paper: --warmup_frac / --decay_frac).
  * Channel-wise log-mel mean/var from the first 8k training spectrograms (Welford), frozen
    -> same, estimated in a pre-pass BEFORE the first step. (v1 of this script trained on
    UN-normalised mels until 8000 BATCHES (512k spectrograms) had passed, which the default
    epoch length never reached, then switched normalisation on at the end of epoch 1.)
  * 1M steps on audio rendered online (every step sees new sounds) -> NOT reproducible from a
    fixed h5: 190k training notes at batch 64 is ~3k steps/epoch, so 1M steps would be ~330
    passes over the same data. --max_steps caps the run; watch the train/val gap.
  * 139 Surge parameters, 3 s stereo -> here 23 parameters, 0.8 s mono (the dataset's limits).
  * lambda_sig is not given in the paper (default 1.0).

v2 also: validation split 0.2 (same as train_bracket / evaluate_holdout_audio — v1's 0.05 let
the JEPA train on the held-out set used to compare models), deterministic SIGReg slices in
validation, cross-modal retrieval accuracy as a validation instrument (chance = 1/batch), and
checkpoints that carry model_kwargs + normaliser so synth_jepa_search.load_synth_jepa can
rebuild everything from one file.
"""
import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from surge_spec import CAT_INDICES, CONT_INDICES, NUM_PARAMS
from synth_jepa_model import SynthJEPA
from synth_jepa_search import WelfordNormalizer


class SurgeH5Dataset(Dataset):
    """Same split as train_bracket.SurgeH5Dataset: the last val_ratio of the file is held out."""
    def __init__(self, h5_path, split="train", val_ratio=0.2):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            total_len = len(f["mel"])
            assert f["params"].shape[1] == NUM_PARAMS
        train_len = total_len - int(total_len * val_ratio)
        self.indices = range(0, train_len) if split == "train" else range(train_len, total_len)
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
        real_idx = self.indices[idx]
        mel = torch.from_numpy(self.h5_file["mel"][real_idx])       # [128, 81]
        params = torch.from_numpy(self.h5_file["params"][real_idx]) # [23]
        return mel, params


class SurgeOnlineDataset(Dataset):
    """Procedurally renders Surge XT audio online per sample using persistent worker synths.

    Seeds are drawn strictly from [min_seed, ...) to guarantee ZERO overlap with the
    validation set (which occupies seeds 5000 + 160000 .. 5000 + 200000 in the H5).
    Workers generate raw audio waveforms on CPU; the GPU calculates ExactGpuMel in batches.
    """
    def __init__(self, length=160000, min_seed=1_000_000, plugin_path=None):
        from surge_spec import DEFAULT_PLUGIN_PATH, SEED_OFFSET

        self.length = length
        self.min_seed = min_seed
        self.plugin_path = plugin_path or DEFAULT_PLUGIN_PATH
        # Assert strict seed disjointness from validation set
        h5_val_max = SEED_OFFSET + 200_000
        assert min_seed > h5_val_max, f"min_seed {min_seed} intersects H5 val range [..., {h5_val_max}]"
        self._synth = None

    def __len__(self):
        return self.length

    def _get_synth(self):
        if self._synth is None:
            from surge_spec import init_synth
            self._synth = init_synth(self.plugin_path, verify=False)
        return self._synth

    def __getitem__(self, idx):
        from surge_spec import draw_patch, patch_to_vector, render_patch

        seed = self.min_seed + idx
        patch = draw_patch(seed)
        synth = self._get_synth()
        audio = render_patch(synth, patch, patch["midi_note"], patch["note_dur"])
        p_vec = patch_to_vector(patch)
        return torch.from_numpy(audio), torch.from_numpy(p_vec)


def prepare_inputs(params: torch.Tensor):
    """[B, 23] stored params -> (continuous in [-1, 1] [B, 20], list of one-hot [B, k]) (Sec. 2)."""
    cont = params[:, CONT_INDICES] * 2.0 - 1.0
    cat_onehots = [F.one_hot(torch.round(params[:, i] * (k - 1)).clamp(0, k - 1).long(), k).float()
                   for i, k in CAT_INDICES]
    return cont, cat_onehots


def estimate_normalizer(ds, n_spectrograms=8000, seed=0, batch=256):
    """Welford pre-pass over the first n_spectrograms of a seeded shuffle of the training set."""
    norm = WelfordNormalizer(n_mels=128)
    order = np.random.default_rng(seed).permutation(len(ds))[:n_spectrograms]
    for s in range(0, len(order), batch):
        norm.update(np.stack([ds[int(i)][0].numpy() for i in order[s:s + batch]]))
    norm.freeze()
    return norm


def wsd_lambda(total_steps, warmup_frac, decay_frac):
    """Warmup-stable-decay: linear warmup, constant, linear decay to zero."""
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


@torch.no_grad()
def validate(model, loader, normalizer, lambda_sig, device, max_batches=50, seed=1234):
    gen = torch.Generator(device=device).manual_seed(seed)
    sums = dict(loss=0.0, lp=0.0, la=0.0, sa=0.0, sp=0.0, r_a2p=0.0, r_p2a=0.0)
    n = 0
    for b, (mel, p) in enumerate(loader):
        if b >= max_batches:
            break
        mel, p = mel.to(device), p.to(device)
        cont, cats = prepare_inputs(p)
        lp, la, sa, sp, za, zp = model(normalizer.normalize(mel), cont, cats, sigreg_generator=gen)
        # cross-modal retrieval within the batch: is the prediction nearest to its own pair?
        tgt = torch.arange(za.shape[0], device=device)
        r_a2p = (torch.cdist(model.f_a2p(za), zp).argmin(dim=1) == tgt).float().mean()
        r_p2a = (torch.cdist(model.f_p2a(zp), za).argmin(dim=1) == tgt).float().mean()
        for k, v in dict(loss=lp + la + lambda_sig * (sa + sp), lp=lp, la=la, sa=sa, sp=sp,
                         r_a2p=r_a2p, r_p2a=r_p2a).items():
            sums[k] += float(v)
        n += 1
    out = {k: v / n for k, v in sums.items()}
    out["retrieval_chance"] = 1.0 / loader.batch_size
    return out


def main():
    parser = argparse.ArgumentParser(description="Train Synth-JEPA model")
    default_h5 = ("/run/media/kim/Kosmos/surge_dataset/surge_bass_200k.h5"
                  if os.path.exists("/run/media/kim/Kosmos/surge_dataset/surge_bass_200k.h5")
                  else "/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5")
    parser.add_argument("--h5_path", type=str, default=default_h5)
    parser.add_argument("--out_dir", type=str, default="/run/media/kim/Mantu/surge_200k_models/synth_jepa_runs")
    parser.add_argument("--run_id", type=str, default="synth_jepa_v2_20ep")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=None, help="stop after this many steps (paper: 1M, online data)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--decay_frac", type=float, default=0.2)
    parser.add_argument("--lambda_sig", type=float, default=1.0)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--ff_dim", type=int, default=1024, help="not given in the paper; 1024 lands near its 53M")
    parser.add_argument("--norm_spectrograms", type=int, default=8000, help="Welford pre-pass size (paper: 8k)")
    parser.add_argument("--val_batches", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--online", action="store_true", help="Render Surge audio online per sample using CPU workers")
    parser.add_argument("--online_samples_per_epoch", type=int, default=160000, help="Epoch length for online training")
    args = parser.parse_args()

    save_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device)
    mode_str = "Online procedural synthesis" if args.online else f"Cached H5 ({args.h5_path})"
    print(f"=== Synth-JEPA Training on {device} ===\nMode: {mode_str}\nCheckpoints: {save_dir}")

    from audio_utils import ExactGpuMel
    gpu_mel_extractor = ExactGpuMel(device=device) if args.online else None

    if args.online:
        train_ds = SurgeOnlineDataset(length=args.online_samples_per_epoch, min_seed=1_000_000)
    else:
        train_ds = SurgeH5Dataset(args.h5_path, split="train")

    val_ds = SurgeH5Dataset(args.h5_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=not args.online,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=True)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    print(f"Estimating channel-wise log-mel statistics over {args.norm_spectrograms} training spectrograms...")
    normalizer = estimate_normalizer(SurgeH5Dataset(args.h5_path, split="train"), args.norm_spectrograms)
    torch.save(normalizer.state_dict(), os.path.join(save_dir, "welford_stats.pt"))

    model_kwargs = dict(embed_dim=args.embed_dim, predictor_hidden=1024, num_audio_layers=args.num_layers,
                        num_param_layers=args.num_layers, num_slices_sigreg=64, ff_dim=args.ff_dim)
    model = SynthJEPA(**model_kwargs).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (paper: 53M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    print(f"Total steps: {total_steps} (~{total_steps * args.batch_size / len(train_ds):.1f} passes over the training set)")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd_lambda(total_steps, args.warmup_frac, args.decay_frac))

    step, best_val, t_start = 0, float("inf"), time.time()
    log_path = os.path.join(save_dir, "val.jsonl")
    open(log_path, "w").close()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, n_b, t0 = dict(loss=0.0, lp=0.0, la=0.0, sa=0.0, sp=0.0), 0, time.time()
        for batch_item1, batch_item2 in train_loader:
            if step >= total_steps:
                break
            step += 1
            if args.online:
                audio_batch = batch_item1.to(device, non_blocking=True)
                p = batch_item2.to(device, non_blocking=True)
                mel = gpu_mel_extractor(audio_batch)
            else:
                mel = batch_item1.to(device, non_blocking=True)
                p = batch_item2.to(device, non_blocking=True)

            cont, cats = prepare_inputs(p)
            lp, la, sa, sp, _za, _zp = model(normalizer.normalize(mel), cont, cats)
            loss = (lp + la) + args.lambda_sig * (sa + sp)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            for k, v in dict(loss=loss, lp=lp, la=la, sa=sa, sp=sp).items():
                sums[k] += float(v.detach())
            n_b += 1
            if step % 200 == 0:
                print(f"Ep {epoch:02d} | Step {step:06d} | Loss {loss.item():.4f} (Pred_P {lp.item():.4f}, Pred_A {la.item():.4f}, "
                      f"SIG_A {sa.item():.4f}, SIG_P {sp.item():.4f}) | LR {scheduler.get_last_lr()[0]:.2e}", flush=True)
        if n_b == 0:
            break

        model.eval()
        val = validate(model, val_loader, normalizer, args.lambda_sig, device, args.val_batches)
        train_avg = {k: v / n_b for k, v in sums.items()}
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch, "step": step, "train": train_avg, "val": val}) + "\n")
        print(f"--- Epoch {epoch:02d} ({time.time() - t0:.1f}s) ---\n"
              f"  Train: Loss {train_avg['loss']:.4f} | Lp {train_avg['lp']:.4f} La {train_avg['la']:.4f} "
              f"Sa {train_avg['sa']:.4f} Sp {train_avg['sp']:.4f}\n"
              f"  Val:   Loss {val['loss']:.4f} | Lp {val['lp']:.4f} La {val['la']:.4f} Sa {val['sa']:.4f} Sp {val['sp']:.4f} "
              f"| retrieval a->p {val['r_a2p']:.3f} p->a {val['r_p2a']:.3f} (chance {val['retrieval_chance']:.3f})", flush=True)

        ckpt = {"epoch": epoch, "step": step, "model_state_dict": model.state_dict(), "model_kwargs": model_kwargs,
                "normalizer": normalizer.state_dict(), "val": val, "val_loss": val["loss"], "args": vars(args)}
        torch.save(ckpt, os.path.join(save_dir, "checkpoint_latest.pt"))
        if val["loss"] < best_val:
            best_val = val["loss"]
            torch.save(ckpt, os.path.join(save_dir, "checkpoint_best.pt"))
            print(f"  [+] New best val loss: {best_val:.4f} -> checkpoint_best.pt")
        if epoch % 5 == 0 or epoch == args.epochs or step >= total_steps:
            torch.save(ckpt, os.path.join(save_dir, f"checkpoint_ep{epoch:02d}.pt"))
        if step >= total_steps:
            break

    print(f"\nTraining completed in {(time.time() - t_start) / 3600:.2f} h ({step} steps). Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
