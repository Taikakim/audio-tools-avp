"""Train a Surge XT inverter (ResMLP point estimate or conditional flow matching).

v2 changes (2026-10-01 review):
  * Categoricals (filter circuit, unison, waveshaper) are learned as classes: CE heads for
    ResMLP, one-hot slots for the flow (param_codec.py). Checkpoints record
    args.param_encoding = "onehot_v2" so loaders rebuild the right head.
  * Validation is comparable across model types. v1 scored ResMLP on its point estimate
    (the MSE-optimal conditional mean) but the flow on ONE fresh random draw per epoch,
    which carries ~2x the posterior variance by construction — so "ResMLP 0.033 vs flow
    0.054" said nothing about which model is better, and best-checkpoint selection for the
    flow chased sampling noise. Now:
      val_score    = weighted mean per-parameter error of the POINT estimate (ResMLP output;
                     flow = mean of K draws for continuous params, majority vote for
                     categoricals). Squared error on continuous, 0/1 on categoricals.
                     Used for checkpoint selection, comparable across models.
      flow extras  = single-draw score and best-of-K score, logged separately.
    Flow draws use a fixed-seed generator so the score is deterministic across epochs.
    Parameter-space scores still cannot see audio equivalence; the real comparison is
    evaluate_holdout_audio.py (renders through Surge and scores audio).
  * Flow time embedding scales t by 1000 (args.time_scale) — see models.py.
  * Optimizer from optimizer.py (single source); input/output layers go to AdamW.
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import param_codec as codec
from models import build_model
from optimizer import build_optimizers
from surge_spec import CAT_INDICES, CONT_INDICES, NUM_PARAMS, PARAM_NAMES, domain_weights

LEADERBOARD_COLS = [
    "run_id", "model", "opt", "param_encoding", "dim", "layers", "lr_main", "lr_sec", "bsz", "epochs",
    "best_epoch", "best_val_score", "val_cont_wmse", "val_cat_acc_mean",
    "flow_single_draw_score", "flow_best_of_k_score", "flow_k", "train_time_s",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SurgeH5Dataset(Dataset):
    """Last `val_ratio` of the file is validation (samples are i.i.d. draws, so a
    contiguous split is fine). evaluate_holdout_audio.py uses the same split."""
    def __init__(self, h5_path, split="train", val_ratio=0.2):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            total_len = len(f["mel"])
            assert f["params"].shape[1] == NUM_PARAMS, f"expected {NUM_PARAMS} params, got {f['params'].shape[1]}"
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


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def resmlp_loss(out, params, w23):
    """Weighted MSE on continuous params + weighted CE per categorical, normalised by total weight."""
    cont_pred, logit_blocks = codec.split_encoded(out)
    cont_true = params[:, CONT_INDICES]
    w_cont = w23[CONT_INDICES]
    total = (w_cont * (cont_pred - cont_true) ** 2).mean(dim=0).sum()
    for logits, cls, (i, _k) in zip(logit_blocks, codec.target_classes(params), CAT_INDICES):
        total = total + w23[i] * F.cross_entropy(logits, cls)
    return total / w23.sum()


def flow_loss(model, mel, params, w_enc):
    x1 = codec.encode(params)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt, target = model.path(x0, x1, t)
    pred = model(xt, t, mel)
    return ((w_enc * (pred - target) ** 2).sum(dim=1) / w_enc.sum()).mean()


# ---------------------------------------------------------------------------
# Validation (comparable across model types)
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(model, model_type, loader, w23, device, max_batches=40, flow_k=8, flow_steps=12, seed=1234):
    gen = torch.Generator(device=device).manual_seed(seed) if model_type == "flow" else None
    sums = {"point": 0.0, "single": 0.0, "best_of_k": 0.0, "cont_wmse": 0.0}
    cat_correct = torch.zeros(len(CAT_INDICES))
    n = 0
    w_cont = w23[CONT_INDICES]
    for step, (mel, params) in enumerate(loader):
        if step >= max_batches:
            break
        mel, params = mel.to(device), params.to(device)
        B = params.shape[0]
        if model_type == "resmlp":
            point = model.predict_params(mel)
        else:
            draws = torch.stack([model.predict_params(mel, num_steps=flow_steps, generator=gen) for _ in range(flow_k)])
            cont_mean = draws[:, :, CONT_INDICES].mean(dim=0)
            # [K, n_cat, B] class indices -> majority vote over the K draws
            draw_classes = torch.stack([torch.stack(codec.target_classes(draws[k])) for k in range(flow_k)])
            classes = torch.mode(draw_classes, dim=0).values
            point = codec.assemble(cont_mean, list(classes))
            draw_scores = torch.stack([(codec.per_param_error(draws[k], params) * w23).mean(dim=1) for k in range(flow_k)])
            sums["single"] += float(draw_scores[0].sum())
            sums["best_of_k"] += float(draw_scores.min(dim=0).values.sum())
        err = codec.per_param_error(point, params)
        sums["point"] += float((err * w23).mean(dim=1).sum())
        sums["cont_wmse"] += float((err[:, CONT_INDICES] * w_cont).sum(dim=1).sum() / w_cont.sum())
        for j, (pc, tc) in enumerate(zip(codec.target_classes(point), codec.target_classes(params))):
            cat_correct[j] += float((pc == tc).sum())
        n += B
    out = {
        "val_score": sums["point"] / n,
        "val_cont_wmse": sums["cont_wmse"] / n,
        "val_cat_acc": {PARAM_NAMES[i]: float(cat_correct[j] / n) for j, (i, _k) in enumerate(CAT_INDICES)},
    }
    if model_type == "flow":
        out["flow_single_draw_score"] = sums["single"] / n
        out["flow_best_of_k_score"] = sums["best_of_k"] / n
    return out


# ---------------------------------------------------------------------------
# Flat Weight Snapshot Helper for Velocity & Direction Tracking
# ---------------------------------------------------------------------------
def get_flat_weights(model):
    """Flattens all trainable parameters into a single 1D tensor."""
    return torch.cat([p.detach().flatten().double() for p in model.parameters() if p.requires_grad])


def save_ckpt(path, model, epoch, global_step, val, args):
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "val": val,
        "val_loss": val["val_score"],
        "args": vars(args),
    }, path)


# ---------------------------------------------------------------------------
# Training Loop with Step-Level Velocity & Param Monitoring
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================================")
    print(f"Starting Run [{args.run_id}]: {args.model_type.upper()} dim={args.hidden_dim} layers={args.num_layers}")
    print(f"Optimizer: {args.opt_family.upper()} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"Param encoding: {args.param_encoding} | time_scale: {args.time_scale}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=================================================================")

    os.makedirs(args.output_dir, exist_ok=True)
    # v2 leaderboard has different columns; never append them under the v1 header.
    leaderboard_path = os.path.join(args.output_dir, "leaderboard_v2.tsv")
    if not os.path.exists(leaderboard_path):
        with open(leaderboard_path, "w") as f:
            f.write("\t".join(LEADERBOARD_COLS) + "\n")

    telemetry_path = os.path.join(args.output_dir, f"{args.run_id}_step_telemetry.tsv")
    with open(telemetry_path, "w") as f:
        f.write("global_step\tepoch\tbatch\tloss\tvelocity\tdir_cosine\tgrad_norm\tweight_norm\tupdate_norm\tlr\n")
    val_log_path = os.path.join(args.output_dir, f"{args.run_id}_val.jsonl")
    open(val_log_path, "w").close()

    train_ds = SurgeH5Dataset(args.h5_path, split="train")
    val_ds = SurgeH5Dataset(args.h5_path, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    w23 = torch.tensor(domain_weights(), device=device)
    w_enc = codec.encoded_weights(w23)

    model = build_model(args.model_type, args.hidden_dim, args.num_layers,
                        encoding=args.param_encoding, time_scale=args.time_scale).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    use_sf = (args.opt_family in ("normuon_sf", "modular"))
    is_modular = (args.opt_family == "modular")

    if args.opt_family == "adamw":
        print("Using standard decoupled AdamW for all parameters")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_adam, weight_decay=args.weight_decay)
        opt_muon = None
        opt_adam = optimizer
        opt_modular = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    elif args.opt_family == "modular":
        print("Using ModularOptimizer: 6-stage pipeline with all modular brakes active except SNR gate:")
        print("  - Stage 1: Momentum Accumulation (beta1=0.9)")
        print("  - Stage 2: Forward Whitening")
        print("  - Stage 3: Core LMO Solver (Spectral with radius scale \\rho_\\ell)")
        print("  - Stage 3b: NorMuon Per-Neuron Row Scaling (unit variance EMA)")
        print("  - Stage 4: Reverse Unwhitening (Pullback with LIFO stack)")
        print("  - Stage 5: Prodigy Escape Velocity (dual-norm coupled, snr_gate=False)")
        print("  - Stage 6: Schedule-Free (c_warmup=200, power r=1), Radial Brake (0.85), Muon-SW & AdamC quadratic decays")
        import sys
        sat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if sat_path not in sys.path:
            sys.path.insert(0, sat_path)
        from stable_audio_tools.training.modular_opt import ModularOptimizer, build_modular_param_groups

        groups = build_modular_param_groups(
            model,
            default_whitening="none",
            spectral_lr=args.lr_muon,
            sign_lr=args.lr_adam,
            colnorm_lr=args.lr_adam,
            spectral_wd=args.weight_decay,
            sign_wd=args.weight_decay,
            colnorm_wd=args.weight_decay,
        )
        opt_modular = ModularOptimizer(
            groups,
            lr=args.lr_adam,
            radial_brake=args.radial_brake,
            normuon=True,
            normuon_beta=0.95,
            schedule_free=True,
            sf_beta=0.9,
            sf_c_warmup=200,
            muon_sw_decay=True,
            adamc_decay=True,
            warmup_steps=args.adam_warmup_steps,
            snr_gate=False,
            escape_velocity=True,
        )
        opt_muon = None
        opt_adam = None
        scheduler = None
        print(opt_modular.summary())
    else:
        opt_muon, opt_adam = build_optimizers(model, lr_muon=args.lr_muon, lr_adam=args.lr_adam,
                                              weight_decay=args.weight_decay, radial_brake=args.radial_brake,
                                              adam_warmup_steps=args.adam_warmup_steps)
        opt_modular = None
        scheduler = None

    best = None
    t0 = time.time()
    global_step = 0
    prev_weights = get_flat_weights(model)
    prev_delta = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        if is_modular:
            opt_modular.train()
        elif use_sf and opt_adam is not None:
            opt_adam.train()

        train_loss = 0.0
        n_batches = 0

        for batch_idx, (mel, params) in enumerate(train_loader):
            global_step += 1
            mel = mel.to(device)
            params = params.to(device)

            if is_modular:
                opt_modular.zero_grad()
            else:
                if opt_muon:
                    opt_muon.zero_grad()
                if opt_adam:
                    opt_adam.zero_grad()

            if args.model_type == "resmlp":
                loss = resmlp_loss(model(mel), params, w23)
            else:
                loss = flow_loss(model, mel, params, w_enc)

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())  # pre-clip norm

            if is_modular:
                opt_modular.step()
            else:
                if opt_muon:
                    opt_muon.step()
                if opt_adam:
                    opt_adam.step()

            loss_val = loss.item()
            train_loss += loss_val
            n_batches += 1

            if global_step % 25 == 0:
                current_weights = get_flat_weights(model)
                current_delta = current_weights - prev_weights
                # displacement over the last 25 steps, expressed per 1,000 steps
                step_vel = float(current_delta.norm().item()) * (1000.0 / 25.0)
                if prev_delta is not None and prev_delta.norm() > 1e-9 and current_delta.norm() > 1e-9:
                    dir_cosine = float((torch.dot(current_delta, prev_delta) / (current_delta.norm() * prev_delta.norm())).item())
                else:
                    dir_cosine = 1.0
                w_norm = float(current_weights.norm().item())
                if is_modular:
                    up_norm = opt_modular.component_telemetry.get("comp/spectral_update_norm", 0.0)
                    curr_lr = opt_modular.param_groups[0].get("lr", 0.0)
                else:
                    up_norm = opt_muon.last_update_norm if opt_muon else 0.0
                    curr_lr = opt_adam.param_groups[0]["lr"] if opt_adam else 0.0

                with open(telemetry_path, "a") as f:
                    f.write(f"{global_step}\t{epoch}\t{batch_idx}\t{loss_val:.6f}\t{step_vel:.4f}\t{dir_cosine:.4f}\t{grad_norm:.4f}\t{w_norm:.4f}\t{up_norm:.4f}\t{curr_lr:.2e}\n")
                if global_step % 100 == 0:
                    ev_info = f" | EV_d: {opt_modular.component_telemetry.get('comp/ev_d', 1.0):.2f}" if is_modular else ""
                    print(f"Step {global_step:6d} [Ep {epoch:2d}/{args.epochs}] | Loss: {loss_val:.5f} | Vel: {step_vel:6.2f} | CosDir: {dir_cosine:+.3f} | ||g||: {grad_norm:.3f} | ||w||: {w_norm:.2f}{ev_info}")
                    sys.stdout.flush()
                prev_delta = current_delta
                prev_weights = current_weights

        train_loss /= n_batches
        if scheduler is not None:
            scheduler.step()

        model.eval()
        if is_modular:
            opt_modular.eval()
        elif use_sf and opt_adam is not None:
            opt_adam.eval()  # swap to the averaged iterate for eval + checkpointing
        val = validate(model, args.model_type, val_loader, w23, device,
                       max_batches=args.val_batches, flow_k=args.flow_val_draws, flow_steps=args.flow_val_steps)
        val.update(epoch=epoch, train_loss=train_loss)
        with open(val_log_path, "a") as f:
            f.write(json.dumps(val) + "\n")

        if best is None or val["val_score"] < best["val_score"]:
            best = dict(val)
            save_ckpt(os.path.join(args.output_dir, f"{args.run_id}_best.pt"), model, epoch, global_step, val, args)
            star = "*"
        else:
            star = " "
        if epoch % 25 == 0 or epoch == args.epochs:
            save_ckpt(os.path.join(args.output_dir, f"{args.run_id}_epoch_{epoch}.pt"), model, epoch, global_step, val, args)

        cat_acc = " ".join(f"{k}={v:.3f}" for k, v in val["val_cat_acc"].items())
        extra = (f" | 1-draw {val['flow_single_draw_score']:.5f} | best-of-{args.flow_val_draws} {val['flow_best_of_k_score']:.5f}"
                 if args.model_type == "flow" else "")
        lr_display = opt_modular.param_groups[0]["lr"] if opt_modular else (opt_adam.param_groups[0]["lr"] if opt_adam else 0.0)
        print(f"=== Epoch {epoch:3d}/{args.epochs:3d} === Train {train_loss:.5f} | Val score {val['val_score']:.5f} {star} "
              f"(cont wMSE {val['val_cont_wmse']:.5f}; acc {cat_acc}){extra} | LR {lr_display:.2e}")
        sys.stdout.flush()

    total_time = time.time() - t0
    print(f"\n[{args.run_id}] Finished in {total_time:.1f}s ({total_time/3600:.2f}h). Best val score: {best['val_score']:.5f} (epoch {best['epoch']})")

    lr_main = args.lr_muon if args.opt_family in ("normuon_sf", "modular") else args.lr_adam
    row = [
        args.run_id, args.model_type, args.opt_family, args.param_encoding, args.hidden_dim, args.num_layers,
        lr_main, args.lr_adam, args.batch_size, args.epochs, best["epoch"], f"{best['val_score']:.5f}",
        f"{best['val_cont_wmse']:.5f}", f"{np.mean(list(best['val_cat_acc'].values())):.4f}",
        f"{best['flow_single_draw_score']:.5f}" if "flow_single_draw_score" in best else "",
        f"{best['flow_best_of_k_score']:.5f}" if "flow_best_of_k_score" in best else "",
        args.flow_val_draws if args.model_type == "flow" else "", f"{total_time:.1f}",
    ]
    with open(leaderboard_path, "a") as f:
        f.write("\t".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5")
    parser.add_argument("--model_type", type=str, default="resmlp", choices=["resmlp", "flow"])
    parser.add_argument("--opt_family", type=str, default="modular", choices=["normuon_sf", "adamw", "modular"],
                        help="modular = full 6-stage ModularOptimizer with all modular brakes active except SNR gate")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_muon", type=float, default=0.01)
    parser.add_argument("--lr_adam", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--radial_brake", type=float, default=0.85)
    parser.add_argument("--adam_warmup_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--param_encoding", type=str, default=codec.ENCODING_V2,
                        choices=[codec.ENCODING_V2], help="v1 (ordinal) is load-only")
    parser.add_argument("--time_scale", type=float, default=1000.0, help="flow time-embedding scale (v1 used 1.0)")
    parser.add_argument("--val_batches", type=int, default=40)
    parser.add_argument("--flow_val_draws", type=int, default=8)
    parser.add_argument("--flow_val_steps", type=int, default=12)
    parser.add_argument("--run_id", type=str, default="G03_resmlp_200ep_normuon_sf")
    parser.add_argument("--output_dir", type=str, default="/run/media/kim/Mantu/surge_200k_models/overtraining_suite")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)
