import os
import sys
import time
import argparse
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import ResMLPInverter, FlowMatchingResMLP
from schedulefree import AdamWScheduleFree

# ---------------------------------------------------------------------------
# Newton-Schulz Quintic Orthogonalization
# ---------------------------------------------------------------------------
def zeropower_via_newtonschulz(grad, ns_steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert len(grad.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    ortho_grad = grad.bfloat16() if grad.dtype != torch.bfloat16 else grad.clone()
    transposed = False
    if ortho_grad.size(0) > ortho_grad.size(1):
        ortho_grad = ortho_grad.T
        transposed = True
        
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
        
    if transposed:
        ortho_grad = ortho_grad.T
    return ortho_grad.to(grad.dtype)

# ---------------------------------------------------------------------------
# NorMuon with Radial Brake & Over-training Safeguards
# ---------------------------------------------------------------------------
class NorMuon(torch.optim.Optimizer):
    """Normalized Muon optimizer with Radial Brake and Muon-SW decay."""
    def __init__(self, params, lr=1e-2, momentum=0.95, weight_decay=0.01, ns_steps=5, radial_brake=0.85):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            ns_steps=ns_steps, radial_brake=radial_brake
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            r_brake = group["radial_brake"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    p.data.mul_(1.0 - lr * wd)
                    
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                # Apply Newton-Schulz orthogonalization on 2D matrices
                if p.ndim == 2:
                    ortho_update = zeropower_via_newtonschulz(buf, ns_steps=ns_steps)
                    scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                    update = ortho_update * (lr * scale)
                else:
                    update = buf * lr
                
                p_old_norm = p.data.norm() if r_brake < 1.0 else None
                p.data.sub_(update)
                
                # Radial Brake: soft-limiting parameter norm expansion
                if r_brake < 1.0 and p_old_norm is not None:
                    p_new_norm = p.data.norm()
                    if p_new_norm > p_old_norm:
                        p_target = p_old_norm + r_brake * (p_new_norm - p_old_norm)
                        scale_factor = float((p_target / p_new_norm.clamp_min(1e-12)).item())
                        p.data.mul_(scale_factor)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SurgeH5Dataset(Dataset):
    def __init__(self, h5_path, split="train", val_ratio=0.2):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            total_len = len(f["mel"])
            val_len = int(total_len * val_ratio)
            train_len = total_len - val_len
            if split == "train":
                self.indices = list(range(0, train_len))
            else:
                self.indices = list(range(train_len, total_len))
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
# Training Loop
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n=================================================================")
    print(f"Starting Run [{args.run_id}]: {args.model_type.upper()} dim={args.hidden_dim} layers={args.num_layers}")
    print(f"Optimizer: {args.opt_family.upper()} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"=================================================================")
    
    os.makedirs(args.output_dir, exist_ok=True)
    leaderboard_path = os.path.join(args.output_dir, "leaderboard.tsv")
    if not os.path.exists(leaderboard_path):
        with open(leaderboard_path, "w") as f:
            f.write("run_id\tmodel\topt\tdim\tlayers\tlr_main\tlr_sec\tbsz\tepochs\tbest_val_loss\ttrain_time_s\n")

    train_ds = SurgeH5Dataset(args.h5_path, split="train")
    val_ds = SurgeH5Dataset(args.h5_path, split="val")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    param_dim = train_ds[0][1].shape[0]
    print(f"Dataset parameter dimension: {param_dim}")
    
    # Domain-weighted loss tensor:
    # High priority to shape (idx 2), cutoff (idx 9), resonance (idx 10), filter (idx 1)
    weights = torch.ones(param_dim, device=device)
    weights[1] = 2.0  # filter circuit
    weights[2] = 3.5  # shape (Saw vs Pulse vs Morph)
    weights[9] = 2.5  # filter cutoff
    weights[10] = 2.0 # filter resonance
    weights[12] = 1.8 # feg amount
    weights[15] = 1.5 # aeg decay
    weights[17] = 1.5 # aeg release
    weights = weights / weights.mean() # normalize so mean weight = 1.0

    if args.model_type == "resmlp":
        model = ResMLPInverter(param_dim=param_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    elif args.model_type == "flow":
        model = FlowMatchingResMLP(param_dim=param_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Build Optimizers
    muon_params = []
    adam_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "norm" not in name.lower() and "embed" not in name.lower():
            muon_params.append(p)
        else:
            adam_params.append(p)

    use_sf = (args.opt_family == "normuon_sf")
    
    if args.opt_family == "adamw":
        # Pure AdamW benchmark with Cosine Annealing LR
        print("Using standard decoupled AdamW for all parameters")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_adam, weight_decay=args.weight_decay)
        opt_muon = None
        opt_adam = optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        # NorMuon + AdamWScheduleFree with Radial Brake & c_warmup control
        print(f"Using NorMuon (lr={args.lr_muon}, radial_brake=0.85) + Schedule-Free AdamW (lr={args.lr_adam})")
        opt_muon = NorMuon(muon_params, lr=args.lr_muon, weight_decay=args.weight_decay, radial_brake=0.85) if muon_params else None
        # sf_c_warmup: 2 * warmup steps (warmup_steps=100) -> 200 steps burn-in before averaging
        opt_adam = AdamWScheduleFree(adam_params, lr=args.lr_adam, weight_decay=args.weight_decay, warmup_steps=100) if adam_params else None
        scheduler = None

    best_val_loss = float("inf")
    t0 = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        if use_sf and opt_adam is not None:
            opt_adam.train()
            
        train_loss = 0.0
        n_batches = 0
        
        for mel, params in train_loader:
            mel = mel.to(device)
            params = params.to(device)
            
            if opt_muon:
                opt_muon.zero_grad()
            if opt_adam:
                opt_adam.zero_grad()
                
            if args.model_type == "resmlp":
                pred_params = model(mel)
                loss = torch.mean(weights * (pred_params - params) ** 2)
            else:
                B = params.shape[0]
                x0 = torch.randn_like(params)
                x1 = params
                t = torch.rand(B, device=device)
                
                t_expanded = t.unsqueeze(-1)
                xt = (1.0 - (1.0 - 1e-5) * t_expanded) * x0 + t_expanded * x1
                target_ut = x1 - (1.0 - 1e-5) * x0
                
                pred_vt = model(xt, t, mel)
                loss = torch.mean(weights * (pred_vt - target_ut) ** 2)
                
            loss.backward()
            
            # Gradient clipping safeguard
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if opt_muon:
                opt_muon.step()
            if opt_adam:
                opt_adam.step()
                
            train_loss += loss.item()
            n_batches += 1
            
        train_loss /= n_batches
        
        if scheduler is not None:
            scheduler.step()
            
        # Validation
        model.eval()
        if use_sf and opt_adam is not None:
            opt_adam.eval() # Swap to averaged deployable iterate x!
            
        val_loss = 0.0
        n_val = 0
        
        with torch.no_grad():
            for val_step, (mel, params) in enumerate(val_loader):
                if val_step >= 40:
                    break
                mel = mel.to(device)
                params = params.to(device)
                
                if args.model_type == "resmlp":
                    pred_params = model(mel)
                    loss = torch.mean(weights * (pred_params - params) ** 2)
                else:
                    pred_params = model.sample(mel, num_steps=12)
                    loss = torch.mean(weights * (pred_params - params) ** 2)
                    
                val_loss += loss.item()
                n_val += 1
                
        val_loss /= n_val
        
        # Checkpoint every 25 epochs + best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.output_dir, f"{args.run_id}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_path)
            star = "*"
        else:
            star = " "
            
        if epoch % 25 == 0 or epoch == args.epochs:
            periodic_path = os.path.join(args.output_dir, f"{args.run_id}_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, periodic_path)
            
        lr_display = opt_adam.param_groups[0]["lr"] if opt_adam else 0.0
        print(f"Epoch {epoch:3d}/{args.epochs:3d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} {star} | LR: {lr_display:.2e}")
        sys.stdout.flush()
        
    total_time = time.time() - t0
    print(f"\n[{args.run_id}] Finished in {total_time:.1f}s ({total_time/3600:.2f}h). Best Val Loss: {best_val_loss:.5f}")
    
    with open(leaderboard_path, "a") as f:
        lr_main = args.lr_muon if args.opt_family == "normuon_sf" else args.lr_adam
        f.write(f"{args.run_id}\t{args.model_type}\t{args.opt_family}\t{args.hidden_dim}\t{args.num_layers}\t{lr_main}\t{args.lr_adam}\t{args.batch_size}\t{args.epochs}\t{best_val_loss:.5f}\t{total_time:.1f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", type=str, default="/run/media/kim/Mantu/surge_dataset/surge_bass_200k.h5")
    parser.add_argument("--model_type", type=str, default="resmlp", choices=["resmlp", "flow"])
    parser.add_argument("--opt_family", type=str, default="normuon_sf", choices=["normuon_sf", "adamw"])
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_muon", type=float, default=0.01)
    parser.add_argument("--lr_adam", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--run_id", type=str, default="G03_resmlp_200ep_sf")
    parser.add_argument("--output_dir", type=str, default="/run/media/kim/Mantu/surge_200k_models/overtraining_suite")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    train(args)
