"""Muon (+ radial brake) for hidden 2D matrices, Schedule-Free AdamW for everything else.

Single source: train_bracket.py imports from here (it used to carry its own copy, and the
two had drifted apart — this file lacked the radial brake).

Naming: earlier versions called this "NorMuon", but it never implemented NorMuon's
per-neuron second-moment normalisation; it is plain Muon (momentum -> Newton-Schulz
orthogonalisation -> aspect-ratio scaling). `NorMuon` is kept as an alias so old imports
still work.
"""
import math

import torch
from schedulefree import AdamWScheduleFree


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


class Muon(torch.optim.Optimizer):
    """Muon for 2D weight matrices, with an optional radial brake.

    radial_brake < 1.0: if a step grows a matrix's Frobenius norm, only that fraction of
    the growth is kept (0.85 = keep 85 %). 1.0 disables it.
    """
    def __init__(self, params, lr=1e-2, momentum=0.95, weight_decay=0.01, ns_steps=5, radial_brake=0.85):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps, radial_brake=radial_brake)
        super().__init__(params, defaults)
        self.last_update_norm = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        up_sq = 0.0
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

                if p.ndim == 2:
                    ortho_update = zeropower_via_newtonschulz(buf, ns_steps=ns_steps)
                    scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                    update = ortho_update * (lr * scale)
                else:
                    update = buf * lr

                up_sq += float(update.pow(2).sum().item())
                p_old_norm = p.data.norm() if r_brake < 1.0 else None
                p.data.sub_(update)

                if p_old_norm is not None:
                    p_new_norm = p.data.norm()
                    if p_new_norm > p_old_norm:
                        p_target = p_old_norm + r_brake * (p_new_norm - p_old_norm)
                        p.data.mul_(float((p_target / p_new_norm.clamp_min(1e-12)).item()))

        self.last_update_norm = math.sqrt(up_sq)
        return loss


NorMuon = Muon  # backward-compatible alias; see module docstring

# Muon is meant for HIDDEN matrices. The input projection (params -> hidden) and the output
# heads (hidden -> params / logits) go to AdamW, as in the reference Muon setups.
ADAM_NAME_KEYS = ("norm", "embed", "param_proj", "head")


def partition_params(model):
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if p.ndim == 2 and not any(k in lname for k in ADAM_NAME_KEYS):
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def build_optimizers(model, lr_muon=1e-2, lr_adam=1e-3, weight_decay=1e-2, radial_brake=0.85, adam_warmup_steps=100):
    """Hidden 2D weights -> Muon; input/output layers, 1D tensors, conv kernels -> AdamWScheduleFree."""
    muon_params, adam_params = partition_params(model)
    print(f"Optimizer partition: {len(muon_params)} hidden 2D matrices -> Muon (lr={lr_muon}, radial_brake={radial_brake})")
    print(f"                     {len(adam_params)} other tensors -> AdamWScheduleFree (lr={lr_adam}, warmup={adam_warmup_steps})")
    opt_muon = Muon(muon_params, lr=lr_muon, weight_decay=weight_decay, radial_brake=radial_brake) if muon_params else None
    opt_adam = (AdamWScheduleFree(adam_params, lr=lr_adam, weight_decay=weight_decay, warmup_steps=adam_warmup_steps)
                if adam_params else None)
    return opt_muon, opt_adam
