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

class NorMuon(torch.optim.Optimizer):
    """Normalized Muon optimizer for 2D weight matrices."""
    def __init__(self, params, lr=1e-2, momentum=0.95, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            
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
                    # Scale update based on matrix aspect ratio
                    scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                    p.data.add_(ortho_update, alpha=-lr * scale)
                else:
                    # Fallback for non-2D tensors
                    p.data.add_(buf, alpha=-lr)

def build_optimizers(model, lr_muon=1e-2, lr_adam=1e-3, weight_decay=1e-2):
    """Partitions model parameters: 2D weights -> NorMuon, 1D/biases -> AdamWScheduleFree."""
    muon_params = []
    adam_params = []
    
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "norm" not in name.lower() and "embed" not in name.lower():
            muon_params.append(p)
        else:
            adam_params.append(p)
            
    print(f"Optimizer partition: {len(muon_params)} 2D weight matrices -> NorMuon (lr={lr_muon})")
    print(f"                     {len(adam_params)} 1D tensors/biases -> AdamWScheduleFree (lr={lr_adam})")
    
    opt_muon = NorMuon(muon_params, lr=lr_muon, weight_decay=weight_decay) if len(muon_params) > 0 else None
    opt_adam = AdamWScheduleFree(adam_params, lr=lr_adam, weight_decay=weight_decay) if len(adam_params) > 0 else None
    
    return opt_muon, opt_adam
