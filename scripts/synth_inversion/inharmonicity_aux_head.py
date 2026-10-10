import torch
import torch.nn as nn
import torch.nn.functional as F

class InharmonicityHead(nn.Module):
    """
    Auxiliary head for predicting inharmonicity from the online encoder's embedding.
    """
    def __init__(self, embed_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus() # Inharmonicity is non-negative, often very small
        )
        
    def forward(self, x):
        """
        x: [batch, time, embed_dim] or [batch, embed_dim]
        returns: [batch, ...] predicted inharmonicity
        """
        return self.net(x).squeeze(-1)

def masked_inharmonicity_loss(pred, target, valid_mask):
    if not (pred.shape == target.shape == valid_mask.shape):
        raise ValueError("pred, target and valid_mask must have identical shapes")
    target = target.detach()
    valid = valid_mask.detach() != 0                       # bool or 0/1 float
    safe = torch.where(valid, target, torch.zeros_like(target))   # NaN/inf never reach the arithmetic
    if not torch.isfinite(safe).all():
        raise ValueError("non-finite target on a row flagged valid")
    sq = (pred - safe) ** 2
    masked = torch.where(valid, sq, torch.zeros_like(sq))
    n = valid.sum().to(pred.dtype)
    return masked.sum() / n.clamp_min(1.0), n               # all-invalid -> exact 0 that keeps its graph
