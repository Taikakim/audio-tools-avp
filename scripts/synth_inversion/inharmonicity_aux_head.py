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
            nn.Sigmoid() # Inharmonicity is [0, 1]
        )
        
    def forward(self, x):
        """
        x: [batch, time, embed_dim] or [batch, embed_dim]
        returns: [batch, ...] predicted inharmonicity
        """
        return self.net(x).squeeze(-1)

def masked_inharmonicity_loss(pred, target, valid_mask):
    """
    pred: [batch, ...] 
    target: [batch, ...]
    valid_mask: [batch, ...] (bool or float 0.0/1.0)
    
    Computes masked MSE loss. 
    Returns loss (scalar) and the number of valid items.
    """
    # Ensure no gradients flow into target or mask
    target = target.detach()
    valid_mask = valid_mask.detach().float()
    
    # Compute per-element loss
    loss = F.mse_loss(pred, target, reduction='none')
    
    # Apply mask
    masked_loss = loss * valid_mask
    
    # Calculate denominator
    valid_count = valid_mask.sum()
    
    if valid_count > 0:
        return masked_loss.sum() / valid_count, valid_count
    else:
        # If no valid targets, return 0 loss that still has grad_fn so DDP doesn't complain,
        # or just 0.0 * pred.sum()
        return 0.0 * pred.sum(), valid_count
