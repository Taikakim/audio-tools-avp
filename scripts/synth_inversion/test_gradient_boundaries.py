import torch
from inharmonicity_aux_head import InharmonicityHead, masked_inharmonicity_loss

# Mock online encoder embedding
embed_dim = 768
batch_size = 4
online_embed = torch.randn(batch_size, embed_dim, requires_grad=True)

# Mock target encoder embedding (should NOT receive gradients)
target_embed = torch.randn(batch_size, embed_dim, requires_grad=True)

# Head
head = InharmonicityHead(embed_dim)

# Mock extracted targets (from InharmonicityExtractor)
# Batch of 4: 2 valid, 2 invalid
targets = torch.tensor([0.1, 0.5, 0.0, 0.0]) 
valid_mask = torch.tensor([True, True, False, False])

# Predict
preds = head(online_embed)

# Loss
loss, valid_count = masked_inharmonicity_loss(preds, targets, valid_mask)
loss.backward()

# Assertions (Section 7.D)
assert online_embed.grad is not None and online_embed.grad.abs().sum() > 0, "Online encoder must receive gradients"
assert target_embed.grad is None, "Target encoder must NOT receive gradients"

# Check that invalid examples contributed zero gradient to the online encoder
# (Since the operation is linear + elementwise up to the sum, online_embed.grad for indices 2 and 3 should be exactly 0)
assert torch.all(online_embed.grad[2] == 0), "Invalid example 2 contributed gradient!"
assert torch.all(online_embed.grad[3] == 0), "Invalid example 3 contributed gradient!"

print("Gradient boundary tests PASSED. Auxiliary head works flawlessly.")
