"""Weight-average (model-soup) onset-density checkpoints.

Averages the tensors in ck["state"] (adapter + conditioner) with given weights;
copies args/scalar_norm/other metadata from the first checkpoint. CPU-only.

    python make_soup.py <out.pt> <ckpt1>:<w1> <ckpt2>:<w2> ...

Weights are normalized internally. Prints the resulting % contributions.
"""
import sys
import torch

out = sys.argv[1]
pairs = []
for a in sys.argv[2:]:
    p, w = a.rsplit(":", 1)
    pairs.append((p, float(w)))
tot = sum(w for _, w in pairs)
pairs = [(p, w / tot) for p, w in pairs]

print(f"[soup] {len(pairs)} checkpoints -> {out}", flush=True)
base = torch.load(pairs[0][0], map_location="cpu", weights_only=False)
state = base["state"]
n_t = sum(1 for v in state.values() if torch.is_tensor(v))
print(f"[soup] state has {len(state)} entries, {n_t} tensors; keys[:3]={list(state)[:3]}", flush=True)

acc = {k: (v.to(torch.float64) * pairs[0][1] if torch.is_tensor(v) else v)
       for k, v in state.items()}
for p, w in pairs[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    for k, v in ck["state"].items():
        if torch.is_tensor(v):
            acc[k] = acc[k] + v.to(torch.float64) * w
for k, v in state.items():
    if torch.is_tensor(v):
        acc[k] = acc[k].to(v.dtype)
base["state"] = acc
torch.save(base, out)
print(f"[soup] saved {out}", flush=True)
for p, w in pairs:
    print(f"   {w * 100:6.2f}%  {p.split('/')[-1]}", flush=True)
