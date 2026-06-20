"""Comparative statistics on trained control-adapter weights — where do heads differ?

Pure CPU. Loads the `state` dict (adapter.{i}.* + conditioner.*) from each checkpoint and
reports per-head norms, the per-layer learning profile, and cross-head cosine similarity.

High cross-head cosine despite different generation evals => the heads learned the SAME thing,
LR mostly changed the speed not the destination (the Flux lesson). Low cosine => genuinely
different solutions.

    python sa3_control/weight_stats.py <tag>=<ckpt.pt> <tag>=<ckpt.pt> ...
"""
import sys
import torch
import numpy as np

if len(sys.argv) < 2:
    sys.exit("usage: weight_stats.py tag=ckpt.pt [tag=ckpt.pt ...]")

heads = {}
for arg in sys.argv[1:]:
    tag, path = arg.split("=", 1)
    heads[tag] = torch.load(path, map_location="cpu", weights_only=False)["state"]

tags = list(heads)
keys = sorted(set.intersection(*[set(h) for h in heads.values()]))
adapter_keys = [k for k in keys if k.startswith("adapter.")]
cond_keys = [k for k in keys if k.startswith("conditioner.")]


def flat(state, ks):
    return torch.cat([state[k].flatten().float() for k in ks]) if ks else torch.tensor([])


# 1. Per-head norms
print("=== per-head L2 norms ===")
print(f"  {'head':22s} {'adapter':>10s} {'conditioner':>12s} {'total':>10s}")
for t in tags:
    a = flat(heads[t], adapter_keys).norm().item()
    c = flat(heads[t], cond_keys).norm().item()
    tot = flat(heads[t], keys).norm().item()
    print(f"  {t:22s} {a:10.3f} {c:12.3f} {tot:10.3f}")

# 2. Cross-head cosine similarity (whole adapter, whole conditioner)
def cos(a, b):
    a, b = a.double(), b.double()            # fp64: fp32 dot of 119M elems loses precision (cos > 1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))

print("\n=== cross-head cosine similarity (1.0 = identical direction) ===")
for what, ks in [("adapter", adapter_keys), ("conditioner", cond_keys)]:
    print(f"  [{what}]")
    print("           " + "".join(f"{t[:10]:>11s}" for t in tags))
    vecs = {t: flat(heads[t], ks) for t in tags}
    for ti in tags:
        row = "".join(f"{cos(vecs[ti], vecs[tj]):>11.3f}" for tj in tags)
        print(f"  {ti[:9]:9s}{row}")

# 3. Per-adapter-layer norm profile (which cross-attn layers learned most)
print("\n=== per-layer adapter norm (which of the 24 cross-attn learned most) ===")
layers = sorted({int(k.split(".")[1]) for k in adapter_keys})
print(f"  layer " + "".join(f"{t[:9]:>10s}" for t in tags))
for li in layers:
    lks = [k for k in adapter_keys if k.startswith(f"adapter.{li}.")]
    row = "".join(f"{flat(heads[t], lks).norm().item():>10.3f}" for t in tags)
    print(f"  {li:5d} {row}")
