"""Install/freeze the control adapters on a loaded StableAudioModel — at runtime,
no edits to the SA3 fork. Wraps every cross-attention module in place.
"""

from __future__ import annotations

from .adapters import ControlledCrossAttention


def _is_cross_attn(m) -> bool:
    # cross-attn has separate to_q + to_kv (and apply_attn); self-attn has fused to_qkv
    return (hasattr(m, "to_q") and hasattr(m, "to_kv")
            and not hasattr(m, "to_qkv") and hasattr(m, "apply_attn"))


def find_cross_attn(root):
    """Return [(parent_module, attr_name, module)] for every SA3 cross-attention
    reachable from `root` (walks nesting via named_modules)."""
    out = []
    for name, m in root.named_modules():
        if name.endswith(".cross_attn") and _is_cross_attn(m):
            parent = root.get_submodule(name.rsplit(".", 1)[0])
            out.append((parent, name.rsplit(".", 1)[1], m))
    return out


def install_adapters(sam, control_dim: int, position_encoding: bool = True):
    """Wrap every cross-attn in `sam.model` with a control adapter (idempotent).
    Returns the list of ControlledCrossAttention wrappers."""
    targets = find_cross_attn(sam.model)
    if not targets:
        raise RuntimeError("no SA3 cross-attention modules found to wrap")
    wrappers = []
    for parent, attr, base in targets:
        if isinstance(base, ControlledCrossAttention):
            wrappers.append(base)
            continue
        dev = next(base.parameters()).device
        dt = next(base.parameters()).dtype
        w = ControlledCrossAttention(base, control_dim, position_encoding).to(device=dev)
        # adapter K/V/out stay in their own (default fp32) dtype unless cast by the caller
        setattr(parent, attr, w)
        wrappers.append(w)
    return wrappers


def freeze_base_train_adapters(sam, wrappers, extra_trainable=()):
    """Freeze ALL base params; unfreeze only the adapter branches (+ extra modules,
    e.g. the control conditioner). Returns the trainable parameter list."""
    sam.model.requires_grad_(False)
    params = []
    for w in wrappers:
        for p in w.adapter.parameters():
            p.requires_grad_(True)
            params.append(p)
    for mod in extra_trainable:
        for p in mod.parameters():
            p.requires_grad_(True)
            params.append(p)
    return params


def adapter_state_dict(wrappers, conditioner=None):
    """Collect only the trainable state (adapter branches + conditioner) for saving."""
    sd = {}
    for i, w in enumerate(wrappers):
        for k, v in w.adapter.state_dict().items():
            sd[f"adapter.{i}.{k}"] = v.detach().cpu()
    if conditioner is not None:
        for k, v in conditioner.state_dict().items():
            sd[f"conditioner.{k}"] = v.detach().cpu()
    return sd
