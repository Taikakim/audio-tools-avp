"""Parameter Group Routing and Functional Block Splitting Engine.

Implements Role-Based Norm Assignment and Block-Aware Splitting:
1. Input Embeddings -> ColNorm
2. Hidden Attention / MLP 2D Weights -> Spectral (UV^T, radius \rho_\ell)
3. Output / Logits Projections -> Sign (\ell_\infty \leftrightarrow \ell_1)
4. AdaLN Modulation Emitters & Biases -> Sign
5. Enforces "Sign-Spectral-Sign" sandwich architecture.
6. Block-aware slicing:
   - split_qkv: splits fused attention [3 * d_head * n_heads, d_model] into 3 blocks.
   - split_adaln: splits fused modulation emitters [k * d_model, d_model] into k blocks.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Sequence
import torch
import torch.nn as nn
from torch import Tensor

# Regexes for architectural layer identification
_EMBEDDING_RE = re.compile(r"(embed|embedding|patch_embed|time_embed|token_embed)", re.IGNORECASE)
_OUTPUT_RE = re.compile(r"(^|\.)(project_out|final_layer|lm_head|output_proj|out_head|final_proj)($|\.)", re.IGNORECASE)
_ADALN_RE = re.compile(r"(^|\.)(global_cond_embedder\.2|modulation|scale_shift_table)($|\.)", re.IGNORECASE)
_FUSED_ATTN_RE = re.compile(r"\.(to_qkv|to_kv|to_q)\.")
_TO_OUT_RE = re.compile(r"\.to_out\.")

MIN_SPECTRAL_DIM = 64


def infer_attention_dim(named_params: Sequence[tuple[str, Tensor]]) -> int | None:
    """Infer the attention model dimension dim for QKV block splitting."""
    for n, p in named_params:
        if _TO_OUT_RE.search(n) and (n.endswith("lora_B") or n.endswith(".weight")):
            return int(p.shape[0])
    outs = [
        int(p.shape[0])
        for n, p in named_params
        if _FUSED_ATTN_RE.search(n) and (n.endswith("lora_B") or n.endswith(".weight"))
    ]
    if outs:
        return math.gcd(*outs) if len(outs) > 1 else outs[0]
    return None


def calculate_radius_scale(shape: torch.Size | tuple[int, ...]) -> float:
    """Calculate the geometric radius scale \rho_\ell = \max(1, \sqrt{d_{out}/d_{in}}).

    This prevents preactivation variance collapse in expansion/reduction layers.
    """
    if len(shape) < 2:
        return 1.0
    d_out, d_in = shape[0], shape[1]
    if d_in <= 0:
        return 1.0
    return max(1.0, math.sqrt(d_out / d_in))


def get_block_count(
    name: str,
    p: Tensor,
    attn_dim: int | None,
    split_qkv: bool = False,
    split_adaln: bool = False,
) -> int:
    """Determine block count for functional block splitting."""
    if p.ndim != 2:
        return 1

    if split_qkv and attn_dim and _FUSED_ATTN_RE.search(name):
        if p.shape[0] % attn_dim == 0 and p.shape[0] // attn_dim > 1:
            return p.shape[0] // attn_dim

    if split_adaln and _ADALN_RE.search(name):
        if p.shape[1] > 0 and p.shape[0] % p.shape[1] == 0 and p.shape[0] // p.shape[1] > 1:
            return p.shape[0] // p.shape[1]

    return 1


def build_modular_param_groups(
    model: nn.Module,
    default_whitening: str = "none",
    spectral_lr: float | None = None,
    sign_lr: float | None = None,
    colnorm_lr: float | None = None,
    spectral_wd: float = 0.01,
    sign_wd: float = 0.0,
    colnorm_wd: float = 0.0,
    split_qkv: bool = True,
    split_adaln: bool = True,
    force_sign: Iterable[str] = (),
) -> list[dict]:
    """Group model parameters according to Role-Based Norm Assignment.

    Routing Rules:
    - Input Embeddings -> ColNorm
    - Output Projections & Logits -> Sign
    - AdaLN Emitters & 1D Biases/Gains -> Sign
    - Hidden 2D Matrices -> Spectral with \rho_\ell = \max(1, \sqrt{d_{out}/d_{in}})

    Returns a list of param group dicts suitable for ModularOptimizer.
    """
    named_trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    attn_dim = infer_attention_dim(named_trainable) if split_qkv else None
    force_sign_pats = [re.compile(pat) for pat in force_sign]

    groups_dict = {
        "colnorm": {"params": [], "param_names": [], "radii": [], "blocks": []},
        "spectral": {"params": [], "param_names": [], "radii": [], "blocks": []},
        "sign": {"params": [], "param_names": [], "radii": [], "blocks": []},
    }

    for name, p in named_trainable:
        # Check explicit overrides
        if any(pat.search(name) for pat in force_sign_pats):
            role = "sign"
        # 1D parameters, biases, norms always use Sign
        elif p.ndim < 2:
            role = "sign"
        # AdaLN modulation emitter always uses Sign (preserves modulation scale)
        elif _ADALN_RE.search(name):
            role = "sign"
        # Input embeddings use ColNorm
        elif _EMBEDDING_RE.search(name) and p.ndim == 2:
            role = "colnorm"
        # Standard 2D linear/conv projections (including DiT output projection) use Spectral
        elif p.ndim == 2 and min(p.shape) >= MIN_SPECTRAL_DIM:
            role = "spectral"
        else:
            role = "sign"

        if role == "spectral":
            radius = calculate_radius_scale(p.shape)
        elif role == "sign" and p.ndim == 2:
            # SCION Table 3: 2D sign layers (output head, modulation) scale by 1 / sqrt(d_in)
            # to preserve width-invariant preactivation update variance
            radius = 1.0 / math.sqrt(p.shape[1]) if p.shape[1] > 0 else 1.0
        else:
            radius = 1.0

        blocks = get_block_count(name, p, attn_dim, split_qkv=split_qkv, split_adaln=split_adaln)

        groups_dict[role]["params"].append(p)
        groups_dict[role]["param_names"].append(name)
        groups_dict[role]["radii"].append(radius)
        groups_dict[role]["blocks"].append(blocks)

    param_groups = []

    # 1. Spectral Group
    if groups_dict["spectral"]["params"]:
        g = {
            "group_type": "spectral",
            "params": groups_dict["spectral"]["params"],
            "param_names": groups_dict["spectral"]["param_names"],
            "radii": groups_dict["spectral"]["radii"],
            "blocks": groups_dict["spectral"]["blocks"],
            "whitening": default_whitening,
            "weight_decay": spectral_wd,
        }
        if spectral_lr is not None:
            g["lr"] = spectral_lr
        param_groups.append(g)

    # 2. ColNorm Group (Embeddings)
    if groups_dict["colnorm"]["params"]:
        g = {
            "group_type": "colnorm",
            "params": groups_dict["colnorm"]["params"],
            "param_names": groups_dict["colnorm"]["param_names"],
            "radii": groups_dict["colnorm"]["radii"],
            "blocks": groups_dict["colnorm"]["blocks"],
            "whitening": "none",
            "weight_decay": colnorm_wd,
        }
        if colnorm_lr is not None:
            g["lr"] = colnorm_lr
        param_groups.append(g)

    # 3. Sign Group (Output, AdaLN, 1D, Biases)
    if groups_dict["sign"]["params"]:
        g = {
            "group_type": "sign",
            "params": groups_dict["sign"]["params"],
            "param_names": groups_dict["sign"]["param_names"],
            "radii": groups_dict["sign"]["radii"],
            "blocks": groups_dict["sign"]["blocks"],
            "whitening": "none",
            "weight_decay": sign_wd,
        }
        if sign_lr is not None:
            g["lr"] = sign_lr
        param_groups.append(g)

    return param_groups


def summarise_modular_groups(groups: list[dict]) -> str:
    """Generate human-readable summary of parameter routing."""
    lines = ["Modular Optimizer Parameter Routing:"]
    for g in groups:
        n_params = sum(p.numel() for p in g["params"])
        lines.append(
            f"  [{g['group_type'].upper():>8}] {len(g['params']):>3} tensors  "
            f"{n_params:>10,d} params  wd={g['weight_decay']}  whitening={g.get('whitening', 'none')}"
        )
        for name, p, rad, blk in zip(
            g.get("param_names", []), g["params"], g.get("radii", []), g.get("blocks", [])
        ):
            extra = f"rho={rad:.2f}" if rad != 1.0 else ""
            if blk > 1:
                extra += f" split={blk}"
            lines.append(f"      {name:<52} {tuple(p.shape)} {extra}")
    return "\n".join(lines)
