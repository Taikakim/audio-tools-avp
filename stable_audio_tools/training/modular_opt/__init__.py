"""Modular Stage-Based Optimizer Framework.

A norm-constrained, geometrically faithful optimizer package implementing the
Transform-Solve-Invert pipeline with role-based parameter routing.

Public API:
    - ModularOptimizer: The main optimizer class (6-stage pipeline).
    - SignLMO, SpectralLMO, ColNormLMO: Linear Minimization Oracles.
    - KLShampooPreconditioner, RotatedSOAPPreconditioner, IdentityPreconditioner.
    - LIFOTransformationStack: Whitening/unwhitening stack.
    - build_modular_param_groups, summarise_modular_groups: Parameter routing.
    - calculate_radius_scale: Geometric radius scaling.
    - NormConstraint, Preconditioner: Protocols for extensibility.

Origin: Kim & Antigravity.Neuromancer
"""

from .optimizer import ModularOptimizer
from .lmo import (
    SignLMO,
    SpectralLMO,
    ColNormLMO,
    newton_schulz_cubic,
    newton_schulz_quintic,
    newton_schulz_cubic5,
    newton_schulz_polar_express,
    spectral_hardcap,
    spectral_clip,
)
from .preconditioners import (
    IdentityPreconditioner,
    KLShampooPreconditioner,
    RotatedSOAPPreconditioner,
    inv_quarter_spd,
)
from .stack import LIFOTransformationStack
from .routing import (
    build_modular_param_groups,
    summarise_modular_groups,
    calculate_radius_scale,
    get_block_count,
    infer_attention_dim,
)
from .protocols import NormConstraint, Preconditioner

__all__ = [
    # Core optimizer
    "ModularOptimizer",
    # LMOs
    "SignLMO",
    "SpectralLMO",
    "ColNormLMO",
    "newton_schulz_cubic",
    "newton_schulz_quintic",
    "newton_schulz_cubic5",
    "newton_schulz_polar_express",
    "spectral_hardcap",
    "spectral_clip",
    # Preconditioners
    "IdentityPreconditioner",
    "KLShampooPreconditioner",
    "RotatedSOAPPreconditioner",
    "inv_quarter_spd",
    # Stack
    "LIFOTransformationStack",
    # Routing
    "build_modular_param_groups",
    "summarise_modular_groups",
    "calculate_radius_scale",
    "get_block_count",
    "infer_attention_dim",
    # Protocols
    "NormConstraint",
    "Preconditioner",
]
