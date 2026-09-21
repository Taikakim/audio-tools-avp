"""ModularOptimizer — 6-Stage Canonical Execution Pipeline.

Implements the Transform-Solve-Invert pipeline for norm-constrained steepest descent:

    Stage 1: Momentum Accumulation (original-space or rotated-space)
    Stage 2: Forward Whitening (push preconditioner factors to LIFO stack)
    Stage 3: Core LMO Solver (Spectral / Sign / ColNorm with block splitting)
    Stage 4: Reverse Unwhitening (pop from LIFO stack — exact Riemannian pullback)
    Stage 5: Prodigy Escape Velocity (dual-norm coupled denominator) + SNR gate
    Stage 6: Outer step-size η, radius scaling ρ_ℓ, weight decay, Schedule-Free

Design principles:
    - Scale invariance: gradient magnitude does not affect step direction.
    - Geometric fidelity: pullbacks preserve optimality under the preconditioned metric.
    - Modularity: each stage can be independently enabled/configured per parameter group.
    - Zero disruption: existing production code (fusion_opt.py, train_lora.py) is never touched.
"""

from __future__ import annotations

import math
from typing import Iterable, Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from .lmo import SignLMO, SpectralLMO, ColNormLMO
from .preconditioners import (
    IdentityPreconditioner,
    KLShampooPreconditioner,
    RotatedSOAPPreconditioner,
)
from .stack import LIFOTransformationStack


# ---------------------------------------------------------------------------
# LMO singleton registry — one instance per type (stateless, safe to share)
# ---------------------------------------------------------------------------
_SIGN_LMO = SignLMO()
_COLNORM_LMO = ColNormLMO()

# SpectralLMO instances are created per-config (algorithm/steps can vary),
# but we keep a default for the common case.
_SPECTRAL_LMO_QUINTIC = SpectralLMO(algorithm="quintic", steps=5)
_SPECTRAL_LMO_CUBIC5 = SpectralLMO(algorithm="cubic5", steps=5)
_SPECTRAL_LMO_CUBIC = SpectralLMO(algorithm="cubic", steps=12)


def _get_lmo(group_type: str, ns_poly: str = "quintic") -> SignLMO | SpectralLMO | ColNormLMO:
    """Return the appropriate LMO for a parameter group type."""
    if group_type == "spectral":
        if ns_poly == "cubic5":
            return _SPECTRAL_LMO_CUBIC5
        elif ns_poly == "quintic":
            return _SPECTRAL_LMO_QUINTIC
        else:
            return _SPECTRAL_LMO_CUBIC
    elif group_type == "colnorm":
        return _COLNORM_LMO
    else:  # "sign" or anything else
        return _SIGN_LMO


# ---------------------------------------------------------------------------
# ModularOptimizer
# ---------------------------------------------------------------------------


class ModularOptimizer(Optimizer):
    """Modular Stage-Based Optimizer implementing the 6-stage Transform-Solve-Invert pipeline.

    Parameter groups MUST contain:
        - group_type: str — "spectral", "sign", or "colnorm"
        - param_names: list[str] — human-readable names for logging
        - radii: list[float] — per-parameter radius scaling ρ_ℓ
        - blocks: list[int] — per-parameter block splitting count

    Optional per-group overrides:
        - whitening: str — "none" | "shampoo" | "soap" (default: "none")
        - weight_decay: float (default: 0.0)
        - lr: float (overrides global lr for this group)
    """

    def __init__(
        self,
        params: Iterable[dict],
        lr: float = 3e-4,
        # Momentum
        beta1: float = 0.9,
        # Preconditioner
        beta_precond: float = 0.95,
        precond_delta: float = 1e-4,
        precond_update_freq: int = 1,
        # Newton-Schulz polynomial for SpectralLMO
        ns_poly: str = "quintic",
        # Prodigy escape velocity
        escape_velocity: bool = False,
        ev_beta: float = 0.999,
        ev_max: float = 2.0,
        # SNR gating
        snr_gate: bool = False,
        snr_beta: float = 0.9,
        snr_floor: float = 0.0,
        # Weight decay
        weight_decay: float = 0.0,
        wd_overtraining: bool = False,
        steps_per_epoch: float | None = None,
        # NorMuon per-neuron row scaling (arXiv:2608.20818, Section 3.2)
        normuon: bool = True,
        normuon_beta: float = 0.95,
        # Schedule-Free averaging
        schedule_free: bool = True,
        sf_beta: float = 0.9,
        sf_c_warmup: int | None = None,
        sf_r: float = 1.0,
        muon_sw_decay: bool = True,
        adamc_decay: bool = True,
        # Warmup
        warmup_steps: int = 0,
        # Variance-Aware Dynamic Dampening (VADD)
        var_dampening_threshold: float | None = None,
        var_dampening_power: float = 1.0,
        var_wd_boost: float = 0.0,
        # Radial Brake (NVIDIA soft limiting on parameter norm growth)
        radial_brake: float = 1.0,
        # Global
        eps: float = 1e-12,
    ):
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta_precond=beta_precond,
            precond_delta=precond_delta,
            precond_update_freq=precond_update_freq,
            ns_poly=ns_poly,
            escape_velocity=escape_velocity,
            ev_beta=ev_beta,
            ev_max=ev_max,
            snr_gate=snr_gate,
            snr_beta=snr_beta,
            snr_floor=snr_floor,
            weight_decay=weight_decay,
            wd_overtraining=wd_overtraining,
            steps_per_epoch=steps_per_epoch,
            normuon=normuon,
            normuon_beta=normuon_beta,
            schedule_free=schedule_free,
            sf_beta=sf_beta,
            sf_c_warmup=sf_c_warmup,
            sf_r=sf_r,
            muon_sw_decay=muon_sw_decay,
            adamc_decay=adamc_decay,
            warmup_steps=warmup_steps,
            var_dampening_threshold=var_dampening_threshold,
            var_dampening_power=var_dampening_power,
            var_wd_boost=var_wd_boost,
            radial_brake=radial_brake,
            eps=eps,
        )
        super().__init__(params, defaults)

        self._step_count = 0
        self.var_dampening_threshold = var_dampening_threshold
        self.var_dampening_power = var_dampening_power
        self.var_wd_boost = var_wd_boost
        self.observed_variance: float | None = None
        self._mode = "train"
        self._comp_telem: dict[str, float] = {}
        self._telem_on: bool = True

        # Validate groups
        for g in self.param_groups:
            gtype = g.get("group_type")
            if gtype not in ("spectral", "sign", "colnorm"):
                raise ValueError(
                    f"ModularOptimizer param groups must set group_type='spectral'|'sign'|'colnorm', "
                    f"got '{gtype}'"
                )

    @property
    def component_telemetry(self) -> dict[str, float]:
        """Return the current step's component telemetry dict."""
        return self._comp_telem

    # ------------------------------------------------------------------
    # Schedule-Free train/eval toggle
    # ------------------------------------------------------------------

    @property
    def uses_sf_averaging(self) -> bool:
        """Whether this optimiser uses Schedule-Free averaging."""
        return any(g.get("schedule_free", False) for g in self.param_groups)

    def train(self) -> None:
        """Switch params to the training point y = (1-β)z + β·x."""
        if self._mode == "train":
            return
        self._mode = "train"
        for group in self.param_groups:
            if not group.get("schedule_free", False):
                continue
            sf_beta = group["sf_beta"]
            for p in group["params"]:
                st = self.state.get(p)
                if st is None or "z" not in st:
                    continue
                p.data.copy_((1 - sf_beta) * st["z"] + sf_beta * st["x"])

    def eval(self) -> None:
        """Switch params to the averaged iterate x (deployable model)."""
        if self._mode == "eval":
            return
        self._mode = "eval"
        for group in self.param_groups:
            if not group.get("schedule_free", False):
                continue
            for p in group["params"]:
                st = self.state.get(p)
                if st is None or "x" not in st:
                    continue
                p.data.copy_(st["x"])

    def set_observed_variance(self, observed_std: float) -> None:
        """Called by training loop before step() to report the batch latent variance."""
        self.observed_variance = observed_std

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Execute the 6-stage Transform-Solve-Invert pipeline for all parameter groups."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Linear warmup factor
        warmup = self.param_groups[0].get("warmup_steps", 0)
        if warmup > 0 and self._step_count < warmup:
            warm_factor = (self._step_count + 1) / warmup
        else:
            warm_factor = 1.0

        telem = {
            "n_params": 0,
            "mom_sq": 0.0,
            "lmo_sq": 0.0,
            "normuon_sq": 0.0,
            "update_sq": 0.0,
            "z_norm_sq": 0.0,
            "x_norm_sq": 0.0,
            "ev_list": [],
            "snr_list": [],
            "var_damp_list": [],
            "radial_brake_list": [],
            "eta_list": [],
            "eff_wd_list": [],
            "wd_decay_list": [],
            "ck_list": [],
        } if self._telem_on else None

        for group in self.param_groups:
            self._group_step(group, warm_factor, telem=telem)

        if self._telem_on and telem and telem["n_params"] > 0:
            mom_norm = math.sqrt(telem["mom_sq"])
            lmo_norm = math.sqrt(telem["lmo_sq"])
            normuon_norm = math.sqrt(telem["normuon_sq"])
            update_norm = math.sqrt(telem["update_sq"])
            z_norm = math.sqrt(telem["z_norm_sq"])
            x_norm = math.sqrt(telem["x_norm_sq"])

            self._comp_telem = {
                "comp/momentum_norm": mom_norm,
                "comp/lmo_norm": lmo_norm,
                "comp/ns_compression_ratio": lmo_norm / max(mom_norm, 1e-12),
                "comp/normuon_gain": normuon_norm / max(lmo_norm, 1e-12),
                "comp/spectral_update_norm": update_norm,
                "comp/weight_norm_z": z_norm,
                "comp/weight_norm_x": x_norm,
                "comp/sf_norm_ratio": x_norm / max(z_norm, 1e-12) if z_norm > 0 else 1.0,
                "comp/ev_d": sum(telem["ev_list"]) / len(telem["ev_list"]) if telem["ev_list"] else 1.0,
                "comp/snr_gate": sum(telem["snr_list"]) / len(telem["snr_list"]) if telem["snr_list"] else 1.0,
                "comp/var_damp_kappa": sum(telem["var_damp_list"]) / len(telem["var_damp_list"]) if telem["var_damp_list"] else 1.0,
                "comp/radial_brake_scale": sum(telem["radial_brake_list"]) / len(telem["radial_brake_list"]) if telem["radial_brake_list"] else 1.0,
                "comp/effective_eta": sum(telem["eta_list"]) / len(telem["eta_list"]) if telem["eta_list"] else 0.0,
                "comp/effective_wd": sum(telem["eff_wd_list"]) / len(telem["eff_wd_list"]) if telem["eff_wd_list"] else 0.0,
                "comp/wd_decay_factor": sum(telem["wd_decay_list"]) / len(telem["wd_decay_list"]) if telem["wd_decay_list"] else 0.0,
                "comp/sf_ck": sum(telem["ck_list"]) / len(telem["ck_list"]) if telem["ck_list"] else 1.0,
            }

        # Schedule-Free: write y back to p.data for next forward pass
        if self._mode == "train":
            for group in self.param_groups:
                if not group.get("schedule_free", False):
                    continue
                sf_beta = group["sf_beta"]
                for p in group["params"]:
                    st = self.state.get(p)
                    if st is None or "z" not in st:
                        continue
                    p.data.copy_(st["z"]).mul_(1.0 - sf_beta).add_(st["x"], alpha=sf_beta)

        self._step_count += 1
        return loss

    # ------------------------------------------------------------------
    # Per-group 6-stage pipeline
    # ------------------------------------------------------------------

    def _group_step(self, group: dict, warm_factor: float, telem: dict | None = None) -> None:
        """Run the full 6-stage pipeline for a single parameter group."""
        group_type: str = group["group_type"]
        param_names: list[str] = group.get("param_names", [])
        radii: list[float] = group.get("radii", [])
        blocks: list[int] = group.get("blocks", [])
        whitening: str = group.get("whitening", "none")

        lr = group["lr"]
        beta1 = group["beta1"]
        ns_poly = group.get("ns_poly", "quintic")
        wd = group.get("weight_decay", 0.0)
        eps = group.get("eps", 1e-12)
        ev_enabled = group.get("escape_velocity", False)
        ev_beta = group.get("ev_beta", 0.999)
        ev_max = group.get("ev_max", 2.0)
        snr_enabled = group.get("snr_gate", False)
        snr_beta = group.get("snr_beta", 0.9)
        snr_floor = group.get("snr_floor", 0.0)
        normuon_enabled = group.get("normuon", True)
        normuon_beta = group.get("normuon_beta", 0.95)
        sf_enabled = group.get("schedule_free", True)
        sf_beta = group.get("sf_beta", 0.9)

        lmo = _get_lmo(group_type, ns_poly)

        for i, p in enumerate(group["params"]):
            if p.grad is None:
                continue

            grad = p.grad
            name = param_names[i] if i < len(param_names) else f"param_{i}"
            radius = radii[i] if i < len(radii) else 1.0
            n_blocks = blocks[i] if i < len(blocks) else 1

            state = self.state[p]

            # ── Lazy state initialization ────────────────────────────
            if len(state) == 0:
                self._init_state(state, p, group, whitening)

            # ── Stage 1: Momentum Accumulation ───────────────────────
            momentum = state["momentum"]
            momentum.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
            # Bias-corrected momentum direction
            bc1 = 1.0 - beta1 ** (self._step_count + 1)
            d_k = momentum / bc1

            # ── Stage 2: Forward Whitening ───────────────────────────
            stack = state["stack"]
            stack.clear()  # Safety: ensure clean stack each step

            preconditioner = state.get("preconditioner")
            if preconditioner is not None and whitening != "none":
                # Update covariance accumulators from raw gradient
                if grad.ndim == 2:
                    preconditioner.update_accumulators(grad)
                    # Forward transform (whitening)
                    d_k = preconditioner.transform(d_k)
                    # Push inverse onto LIFO stack
                    stack.push(
                        name=preconditioner.name,
                        inverse_fn=preconditioner.inverse_transform,
                    )

            # ── Stage 3: Core LMO Solver with Block Splitting ────────
            if n_blocks > 1 and d_k.ndim == 2:
                # Block-aware splitting (QKV or AdaLN)
                block_size = d_k.shape[0] // n_blocks
                lmo_blocks = []
                for b_idx in range(n_blocks):
                    start = b_idx * block_size
                    end = start + block_size
                    block_grad = d_k[start:end]
                    block_radius = radius  # Same radius for each sub-block
                    lmo_result = lmo.compute_lmo(block_grad, radius=block_radius)
                    lmo_blocks.append(lmo_result)
                M = torch.cat(lmo_blocks, dim=0)
            else:
                # Standard single-block LMO
                if group_type == "spectral" and d_k.ndim != 2:
                    # Spectral LMO requires 2D — fall back to Sign for non-2D
                    M = _SIGN_LMO.compute_lmo(d_k, radius=radius)
                else:
                    M = lmo.compute_lmo(d_k, radius=radius)
            M_lmo = M

            # ── Stage 3b: NorMuon Per-Neuron Row Scaling ─────────────
            # (Li & Han 2026 arXiv:2608.20818, Section 3.2 / FusionOpt)
            # Normalizes each neuron row to unit variance under EMA.
            # Bounds per-neuron velocity in DiTs, preventing harsh timbres and glitches.
            if normuon_enabled and group_type == "spectral" and M.ndim == 2:
                if "normuon_r" not in state:
                    state["normuon_r"] = torch.zeros(M.shape[0], device=M.device, dtype=torch.float32)
                rn = state["normuon_r"]
                row_ss = (M * M).sum(dim=-1).float()
                rn.mul_(normuon_beta).add_(row_ss, alpha=(1.0 - normuon_beta))
                bc_r = 1.0 - normuon_beta ** (self._step_count + 1)
                r_hat = rn / max(bc_r, 1e-12)
                M = M / r_hat.clamp_min(eps).sqrt().unsqueeze(-1).to(M.dtype)

            # ── Stage 4: Reverse Unwhitening (Pullback) ──────────────
            if not stack.is_empty():
                M = stack.unwhiten(M)

            # ── Stage 5: Prodigy Escape Velocity + SNR Gate ──────────
            ev_mult = 1.0
            if ev_enabled:
                ev_mult = self._escape_velocity_step(
                    state, p, grad, lmo, group_type, ev_beta, ev_max, eps, M=M
                )

            snr_mult = 1.0
            if snr_enabled:
                snr_mult = self._snr_gate_step(state, grad, snr_beta, snr_floor, eps)

            # ── Variance-Aware Dynamic Dampening (VADD) ─────────────
            var_damp = 1.0
            var_thresh = group.get("var_dampening_threshold") or self.var_dampening_threshold
            if var_thresh is not None and self.observed_variance is not None:
                if self.observed_variance > var_thresh:
                    power = group.get("var_dampening_power", self.var_dampening_power)
                    var_damp = (var_thresh / max(self.observed_variance, 1e-8)) ** power
                    # Option 1: both - Emergency Escape Velocity brake + multiplier throttle
                    ev_mult = ev_mult * var_damp
                    if ev_enabled and "ev_d" in state:
                        state["ev_d"] = max(1.0, state["ev_d"] * var_damp)

            # ── Stage 6: Outer Step-Size, Radius, Scaled Weight Decay ───────
            # Effective step = lr * warmup * escape_velocity * snr_gate * var_damp
            eta = lr * warm_factor * ev_mult * snr_mult * var_damp

            group_type = group.get("group_type", "spectral")
            is_spectral = (group_type == "spectral")
            is_sign = (group_type == "sign")

            # Overtraining-aware weight decay base scaling: wd_t = wd_base * sqrt(f_t)
            # where f_t = max(1.0, current_step / steps_per_epoch) (Everett & Qiu 2026, arXiv:2609.04577)
            eff_wd = wd
            if wd > 0 and group.get("wd_overtraining", False):
                spe = group.get("steps_per_epoch")
                if spe and spe > 0:
                    f_t = max(1.0, (self._step_count + 1) / spe)
                    eff_wd = wd * math.sqrt(f_t)

            # Boost weight decay during high-variance episodes
            if var_damp < 1.0 and eff_wd > 0:
                boost = group.get("var_wd_boost", self.var_wd_boost)
                if boost > 0:
                    eff_wd = eff_wd * (1.0 + boost * (1.0 - var_damp))

            # Branching decay scaling:
            # - Muon (spectral): Muon-SW quadratic scaling (eta^2 / eta_max) to preserve Robbins-Monro stationarity (Apte 2026)
            # - AdamW (sign): ScheduleFree+ AdamC quadratic scaling (eta^2 / eta_max) on query point y (Defazio 2026)
            # - Other (colnorm / fallback): linear decay (eta)
            if eff_wd > 0:
                eta_max = lr
                if is_spectral and group.get("muon_sw_decay", True):
                    wd_decay_factor = eff_wd * (eta * eta / max(eta_max, 1e-12))
                elif is_sign and group.get("adamc_decay", True):
                    wd_decay_factor = eff_wd * (eta * eta / max(eta_max, 1e-12))
                else:
                    wd_decay_factor = eff_wd * eta
            else:
                wd_decay_factor = 0.0

            if sf_enabled:
                # Schedule-Free: update z (fast iterate) and x (average iterate)
                z = state["z"]
                x = state["x"]

                # Apply weight decay
                if wd_decay_factor > 0:
                    if is_sign and group.get("adamc_decay", True):
                        # ScheduleFree+ AdamC: decay towards query point y (stored in p.data)
                        z.add_(p.data, alpha=-wd_decay_factor)
                    else:
                        # Muon-SW / standard: decay on fast iterate z
                        z.add_(z, alpha=-wd_decay_factor)

                # Radial brake soft-limiting (NVIDIA RadialBrakeHook)
                r_brake = group.get("radial_brake", 1.0)
                z_old_norm = z.norm() if r_brake < 1.0 else None
                radial_scale = 1.0

                # Apply direction update to z
                z.add_(M, alpha=-eta)

                if r_brake < 1.0:
                    z_new_norm = z.norm()
                    if z_new_norm > z_old_norm:
                        z_target = z_old_norm + r_brake * (z_new_norm - z_old_norm)
                        radial_scale = float((z_target / z_new_norm.clamp_min(1e-12)).item())
                        z.mul_(radial_scale)

                # ScheduleFree+ Averaging: C_warmup burn-in + power weighting (r=1)
                c_warmup = group.get("sf_c_warmup")
                if c_warmup is None:
                    c_warmup = 2 * group.get("warmup_steps", 0)

                if self._step_count < c_warmup:
                    # Burn-in phase: do not average; keep x tracking z directly
                    ck = 1.0
                else:
                    # Power-weighted averaging: w_t = (t - c_warmup)^r
                    # For r=1: ck = 2 / (t_eff + 2)
                    t_eff = self._step_count - c_warmup + 1
                    r = group.get("sf_r", 1.0)
                    ck = (r + 1.0) / (t_eff + r + 1.0)

                x.mul_(1.0 - ck).add_(z, alpha=ck)
            else:
                # Standard non-SF update
                if wd_decay_factor > 0:
                    p.data.add_(p.data, alpha=-wd_decay_factor)

                r_brake = group.get("radial_brake", 1.0)
                p_old_norm = p.data.norm() if r_brake < 1.0 else None
                radial_scale = 1.0

                p.data.add_(M, alpha=-eta)

                if r_brake < 1.0:
                    p_new_norm = p.data.norm()
                    if p_new_norm > p_old_norm:
                        p_target = p_old_norm + r_brake * (p_new_norm - p_old_norm)
                        radial_scale = float((p_target / p_new_norm.clamp_min(1e-12)).item())
                        p.data.mul_(radial_scale)

            if telem is not None:
                telem["n_params"] += 1
                telem["mom_sq"] += float((d_k * d_k).sum().item())
                telem["lmo_sq"] += float((M_lmo * M_lmo).sum().item())
                telem["normuon_sq"] += float((M * M).sum().item())
                telem["update_sq"] += float((M * eta).pow(2).sum().item())
                if sf_enabled:
                    telem["z_norm_sq"] += float((z * z).sum().item())
                    telem["x_norm_sq"] += float((x * x).sum().item())
                else:
                    p_sq = float((p.data * p.data).sum().item())
                    telem["z_norm_sq"] += p_sq
                    telem["x_norm_sq"] += p_sq
                telem["ev_list"].append(float(ev_mult))
                telem["snr_list"].append(float(snr_mult))
                telem["var_damp_list"].append(float(var_damp))
                telem["radial_brake_list"].append(float(radial_scale))
                telem["eta_list"].append(float(eta))
                telem["eff_wd_list"].append(float(eff_wd))
                telem["wd_decay_list"].append(float(wd_decay_factor))
                telem["ck_list"].append(float(ck if sf_enabled else 1.0))

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def _init_state(
        self,
        state: dict[str, Any],
        p: Tensor,
        group: dict,
        whitening: str,
    ) -> None:
        """Initialize per-parameter optimizer state."""
        # Momentum buffer (same dtype/device as parameter)
        state["momentum"] = torch.zeros_like(p.data)

        # LIFO stack for whitening/unwhitening
        state["stack"] = LIFOTransformationStack()

        # Preconditioner (only for 2D tensors with whitening enabled)
        if whitening != "none" and p.ndim == 2:
            shape = (p.shape[0], p.shape[1])
            beta_p = group.get("beta_precond", 0.95)
            delta = group.get("precond_delta", 1e-4)
            freq = group.get("precond_update_freq", 1)

            if whitening == "shampoo":
                state["preconditioner"] = KLShampooPreconditioner(
                    shape=shape,
                    device=p.device,
                    dtype=p.dtype,
                    beta=beta_p,
                    delta=delta,
                    update_freq=freq,
                )
            elif whitening == "soap":
                state["preconditioner"] = RotatedSOAPPreconditioner(
                    shape=shape,
                    device=p.device,
                    dtype=p.dtype,
                    beta=beta_p,
                    delta=delta,
                    update_freq=freq,
                )
            else:
                state["preconditioner"] = IdentityPreconditioner()
        else:
            state["preconditioner"] = None

        # Escape velocity state
        if group.get("escape_velocity", False):
            state["w0"] = p.data.clone()  # Initial weights
            state["ev_numerator"] = 0.0  # Running numerator
            state["ev_denominator"] = 0.0  # Running denominator
            state["ev_d"] = 1.0  # Current escape velocity multiplier

        # SNR gate state
        if group.get("snr_gate", False):
            state["snr_ema"] = torch.zeros_like(p.data)
            state["snr_sq_ema"] = torch.zeros_like(p.data)

        # Schedule-Free state
        if group.get("schedule_free", False):
            state["z"] = p.data.clone()  # Fast iterate
            state["x"] = p.data.clone()  # Averaged iterate

        # NorMuon per-neuron row scaling state
        if group.get("normuon", True) and group.get("group_type") == "spectral" and p.ndim == 2:
            state["normuon_r"] = torch.zeros(p.shape[0], device=p.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Stage 5 internals
    # ------------------------------------------------------------------

    def _escape_velocity_step(
        self,
        state: dict,
        p: Tensor,
        grad: Tensor,
        lmo: SignLMO | SpectralLMO | ColNormLMO,
        group_type: str,
        ev_beta: float,
        ev_max: float,
        eps: float,
        M: Tensor | None = None,
    ) -> float:
        """Prodigy escape velocity with dual-norm-coupled denominator.

        d_{k+1} = max(d_k, sum(α_τ <G_τ, W_0 - W_τ>) / sum(α_τ ||G_τ||_*))

        The denominator uses the EXACT dual norm corresponding to the layer's LMO:
            - ℓ_1 dual for Sign layers
            - Nuclear norm for Spectral layers
            - (2,1) norm for ColNorm layers

        This is mathematically mandated by Fenchel-Young duality to ensure the
        estimated distance d_t matches the geometry of each parameter subspace.
        """
        w0 = state["w0"]
        # Inner product <G_k, W_0 - W_k>
        if self._step_count == 0:
            ip = 0.0
        else:
            ip = float(torch.sum(grad * (w0 - p.data)).item())

        # Dual norm ||G_k||_* (geometry-matched)
        dual_norm = lmo.compute_dual_norm(grad, M=M) if hasattr(lmo, "compute_dual_norm") else float(grad.norm().item())

        # EMA update of numerator and denominator
        alpha_t = 1.0  # Uniform weighting (could use 1/(k+1) for bias correction)
        state["ev_numerator"] = ev_beta * state["ev_numerator"] + alpha_t * ip
        state["ev_denominator"] = ev_beta * state["ev_denominator"] + alpha_t * dual_norm

        # Escape velocity: d only grows, clamped to [1.0, ev_max]
        if state["ev_denominator"] > eps:
            d_hat = state["ev_numerator"] / state["ev_denominator"]
            state["ev_d"] = min(ev_max, max(state["ev_d"], d_hat))

        return state["ev_d"]

    def _snr_gate_step(
        self,
        state: dict,
        grad: Tensor,
        snr_beta: float,
        snr_floor: float,
        eps: float,
    ) -> float:
        """SNR gate: scale update by signal-to-noise ratio of raw gradient.

        Measures |EMA(G)| / sqrt(EMA(G²)) per element, averaged across the tensor.
        When gradients degrade to noise, the ratio drops toward 0, braking the update.
        """
        ema = state["snr_ema"]
        sq_ema = state["snr_sq_ema"]

        # Update EMAs
        ema.mul_(snr_beta).add_(grad, alpha=(1 - snr_beta))
        sq_ema.mul_(snr_beta).addcmul_(grad, grad, value=(1 - snr_beta))

        # Per-element SNR = |EMA(G)| / sqrt(EMA(G²))
        signal = ema.abs()
        noise = sq_ema.sqrt().clamp_min(eps)
        snr = (signal / noise).mean().item()

        # Floor + clamp
        snr = max(snr, snr_floor)
        return min(snr, 1.0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_step_count(self) -> int:
        """Return the current global step count."""
        return self._step_count

    def summary(self) -> str:
        """Return a human-readable summary of the optimizer configuration."""
        lines = [
            f"ModularOptimizer (step={self._step_count})",
            f"  schedule_free={self.uses_sf_averaging}",
        ]
        for i, g in enumerate(self.param_groups):
            n_params = sum(p.numel() for p in g["params"])
            lines.append(
                f"  Group {i} [{g['group_type']:>8}]: "
                f"{len(g['params'])} tensors, {n_params:,d} params, "
                f"lr={g['lr']:.2e}, wd={g.get('weight_decay', 0)}, "
                f"whitening={g.get('whitening', 'none')}"
            )
        return "\n".join(lines)
