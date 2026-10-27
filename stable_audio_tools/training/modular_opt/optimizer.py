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
import warnings
from typing import Iterable, Any

import torch
from torch import Tensor
from torch.optim import Optimizer

from .lmo import SignLMO, SpectralLMO, ColNormLMO
from .preconditioners import (
    IdentityPreconditioner,
    ShampooPreconditioner,
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


_NS_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_SPECTRAL_LMO_CACHE: dict[tuple[str, str], SpectralLMO] = {}


def _get_lmo(group_type: str, ns_poly: str = "quintic", ns_dtype: str = "fp32") -> SignLMO | SpectralLMO | ColNormLMO:
    """Return the appropriate LMO for a parameter group type."""
    if group_type == "spectral" and ns_dtype != "fp32":
        # C7 (KUANG 2026-10-17): Newton-Schulz in bf16/fp16; fp32 keeps the singletons below.
        key = (ns_poly, ns_dtype)
        if key not in _SPECTRAL_LMO_CACHE:
            algo = ns_poly if ns_poly in ("cubic5", "quintic") else "cubic"
            _SPECTRAL_LMO_CACHE[key] = SpectralLMO(algorithm=algo, steps=12 if algo == "cubic" else 5,
                                                   compute_dtype=_NS_DTYPES[ns_dtype])
        return _SPECTRAL_LMO_CACHE[key]
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



def _magnitude_step(t: Tensor, M: Tensor, step: float, group: dict, name: str) -> None:
    """Apply the Stage-6 update  t <- t - step*M  in place, or its multiplicative form for DoRA
    magnitudes.

    WHY (C, 2026-09-24). A sign-group step moves every element by the same absolute amount. On
    goa3_avp_r256_2026-09-23 (lr 6e-4, ~6900 steps) that walked the small global-conditioning
    magnitudes (mean |m| 0.13) through zero -- 592 non-positive rows in to_global_embed.0 by step
    6340 -- while the ~2.4 transformer-block magnitudes were unaffected; then the loss went NaN.
    With magnitude_update="multiplicative", params named *.magnitude take the step in log space:
        t <- t * exp(-step * M)
    i.e. a relative change of about step*|M| per element, the same for a 0.13 and a 2.4 scalar,
    and the sign of t can never change (exponentiated gradient / Madam, Bernstein et al. 2020).
    A DoRA magnitude is a row norm, so it should never need to cross zero. For small steps this equals
    the additive update scaled by |t|: t*exp(-sM) ~ t - s*t*M.
    """
    if group.get("magnitude_update", "additive") == "multiplicative" and name.endswith("magnitude"):
        t.mul_(torch.exp(M.to(t.dtype) * (-step)))
    else:
        t.add_(M, alpha=-step)


def _normuon_rows(M: Tensor, state: dict, mode: str, beta: float, step_count: int, eps: float,
                  n_blocks: int = 1) -> Tensor:
    """Stage 3b, NorMuon per-neuron row scaling, in one of two forms (C2, KUANG 2026-10-17).

    "row_unit" (the historical form, bit-identical): every row is divided by the square root of a
    bias-corrected EMA of its SUM of squares, so rows come out unit length and the matrix's total size
    is not restored. Same step size as dion on square/tall matrices; sqrt(d_in/d_out) larger on wide
    ones (docs/TRAINING.md F23).

    "dion" (microsoft/dion normuon.py::_normuon_normalization_core): EMA of the row MEAN square, divide,
    then rescale to the input's Frobenius norm (per row block when the matrix is block-split, like
    dion's split_sizes). Runs in fp32, as Dion3 section 6 found necessary.
    """
    if mode == "dion":
        if "normuon_v" not in state:
            state["normuon_v"] = torch.zeros(M.shape[0], device=M.device, dtype=torch.float32)
        v = state["normuon_v"]
        M32 = M.float()
        v.lerp_((M32 * M32).mean(dim=-1), 1.0 - beta)
        U = M32 / (v.sqrt() + 1e-8).unsqueeze(-1)
        if n_blocks > 1 and M.shape[0] % n_blocks == 0:
            Mb = M32.view(n_blocks, M.shape[0] // n_blocks, -1)
            Ub = U.view_as(Mb)
            U = (Ub * (Mb.norm(dim=(1, 2), keepdim=True)
                       / Ub.norm(dim=(1, 2), keepdim=True).clamp_min(1e-8))).view_as(M32)
        else:
            U = U * (M32.norm() / U.norm().clamp_min(1e-8))
        return U.to(M.dtype)
    if "normuon_r" not in state:
        state["normuon_r"] = torch.zeros(M.shape[0], device=M.device, dtype=torch.float32)
    rn = state["normuon_r"]
    row_ss = (M * M).sum(dim=-1).float()
    rn.mul_(beta).add_(row_ss, alpha=(1.0 - beta))
    bc_r = 1.0 - beta ** (step_count + 1)
    r_hat = rn / max(bc_r, 1e-12)
    return M / r_hat.clamp_min(eps).sqrt().unsqueeze(-1).to(M.dtype)


def _adam_direction(state: dict, grad: Tensor, group: dict, step_count: int, sf_enabled: bool) -> Tensor:
    """group_type="adam" (C3, KUANG 2026-10-17): element-wise Adam direction for the parameters the
    published SF-NorMuon recipe (arXiv 2605.23061, Table 5) runs on Schedule-Free AdamW.

    Under Schedule-Free there is NO momentum buffer (the y/z interpolation is the momentum, Defazio et
    al. 2024, Alg. 2): the direction is g / (sqrt(v_hat) + eps). Without Schedule-Free it is plain Adam
    with momentum group["adam_momentum"] (default: the group's beta1). Update RMS is a little below
    lr per element, i.e. the scale the "rms0.2" spectral update is matched to.
    """
    b2 = group.get("adam_beta2", 0.99)
    a_eps = group.get("adam_eps", 1e-8)
    mom = group.get("adam_momentum")
    if mom is None:
        mom = 0.0 if sf_enabled else group["beta1"]
    g = grad.float()
    if "adam_v" not in state:
        state["adam_v"] = torch.zeros_like(g)
    v = state["adam_v"]
    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
    v_hat = v / (1.0 - b2 ** (step_count + 1))
    if mom > 0:
        if "adam_m" not in state:
            state["adam_m"] = torch.zeros_like(g)
        m = state["adam_m"]
        m.mul_(mom).add_(g, alpha=1.0 - mom)
        d = m / (1.0 - mom ** (step_count + 1))
    else:
        d = g
    return (d / (v_hat.sqrt() + a_eps)).to(grad.dtype)


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
        # Mousse reference (github.com/Anti-Entrophic/Mousse, dion/dion/mousse.py:80-81)
        # ships shampoo_update_freq=10 and shampoo_alpha=0.125. With the bottleneck the
        # eigh is 128x128, so there is no reason to refresh less often than the reference.
        precond_update_freq: int = 10,
        precond_alpha: float = 0.125,
        precond_bottleneck: bool = True,
        precond_max_dim: int = 1024,
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
        # DoRA magnitude update geometry: "additive" (m -= eta*M, the historical behaviour) or
        # "multiplicative" (m *= exp(-eta*M): a step in log m -- scale-relative, and m can never
        # cross zero). See _magnitude_step.
        magnitude_update: str = "additive",
        # C-series options (KUANG 2026-10-17, docs/modular-opt-test-plan-2026-10-17.md). Every default
        # reproduces the historical step bit-for-bit (tests/test_modular_c_series.py).
        # normuon_mode: "row_unit" (historical) | "dion" (microsoft/dion NorMuon + spectral_norm lr).
        normuon_mode: str = "row_unit",
        # update_scale: "native" (radius / NorMuon decide the size) | "rms0.2" (SF-NorMuon Alg. 1 line 13:
        # every spectral matrix's update rescaled to RMS 0.2*lr per element, Adam-matched).
        update_scale: str = "native",
        # normuon_after_unwhiten: run NorMuon after the Shampoo pullback, so the per-neuron equalisation
        # is what reaches the weights (historical: before, in the whitened frame).
        normuon_after_unwhiten: bool = False,
        # ns_dtype: dtype of the Newton-Schulz iteration only: "fp32" | "bf16" | "fp16" (C7, WINTERMUTE).
        ns_dtype: str = "fp32",
        # group_type="adam" groups (routing scalar_opt="adam"): second-moment beta, eps, momentum
        # (None = none under Schedule-Free, beta1 otherwise).
        adam_beta2: float = 0.99,
        adam_eps: float = 1e-8,
        adam_momentum: float | None = None,
        # state_dtype (C1, KUANG 2026-10-17): "param" (historical: momentum and Schedule-Free z/x in the
        # parameter's dtype) | "fp32": optimizer state and the update arithmetic in fp32 for low-precision
        # parameters (fp32 master copy; the bf16 parameter is written from it each step, as FusionOpt
        # does). For bf16 adapters this keeps the steps that bf16 storage rounds away (TRAINING.md F21).
        state_dtype: str = "param",
        # ns_batch: run the Newton-Schulz of all same-shape spectral matrices of a group as one batched call
        # (speed; same per-matrix arithmetic up to floating-point reassociation). Default off.
        ns_batch: bool = False,
        # strict_group_lrs (CONTINUITY 2026-10-27, Kim): a spectral group and a 1-D group (sign / colnorm / adam)
        # on ONE learning rate get very different per-weight steps (the sign step moves each entry by the full lr,
        # a spectral entry by ~lr/sqrt(max(m, n)); guide section 4). By default this only WARNS; True raises, so a
        # uniform lr needs an explicit choice (give the groups their own "lr", or leave strict off).
        strict_group_lrs: bool = False,
        # Global
        eps: float = 1e-12,
    ):
        if state_dtype not in ("param", "fp32"):
            raise ValueError(f"state_dtype must be 'param' or 'fp32', got {state_dtype!r}")
        if normuon_mode not in ("row_unit", "dion"):
            raise ValueError(f"normuon_mode must be 'row_unit' or 'dion', got {normuon_mode!r}")
        if update_scale not in ("native", "rms0.2"):
            raise ValueError(f"update_scale must be 'native' or 'rms0.2', got {update_scale!r}")
        if ns_dtype not in _NS_DTYPES:
            raise ValueError(f"ns_dtype must be one of {sorted(_NS_DTYPES)}, got {ns_dtype!r}")
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta_precond=beta_precond,
            precond_delta=precond_delta,
            precond_update_freq=precond_update_freq,
            precond_alpha=precond_alpha,
            precond_bottleneck=precond_bottleneck,
            precond_max_dim=precond_max_dim,
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
            magnitude_update=magnitude_update,
            normuon_mode=normuon_mode,
            update_scale=update_scale,
            normuon_after_unwhiten=normuon_after_unwhiten,
            ns_dtype=ns_dtype,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
            adam_momentum=adam_momentum,
            state_dtype=state_dtype,
            ns_batch=ns_batch,
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
            if gtype not in ("spectral", "sign", "colnorm", "adam"):
                raise ValueError(
                    f"ModularOptimizer param groups must set group_type='spectral'|'sign'|'colnorm'|'adam', "
                    f"got '{gtype}'"
                )
        self._check_group_lrs(strict_group_lrs)


    def state_dict(self):
        """Adds _step_count, which drives LR warmup, the Schedule-Free c_warmup burn-in and the
        momentum bias corrections. Without it a resume restarts all three from step 0."""
        sd = super().state_dict()
        sd["modular_step_count"] = self._step_count
        return sd

    def load_state_dict(self, state_dict) -> None:
        """torch moves the tensors it finds directly in each param's state to the param's
        device, but not tensors held as attributes of objects stored there (the Shampoo/SOAP
        preconditioners). A checkpoint loads onto the CPU first, so without this a resumed
        run kept C/P/L/R on the CPU and died on the first step (tests/test_modular_resume_device.py)."""
        state_dict = dict(state_dict)
        step_count = state_dict.pop("modular_step_count", None)
        # state_dtype="fp32": torch's load_state_dict casts every floating state tensor to the PARAMETER's
        # dtype, which would turn the fp32 master state back into bf16 on resume. Keep the saved fp32
        # tensors and put them back after the base class has run.
        keep = {}
        saved_groups = state_dict.get("param_groups", [])
        if any(g.get("state_dtype") == "fp32" for g in saved_groups):
            idx2p = {}
            for sg, g in zip(saved_groups, self.param_groups):
                for i, p in zip(sg["params"], g["params"]):
                    idx2p[i] = p
            for i, st in state_dict.get("state", {}).items():
                p = idx2p.get(i)
                if p is None or p.dtype == torch.float32:
                    continue
                for k, v in st.items():
                    if torch.is_tensor(v) and v.dtype == torch.float32 and v.ndim > 0:
                        keep[(p, k)] = v
        super().load_state_dict(state_dict)
        for (p, k), v in keep.items():
            self.state[p][k] = v.to(p.device).clone()
        if step_count is not None:
            self._step_count = int(step_count)
        for p, st in self.state.items():
            for v in st.values():
                if torch.is_tensor(v) or not hasattr(v, "__dict__"):
                    continue
                for k, t in vars(v).items():
                    if torch.is_tensor(t) and t.device != p.device:
                        setattr(v, k, t.to(p.device))

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
            "groups": {},
        } if self._telem_on else None

        for gi, group in enumerate(self.param_groups):
            self._group_step(group, warm_factor, telem=telem, gkey=f"g{gi}_{group['group_type']}")

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
            # C4 (KUANG 2026-10-17): per-group step size vs weight size. The pooled numbers above
            # cannot tell a damped group from a hot one (docs/TRAINING.md F24).
            for k, (u_sq, w_sq, n) in telem["groups"].items():
                self._comp_telem[f"comp/{k}/update_rms"] = math.sqrt(u_sq / max(n, 1))
                self._comp_telem[f"comp/{k}/weight_rms"] = math.sqrt(w_sq / max(n, 1))
                self._comp_telem[f"comp/{k}/rel_update"] = math.sqrt(u_sq / max(w_sq, 1e-30))
            fails = 0
            for st in self.state.values():
                pc = st.get("preconditioner") if isinstance(st, dict) else None
                fails += int(getattr(pc, "eigh_failures", 0) or 0)
            self._comp_telem["comp/eigh_failures"] = float(fails)

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
                    if st["z"].dtype != p.dtype:
                        # state_dtype="fp32": form y in fp32, round once into the bf16 parameter
                        p.data.copy_(st["z"].mul(1.0 - sf_beta).add_(st["x"], alpha=sf_beta))
                    else:
                        p.data.copy_(st["z"]).mul_(1.0 - sf_beta).add_(st["x"], alpha=sf_beta)

        self._step_count += 1
        return loss

    # ------------------------------------------------------------------
    # Per-group 6-stage pipeline
    # ------------------------------------------------------------------

    def _momentum_whiten(self, state, grad, beta1, whitening):
        """Stages 1-2 (momentum, forward whitening). Pushes the pullback onto the state's LIFO stack."""
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
        return d_k

    def _batched_spectral_lmo(self, group, lmo) -> dict:
        """ns_batch=True (KUANG 2026-10-17): run Stages 1-2 for every spectral parameter of the group, then the
        Newton-Schulz of all same-shape matrices (and same-shape QKV/AdaLN blocks) as ONE batched call.
        Returns {param: (d_k, M or None)}; M is None where the per-matrix path must still run (non-2-D).
        The arithmetic per matrix is the per-matrix LMO's; only the call pattern changes.
        """
        from .lmo import newton_schulz_batched
        param_names = group.get("param_names", [])
        radii = group.get("radii", [])
        blocks = group.get("blocks", [])
        whitening = group.get("whitening", "none")
        dion = group.get("normuon_mode", "row_unit") == "dion"
        pre, items = {}, {}
        for i, p in enumerate(group["params"]):
            if p.grad is None:
                continue
            grad = p.grad
            if group.get("state_dtype", "param") == "fp32" and p.dtype != torch.float32:
                grad = grad.float()
            state = self.state[p]
            if len(state) == 0:
                self._init_state(state, p, group, whitening)
            d_k = self._momentum_whiten(state, grad, group["beta1"], whitening)
            pre[p] = [d_k, None]
            if d_k.ndim != 2:
                continue
            nb = blocks[i] if i < len(blocks) else 1
            nb = nb if nb > 1 else 1
            r = 1.0 if dion else (radii[i] if i < len(radii) else 1.0)
            bs = d_k.shape[0] // nb
            for b in range(nb):
                sub = d_k[b * bs:(b + 1) * bs] if nb > 1 else d_k
                items.setdefault((tuple(sub.shape), sub.dtype), []).append((p, b, nb, r, sub))
        out_blocks: dict = {}
        for _key, lst in items.items():
            ortho = newton_schulz_batched(torch.stack([it[4] for it in lst]), algorithm=lmo.algorithm,
                                          steps=lmo.steps, eps=lmo.eps, compute_dtype=lmo.compute_dtype)
            for j, (p, b, nb, r, _sub) in enumerate(lst):
                out_blocks.setdefault(p, [None] * nb)[b] = r * ortho[j]
        for p, bl in out_blocks.items():
            pre[p][1] = bl[0] if len(bl) == 1 else torch.cat(bl, dim=0)
        return pre

    def _spectral_sign_direction(self, state, grad, group, group_type, lmo, radius, n_blocks, beta1,
                                 whitening, normuon_enabled, normuon_beta, eps, pre=None):
        """Stages 1-4 for the spectral / sign / colnorm groups. Returns (d_k, M, M_lmo).

        With every C-series option at its default this is the historical inline code, unchanged.
        `pre` = (d_k, M) from _batched_spectral_lmo: Stages 1-2 (and Stage 3 when M is not None) are done.
        """
        normuon_mode = group.get("normuon_mode", "row_unit")
        update_scale = group.get("update_scale", "native")
        after_unwhiten = group.get("normuon_after_unwhiten", False)
        is_spectral_2d = group_type == "spectral" and grad.ndim == 2

        if pre is None:
            d_k = self._momentum_whiten(state, grad, beta1, whitening)
        else:
            d_k = pre[0]
        stack = state["stack"]

        # ── Stage 3: Core LMO Solver with Block Splitting ────────
        # dion mode takes dion's own lr rule (sqrt(d_out/d_in), applied below), so the radius is 1 here.
        lmo_radius = 1.0 if (is_spectral_2d and normuon_mode == "dion") else radius
        if pre is not None and pre[1] is not None:
            M = pre[1]
        elif n_blocks > 1 and d_k.ndim == 2:
            # Block-aware splitting (QKV or AdaLN)
            block_size = d_k.shape[0] // n_blocks
            lmo_blocks = []
            for b_idx in range(n_blocks):
                start = b_idx * block_size
                end = start + block_size
                block_grad = d_k[start:end]
                lmo_result = lmo.compute_lmo(block_grad, radius=lmo_radius)  # same radius per block
                lmo_blocks.append(lmo_result)
            M = torch.cat(lmo_blocks, dim=0)
        else:
            # Standard single-block LMO
            if group_type == "spectral" and d_k.ndim != 2:
                # Spectral LMO requires 2D — fall back to Sign for non-2D
                M = _SIGN_LMO.compute_lmo(d_k, radius=radius)
            else:
                M = lmo.compute_lmo(d_k, radius=lmo_radius)
        M_lmo = M

        do_normuon = normuon_enabled and group_type == "spectral" and M.ndim == 2

        # ── Stage 3b: NorMuon Per-Neuron Row Scaling ─────────────
        # (Li & Han 2026 arXiv:2608.20818, Section 3.2 / FusionOpt)
        # Normalizes each neuron row to unit variance under EMA.
        # Bounds per-neuron velocity in DiTs, preventing harsh timbres and glitches.
        if do_normuon and not (after_unwhiten and not stack.is_empty()):
            M = _normuon_rows(M, state, normuon_mode, normuon_beta, self._step_count, eps, n_blocks)

        # ── Stage 4: Reverse Unwhitening (Pullback) ──────────────
        if not stack.is_empty():
            # Mousse (arXiv:2603.09697) Algorithm 1 lines 8 and 10. The whitening is
            # applied on the way in AND again on the way out (line 9 reuses the same
            # negative powers, which is correct -- see preconditioners.py), so the
            # magnitude the LMO produced is destroyed by the pullback. Save it and
            # restore it, or the step size is whatever the curvature happens to be.
            # Its absence is the likeliest reason naive whitening blows up.
            gamma = M.norm()
            M = stack.unwhiten(M)
            M = M * (gamma / M.norm().clamp_min(eps))
            # C5: NorMuon after the pullback, so the per-neuron equalisation reaches the weights
            # (the left-side pullback re-mixes the rows of square/wide matrices).
            if do_normuon and after_unwhiten:
                M = _normuon_rows(M, state, normuon_mode, normuon_beta, self._step_count, eps, n_blocks)

        # ── Stage 4b: step-size convention for spectral matrices (C2/C3) ──
        if is_spectral_2d:
            if update_scale == "rms0.2":
                # SF-NorMuon Alg. 1 line 13: eta_hat = 0.2 * eta * sqrt(m*n) / ||P||_F, i.e. update RMS
                # 0.2*lr per element whatever the shape (Adam-matched; replaces radius and dion's rule).
                m_, n_ = M.shape
                M = M * (0.2 * math.sqrt(m_ * n_) / M.float().norm().clamp_min(1e-12)).to(M.dtype)
            elif normuon_mode == "dion":
                # dion adjust_lr="spectral_norm": lr * sqrt(d_out / d_in), per row block when split.
                d_out = M.shape[0] // n_blocks if n_blocks > 1 else M.shape[0]
                M = M * math.sqrt(d_out / M.shape[1])
        return d_k, M, M_lmo

    def _group_step(self, group: dict, warm_factor: float, telem: dict | None = None, gkey: str = "") -> None:
        """Run the full 6-stage pipeline for a single parameter group."""
        group_type: str = group["group_type"]
        param_names: list[str] = group.get("param_names", [])
        radii: list[float] = group.get("radii", [])
        # Per-parameter step multipliers (routing.py lora_a_lr_mult). Applied to the final step
        # only, AFTER NorMuon: NorMuon row-normalises M, so any scale applied before it cancels.
        lr_mults: list[float] = group.get("lr_mults", [])
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

        lmo = _get_lmo(group_type, ns_poly, group.get("ns_dtype", "fp32"))
        pre_all = (self._batched_spectral_lmo(group, lmo)
                   if group.get("ns_batch", False) and group_type == "spectral" else {})

        for i, p in enumerate(group["params"]):
            if p.grad is None:
                continue

            grad = p.grad
            # C1: state_dtype="fp32" runs every stage in fp32 for a low-precision parameter.
            master32 = group.get("state_dtype", "param") == "fp32" and p.dtype != torch.float32
            if master32:
                grad = grad.float()
            name = param_names[i] if i < len(param_names) else f"param_{i}"
            radius = radii[i] if i < len(radii) else 1.0
            lr_mult = lr_mults[i] if i < len(lr_mults) else 1.0
            n_blocks = blocks[i] if i < len(blocks) else 1

            state = self.state[p]

            # ── Lazy state initialization ────────────────────────────
            if len(state) == 0:
                self._init_state(state, p, group, whitening)

            if group_type == "adam":
                # C3: element-wise Adam direction; no whitening, no LMO, no NorMuon. Stages 5-6 below
                # (lr, warmup, wd at z, Schedule-Free averaging) are shared with the other groups.
                d_k = grad
                M = _adam_direction(state, grad, group, self._step_count, sf_enabled)
                M_lmo = M
                stack = state["stack"]
                stack.clear()
            else:
                d_k, M, M_lmo = self._spectral_sign_direction(
                    state, grad, group, group_type, lmo, radius, n_blocks, beta1, whitening,
                    normuon_enabled, normuon_beta, eps, pre=pre_all.get(p))

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
                _magnitude_step(z, M, eta * lr_mult, group, name)

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
                # Standard non-SF update (on the fp32 master copy when state_dtype="fp32")
                w = state["w32"] if master32 else p.data
                if wd_decay_factor > 0:
                    w.add_(w, alpha=-wd_decay_factor)

                r_brake = group.get("radial_brake", 1.0)
                p_old_norm = w.norm() if r_brake < 1.0 else None
                radial_scale = 1.0

                _magnitude_step(w, M, eta * lr_mult, group, name)

                if r_brake < 1.0:
                    p_new_norm = w.norm()
                    if p_new_norm > p_old_norm:
                        p_target = p_old_norm + r_brake * (p_new_norm - p_old_norm)
                        radial_scale = float((p_target / p_new_norm.clamp_min(1e-12)).item())
                        w.mul_(radial_scale)
                if master32:
                    p.data.copy_(w)

            if telem is not None:
                telem["n_params"] += 1
                telem["mom_sq"] += float((d_k * d_k).sum().item())
                telem["lmo_sq"] += float((M_lmo * M_lmo).sum().item())
                telem["normuon_sq"] += float((M * M).sum().item())
                telem["update_sq"] += float((M * (eta * lr_mult)).pow(2).sum().item())
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
                # C4: per-group step and weight size (weights = z under Schedule-Free).
                acc = telem["groups"].setdefault(gkey or group_type, [0.0, 0.0, 0])
                acc[0] += float((M.float() * (eta * lr_mult)).pow(2).sum().item())
                w_ref = z if sf_enabled else p.data
                acc[1] += float(w_ref.float().pow(2).sum().item())
                acc[2] += p.numel()

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
        # C1: state_dtype="fp32" keeps every state tensor of a low-precision parameter in fp32.
        master32 = group.get("state_dtype", "param") == "fp32" and p.dtype != torch.float32
        sd = torch.float32 if master32 else p.dtype
        # Momentum buffer (parameter's dtype, or fp32 under state_dtype="fp32")
        state["momentum"] = torch.zeros_like(p.data, dtype=sd)
        if master32 and not group.get("schedule_free", False):
            state["w32"] = p.data.float().clone()  # fp32 master weights (non-Schedule-Free path)

        # LIFO stack for whitening/unwhitening
        state["stack"] = LIFOTransformationStack()

        # Preconditioner (only for 2D tensors with whitening enabled)
        if whitening != "none" and p.ndim == 2:
            shape = (p.shape[0], p.shape[1])
            beta_p = group.get("beta_precond", 0.95)
            delta = group.get("precond_delta", 1e-4)
            # Default 10, matching the Mousse reference implementation, not 1:
            # at freq 1 this is an eigendecomposition per tensor per step.
            freq = group.get("precond_update_freq", 10)

            alpha = group.get("precond_alpha", 0.125)
            bottleneck = group.get("precond_bottleneck", True)
            max_dim = group.get("precond_max_dim", 1024)

            if whitening == "shampoo":
                state["preconditioner"] = ShampooPreconditioner(
                    shape=shape,
                    device=p.device,
                    dtype=sd,
                    beta=beta_p,
                    delta=delta,
                    update_freq=freq,
                    alpha=alpha,
                    bottleneck=bottleneck,
                    max_dim=max_dim,
                )
            elif whitening == "soap":
                state["preconditioner"] = RotatedSOAPPreconditioner(
                    shape=shape,
                    device=p.device,
                    dtype=sd,
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
            state["w0"] = p.data.to(sd).clone()  # Initial weights
            state["ev_numerator"] = 0.0  # Running numerator
            state["ev_denominator"] = 0.0  # Running denominator
            state["ev_d"] = 1.0  # Current escape velocity multiplier

        # SNR gate state
        if group.get("snr_gate", False):
            state["snr_ema"] = torch.zeros_like(p.data, dtype=sd)
            state["snr_sq_ema"] = torch.zeros_like(p.data, dtype=sd)

        # Schedule-Free state
        if group.get("schedule_free", False):
            state["z"] = p.data.to(sd).clone()  # Fast iterate
            state["x"] = p.data.to(sd).clone()  # Averaged iterate

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

    def set_step_count(self, n: int) -> None:
        """For checkpoints written before _step_count was saved: the trainer sets it from
        Lightning's restored global_step (both count optimizer steps)."""
        self._step_count = int(n)

    def _check_group_lrs(self, strict: bool) -> None:
        """Warn (or raise, when strict) if a spectral group and a 1-D group share one learning rate.

        Not an estimate of step sizes: it only detects the situation the guide's section 4 describes as a
        trap, a spectral matrix and a sign / colnorm / adam list on the same lr (per-weight steps differ by
        ~sqrt(max(m, n)) or more). Give the groups their own "lr" (routing / --scalar-lr style options) to
        silence it. Pure check: it changes nothing about the step.
        """
        types = {g["group_type"] for g in self.param_groups if g["params"]}
        lrs = {float(g["lr"]) for g in self.param_groups if g["params"]}
        if "spectral" not in types or not (types & {"sign", "colnorm", "adam"}) or len(lrs) != 1:
            return
        msg = (
            f"ModularOptimizer: the spectral group and the {'/'.join(sorted(types - {'spectral'}))} group(s) share "
            f"one learning rate ({next(iter(lrs)):.2e}). The same number gives very different per-weight steps "
            f"(a sign step moves every entry by the full lr, a spectral entry by about lr/sqrt(max(m, n))), so "
            f"biases and gains can run away first. Set a separate lr per group, or pass strict_group_lrs=False "
            f"and accept it explicitly."
        )
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

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
