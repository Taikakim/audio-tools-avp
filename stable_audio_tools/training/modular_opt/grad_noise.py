"""Half-batch gradient-noise meter (C6; docs/modular-opt-test-plan-2026-10-17.md section N).

Owner: KUANG (created 2026-10-17, Kim's idea: measure whether a batch carries more signal than noise
instead of borrowing an LLM batch size).

The trainer hands over two gradients computed from two independent halves of the same step's batch
(each the MEAN gradient over its half). Per 2-D matrix and pooled over all of them this logs:

- cos(g1, g2): noise differs between the halves, the signal does not, so this is the share of signal
  seen at HALF the batch;
- noise_ratio = |g1 - g2|^2 / |g1 + g2|^2;
- the noise scale B_simple = tr(Sigma) / |G|^2 (McCandlish et al. 2018, "An Empirical Model of
  Large-Batch Training", Appendix A.1), from the two batch sizes b (a half) and 2b (both halves):
      |G|^2_est = 2 |G_2b|^2 - |G_b|^2,        S_est = 2 b (|G_b|^2 - |G_2b|^2)
  in units of SAMPLES. Both estimates are noisy, so B_simple_ema uses EMAs of S and |G|^2 (as the
  paper recommends), not an EMA of their ratio. Below B_simple a bigger batch gives nearly
  proportional progress per step; well above it, compute is wasted;
- cos_fullbatch_est = |G|^2 / (|G|^2 + S / (2b)): the cosine two independent FULL batches would
  have, i.e. the half-batch reading converted to the batch actually trained with;
- cos_null: the pooled cosine between half 1 of the PREVIOUS measurement and half 2 of this one,
  on a fixed random subset of coordinates. Two unrelated steps share only the slowly changing part
  of the gradient, so "near zero" is read against this, not against a guessed threshold.

Nothing here changes training: the meter only reads gradients. At batch 1 per half (the FiLM recipe's
two micro-batches) the halves differ by crop as well as by noise; that IS per-sample gradient noise,
which is what B_simple is defined from, but b = 1 makes each reading very noisy (read the EMA).
"""
from __future__ import annotations

import json
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor


class GradNoiseMeter:
    def __init__(self, named_params: Iterable[tuple[str, Tensor]], every: int = 50, out_path: str | None = None,
                 null_coords: int = 4096, ema: float = 0.9, seed: int = 0):
        self.names, self.params = [], []
        for n, p in named_params:
            if p.requires_grad and p.ndim >= 2:
                self.names.append(n)
                self.params.append(p)
        self.every = int(every)
        self.out_path = out_path
        self.ema = float(ema)
        self._s_ema = None
        self._g2_ema = None
        self._prev_sub = None
        gen = torch.Generator().manual_seed(seed)
        self._sub_idx = []
        for p in self.params:
            n = p.numel()
            k = min(n, max(1, null_coords // max(1, len(self.params))))
            self._sub_idx.append(torch.randperm(n, generator=gen)[:k])

    def due(self, step: int) -> bool:
        return self.every > 0 and step % self.every == 0 and len(self.params) > 0

    def capture(self) -> list[Tensor | None]:
        """Copy of the tracked params' current .grad (for trainers that build the halves from
        accumulated micro-batches: capture after the first half, subtract from the total later)."""
        return [None if p.grad is None else p.grad.detach().float().clone() for p in self.params]

    @torch.no_grad()
    def measure(self, step: int, g1: Sequence[Tensor | None], g2: Sequence[Tensor | None], b_half: float,
                extra: dict | None = None) -> dict:
        """g1, g2: per tracked param, the MEAN gradient over each half (None = no gradient, skipped).
        b_half: samples per half. Returns the pooled summary (also written to out_path as JSONL)."""
        per = []
        s11 = s22 = s12 = sf = 0.0
        sub1, sub2 = [], []
        for name, a, b, idx in zip(self.names, g1, g2, self._sub_idx):
            if a is None or b is None:
                continue
            a = a.float()
            b = b.float()
            n11 = float((a * a).sum())
            n22 = float((b * b).sum())
            n12 = float((a * b).sum())
            full = 0.25 * (n11 + n22 + 2 * n12)            # |(a + b) / 2|^2
            s11 += n11
            s22 += n22
            s12 += n12
            sf += full
            gb = 0.5 * (n11 + n22)
            g2_est = 2 * full - gb
            s_est = 2 * b_half * (gb - full)
            per.append({"name": name, "cos": n12 / math.sqrt(max(n11 * n22, 1e-30)),
                        "b_simple_raw": (s_est / g2_est) if g2_est > 0 else float("nan")})
            idx = idx.to(a.device)
            sub1.append(a.flatten()[idx])
            sub2.append(b.flatten()[idx])
        if not per:
            return {}
        gb = 0.5 * (s11 + s22)
        g2_est = 2 * sf - gb
        s_est = 2 * b_half * (gb - sf)
        d = self.ema
        self._s_ema = s_est if self._s_ema is None else d * self._s_ema + (1 - d) * s_est
        self._g2_ema = g2_est if self._g2_ema is None else d * self._g2_ema + (1 - d) * g2_est
        cos_null = float("nan")
        cur1, cur2 = torch.cat(sub1), torch.cat(sub2)
        if self._prev_sub is not None and self._prev_sub.numel() == cur2.numel():
            cos_null = float(torch.nn.functional.cosine_similarity(self._prev_sub, cur2, dim=0))
        self._prev_sub = cur1
        cosines = sorted(r["cos"] for r in per)
        out = {
            "grad_noise/cos_pooled": s12 / math.sqrt(max(s11 * s22, 1e-30)),
            "grad_noise/cos_median": cosines[len(cosines) // 2],
            "grad_noise/cos_null": cos_null,
            "grad_noise/noise_ratio": (s11 + s22 - 2 * s12) / max(s11 + s22 + 2 * s12, 1e-30),
            "grad_noise/b_simple_raw": (s_est / g2_est) if g2_est > 0 else float("nan"),
            "grad_noise/b_simple_ema": (self._s_ema / self._g2_ema) if self._g2_ema > 0 else float("nan"),
            "grad_noise/cos_fullbatch_est": (g2_est / (g2_est + s_est / (2 * b_half)))
            if g2_est > 0 and (g2_est + s_est / (2 * b_half)) > 0 else float("nan"),
            "grad_noise/b_half": float(b_half),
        }
        if self.out_path:
            rec = {"step": int(step), **{k.split("/", 1)[1]: v for k, v in out.items()}, "per_matrix": per}
            if extra:
                rec.update(extra)
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return out


# ---------------------------------------------------------------------------------------------------
# Direction-resolved agreement (test plan section P; KUANG 2026-10-17)
# ---------------------------------------------------------------------------------------------------
DEFAULT_BANDS = ((0, 1), (1, 4), (4, 16), (16, 32))


def momentum_basis(M: Tensor, k: int = 32) -> dict:
    """Singular basis of a (pre-polar) momentum matrix M = U S V^T, top k kept.

    Use a momentum built from batches that are NOT the ones being compared (a third sample group, or the
    saved optimizer momentum), so the directions are not selected on the gradients they are tested with.
    """
    U, S, Vh = torch.linalg.svd(M.float(), full_matrices=False)
    k = min(k, S.numel())
    return {"U": U[:, :k], "S": S, "V": Vh[:k].T, "k": k}


def _band_block(G: Tensor, U: Tensor, V: Tensor) -> Tensor:
    return U.T @ G @ V


@torch.no_grad()
def direction_agreement(gA: Tensor, gB: Tensor, basis: dict, bands=DEFAULT_BANDS, gap_tol: float = 0.05) -> dict:
    """Compare two independent gradients (each a mean over its own samples) on a FIXED momentum basis.

    Per band of ranks [lo, hi) the band block C = U_b^T G V_b is compared as a whole (cosine of the two
    blocks). That cosine is invariant to rotations inside the band, so near-degenerate singular vectors
    inside a band need no identification; a band EDGE between near-equal singular values is flagged
    (edge_ratio = S[lo-1] / S[lo] < 1 + gap_tol) because the split there is unstable.
    Also per band: share of each gradient's energy, and `agree_momentum` = fraction of the band's ranks
    whose diagonal coefficient u_i^T G v_i is positive for BOTH gradients (the momentum's descent
    direction -u_i v_i^T then lowers both losses; SAMuon appendix I's sign alignment).
    `residual` = everything outside the top-k subspace pair, G - U_k U_k^T G V_k V_k^T, labelled as such:
    it is the bulk beyond rank k, not measured direction by direction.
    """
    gA = gA.float()
    gB = gB.float()
    U, V, S, k = basis["U"].to(gA.device), basis["V"].to(gA.device), basis["S"].to(gA.device), basis["k"]
    eA, eB = float((gA * gA).sum()), float((gB * gB).sum())
    out = {}
    for lo, hi in bands:
        if lo >= k:
            continue
        hi = min(hi, k)
        cA = _band_block(gA, U[:, lo:hi], V[:, lo:hi])
        cB = _band_block(gB, U[:, lo:hi], V[:, lo:hi])
        nA, nB = float(cA.norm()), float(cB.norm())
        dA, dB = torch.diagonal(cA), torch.diagonal(cB)
        edge = float(S[lo - 1] / S[lo].clamp_min(1e-30)) if lo > 0 else float("inf")
        out[f"r{lo + 1}-{hi}"] = {
            "cos": float((cA * cB).sum()) / max(nA * nB, 1e-30),
            "energy_A": nA * nA / max(eA, 1e-30),
            "energy_B": nB * nB / max(eB, 1e-30),
            "agree_momentum": float(((dA > 0) & (dB > 0)).float().mean()),
            "sign_agree": float((torch.sign(dA) == torch.sign(dB)).float().mean()),
            "edge_ratio": edge,
            "edge_unstable": bool(edge < 1.0 + gap_tol),
        }
    Uk, Vk = U[:, :k], V[:, :k]
    rA = gA - Uk @ (Uk.T @ gA @ Vk) @ Vk.T
    rB = gB - Uk @ (Uk.T @ gB @ Vk) @ Vk.T
    nA, nB = float(rA.norm()), float(rB.norm())
    out["residual"] = {"cos": float((rA * rB).sum()) / max(nA * nB, 1e-30),
                       "energy_A": nA * nA / max(eA, 1e-30), "energy_B": nB * nB / max(eB, 1e-30),
                       "beyond_rank": k}
    return out


@torch.no_grad()
def momentum_spectra(opt, named_params: Iterable[tuple[str, Tensor]], k: int = 32) -> list[dict]:
    """Singular values of the optimizer's pre-polar momentum buffers (reading only; nothing is mutated).

    Returns per matrix: top-k singular values normalised by the largest, stable rank ||M||_F^2 / s_1^2, and
    the top-1 energy share. This is the spectrum BEFORE Newton-Schulz; the update's own spectrum is flat
    by construction (up to NorMuon's row scaling) and says nothing about the gradient's information.
    """
    rows = []
    for name, p in named_params:
        st = opt.state.get(p) if hasattr(opt, "state") else None
        m = None if not st else st.get("momentum", st.get("m"))
        if m is None or m.ndim != 2:
            continue
        s = torch.linalg.svdvals(m.float())
        if s.numel() == 0 or float(s[0]) == 0.0:
            continue
        e = float((s * s).sum())
        rows.append({"name": name, "shape": list(m.shape), "s_top": (s[:k] / s[0]).tolist(),
                     "stable_rank": e / float(s[0] ** 2), "top1_energy": float(s[0] ** 2) / e})
    return rows
