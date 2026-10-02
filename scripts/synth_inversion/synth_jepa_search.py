"""Synth-JEPA renderer-free parameter search (Sec. 2.2 of Hayes, Tian, Lattner, arXiv:2609.31024).

1. Given target audio y*, compute z_a* = E_a(y*) once.
2. Score candidates x with D_JEPA(y*, x) = MSE(z_a*, f_p->a(E_p(x)))  (Eq. 4; no synth calls).
3. Two stages (paper): half the evaluation budget evolves a population of 32 with JADE
   (mutation/crossover on continuous params; categoricals copied from another candidate or
   resampled); then the 8 best are refined by Adam on their continuous params (lr 0.1, cosine
   decay to zero), pruning the worse half three times at evenly spaced intervals; the sole
   remaining candidate is returned and rendered once.

v2 (2026-10-02 review, checked against the paper PDF):
  * Search box = the TRAINING SUPPORT (surge_spec.CONT_BOUNDS), not all of [-1, 1]. The paper
    samples a uniform prior over each parameter's full range, so its search box and its
    training support coincide. Our dataset draws narrower ranges and spike-and-slab values, so
    a full-range search spends budget where E_p never saw data — exactly where a learned
    objective is least trustworthy and easiest to exploit.
  * Couplings applied to every candidate before scoring (detune=0 without unison, drive=0
    without a waveshaper, delay feedback=0 without delay), as in every training sample.
  * The MIDI note can be pinned (search(midi_note=...)). Pitch is in the parameter vector
    here; unpinned, the search can match timbre by moving the pitch, and the render (at the
    true note) is then not what was scored.
  * Budget counts every objective evaluation (initial population, pruning re-scores, final
    score), and stage 2's step count is set so its evaluations fill its half of the budget
    given the shrinking pool (v1 used ~60 % of it).
  * Stage-2 pruning keeps the cosine schedule and each survivor's Adam moments (v1 rebuilt
    the optimiser at each prune, which detached the scheduler: LR stayed constant afterwards).
  * The returned candidate is the arg-min of a final re-score, not index 0.
  * JADE adaptation follows Zhang & Sanderson: mu_F updated with the Lehmer mean, F redrawn
    while <= 0 and truncated at 1, r1 != r2 != i, at least one crossed gene. (No external
    archive — "optional" in JADE; the paper does not say it used one.)
  * WelfordNormalizer is CHANNEL-wise (one mean/var per mel band) as in Sec. 3.1, not per
    (band, frame) element, and refuses to normalise before its statistics are frozen.
"""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from surge_spec import (CAT_INDICES, CONT_BOUNDS, CONT_INDICES, NOTE_HIGH, NOTE_LOW, NUM_PARAMS, PARAM_NAMES,
                        canonicalize_vector, vector_to_patch)


class WelfordNormalizer:
    """Channel-wise (per mel band) running mean/variance, Welford/Chan merge, then frozen.

    Sec. 3.1: "we estimate channel-wise mean and variance over the first 8k training
    spectrograms using Welford's online algorithm and freeze the resulting statistics".
    """
    def __init__(self, n_mels: int = 128, eps: float = 1e-5):
        self.count = 0
        self.mean = np.zeros(n_mels, dtype=np.float64)
        self.M2 = np.zeros(n_mels, dtype=np.float64)
        self.eps = eps
        self.frozen = False
        self.frozen_mean = None
        self.frozen_std = None
        self.n_spectrograms = 0

    def update(self, batch_mels: np.ndarray):
        """batch_mels: [B, n_mels, T]. Merges all B*T frames per band (Chan et al. parallel update)."""
        if self.frozen:
            return
        x = np.asarray(batch_mels, dtype=np.float64)
        frames = x.transpose(1, 0, 2).reshape(x.shape[1], -1)  # [n_mels, B*T]
        n_b = frames.shape[1]
        mean_b = frames.mean(axis=1)
        m2_b = ((frames - mean_b[:, None]) ** 2).sum(axis=1)
        n = self.count + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * n_b / n
        self.M2 = self.M2 + m2_b + delta ** 2 * self.count * n_b / n
        self.count = n
        self.n_spectrograms += x.shape[0]

    def freeze(self):
        self.frozen = True
        variance = self.M2 / max(1, self.count - 1)
        self.frozen_mean = torch.from_numpy(self.mean.astype(np.float32))
        self.frozen_std = torch.from_numpy(np.sqrt(variance + self.eps).astype(np.float32))

    def normalize(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if not self.frozen:
            raise RuntimeError("WelfordNormalizer used before freeze(); estimate the statistics first")
        mean = self.frozen_mean.to(mel_tensor.device)[:, None]  # broadcast over time
        std = self.frozen_std.to(mel_tensor.device)[:, None]
        return (mel_tensor - mean) / std

    def state_dict(self) -> dict:
        return {"mean": self.frozen_mean, "std": self.frozen_std, "n_spectrograms": self.n_spectrograms}

    @classmethod
    def from_state_dict(cls, state: dict) -> "WelfordNormalizer":
        norm = cls(n_mels=int(state["mean"].shape[0]))
        norm.frozen, norm.frozen_mean, norm.frozen_std = True, state["mean"], state["std"]
        norm.n_spectrograms = int(state.get("n_spectrograms", 0))
        return norm


def load_synth_jepa(ckpt_path: str, device: str = "cpu"):
    """-> (SynthJEPA model in eval mode, frozen WelfordNormalizer) from a train_synth_jepa.py checkpoint."""
    from synth_jepa_model import SynthJEPA

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "normalizer" not in ckpt or "model_kwargs" not in ckpt:
        raise ValueError(f"{ckpt_path}: no normalizer/model_kwargs in checkpoint (written by train_synth_jepa.py v1?)")
    model = SynthJEPA(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), WelfordNormalizer.from_state_dict(ckpt["normalizer"])


def _to_pm1(v01):
    return 2.0 * v01 - 1.0


class SynthJEPASearcher:
    """Renderer-free hybrid JADE + Adam searcher over the training support."""

    def __init__(self, model, normalizer: WelfordNormalizer, device="cuda:0", seed: int = 0):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.normalizer = normalizer
        self.num_cont = len(CONT_INDICES)
        self.cat_sizes = [k for _, k in CAT_INDICES]
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        cont_names = [PARAM_NAMES[i] for i in CONT_INDICES]
        self.lo = torch.tensor([_to_pm1(CONT_BOUNDS[n][0]) for n in cont_names], device=self.device)
        self.hi = torch.tensor([_to_pm1(CONT_BOUNDS[n][1]) for n in cont_names], device=self.device)
        pos = {n: j for j, n in enumerate(cont_names)}
        self._detune, self._drive = pos["unison_detune"], pos["drive"]
        self._delay_mix, self._delay_fb = pos["delay_mix"], pos["delay_fb"]
        self._note = pos["midi_note"]
        self._cat_pos = {PARAM_NAMES[i]: j for j, (i, _k) in enumerate(CAT_INDICES)}
        self._note_value = None
        self.n_evals = 0

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self.gen)

    def _randint(self, high, shape):
        return torch.randint(0, high, shape, device=self.device, generator=self.gen)

    def _onehots(self, cats: List[torch.Tensor]):
        return [F.one_hot(c, k).float() for c, k in zip(cats, self.cat_sizes)]

    def _constrain(self, cont: torch.Tensor, cats: List[torch.Tensor]) -> torch.Tensor:
        """Clamp to the training box, apply the training couplings, pin the note. Out-of-place."""
        cont = torch.maximum(torch.minimum(cont, self.hi), self.lo).clone()
        zero = torch.full_like(cont[:, 0], -1.0)  # vector value 0 is -1 in [-1, 1] space
        cont[:, self._detune] = torch.where(cats[self._cat_pos["unison"]] == 1, cont[:, self._detune], zero)
        cont[:, self._drive] = torch.where(cats[self._cat_pos["waveshaper_type"]] > 0, cont[:, self._drive], zero)
        cont[:, self._delay_fb] = torch.where(cont[:, self._delay_mix] > -1.0, cont[:, self._delay_fb], zero)
        if self._note_value is not None:
            cont[:, self._note] = self._note_value
        return cont

    def score_candidates(self, za_target: torch.Tensor, cont: torch.Tensor, cats: List[torch.Tensor]) -> torch.Tensor:
        """D_JEPA(y*, x) = MSE(z_a*, f_p->a(E_p(x))) per candidate (Eq. 4). Counts evaluations."""
        self.n_evals += cont.shape[0]
        z_hat_a = self.model.predict_audio_latent(self.model.encode_params(cont, self._onehots(cats)))
        return F.mse_loss(z_hat_a, za_target.expand_as(z_hat_a), reduction="none").mean(dim=-1)

    @torch.no_grad()
    def encode_target(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        return self.model.encode_audio(self.normalizer.normalize(mel.to(self.device)))  # [1, 512]

    # ------------------------------------------------------------------ search
    def search(self, target_mel: torch.Tensor, midi_note: Optional[int] = None,
               total_eval_budget: int = 2048, pop_size: int = 32, n_refine: int = 8) -> dict:
        """Returns {'patch', 'vector' (23-d), 'd_jepa' (final objective), 'n_evals'}."""
        self.n_evals = 0
        self._note_value = (None if midi_note is None else
                            float(_to_pm1((midi_note - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW))))
        za_target = self.encode_target(target_mel)
        stage1_budget = total_eval_budget // 2

        # ---- Stage 1: JADE (current-to-pbest/1/bin), uniform init over the training box
        cats_pop = [self._randint(k, (pop_size,)) for k in self.cat_sizes]
        cont_pop = self._constrain(self.lo + (self.hi - self.lo) * self._rand(pop_size, self.num_cont), cats_pop)
        with torch.no_grad():
            scores = self.score_candidates(za_target, cont_pop, cats_pop)

        mu_cr, mu_f, c = 0.5, 0.5, 0.1
        p_best_count = max(2, int(round(0.05 * pop_size)))
        idx = torch.arange(pop_size, device=self.device)
        while self.n_evals + pop_size <= stage1_budget:
            cr = (mu_cr + 0.1 * torch.randn(pop_size, device=self.device, generator=self.gen)).clamp(0.0, 1.0)
            f_val = torch.zeros(pop_size, device=self.device)
            todo = torch.ones(pop_size, dtype=torch.bool, device=self.device)
            while todo.any():  # F ~ Cauchy(mu_f, 0.1): redraw while <= 0, truncate at 1
                draw = mu_f + 0.1 * torch.tan(math.pi * (self._rand(pop_size) - 0.5))
                f_val = torch.where(todo, draw, f_val)
                todo = f_val <= 0
            f_val = f_val.clamp(max=1.0)

            pbest = torch.topk(scores, p_best_count, largest=False).indices[self._randint(p_best_count, (pop_size,))]
            r1 = (idx + 1 + self._randint(pop_size - 1, (pop_size,))) % pop_size  # r1 != i
            r2 = (idx + 1 + self._randint(pop_size - 1, (pop_size,))) % pop_size  # r2 != i
            r2 = torch.where(r2 == r1, (r2 + 1) % pop_size, r2)                     # r2 != r1
            r2 = torch.where(r2 == idx, (r2 + 1) % pop_size, r2)                    # (re-check i)
            fv = f_val.unsqueeze(-1)
            v = cont_pop + fv * (cont_pop[pbest] - cont_pop) + fv * (cont_pop[r1] - cont_pop[r2])
            mask = self._rand(pop_size, self.num_cont) < cr.unsqueeze(-1)
            mask[idx, self._randint(self.num_cont, (pop_size,))] = True  # binomial crossover: >= 1 gene
            u_cont = torch.where(mask, v, cont_pop)

            u_cats = []  # categoricals: copied from another candidate or resampled (Sec. 2.2)
            for j, k in enumerate(self.cat_sizes):
                copied = cats_pop[j][self._randint(pop_size, (pop_size,))]
                u_cats.append(torch.where(self._rand(pop_size) < 0.15, self._randint(k, (pop_size,)), copied))
            u_cont = self._constrain(u_cont, u_cats)

            with torch.no_grad():
                u_scores = self.score_candidates(za_target, u_cont, u_cats)
            better = u_scores < scores
            cont_pop[better] = u_cont[better]
            for j in range(len(self.cat_sizes)):
                cats_pop[j][better] = u_cats[j][better]
            scores[better] = u_scores[better]
            if better.any():
                s_cr, s_f = cr[better], f_val[better]
                mu_cr = (1 - c) * mu_cr + c * float(s_cr.mean())
                mu_f = (1 - c) * mu_f + c * float((s_f ** 2).sum() / s_f.sum())  # Lehmer mean

        # ---- Stage 2: Adam on the best n_refine, pruning the worse half 3x at evenly spaced intervals
        top = torch.topk(scores, min(n_refine, pop_size), largest=False).indices
        active = cont_pop[top].clone().requires_grad_(True)
        active_cats = [c_[top].clone() for c_ in cats_pop]
        n0 = active.shape[0]
        pools = [n0, max(1, n0 // 2), max(1, n0 // 4)]          # pool size in each third of the steps
        overhead = sum(pools) + 1                                # pruning re-scores + final score
        remaining = total_eval_budget - self.n_evals - overhead
        total_steps = max(3, (3 * remaining) // sum(pools))
        prune_at = {total_steps // 3 - 1, (2 * total_steps) // 3 - 1, total_steps - 1}

        opt = torch.optim.Adam([active], lr=0.1)
        for step in range(total_steps):
            for g in opt.param_groups:  # cosine decay from 0.1 to zero
                g["lr"] = 0.1 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
            opt.zero_grad()
            self.score_candidates(za_target, self._constrain(active, active_cats), active_cats).sum().backward()
            opt.step()
            with torch.no_grad():
                active.copy_(self._constrain(active, active_cats))

            if step in prune_at and active.shape[0] > 1:
                with torch.no_grad():
                    cur = self.score_candidates(za_target, active, active_cats)
                keep = torch.topk(cur, max(1, active.shape[0] // 2), largest=False).indices
                state = opt.state[active]
                new_active = active.detach()[keep].clone().requires_grad_(True)
                new_opt = torch.optim.Adam([new_active], lr=opt.param_groups[0]["lr"])
                if state:  # carry each survivor's Adam moments and the step count
                    step_count = state["step"]
                    new_opt.state[new_active] = {
                        "step": step_count.clone() if torch.is_tensor(step_count) else step_count,
                        "exp_avg": state["exp_avg"][keep].clone(),
                        "exp_avg_sq": state["exp_avg_sq"][keep].clone(),
                    }
                active, opt = new_active, new_opt
                active_cats = [c_[keep] for c_ in active_cats]

        with torch.no_grad():
            final_scores = self.score_candidates(za_target, active, active_cats)
        b = int(torch.argmin(final_scores))
        vec = np.zeros(NUM_PARAMS, dtype=np.float32)
        vec[CONT_INDICES] = (active[b].detach().cpu().numpy() + 1.0) / 2.0
        for (i, k), c_ in zip(CAT_INDICES, active_cats):
            vec[i] = float(c_[b]) / (k - 1)
        vec = canonicalize_vector(np.clip(vec, 0.0, 1.0))
        return {"patch": vector_to_patch(vec), "vector": vec, "d_jepa": float(final_scores[b]),
                "n_evals": int(self.n_evals)}
