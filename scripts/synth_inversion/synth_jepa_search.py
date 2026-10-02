"""Synth-JEPA Renderer-Free Parameter Search: JADE Evolution + Differentiable Adam Refinement.

Section 2.2 of Hayes et al. (arXiv:2609.31024):
1. Given target audio y*, compute z_a* = E_a(y*) once.
2. Candidate parameters are scored via:
   D_JEPA(y*, x) = MSE(z_a*, f_p->a(E_p(x)))
   (pure PyTorch forward pass, zero calls to Surge XT).
3. Two-Stage Hybrid Optimization:
   Stage 1: Evolve population of 32 candidates with JADE (Adaptive Differential Evolution).
   Stage 2: 8 best candidates are refined by gradient descent on continuous parameters
            using Adam (lr=0.1, cosine decay to zero), pruned 3 times (8 -> 4 -> 2 -> 1).
4. The sole remaining candidate is returned and rendered through Surge XT once.
"""
import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from surge_spec import CAT_INDICES, CONT_INDICES, NUM_PARAMS, LP_FILTERS, WAVESHAPER_TYPES, render_patch


class WelfordNormalizer:
    """Online running mean and variance estimator (Welford 1962)."""
    def __init__(self, shape=(128, 81), eps=1e-5):
        self.count = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)
        self.eps = eps
        self.frozen = False
        self.frozen_mean = None
        self.frozen_std = None

    def update(self, batch_mels: np.ndarray):
        if self.frozen:
            return
        for mel in batch_mels:
            self.count += 1
            delta = mel - self.mean
            self.mean += delta / self.count
            delta2 = mel - self.mean
            self.M2 += delta * delta2

    def freeze(self):
        self.frozen = True
        variance = self.M2 / max(1, self.count - 1)
        self.frozen_mean = torch.from_numpy(self.mean.astype(np.float32))
        self.frozen_std = torch.from_numpy(np.sqrt(variance + self.eps).astype(np.float32))

    def normalize(self, mel_tensor: torch.Tensor) -> torch.Tensor:
        if not self.frozen:
            return mel_tensor
        device = mel_tensor.device
        mean = self.frozen_mean.to(device)
        std = self.frozen_std.to(device)
        return (mel_tensor - mean) / std


class SynthJEPASearcher:
    """Renderer-free hybrid JADE + Adam searcher."""
    def __init__(self, model, normalizer: WelfordNormalizer, device="cuda:0"):
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.num_cont = len(CONT_INDICES)
        self.cat_sizes = [k for _, k in CAT_INDICES]

    @torch.no_grad()
    def encode_target(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [1, 128, 81] or [128, 81]
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        norm_mel = self.normalizer.normalize(mel.to(self.device))
        return self.model.encode_audio(norm_mel)  # [1, 512]

    def score_candidates(
        self,
        za_target: torch.Tensor,
        cont_tensor: torch.Tensor,
        cat_onehots: List[torch.Tensor],
    ) -> torch.Tensor:
        """Computes D_JEPA(y*, x) = MSE(z_a*, f_p->a(E_p(x))) for B candidates."""
        # cont_tensor: [B, 20] in [-1, 1]
        zp = self.model.encode_params(cont_tensor, cat_onehots)  # [B, 512]
        z_hat_a = self.model.predict_audio_latent(zp)            # [B, 512]
        # MSE per candidate
        return F.mse_loss(z_hat_a, za_target.expand_as(z_hat_a), reduction="none").mean(dim=-1)  # [B]

    def search(
        self,
        target_mel: torch.Tensor,
        total_eval_budget: int = 2048,
        pop_size: int = 32,
    ) -> dict:
        """Executes the two-stage search: JADE evolution then Adam gradient refinement."""
        za_target = self.encode_target(target_mel)

        # -------------------------------------------------------------
        # Stage 1: JADE Evolutionary Search (50% of budget)
        # -------------------------------------------------------------
        stage1_budget = total_eval_budget // 2
        num_generations = max(1, stage1_budget // pop_size)

        # Initialize population uniformly in [-1, 1] for continuous params
        cont_pop = torch.empty(pop_size, self.num_cont, device=self.device).uniform_(-1.0, 1.0)
        cat_pop_indices = [
            torch.randint(0, k, (pop_size,), device=self.device) for k in self.cat_sizes
        ]

        def get_cat_onehots(indices_list):
            return [F.one_hot(idx, k).float() for idx, k in zip(indices_list, self.cat_sizes)]

        with torch.no_grad():
            scores = self.score_candidates(za_target, cont_pop, get_cat_onehots(cat_pop_indices))

        # JADE parameter adaptation memories
        mu_cr = 0.5
        mu_f = 0.5
        c_rate = 0.1

        for gen in range(num_generations):
            # Sample CR_i ~ Normal(mu_cr, 0.1), F_i ~ Cauchy(mu_f, 0.1)
            cr = torch.normal(mu_cr, 0.1, size=(pop_size,), device=self.device).clamp(0.0, 1.0)
            f_val = mu_f + 0.1 * torch.tan(torch.pi * (torch.rand(pop_size, device=self.device) - 0.5))
            f_val = f_val.clamp(0.1, 1.0)

            # JADE current-to-pbest mutation
            p_best_count = max(2, int(0.05 * pop_size))
            best_indices = torch.topk(scores, p_best_count, largest=False).indices

            # Mutate continuous controls
            r1 = torch.randint(0, pop_size, (pop_size,), device=self.device)
            r2 = torch.randint(0, pop_size, (pop_size,), device=self.device)
            pbest = best_indices[torch.randint(0, p_best_count, (pop_size,), device=self.device)]

            v_cont = cont_pop + f_val.unsqueeze(-1) * (cont_pop[pbest] - cont_pop) + f_val.unsqueeze(-1) * (cont_pop[r1] - cont_pop[r2])
            v_cont = v_cont.clamp(-1.0, 1.0)

            # Crossover
            mask = torch.rand_like(cont_pop) < cr.unsqueeze(-1)
            u_cont = torch.where(mask, v_cont, cont_pop)

            # Categorical variation: copy from random or resample
            u_cat_indices = []
            for j, k in enumerate(self.cat_sizes):
                cat_copy = cat_pop_indices[j][torch.randint(0, pop_size, (pop_size,), device=self.device)]
                resample_mask = torch.rand(pop_size, device=self.device) < 0.15
                new_cat = torch.where(resample_mask, torch.randint(0, k, (pop_size,), device=self.device), cat_copy)
                u_cat_indices.append(new_cat)

            with torch.no_grad():
                u_scores = self.score_candidates(za_target, u_cont, get_cat_onehots(u_cat_indices))

            # Selection
            better = u_scores < scores
            cont_pop[better] = u_cont[better]
            for j in range(len(self.cat_sizes)):
                cat_pop_indices[j][better] = u_cat_indices[j][better]
            scores[better] = u_scores[better]

            if better.any():
                mu_cr = (1 - c_rate) * mu_cr + c_rate * float(cr[better].mean().item())
                mu_f = (1 - c_rate) * mu_f + c_rate * float(f_val[better].mean().item())

        # -------------------------------------------------------------
        # Stage 2: Differentiable Adam Refinement (50% of budget)
        # -------------------------------------------------------------
        # Pick top 8 candidates
        top8_indices = torch.topk(scores, 8, largest=False).indices
        active_cont = cont_pop[top8_indices].clone().detach().requires_grad_(True)
        active_cats = [cat_pop_indices[j][top8_indices].clone().detach() for j in range(len(self.cat_sizes))]

        stage2_budget = total_eval_budget - stage1_budget
        n_candidates = 8
        total_steps = stage2_budget // n_candidates
        prune_intervals = [total_steps // 3, (2 * total_steps) // 3, total_steps - 1]

        optimizer = torch.optim.Adam([active_cont], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

        for step in range(total_steps):
            optimizer.zero_grad()
            cat_onehots = get_cat_onehots(active_cats)
            loss_cands = self.score_candidates(za_target, active_cont, cat_onehots)
            loss = loss_cands.sum()
            loss.backward()

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                active_cont.clamp_(-1.0, 1.0)

            # Prune worst half at 3 evenly spaced intervals
            if step in prune_intervals and active_cont.shape[0] > 1:
                with torch.no_grad():
                    cur_scores = self.score_candidates(za_target, active_cont, get_cat_onehots(active_cats))
                    keep_n = max(1, active_cont.shape[0] // 2)
                    best_sub = torch.topk(cur_scores, keep_n, largest=False).indices

                    active_cont = active_cont[best_sub].clone().detach().requires_grad_(True)
                    active_cats = [active_cats[j][best_sub].clone().detach() for j in range(len(self.cat_sizes))]
                    optimizer = torch.optim.Adam([active_cont], lr=optimizer.param_groups[0]["lr"])

        # Best candidate
        final_cont = active_cont[0].detach().cpu().numpy()  # [-1, 1] mapped
        # Unscale from [-1, 1] to [0, 1]
        final_cont_01 = (final_cont + 1.0) / 2.0
        final_cats = [int(active_cats[j][0].item()) for j in range(len(self.cat_sizes))]

        # Assemble full 23-d patch dictionary
        patch = self.assemble_patch(final_cont_01, final_cats)
        return patch

    def assemble_patch(self, cont_01: np.ndarray, cat_classes: List[int]) -> dict:
        """Transforms continuous [0, 1] and categorical class indices into a Surge patch dict."""
        from surge_spec import NOTE_LOW, NOTE_HIGH, LP_FILTERS, WAVESHAPER_TYPES

        p = np.zeros(NUM_PARAMS, dtype=np.float32)
        p[CONT_INDICES] = cont_01

        filter_idx = cat_classes[0]
        unison = bool(cat_classes[1] == 1)
        ws_idx = cat_classes[2]

        return dict(
            midi_note=int(round(NOTE_LOW + p[0] * (NOTE_HIGH - NOTE_LOW))),
            filter_idx=filter_idx,
            shape=float(p[2]),
            width=float(p[3]),
            sub_mix=float(p[4]),
            sync=float(p[5]),
            fm_depth=float(p[6]),
            unison=unison,
            unison_detune=float(p[8] * 0.35) if unison else 0.0,
            cutoff=float(p[9]),
            resonance=float(p[10]),
            keytrack_raw=float(0.5 + p[11] * 0.5),
            feg_amount=float(p[12]),
            feg_decay=float(p[13]),
            feg_sustain=float(p[14]),
            aeg_decay=float(p[15]),
            aeg_sustain=float(p[16]),
            aeg_release=float(p[17]),
            ws_idx=ws_idx,
            drive_raw=float(0.50 + p[19] * 0.32),
            chorus_mix=float(p[20]),
            delay_mix=float(p[21]),
            delay_fb=float(p[22]),
        )
