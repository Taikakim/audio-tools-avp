# Owner: KUANG (created 2026-10-15)
"""Flow-seeded, batched Synth-JEPA search: a replacement for the paper's JADE stage (V5_PROPOSAL_REVIEW.md, sec. 6).

The paper (arXiv 2609.31024, sec. 2.2) spends half of a 2 048-evaluation budget per target on JADE (population 32,
about 32 generations) to find the right region, categoricals included, then refines the 8 best with Adam (lr 0.1,
cosine decay to 0), pruning the worse half three times. synth_jepa_search_v3.SynthJEPASearcher implements that, one
target at a time.

Here the global stage is the flow inverter: K flow samples per target (MIDI pinned, flow_sampling_v5) already carry
the categoricals and the region, so no evolutionary stage is needed. Then the paper's Adam stage runs on all targets
AT ONCE: candidates of every target are one batch, Adam is written out by hand so pruning is an index selection, the
note is pinned per row, and the training couplings (detune needs unison, drive needs a waveshaper, delay feedback needs
delay) and the training box are applied after every step, as in synth_jepa_search_v3. Optionally a share of the seed
pool is drawn uniformly from the box (--random_frac) in case the flow misses a mode.

Budget per target counts objective evaluations as the paper does: the initial scoring of the seeds, every Adam step
per active candidate, the pruning re-scores and the final score. Flow sampling is NOT in that count (it is a separate
cost; its time is reported separately). The best candidate seen at any scoring point is returned, so the result is
never worse (by the objective) than the best seed.

  searcher = FlowSeededSearcher(jepa, normalizer, flow, device="cuda:0", bounds=checkpoint_bounds(jepa_path))
  out = searcher.search(mel [T, 128, 81], midi [T], budget=512, n_seeds=32)   # out["vectors"] [T, P], ...

Status: CPU-tested on tiny random models (shapes, note pinning, box and couplings, budget accounting, never worse than
the seeds, determinism, batched == one-by-one). NOT run on real checkpoints; whether it matches JADE + Adam on rendered
audio is what audio_eval_v5.py --seeded decides. Timing: `python flow_seeded_search_v5.py --bench ...` on the desktop
(random weights; keep --targets modest on small GPUs).
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import param_codec_v3 as codec  # noqa: E402
from surge_spec_v3 import CAT_INDICES, CONT_BOUNDS, CONT_INDICES, NOTE_HIGH, NOTE_LOW, PARAM_NAMES  # noqa: E402


def _pm1(v01):
    return 2.0 * v01 - 1.0


class FlowSeededSearcher:
    def __init__(self, jepa, normalizer, flow, device="cuda:0", bounds=None):
        self.device = torch.device(device)
        self.jepa = jepa.to(self.device).eval()
        for p in self.jepa.parameters():
            p.requires_grad_(False)
        self.normalizer = normalizer
        self.flow = flow.to(self.device).eval() if flow is not None else None
        names = [PARAM_NAMES[i] for i in CONT_INDICES]
        box = dict(CONT_BOUNDS) if bounds is None else {n: tuple(bounds[n]) for n in names}
        self.lo = torch.tensor([_pm1(box[n][0]) for n in names], device=self.device)
        self.hi = torch.tensor([_pm1(box[n][1]) for n in names], device=self.device)
        pos = {n: j for j, n in enumerate(names)}
        self.cat_sizes = [k for _, k in CAT_INDICES]
        cat_pos = {PARAM_NAMES[i]: j for j, (i, _k) in enumerate(CAT_INDICES)}
        self._note = pos["midi_note"]
        self._couplings = []                                          # (continuous slot, condition)
        if "unison_detune" in pos and "unison" in cat_pos:
            self._couplings.append((pos["unison_detune"], ("cat_eq", cat_pos["unison"], 1)))
        if "drive" in pos and "waveshaper_type" in cat_pos:
            self._couplings.append((pos["drive"], ("cat_gt", cat_pos["waveshaper_type"], 0)))
        if "delay_fb" in pos and "delay_mix" in pos:
            self._couplings.append((pos["delay_fb"], ("cont_gt", pos["delay_mix"], -1.0)))
        self.n_evals = None

    # ------------------------------------------------------------------------------------------- helpers
    def vectors_to_search(self, vecs):
        """[N, P] storage vectors in [0, 1] -> (continuous in [-1, 1] [N, C], list of class tensors)."""
        v = torch.as_tensor(vecs, dtype=torch.float32, device=self.device)
        cont = _pm1(v[:, CONT_INDICES])
        cats = [torch.round(v[:, i] * (k - 1)).clamp(0, k - 1).long() for i, k in CAT_INDICES]
        return cont, cats

    def search_to_vectors(self, cont, cats):
        return codec.assemble(((cont + 1.0) / 2.0).clamp(0.0, 1.0), cats)

    def constrain(self, cont, cats, note):
        """Training box, couplings, per-row pinned note (note in [-1, 1], one per row). Out of place."""
        cont = torch.maximum(torch.minimum(cont, self.hi), self.lo).clone()
        off = torch.full_like(cont[:, 0], -1.0)                     # vector value 0 is -1 here
        for slot, (kind, j, val) in self._couplings:
            if kind == "cat_eq":
                keep = cats[j] == val
            elif kind == "cat_gt":
                keep = cats[j] > val
            else:
                keep = cont[:, j] > val
            cont[:, slot] = torch.where(keep, cont[:, slot], off)
        cont[:, self._note] = note
        return cont

    def objective(self, za, cont, cats):
        """D_JEPA per row: MSE(z_a target of the row, f_p->a(E_p(x)))."""
        onehots = [F.one_hot(c, k).float() for c, k in zip(cats, self.cat_sizes)]
        z_hat = self.jepa.predict_audio_latent(self.jepa.encode_params(cont, onehots))
        return ((z_hat - za) ** 2).mean(dim=-1)

    @staticmethod
    def schedule(budget, n0, prunes=3):
        """Adam steps per phase so initial + steps + pruning re-scores + final score fit the per-target budget."""
        pools = [max(1, n0 // 2 ** i) for i in range(prunes + 1)]
        fixed = n0 + sum(pools[:-1]) + pools[-1]                    # seeds scored, pruning re-scores, final score
        per_phase = max(1, (budget - fixed) // sum(pools))
        return pools, per_phase

    # -------------------------------------------------------------------------------------------- search
    def search(self, mel, midi, budget=512, n_seeds=32, random_frac=0.0, steps=25, tau=1.0, lr=0.1, seed=0,
               seeds=None):
        """mel [T, 128, 81], midi [T]. seeds: optional [T, K, P] vectors instead of sampling the flow."""
        t_start = time.perf_counter()
        T = mel.shape[0]
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        n_rand = int(round(random_frac * n_seeds))
        n_flow = n_seeds - n_rand
        t0 = time.perf_counter()
        if seeds is None:
            from flow_sampling_v5 import sample_candidates
            if self.flow is None:
                raise ValueError("no flow model: pass seeds=")
            seeds = sample_candidates(self.flow, mel, midi, k=n_flow, steps=steps, seed=seed, tau=tau)
        seeds = np.asarray(seeds, dtype=np.float32)[:, :n_flow]
        t_flow = time.perf_counter() - t0
        K = n_seeds
        cont, cats = self.vectors_to_search(seeds.reshape(T * n_flow, -1))
        cont, cats = cont.view(T, n_flow, -1), [c.view(T, n_flow) for c in cats]
        if n_rand:
            rc = self.lo + (self.hi - self.lo) * torch.rand(T, n_rand, cont.shape[-1], device=self.device, generator=gen)
            cont = torch.cat([cont, rc], 1)
            cats = [torch.cat([c, torch.randint(0, k, (T, n_rand), device=self.device, generator=gen)], 1)
                    for c, k in zip(cats, self.cat_sizes)]
        note = _pm1(torch.as_tensor((np.asarray(midi, dtype=np.float64) - NOTE_LOW) / (NOTE_HIGH - NOTE_LOW),
                                    dtype=torch.float32, device=self.device).clamp(0, 1))
        with torch.no_grad():
            za = self.jepa.encode_audio(self.normalizer.normalize(mel.to(self.device)))       # [T, D]

        flat = lambda x: x.reshape(-1, x.shape[-1])                                         # noqa: E731
        rows = lambda n: torch.arange(T, device=self.device).repeat_interleave(n)          # noqa: E731
        cont = self.constrain(flat(cont), [c.reshape(-1) for c in cats], note[rows(K)]).view(T, K, -1)
        with torch.no_grad():
            score = self.objective(za[rows(K)], flat(cont), [c.reshape(-1) for c in cats]).view(T, K)
        n_evals = K
        best_s, best_i = score.min(dim=1)
        best_cont = cont[torch.arange(T), best_i].clone()
        best_cats = [c[torch.arange(T), best_i].clone() for c in cats]

        def keep_best(s, cn, ct):
            nonlocal best_s, best_cont, best_cats
            m, i = s.min(dim=1)
            better = m < best_s
            ar = torch.arange(T, device=self.device)
            best_s = torch.where(better, m, best_s)
            best_cont = torch.where(better[:, None], cn[ar, i], best_cont)
            best_cats = [torch.where(better, c[ar, i], b) for c, b in zip(ct, best_cats)]

        # ---- Adam on everything, pruning the worse half 3x (paper sec. 2.2), batched over targets
        pools, per_phase = self.schedule(budget, K)
        n = min(pools[0], K)
        order = torch.argsort(score, dim=1)[:, :n]
        x = torch.gather(cont, 1, order[..., None].expand(-1, -1, cont.shape[-1])).clone()
        ct = [torch.gather(c, 1, order) for c in cats]
        m_ = torch.zeros_like(x)
        v_ = torch.zeros_like(x)
        total = per_phase * len(pools)
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = 0
        for phase, pool in enumerate(pools):
            for _ in range(per_phase):
                step += 1
                lr_t = lr * 0.5 * (1.0 + math.cos(math.pi * (step - 1) / total))
                xv = x.detach().requires_grad_(True)
                ctf = [c.reshape(-1) for c in ct]
                cur = self.constrain(flat(xv), ctf, note[rows(xv.shape[1])])
                loss = self.objective(za[rows(xv.shape[1])], cur, ctf)
                g, = torch.autograd.grad(loss.sum(), xv)
                n_evals += xv.shape[1]
                with torch.no_grad():
                    keep_best(loss.detach().view(T, -1), cur.detach().view(T, xv.shape[1], -1), ct)
                    m_ = b1 * m_ + (1 - b1) * g
                    v_ = b2 * v_ + (1 - b2) * g * g
                    upd = (m_ / (1 - b1 ** step)) / ((v_ / (1 - b2 ** step)).sqrt() + eps)
                    x = self.constrain(flat(x - lr_t * upd), ctf, note[rows(x.shape[1])]).view(T, x.shape[1], -1)
            with torch.no_grad():                                    # re-score, then prune (or final score)
                ctf = [c.reshape(-1) for c in ct]
                s = self.objective(za[rows(x.shape[1])], flat(x), ctf).view(T, -1)
                n_evals += x.shape[1]
                keep_best(s, x, ct)
                if phase < len(pools) - 1:
                    k_ = pools[phase + 1]
                    idx = torch.topk(s, k_, dim=1, largest=False).indices
                    gi = idx[..., None].expand(-1, -1, x.shape[-1])
                    x, m_, v_ = torch.gather(x, 1, gi), torch.gather(m_, 1, gi), torch.gather(v_, 1, gi)
                    ct = [torch.gather(c, 1, idx) for c in ct]
        self.n_evals = n_evals
        vec = self.search_to_vectors(best_cont, best_cats)
        return {"vectors": vec.detach().cpu().numpy(), "d_jepa": best_s.detach().cpu().numpy(),
                "seed_best": score.min(dim=1).values.detach().cpu().numpy(), "n_evals_per_target": n_evals,
                "adam_steps": total, "time_flow_s": t_flow, "time_total_s": time.perf_counter() - t_start}


# ------------------------------------------------------------------------------------------------- bench
def bench(a):
    """Random-weight timing on the current device: paper JADE + Adam (one target at a time) vs flow-seeded batched
    search, for a few budgets. Timing only; results are meaningless."""
    import models_v4 as m4
    from synth_jepa_model import SynthJEPA
    from synth_jepa_search_v3 import SynthJEPASearcher, WelfordNormalizer
    dev = a.device
    kw = dict(embed_dim=768, ff_dim=2048, num_audio_layers=8, num_param_layers=8, predictor_hidden=1024) \
        if a.size == "base" else dict(embed_dim=384, ff_dim=1024, num_audio_layers=4, num_param_layers=4,
                                      predictor_hidden=768)
    fd = 768 if a.size == "base" else 384
    fl = 8 if a.size == "base" else 4
    torch.manual_seed(0)
    jepa = SynthJEPA(num_slices_sigreg=64, in_frames=81, **kw).to(dev).eval()
    flow = m4.build_model_v4(hidden_dim=fd, num_layers=fl, encoder="temporal_asp").to(dev).eval()
    norm = WelfordNormalizer()
    norm.update(np.random.randn(16, 128, 81).astype(np.float32))
    norm.freeze()
    mel = torch.rand(a.targets, 128, 81) * 2 - 1
    midi = np.full(a.targets, 36)

    def sync():
        if str(dev).startswith("cuda"):
            torch.cuda.synchronize()

    js = SynthJEPASearcher(jepa, norm, device=dev, seed=0)
    js.search(mel[0], midi_note=36, total_eval_budget=128)             # warm-up
    sync()
    t0 = time.perf_counter()
    for i in range(a.jade_targets):
        js.search(mel[i], midi_note=36, total_eval_budget=2048)
    sync()
    t_jade = (time.perf_counter() - t0) / a.jade_targets
    print(f"paper JADE + Adam, budget 2048: {t_jade:.2f} s per target (one target at a time, {a.jade_targets} timed)")
    fs = FlowSeededSearcher(jepa, norm, flow, device=dev)
    fs.search(mel[:2], midi[:2], budget=64, n_seeds=8)                 # warm-up
    for b in a.budgets:
        sync()
        out = fs.search(mel, midi, budget=b, n_seeds=a.seeds)
        sync()
        print(f"flow-seeded, budget {b}, {a.targets} targets batched: {out['time_total_s'] / a.targets:.3f} s per target "
              f"(flow sampling {out['time_flow_s'] / a.targets:.3f} s), {out['n_evals_per_target']} evaluations, "
              f"{out['adam_steps']} Adam steps")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true", help="timing with random weights (the only CLI mode)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", choices=("base", "small"), default="base")
    ap.add_argument("--targets", type=int, default=16, help="targets searched in one batch")
    ap.add_argument("--jade_targets", type=int, default=4, help="targets timed for the paper's search")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--budgets", type=int, nargs="+", default=[256, 512, 1024])
    a = ap.parse_args(argv)
    if not a.bench:
        raise SystemExit("use --bench for timing; searches run through audio_eval_v5.py --seeded")
    bench(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
