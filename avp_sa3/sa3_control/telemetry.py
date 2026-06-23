"""Rich training telemetry → wandb, for when the training paradigm is still under
development and you want to keep options open for *future* analysis.

Design principle: **log broadly now — you can't backfill.** Cheap scalars every log step;
per-layer norms periodically; weight/grad distributions and the weight-space trajectory
sparsely. Everything is namespaced so wandb can group/filter (`wnorm/*`, `dist_init/*`,
`traj/*`, `fusion/*`).

Tiers (cadences are step counts; all should divide the caller's log_every):
  scalar  every log step   loss, gnorm, it/s, lr, epoch, + FusionOpt internals
  layer   every `layer_every`   per-tensor weight-norm, grad-norm, grad/weight, dist-from-init
  hist    every `hist_every`    per-tensor weight + grad histograms (distribution drift)
  traj    every `traj_every`    global weight-space velocity / path-length / net / efficiency
                                (the live version of checkpoint_trajectory_stats.py)

FusionOpt: read non-invasively via its public `.components` + the `_loss_ema`/`_gnorm_ema`
EMAs → recover the **Polyak adaptive step** (loss_ema / gnorm_ema). Shows what the optimizer
is actually doing to the step size over training, distinct from the nominal LR.
"""
from __future__ import annotations

import torch


class TrainTelemetry:
    def __init__(self, trainable_mod, wb, *, scalar_every=20, layer_every=200,
                 hist_every=2000, traj_every=5400, optimizer=None):
        self.wb = wb
        self.opt = optimizer
        self.scalar_every = scalar_every
        self.layer_every = layer_every
        self.hist_every = hist_every
        self.traj_every = traj_every
        self.named = [(n, p) for n, p in trainable_mod.named_parameters() if p.requires_grad]
        if wb is None:
            return  # no-op; skip the (sizeable) init snapshot when not logging
        # init snapshot on CPU for distance-from-init; flat copy for velocity-per-checkpoint
        self.W0 = {n: p.detach().float().cpu().clone() for n, p in self.named}
        self._prev_flat = self._flat_cpu()
        self._path_len = 0.0
        n_params = sum(p.numel() for _, p in self.named)
        print(f"[telemetry] tracking {len(self.named)} tensors / {n_params / 1e6:.1f}M params "
              f"— layers@{layer_every} hist@{hist_every} traj@{traj_every}", flush=True)

    def _flat_cpu(self):
        return torch.cat([p.detach().float().reshape(-1).cpu() for _, p in self.named])

    def _flat_W0(self):
        return torch.cat([self.W0[n].reshape(-1) for n, _ in self.named])

    def _fusion(self):
        o, d = self.opt, {}
        comps = getattr(o, "components", None)
        if comps is None:                       # not a FusionOpt
            return d
        d["fusion/n_components"] = len(comps)
        le, ge = getattr(o, "_loss_ema", None), getattr(o, "_gnorm_ema", None)
        try:
            if le is not None:
                d["fusion/loss_ema"] = float(le)
            if ge is not None:
                d["fusion/gnorm_ema"] = float(ge)
            if le is not None and ge is not None:
                d["fusion/gamma_step"] = float(le) / (float(ge) + 1e-12)   # Polyak adaptive step size
        except Exception:
            pass
        d.update(getattr(o, "_comp_telem", {}) or {})   # per-component update-magnitude profile (comp/*)
        return d

    @torch.no_grad()
    def log(self, step, *, loss, gnorm, it_s, lr, epoch):
        if self.wb is None:
            return
        d = {"loss": loss, "gnorm": gnorm, "it_s": it_s, "lr": lr, "epoch": epoch}
        if self.opt is not None:
            d.update(self._fusion())

        if step % self.layer_every == 0:        # per-layer norms + distance-from-init
            sw = sg = sd = 0.0
            for n, p in self.named:
                w = float(p.detach().float().norm())
                g = float(p.grad.detach().float().norm()) if p.grad is not None else 0.0
                di = float((p.detach().float().cpu() - self.W0[n]).norm())
                d[f"wnorm/{n}"] = w
                d[f"gnorm_layer/{n}"] = g
                d[f"g_over_w/{n}"] = g / (w + 1e-9)      # relative learning signal per layer
                d[f"dist_init/{n}"] = di                # the layer's trajectory length from init
                sw += w * w; sg += g * g; sd += di * di
            d["global/wnorm"] = sw ** 0.5
            d["global/gnorm"] = sg ** 0.5
            d["global/dist_init"] = sd ** 0.5

        if step % self.hist_every == 0:         # weight + grad distributions (drift, dead units)
            for n, p in self.named:
                d[f"whist/{n}"] = self.wb.Histogram(p.detach().float().cpu().numpy())
                if p.grad is not None:
                    d[f"ghist/{n}"] = self.wb.Histogram(p.grad.detach().float().cpu().numpy())

        if step > 0 and step % self.traj_every == 0:    # global weight-space trajectory (live)
            cur = self._flat_cpu()
            vel = float((cur - self._prev_flat).norm())
            self._path_len += vel
            net = float((cur - self._flat_W0()).norm())
            d["traj/velocity"] = vel
            d["traj/path_len"] = self._path_len
            d["traj/net_disp"] = net
            d["traj/path_efficiency"] = net / (self._path_len + 1e-9)   # low ⇒ wandering basin
            self._prev_flat = cur

        try:
            self.wb.log(d, step=step)
        except Exception as e:               # telemetry must NEVER crash a multi-hour training run
            if not getattr(self, "_logwarn", False):
                print(f"[telemetry] wandb.log failed (continuing training): {e}", flush=True)
                self._logwarn = True
