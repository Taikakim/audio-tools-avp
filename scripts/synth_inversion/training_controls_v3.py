"""Accuracy controls DURING training for the joint JEPA + flow trainer (train_realistic_bass_overnight.py).

Why: both training losses live in parameter / latent space and every sample is an independent prior
draw, so the models almost never see two sounds that differ in ONE audible control. The renderer-in-the-
loop refinement (refine.py) found cutoff off by a median ~9 semitones on 24 real stems, then amp sustain
and the envelopes. These controls put that axis-wise resolution into training:

  LadderDataset      "bracketed" minimal-pair batches (Kim's recipe, 2026-10-06): anchors from the prior,
                     each swept in 16 steps along one audible axis with everything else fixed:
                       cutoff_low   cutoff 0.08..0.30 raw (~25..130 Hz), the region the mel front end
                                    resolves worst and where real basses sit
                       aeg_sustain  sustain stepped down to 0; decay drawn between a 1/64 and a 1/8 note
                                    at 100 BPM (37.5..300 ms -> raw 0.25..0.48 on Surge's log2 scale)
                       feg_amount   cutoff low (<0.30), filter sustain 0, filter decay in the same 0.25..
                                    0.48 bracket, FEG amount laddered up over the prior range
                       aeg_decay    decay 0.25..0.48 with sustain held low so the decay is audible
                     (FEG/AEG attack are not in the 23-d space, so "random attack" is not available.)
  ordinal_loss       on a ladder, the AUDIO embedding's distance from the ladder's first step must grow
                     with the step index -- the JEPA geometry the search and the refinement walk on.
                     SIGReg is NOT applied to ladder batches: it assumes an i.i.d. batch and would push
                     the near-duplicates apart.
  ot_couple          minibatch optimal-transport noise<->target pairing for the flow (Hungarian), as in
                     the reference flow (synth-permutations src/data/ot.py); straighter paths.
  EMA                exponential moving average of the online weights, half-life in steps.
  ladder_validation  on FIXED held-out ladders: Spearman(step index, ||z_a,k - z_a,0||) for the JEPA and
                     Spearman(true axis value, flow point estimate of that axis) for the flow -- "does it
                     get cutoff / sustain / FEG / decay right", per axis.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from torch.utils.data import Dataset

from surge_spec_v3 import CONT_BOUNDS, NOTE_DUR_RANGE, PARAM_INDEX, canonicalize_vector, vector_to_patch

AXES = ("cutoff_low", "aeg_sustain", "feg_amount", "aeg_decay")
AXIS_PARAM = {"cutoff_low": "cutoff", "aeg_sustain": "aeg_sustain", "feg_amount": "feg_amount",
              "aeg_decay": "aeg_decay"}
ENV_BRACKET = (0.25, 0.48)  # 1/64 .. 1/8 note at 100 BPM, Surge log2-seconds raw


def _clip(name, v):
    lo, hi = CONT_BOUNDS[name]
    return float(np.clip(v, lo, hi))


def make_ladder(base_vec, axis, n_steps, rng):
    """[n_steps, 23] vectors: base_vec swept along `axis` (ascending axis value)."""
    v = np.array(base_vec, dtype=np.float32, copy=True)
    I = PARAM_INDEX
    if axis == "cutoff_low":
        vals = np.linspace(0.08, 0.30, n_steps)
    elif axis == "aeg_sustain":
        v[I["aeg_decay"]] = _clip("aeg_decay", rng.uniform(*ENV_BRACKET))
        vals = np.linspace(0.0, CONT_BOUNDS["aeg_sustain"][1], n_steps)
    elif axis == "feg_amount":
        v[I["cutoff"]] = _clip("cutoff", rng.uniform(0.08, 0.30))
        v[I["feg_sustain"]] = 0.0
        v[I["feg_decay"]] = _clip("feg_decay", rng.uniform(*ENV_BRACKET))
        vals = np.linspace(*CONT_BOUNDS["feg_amount"], n_steps)
    elif axis == "aeg_decay":
        v[I["aeg_sustain"]] = _clip("aeg_sustain", rng.uniform(0.0, 0.2))
        vals = np.linspace(*ENV_BRACKET, n_steps)
    else:
        raise ValueError(axis)
    name = AXIS_PARAM[axis]
    out = np.repeat(v[None], n_steps, axis=0)
    out[:, I[name]] = [_clip(name, x) for x in vals]
    return canonicalize_vector(out)


class LadderDataset(Dataset):
    """Each item is a whole ladder batch: n_anchors x n_steps renders, one axis per anchor."""

    def __init__(self, manifold_path, plugin_path, n_anchors=4, n_steps=16, length=10_000_000, split="train"):
        self.manifold_path, self.plugin_path, self.length, self.split = manifold_path, plugin_path, length, split
        self.n_anchors, self.n_steps = n_anchors, n_steps
        self._synth = self._prior = None

    def __len__(self):
        return self.length

    def build(self, rng, synth, prior):
        from surge_spec_v3 import render_patch
        audio, vecs, axis_ids = [], [], []
        for _ in range(self.n_anchors):
            _, base, midi_note, _ = prior.sample_patch_and_midi(rng)
            note_dur = float(rng.uniform(*NOTE_DUR_RANGE))
            ax = int(rng.randint(len(AXES)))
            for vec in make_ladder(base, AXES[ax], self.n_steps, rng):
                patch = vector_to_patch(vec)
                audio.append(render_patch(synth, patch, int(midi_note), note_dur))
                vecs.append(vec)
                axis_ids.append(ax)
        return (torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(vecs).astype(np.float32)),
                torch.tensor(axis_ids))

    def __getitem__(self, idx):
        from realistic_bass_prior import RealisticBassPrior
        from surge_spec_v3 import init_synth
        if self._synth is None:
            self._synth = init_synth(self.plugin_path, verify=False)
            self._prior = RealisticBassPrior(self.manifold_path, split=self.split)
        return self.build(np.random, self._synth, self._prior)


def ordinal_loss(za, n_steps, margin=0.01):
    """za [n_ladders * n_steps, D] in ladder order. Hinge on every ordered pair (i < j) of a ladder:
    the distance of step j from step 0 must exceed that of step i (scale-free: divided by the ladder's
    mean distance, detached)."""
    z = za.view(-1, n_steps, za.shape[-1])
    d = (z[:, 1:] - z[:, :1]).norm(dim=-1)                      # [L, n_steps-1]
    d = d / d.mean(dim=1, keepdim=True).detach().clamp_min(1e-6)
    i, j = torch.triu_indices(d.shape[1], d.shape[1], offset=1, device=d.device)
    return F.relu(d[:, i] - d[:, j] + margin * (j - i).float()).mean()


def ot_couple(x0, x1):
    """Permute the noise so noise[k] is the OT partner of target x1[k] (minibatch Hungarian)."""
    cost = torch.cdist(x0.detach().float(), x1.detach().float()).cpu().numpy()
    r, c = linear_sum_assignment(cost)
    out = torch.empty_like(x0)
    out[torch.as_tensor(c, device=x0.device)] = x0[torch.as_tensor(r, device=x0.device)]
    return out


class EMA:
    """EMA of parameters (and a copy of buffers), half-life in optimizer steps, with the usual early
    warm-up (beta_t = min(beta, (1 + t) / (10 + t))) so step-0 weights do not dominate."""

    def __init__(self, model, halflife_steps):
        self.beta = 0.5 ** (1.0 / max(1.0, float(halflife_steps)))
        self.halflife = halflife_steps
        self.n = 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        b = min(self.beta, (1.0 + self.n) / (10.0 + self.n))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point and k in self.shadow:
                self.shadow[k].mul_(b).add_(v.detach().float(), alpha=1.0 - b)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return {"shadow": self.shadow, "n": self.n, "beta": self.beta, "halflife": self.halflife}

    def load_state_dict(self, sd):
        self.shadow, self.n, self.beta, self.halflife = sd["shadow"], sd["n"], sd["beta"], sd["halflife"]

    def swapped(self, model):
        """Context manager: model holds the EMA weights inside the block, its own weights after."""
        ema = self

        class _Swap:
            def __enter__(self_):
                self_.backup = copy.deepcopy(model.state_dict())
                model.load_state_dict({k: v.to(self_.backup[k].dtype) for k, v in ema.shadow.items()})
                return model

            def __exit__(self_, *exc):
                model.load_state_dict(self_.backup)
                return False
        return _Swap()


def render_ladder_set(synth, prior, n_anchors_per_axis, n_steps, seed):
    """Fixed held-out ladders: every axis gets the same number of anchors."""
    from surge_spec_v3 import render_patch
    rng = np.random.RandomState(seed)
    audio, vecs, axis_ids = [], [], []
    for ax, axis in enumerate(AXES):
        for _ in range(n_anchors_per_axis):
            _, base, midi_note, _ = prior.sample_patch_and_midi(rng)
            note_dur = float(rng.uniform(*NOTE_DUR_RANGE))
            for vec in make_ladder(base, axis, n_steps, rng):
                audio.append(render_patch(synth, vector_to_patch(vec), int(midi_note), note_dur))
                vecs.append(vec)
                axis_ids.append(ax)
    return torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(vecs).astype(np.float32)), \
        np.array(axis_ids)


@torch.no_grad()
def ladder_validation(jepa, flow, normalizer, mel_fn, ladder_set, n_steps, device, codec,
                      flow_draws=8, flow_steps=12, chunk=128):
    audio, vecs, axis_ids = ladder_set
    was_j, was_f = jepa.training, flow.training
    jepa.eval()
    flow.eval()
    za_all, est_all = [], []
    gen = torch.Generator(device=device).manual_seed(4321)
    for s in range(0, len(audio), chunk):
        mel = mel_fn(audio[s:s + chunk].to(device))
        za_all.append(jepa.encode_audio(normalizer.normalize(mel)))
        draws = torch.stack([codec.decode(flow.sample(mel, num_steps=flow_steps, generator=gen))
                             for _ in range(flow_draws)])
        est_all.append(draws.mean(0))
    jepa.train(was_j)
    flow.train(was_f)
    za = torch.cat(za_all).view(-1, n_steps, za_all[0].shape[-1])
    est = torch.cat(est_all).cpu().numpy().reshape(-1, n_steps, vecs.shape[-1])
    true = vecs.numpy().reshape(-1, n_steps, vecs.shape[-1])
    ax_of_ladder = axis_ids.reshape(-1, n_steps)[:, 0]
    dist = (za[:, 1:] - za[:, :1]).norm(dim=-1).cpu().numpy()
    out = {}
    for ax, axis in enumerate(AXES):
        rows = np.where(ax_of_ladder == ax)[0]
        if not len(rows):
            continue
        k = PARAM_INDEX[AXIS_PARAM[axis]]
        j_rho = [spearmanr(np.arange(1, n_steps), dist[r]).correlation for r in rows]
        f_rho = [spearmanr(true[r, :, k], est[r, :, k]).correlation for r in rows]
        f_mae = [float(np.mean(np.abs(true[r, :, k] - est[r, :, k]))) for r in rows]
        out[f"ladder_{axis}_jepa_rho"] = float(np.nanmean(j_rho))
        out[f"ladder_{axis}_flow_rho"] = float(np.nanmean(f_rho))
        out[f"ladder_{axis}_flow_mae"] = float(np.mean(f_mae))
    out["ladder_flow_rho_mean"] = float(np.nanmean([v for k, v in out.items() if k.endswith("flow_rho")]))
    out["ladder_jepa_rho_mean"] = float(np.nanmean([v for k, v in out.items() if k.endswith("jepa_rho")]))
    return out
