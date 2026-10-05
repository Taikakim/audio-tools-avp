"""Joint online training of Synth-JEPA and the flow-matching inverter on the real-preset bass prior.

Audio is rendered by real Surge XT in DataLoader workers (one persistent plugin per worker) from
realistic_bass_prior.py (training split of the presets); log-mel is computed on the GPU with
ExactGpuMel (matches audio_utils.make_mel_spec). Both models see the same batch each step.

v2 (2026-10-02 review). What changed and why:
  * Validation exists. Two FIXED sets are rendered once at startup: notes from HELD-OUT presets
    (the prior's 'val' split) and notes from training presets. Every --val_interval_steps both are
    scored: JEPA losses (fixed SIGReg slices), flow loss (fixed noise/t), and retrieval over the
    WHOLE set (chance 1/val_size, not 1/batch). The train-preset vs held-out-preset gap is the
    number that says whether the model generalises beyond the ~400 archetypes or memorises them.
    v1 logged only training loss and in-batch retrieval (chance 1/32).
  * A fresh start works (v1 referenced `normalizer` before creating it: NameError).
  * Resume is exact: both optimisers, the normaliser, step, elapsed time and the best score are in
    checkpoint_latest.pt; files are written to a temp name and renamed, so a crash mid-save cannot
    corrupt the file the next start resumes from. A checkpoint from another prior (manifold md5) or
    from the v1 script is refused instead of silently mixing distributions.
  * LR schedule: linear warmup (--warmup_steps), constant, linear decay to 0 over the last
    --decay_frac of the time budget (WSD; the run is time-budgeted, so the schedule is too).
    v1 ran constant LR, and its watchdog halved it at random moments.
  * The flow model is trained on the UN-normalised mel, like every other flow model here, and is
    exported as flow_latest.pt / flow_best.pt in the format models.load_inverter() reads, so
    evaluate_holdout_audio.py --ckpt works on it. The JEPA is exported as jepa_latest.pt /
    jepa_best.pt in the format synth_jepa_search.load_synth_jepa() reads, carrying the prior's
    support box ('prior_bounds') for the search.
  * Non-finite loss or gradient: the step is skipped (weights never see a NaN); after
    --max_bad_steps consecutive skips the process exits with code 3 so the watchdog restarts it
    from the last good checkpoint.
  * Rolling step checkpoints are pruned to the last --keep_last.
  * run_meta.json is written at a fresh launch (purpose, prior, args).
"""
import argparse
import glob
import json
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import param_codec as codec  # noqa: E402
from audio_utils import ExactGpuMel  # noqa: E402
from models import build_model  # noqa: E402
from realistic_bass_prior import MANIFOLD_PATH, RealisticBassPrior, manifold_md5  # noqa: E402
from surge_spec import (  # noqa: E402
    CAT_INDICES, CONT_INDICES, DEFAULT_PLUGIN_PATH, LP_FILTERS, NOTE_DUR_RANGE,
    NUM_PARAMS, PARAM_INDEX, WAVESHAPER_TYPES, canonicalize_vector, domain_weights,
    init_synth, render_patch, vector_to_patch,
)
from synth_jepa_model import SynthJEPA  # noqa: E402
from synth_jepa_search import WelfordNormalizer  # noqa: E402

FORMAT_VERSION = 2
JEPA_KWARGS = dict(embed_dim=512, predictor_hidden=1024, num_audio_layers=8, num_param_layers=8,
                   num_slices_sigreg=64, ff_dim=1024, in_frames=81)
FLOW_KWARGS = dict(model_type="flow", hidden_dim=512, num_layers=8, encoding=codec.ENCODING_V2, time_scale=1000.0)

_STOP_REQUESTED = False


def _sig_handler(sig, frame):
    global _STOP_REQUESTED
    print(f"\n[!] Signal {sig} received. Saving checkpoint and stopping...", flush=True)
    _STOP_REQUESTED = True


def sample_unconstrained_random_patch(rng=None):
    """Draw a completely unconstrained random patch across the full 23-parameter space.
    Not constrained to empirical bass distributions or proxy-bass heuristic ranges."""
    rng = np.random if rng is None else rng
    vec = rng.uniform(0.0, 1.0, size=NUM_PARAMS).astype(np.float32)
    filter_choice = int(rng.randint(0, len(LP_FILTERS)))
    vec[PARAM_INDEX["filter_type"]] = filter_choice / max(1, len(LP_FILTERS) - 1)
    unison_choice = int(rng.randint(0, 2))
    vec[PARAM_INDEX["unison"]] = float(unison_choice)
    ws_choice = int(rng.randint(0, len(WAVESHAPER_TYPES)))
    vec[PARAM_INDEX["waveshaper_type"]] = ws_choice / max(1, len(WAVESHAPER_TYPES) - 1)
    # no FX: they poison the persistent worker's Surge state (surge_spec.apply_patch), and the preset
    # prior never uses them, so random patches must not either
    for fx in ("chorus_mix", "delay_mix", "delay_fb"):
        vec[PARAM_INDEX[fx]] = 0.0
    vec = canonicalize_vector(vec)
    patch = vector_to_patch(vec)
    midi_note = patch["midi_note"]
    note_dur = float(rng.uniform(*NOTE_DUR_RANGE))
    return patch, vec, midi_note, note_dur


class RealisticOnlineDataset(Dataset):
    """Renders prior draws online; one persistent Surge instance per worker (persistent_workers)."""

    def __init__(self, manifold_path, plugin_path=DEFAULT_PLUGIN_PATH, length=10_000_000, random_patch_frac=0.0):
        self.manifold_path = manifold_path
        self.plugin_path = plugin_path
        self.length = length
        self.random_patch_frac = float(random_patch_frac)
        self._synth = None
        self._prior = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self._synth is None:
            self._synth = init_synth(self.plugin_path, verify=False)  # enums verified once in the main process
            self._prior = RealisticBassPrior(self.manifold_path, split="train")
        if self.random_patch_frac > 0.0 and np.random.rand() < self.random_patch_frac:
            patch, vec, midi_note, note_dur = sample_unconstrained_random_patch()
        else:
            patch, vec, midi_note, note_dur = self._prior.sample_patch_and_midi()
        audio = render_patch(self._synth, patch, midi_note, note_dur)
        return torch.from_numpy(audio), torch.from_numpy(vec)


def _worker_init(_worker_id):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the main process handles Ctrl-C / SIGTERM
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def render_fixed_set(synth, prior, n, seed):
    rng = np.random.RandomState(seed)
    audio, vecs = [], []
    for _ in range(n):
        patch, vec, midi_note, note_dur = prior.sample_patch_and_midi(rng)
        audio.append(render_patch(synth, patch, midi_note, note_dur))
        vecs.append(vec)
    return torch.from_numpy(np.stack(audio)), torch.from_numpy(np.stack(vecs))


def prepare_jepa_inputs(params: torch.Tensor):
    """[B, 23] stored params -> (continuous in [-1, 1] [B, 20], list of one-hot [B, k])."""
    cont = params[:, CONT_INDICES] * 2.0 - 1.0
    cat_onehots = [F.one_hot(torch.round(params[:, i] * (k - 1)).clamp(0, k - 1).long(), k).float()
                   for i, k in CAT_INDICES]
    return cont, cat_onehots


def flow_loss(model, mel, params, w_enc, generator=None):
    """Conditional flow-matching loss (OT path), domain-weighted, on the encoded layout."""
    x1 = codec.encode(params)
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    xt, target = model.path(x0, x1, t)
    pred = model(xt, t, mel)
    return ((w_enc * (pred - target) ** 2).sum(dim=1) / w_enc.sum()).mean()


def lr_factor(step, frac, warmup_steps, decay_frac):
    warm = 1.0 if warmup_steps <= 0 else min(1.0, (step + 1) / max(1, warmup_steps))
    decay = 1.0 if decay_frac <= 0.0 or frac < 1.0 - decay_frac else max(0.0, (1.0 - frac) / max(decay_frac, 1e-9))
    return warm * decay


@torch.no_grad()
def validate(jepa, flow, normalizer, mel_fn, audio, params, w_enc, lambda_sig, device, chunk=128):
    """Fixed-set scores. Retrieval is over the whole set (chance = 1/len)."""
    jepa.eval()
    flow.eval()
    gen = torch.Generator(device=device).manual_seed(1234)
    sums, n, za_all, zp_all = {}, 0, [], []
    for s in range(0, len(audio), chunk):
        a = audio[s:s + chunk].to(device)
        p = params[s:s + chunk].to(device)
        mel = mel_fn(a)
        cont, cats = prepare_jepa_inputs(p)
        lp, la, sa, sp, za, zp = jepa(normalizer.normalize(mel), cont, cats, sigreg_generator=gen)
        fl = flow_loss(flow, mel, p, w_enc, generator=gen)
        b = a.shape[0]
        for k, v in dict(jepa=lp + la + lambda_sig * (sa + sp), lp=lp, la=la, sig_a=sa, sig_p=sp, flow=fl).items():
            sums[k] = sums.get(k, 0.0) + float(v) * b
        n += b
        za_all.append(za)
        zp_all.append(zp)
    za, zp = torch.cat(za_all), torch.cat(zp_all)
    tgt = torch.arange(len(za), device=device)
    out = {k: v / n for k, v in sums.items()}
    out["r_a2p"] = float((torch.cdist(jepa.f_a2p(za), zp).argmin(dim=1) == tgt).float().mean())
    out["r_p2a"] = float((torch.cdist(jepa.f_p2a(zp), za).argmin(dim=1) == tgt).float().mean())
    out["chance"] = 1.0 / len(za)
    jepa.train()
    flow.train()
    return out


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="/run/media/kim/Mantu/surge_200k_models/overnight_realistic_bass_v2")
    ap.add_argument("--manifold", default=MANIFOLD_PATH)
    ap.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    ap.add_argument("--hours", type=float, default=8.0, help="total time budget, including resumed time")
    ap.add_argument("--max_steps", type=int, default=0, help=">0 stops after this many steps (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--min_batch_size", type=int, default=2, help="minimum batch size when decaying batch size")
    ap.add_argument("--batch_schedule", choices=["constant", "linear_decay", "power2_decay"], default="constant",
                    help="batch size schedule over the time budget: constant, linear_decay, or power2_decay")
    ap.add_argument("--resume_from", default="", help="explicit checkpoint path to resume from (overrides latest)")
    ap.add_argument("--optimizer", choices=["modular", "adamw", "fusion"], default="modular",
                    help="optimizer choice: modular (ModularOptimizer with SCION LMOs), adamw, or fusion")
    ap.add_argument("--whitening", choices=["shampoo", "soap", "none"], default="shampoo",
                    help="whitening preconditioner for ModularOptimizer (default: shampoo)")
    ap.add_argument("--random_patch_frac", type=float, default=0.05,
                    help="fraction of training draws from unconstrained random parameter space (default: 0.05)")
    ap.add_argument("--snr_gate", action="store_true", default=False,
                    help="enable SNR gating (default: False, disabled)")
    ap.add_argument("--ns_poly", default="cubic5", choices=["quintic", "cubic5", "cubic"],
                    help="Newton-Schulz polynomial for SpectralLMO (default: cubic5)")
    ap.add_argument("--radial_brake", type=float, default=0.8,
                    help="soft limiting on parameter norm growth (default: 0.8)")
    ap.add_argument("--var_dampening", type=float, default=1.15,
                    help="Variance-Aware Dynamic Dampening threshold (default: 1.15)")
    ap.add_argument("--sf_c_warmup", type=int, default=1000,
                    help="steps before Schedule-Free Polyak averaging engages (default: 1000)")
    ap.add_argument("--sf_r", type=float, default=1.0,
                    help="Schedule-Free learning rate weighting (default: 1.0)")
    ap.add_argument("--wd_jepa", type=float, default=0.02,
                    help="weight decay on JEPA spectral weights (default: 0.02)")
    ap.add_argument("--wd_flow", type=float, default=1e-4,
                    help="weight decay on Flow spectral weights (default: 1e-4)")
    ap.add_argument("--wd_overtraining", action="store_true", default=True,
                    help="enable overtraining-aware weight decay scaling (default: True)")
    ap.add_argument("--fusion_components", default="shampoo,sf,normuon,ns5,mona",
                    help="comma-separated list of FusionOpt components to enable")
    ap.add_argument("--ff_dim", type=int, default=1152,
                    help="feedforward dimension for JEPA (1152 matches ~53M paper size, 1024 is default ~50M)")
    ap.add_argument("--total_steps", type=int, default=0,
                    help="total steps for WSD decay schedule (0 calculates from hours and batch size)")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4,
                    help="batches loaded in advance by each worker (default: 4)")
    ap.add_argument("--lr_jepa", type=float, default=9e-5)
    ap.add_argument("--lr_flow", type=float, default=6e-5)
    ap.add_argument("--lambda_sig", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--decay_frac", type=float, default=0.2, help="final fraction of the time budget with LR decay")
    ap.add_argument("--norm_batches", type=int, default=250, help="batches for the mel normaliser (paper: first 8k)")
    ap.add_argument("--val_size", type=int, default=512, help="notes per fixed validation set")
    ap.add_argument("--val_interval_steps", type=int, default=1000)
    ap.add_argument("--checkpoint_interval_steps", type=int, default=1000)
    ap.add_argument("--keep_last", type=int, default=3, help="rolling checkpoint_step_*.pt files kept")
    ap.add_argument("--log_interval_steps", type=int, default=50)
    ap.add_argument("--max_bad_steps", type=int, default=20)
    ap.add_argument("--allow_prior_change", action="store_true", help="resume although the manifold changed")
    ap.add_argument("--purpose", default="", help="one line for run_meta.json: what question this run answers")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return ap


def train(args):
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    os.makedirs(args.out_dir, exist_ok=True)
    log_file = open(os.path.join(args.out_dir, "training.log"), "a", buffering=1)

    def log(msg):
        line = f"{time.strftime('[%Y-%m-%d %H:%M:%S]')} {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    device = torch.device(args.device)
    prior_md5 = manifold_md5(args.manifold)
    train_prior = RealisticBassPrior(args.manifold, split="train")
    val_prior = RealisticBassPrior(args.manifold, split="val")
    prior_bounds = train_prior.support_bounds()
    log("=" * 70)
    log(f"Realistic-bass joint training on {device} | prior {args.manifold} (md5 {prior_md5[:10]}, "
        f"{train_prior.N} train / {val_prior.N} held-out presets, calibrated={train_prior.calibrated}, "
        f"random_patch_frac={args.random_patch_frac:.1%})")
    if not train_prior.calibrated:
        log("WARNING: manifold was built with --no_plugin (unverified unit conversions)")

    mel_fn = ExactGpuMel(device=device)
    jepa_kwargs = dict(embed_dim=512, predictor_hidden=1024, num_audio_layers=8, num_param_layers=8,
                       num_slices_sigreg=64, ff_dim=args.ff_dim, in_frames=81)
    jepa = SynthJEPA(**jepa_kwargs).to(device)
    flow = build_model(**FLOW_KWARGS).to(device)

    if args.optimizer == "modular":
        from stable_audio_tools.training.modular_opt import (
            ModularOptimizer,
            build_modular_param_groups,
            summarise_modular_groups,
        )

        jepa_groups = build_modular_param_groups(
            jepa,
            default_whitening=args.whitening,
            spectral_lr=args.lr_jepa,
            sign_lr=args.lr_jepa,
            colnorm_lr=args.lr_jepa,
            spectral_wd=args.wd_jepa,
            sign_wd=0.0,
            colnorm_wd=0.0,
        )
        flow_groups = build_modular_param_groups(
            flow,
            default_whitening=args.whitening,
            spectral_lr=args.lr_flow,
            sign_lr=args.lr_flow,
            colnorm_lr=args.lr_flow,
            spectral_wd=args.wd_flow,
            sign_wd=0.0,
            colnorm_wd=0.0,
        )
        opt_jepa = ModularOptimizer(
            jepa_groups,
            lr=args.lr_jepa,
            schedule_free=True,
            sf_c_warmup=args.sf_c_warmup,
            sf_r=args.sf_r,
            normuon=True,
            snr_gate=args.snr_gate,
            ns_poly=args.ns_poly,
            radial_brake=args.radial_brake,
            var_dampening_threshold=args.var_dampening,
            wd_overtraining=args.wd_overtraining,
            warmup_steps=args.warmup_steps,
        )
        opt_flow = ModularOptimizer(
            flow_groups,
            lr=args.lr_flow,
            schedule_free=True,
            sf_c_warmup=args.sf_c_warmup,
            sf_r=args.sf_r,
            normuon=True,
            snr_gate=args.snr_gate,
            ns_poly=args.ns_poly,
            radial_brake=args.radial_brake,
            var_dampening_threshold=args.var_dampening,
            wd_overtraining=args.wd_overtraining,
            warmup_steps=args.warmup_steps,
        )
        log(f"Optimizer: ModularOptimizer (ns_poly={args.ns_poly}, whitening={args.whitening}, "
            f"radial_brake={args.radial_brake}, var_damp={args.var_dampening}, sf_c_warmup={args.sf_c_warmup}, "
            f"wd_ot={args.wd_overtraining}, snr_gate={args.snr_gate})")
    elif args.optimizer == "fusion":
        from stable_audio_tools.training.fusion_groups import build_fusion_param_groups
        from stable_audio_tools.training.fusion_opt import FusionOpt

        jepa_groups = build_fusion_param_groups(jepa, spectral_lr=args.lr_jepa, scalar_lr=args.lr_jepa,
                                                spectral_wd=0.01, scalar_wd=0.0)
        flow_groups = build_fusion_param_groups(flow, spectral_lr=args.lr_flow, scalar_lr=args.lr_flow,
                                                spectral_wd=1e-4, scalar_wd=0.0)
        components = set(c.strip() for c in args.fusion_components.split(",") if c.strip())
        tot_steps = args.total_steps if args.total_steps > 0 else max(1000, int(args.hours * 3600 * 1.5))
        opt_jepa = FusionOpt(jepa_groups, lr=args.lr_jepa, components=components,
                             decay_schedule="wsd", total_steps=tot_steps, decay_start_frac=0.8)
        opt_flow = FusionOpt(flow_groups, lr=args.lr_flow, components=components,
                             decay_schedule="wsd", total_steps=tot_steps, decay_start_frac=0.8)
        log(f"Optimizer: FusionOpt with components {sorted(components)} (decay_schedule=wsd, total_steps={tot_steps})")
    else:
        opt_jepa = torch.optim.AdamW(jepa.parameters(), lr=args.lr_jepa, weight_decay=0.05)
        opt_flow = torch.optim.AdamW(flow.parameters(), lr=args.lr_flow, weight_decay=1e-4)
        log("Optimizer: AdamW")

    w_enc = codec.encoded_weights(torch.from_numpy(domain_weights()).to(device))
    log(f"JEPA {sum(p.numel() for p in jepa.parameters()) / 1e6:.2f}M params (ff_dim={args.ff_dim}), "
        f"flow {sum(p.numel() for p in flow.parameters()) / 1e6:.2f}M params")

    # Fixed validation sets (main-process synth; verifies the enum mapping once)
    synth = init_synth(args.plugin, verify=True)
    t0 = time.time()
    val_sets = {"heldout": render_fixed_set(synth, val_prior, args.val_size, seed=1),
                "trainpresets": render_fixed_set(synth, train_prior, args.val_size, seed=2)}
    log(f"Rendered fixed validation sets ({args.val_size} notes each) in {time.time() - t0:.1f}s")
    del synth
    import gc
    gc.collect()

    loader = DataLoader(RealisticOnlineDataset(args.manifold, args.plugin, random_patch_frac=args.random_patch_frac),
                        batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True,
                        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                        persistent_workers=args.num_workers > 0, worker_init_fn=_worker_init,
                        # spawn, not fork: the main process has loaded Surge (validation renders),
                        # and a forked copy of the plugin/audio state can deadlock
                        multiprocessing_context="spawn" if args.num_workers > 0 else None)
    batches = iter(loader)

    latest = os.path.join(args.out_dir, "checkpoint_latest.pt")
    ckpt_target = args.resume_from if args.resume_from else latest
    step, elapsed_prev, best = 0, 0.0, float("inf")
    if os.path.exists(ckpt_target):
        ckpt = torch.load(ckpt_target, map_location=device, weights_only=False)
        if ckpt.get("format_version", 1) < FORMAT_VERSION:
            raise SystemExit(f"{ckpt_target} was written by the v1 trainer (no optimiser state, buggy prior); "
                             "start a new run in a new --out_dir.")
        if ckpt["prior_md5"] != prior_md5 and not args.allow_prior_change:
            raise SystemExit(f"{ckpt_target} was trained on a different manifold (md5 {ckpt['prior_md5'][:10]}); "
                             "use a new --out_dir, or --allow_prior_change to continue anyway.")
        jepa.load_state_dict(ckpt["jepa_state_dict"])
        flow.load_state_dict(ckpt["flow_state_dict"])
        opt_jepa.load_state_dict(ckpt["opt_jepa"])
        opt_flow.load_state_dict(ckpt["opt_flow"])
        normalizer = WelfordNormalizer.from_state_dict(ckpt["normalizer"])
        step = ckpt["step"]
        elapsed_prev = 0.0 if args.resume_from else ckpt["elapsed_s"]
        best = ckpt.get("best_val", float("inf"))
        log(f"Resumed at step {step} from {os.path.basename(ckpt_target)} "
            f"({ckpt['elapsed_s'] / 3600:.2f} h previously logged, best held-out JEPA loss {best:.4f})")
    else:
        log(f"Estimating mel normaliser over {args.norm_batches} batches...")
        normalizer = WelfordNormalizer()
        for _ in range(args.norm_batches):
            aud, _ = next(batches)
            normalizer.update(mel_fn(aud.to(device)).cpu().numpy())
        normalizer.freeze()
        with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
            json.dump({"run": os.path.basename(os.path.normpath(args.out_dir)),
                       "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "purpose": args.purpose or None, "status": "running",
                       "script": os.path.abspath(__file__), "args": vars(args),
                       "prior": {"manifold": args.manifold, "md5": prior_md5, "train_presets": train_prior.N,
                                 "heldout_presets": val_prior.N, "calibrated": train_prior.calibrated},
                       "result": None, "kim_feedback": None}, f, indent=2)

    def resume_state(val_best):
        return {"format_version": FORMAT_VERSION, "step": step, "elapsed_s": elapsed_prev + time.time() - start,
                "jepa_state_dict": jepa.state_dict(), "flow_state_dict": flow.state_dict(),
                "opt_jepa": opt_jepa.state_dict(), "opt_flow": opt_flow.state_dict(),
                "normalizer": normalizer.state_dict(), "prior_md5": prior_md5, "prior_bounds": prior_bounds,
                "best_val": val_best, "args": vars(args), "jepa_kwargs": jepa_kwargs}

    def export(tag):
        meta = {"step": step, "prior": args.manifold, "prior_md5": prior_md5}
        atomic_save({"model_state_dict": jepa.state_dict(), "model_kwargs": jepa_kwargs,
                     "normalizer": normalizer.state_dict(), "prior_bounds": prior_bounds, **meta},
                    os.path.join(args.out_dir, f"jepa_{tag}.pt"))
        atomic_save({"model_state": flow.state_dict(), "prior_bounds": prior_bounds,
                     "args": {"model_type": "flow", "param_encoding": codec.ENCODING_V2, "time_scale": 1000.0}, **meta},
                    os.path.join(args.out_dir, f"flow_{tag}.pt"))

    def save_checkpoint():
        if hasattr(opt_jepa, "eval"):
            opt_jepa.eval()
        if hasattr(opt_flow, "eval"):
            opt_flow.eval()
        state = resume_state(best)
        atomic_save(state, os.path.join(args.out_dir, f"checkpoint_step_{step:07d}.pt"))
        atomic_save(state, latest)
        export("latest")
        for old in sorted(glob.glob(os.path.join(args.out_dir, "checkpoint_step_*.pt")))[:-args.keep_last]:
            os.remove(old)
        if hasattr(opt_jepa, "train"):
            opt_jepa.train()
        if hasattr(opt_flow, "train"):
            opt_flow.train()

    total_s = args.hours * 3600.0
    start = time.time()
    run = {k: 0.0 for k in ("jepa", "flow", "r_a2p", "r_p2a")}
    bad = {"jepa": 0, "flow": 0}
    log("Entering main training loop...")
    if hasattr(opt_jepa, "train"):
        opt_jepa.train()
    if hasattr(opt_flow, "train"):
        opt_flow.train()

    while not _STOP_REQUESTED:
        elapsed = elapsed_prev + time.time() - start
        if elapsed >= total_s or (args.max_steps and step >= args.max_steps):
            break
        if args.optimizer == "adamw":
            f = lr_factor(step, elapsed / total_s, args.warmup_steps, args.decay_frac)
            for g in opt_jepa.param_groups:
                g["lr"] = args.lr_jepa * f
            for g in opt_flow.param_groups:
                g["lr"] = args.lr_flow * f
        else:
            f = 1.0

        # Dynamic batch schedule
        time_frac = min(1.0, max(0.0, elapsed / max(1e-6, total_s)))
        if args.batch_schedule == "linear_decay":
            curr_b = int(round(args.batch_size - time_frac * (args.batch_size - args.min_batch_size)))
        elif args.batch_schedule == "power2_decay":
            # 32 -> 16 -> 8 -> 4 -> 2 across geometric time intervals
            stages = [32, 16, 8, 4, 2]
            stage_idx = min(len(stages) - 1, int(time_frac * len(stages)))
            curr_b = max(args.min_batch_size, stages[stage_idx])
        else:
            curr_b = args.batch_size
        curr_b = max(args.min_batch_size, min(args.batch_size, curr_b))

        audio, params = next(batches)
        if curr_b < audio.shape[0]:
            audio = audio[:curr_b]
            params = params[:curr_b]
        audio = audio.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        mel = mel_fn(audio)

        opt_jepa.zero_grad(set_to_none=True)
        cont, cats = prepare_jepa_inputs(params)
        lp, la, sa, sp, za, zp = jepa(normalizer.normalize(mel), cont, cats)
        j_loss = lp + la + args.lambda_sig * (sa + sp)
        j_loss.backward()
        gn = nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)
        if torch.isfinite(j_loss) and torch.isfinite(gn):
            opt_jepa.step()
            bad["jepa"] = 0
        else:
            bad["jepa"] += 1
            log(f"Step {step}: non-finite JEPA loss/grad ({float(j_loss):.4g}/{float(gn):.4g}); step skipped")

        opt_flow.zero_grad(set_to_none=True)
        f_loss = flow_loss(flow, mel, params, w_enc)
        f_loss.backward()
        gn = nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        if torch.isfinite(f_loss) and torch.isfinite(gn):
            opt_flow.step()
            bad["flow"] = 0
        else:
            bad["flow"] += 1
            log(f"Step {step}: non-finite flow loss/grad ({float(f_loss):.4g}/{float(gn):.4g}); step skipped")
        if max(bad.values()) >= args.max_bad_steps:
            log(f"{args.max_bad_steps} consecutive non-finite steps; exiting (code 3) without saving")
            log_file.close()
            sys.exit(3)

        with torch.no_grad():
            tgt = torch.arange(za.shape[0], device=device)
            run["r_a2p"] += float((torch.cdist(jepa.f_a2p(za), zp).argmin(dim=1) == tgt).float().mean())
            run["r_p2a"] += float((torch.cdist(jepa.f_p2a(zp), za).argmin(dim=1) == tgt).float().mean())
        run["jepa"] += float(j_loss.detach())
        run["flow"] += float(f_loss.detach())
        step += 1

        if step % args.log_interval_steps == 0:
            k = args.log_interval_steps
            el = elapsed_prev + time.time() - start
            log(f"Step {step:7d} | {el / 3600:5.2f}h / {args.hours:.2f}h | lr x{f:.3f} | batch {curr_b:2d} | "
                f"JEPA {run['jepa'] / k:.4f} | Flow {run['flow'] / k:.4f} | "
                f"in-batch R a2p {100 * run['r_a2p'] / k:5.1f}% p2a {100 * run['r_p2a'] / k:5.1f}% "
                f"(chance {100 / curr_b:.1f}%)")
            run = {key: 0.0 for key in run}

        if step % args.val_interval_steps == 0:
            if hasattr(opt_jepa, "eval"):
                opt_jepa.eval()
            if hasattr(opt_flow, "eval"):
                opt_flow.eval()
            res = {name: validate(jepa, flow, normalizer, mel_fn, a, p, w_enc, args.lambda_sig, device)
                   for name, (a, p) in val_sets.items()}
            h, tr = res["heldout"], res["trainpresets"]
            log(f"VAL step {step} | held-out presets: JEPA {h['jepa']:.4f} flow {h['flow']:.4f} "
                f"R a2p {100 * h['r_a2p']:.1f}% p2a {100 * h['r_p2a']:.1f}% (chance {100 * h['chance']:.2f}%) | "
                f"train presets: JEPA {tr['jepa']:.4f} flow {tr['flow']:.4f} | gap {h['jepa'] - tr['jepa']:+.4f}")
            with open(os.path.join(args.out_dir, "val.jsonl"), "a") as fh:
                fh.write(json.dumps({"step": step, "time": time.time(), **{f"{n}/{k}": v for n, r in res.items()
                                                                           for k, v in r.items()}}) + "\n")
            if h["jepa"] < best:
                best = h["jepa"]
                atomic_save(resume_state(best), os.path.join(args.out_dir, "checkpoint_best.pt"))
                export("best")
                log(f"--> new best held-out JEPA loss {best:.4f}: checkpoint_best.pt, jepa_best.pt, flow_best.pt")
            if hasattr(opt_jepa, "train"):
                opt_jepa.train()
            if hasattr(opt_flow, "train"):
                opt_flow.train()

        if step % args.checkpoint_interval_steps == 0:
            save_checkpoint()
            log(f"--> checkpoint at step {step}")

    if hasattr(opt_jepa, "eval"):
        opt_jepa.eval()
    if hasattr(opt_flow, "eval"):
        opt_flow.eval()
    save_checkpoint()
    atomic_save(resume_state(best), os.path.join(args.out_dir, "checkpoint_final.pt"))
    log(f"Stopped at step {step} ({(elapsed_prev + time.time() - start) / 3600:.2f} h); "
        f"best held-out JEPA loss {best:.4f}")
    log_file.close()


if __name__ == "__main__":
    train(build_parser().parse_args())
