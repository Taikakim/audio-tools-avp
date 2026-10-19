# Owner: KUANG (created 2026-10-13)
"""Flow-DPO: preference fine-tuning of the flow inverter on rendered, scored candidates (V5_PROPOSAL_REVIEW.md, sec. 6).

Objective: Diffusion-DPO (Wallace et al. 2023) written for conditional flow matching. For a condition c (target mel),
a preferred vector x_w and a dispreferred x_l (both from dpo_pairs_v5.py: rendered, scored against the target), one
time t and one noise x0 SHARED by both:
    e_m(x) = weighted mean squared velocity error of model m at (x_t, t, c)   (codec layout, the trainer's domain weights)
    delta  = [e_policy(x_w) - e_ref(x_w)] - [e_policy(x_l) - e_ref(x_l)]
    L_dpo  = -log sigmoid(-beta * delta)
The policy is rewarded for lowering its error on the winner RELATIVE to the frozen reference more than on the loser.
beta sets how far it may move from the reference before the loss saturates: delta ~ -2 / beta is where the gradient has
mostly vanished, so beta = 100 allows a relative error shift of ~0.02 (flow MSE is ~0.4). Sweep {30, 100, 300}.
Optional anchor: --sft_weight adds the plain flow-matching loss on the winner (or, for synthetic conditions, on the
true vector), which keeps the policy on the training distribution.

Choices, with reasons:
  * MSE inside DPO, whatever the base run used (MAE): the DPO derivation needs the Gaussian likelihood form.
  * The policy runs in eval mode (BatchNorm uses its running statistics, frozen; dropout is 0 anyway) while its weights
    train: then policy == reference exactly at step 0 (delta = 0, loss = log 2), and batch statistics of the winner /
    loser mix cannot leak into the comparison.
  * AdamW, small learning rate, short run: this is a fine-tune of a converged model, not training.
  * Pairs per condition: best vs worst (default) or every pair whose score gap exceeds the margin. The margin is
    --margin + --noise_mult x the condition's render-noise floor when the pairs file has one (synthetic conditions),
    so pairs that differ only by Surge's render-to-render noise are dropped.
  * Conditions are split into train / validation by condition, not by pair.

Outputs (DIR): flow_dpo_latest.pt (and flow_dpo_step_N.pt every --save_every) in the trainer's export format, so
load_inverter_v4, dpo_pairs_v5 (next round) and audio_eval_v5 read it; dpo_log.jsonl. With --link_run RUN, DIR also
gets jepa_online_latest.pt (copied from RUN, kind/tag via --link_kind/--link_tag), flow_online_latest.pt (the DPO
flow, step number = the JEPA's) and run_meta.json, so heldout_eval_v4 --run DPO=DIR --kinds online scores it in
parameter space next to the source run.

Status: CPU-tested on tiny random models (delta = 0 at init, the loss falls on a synthetic preference, the reference
stays frozen, export round trip). NOT run on real pairs; whether it improves rendered audio is what audio_eval_v5 and
the listening test decide.

Layouts (KUANG 2026-10-19): codec and domain weights follow the flow's layout (layout_v6), so v3, v4 and v5 flows
all work; the pairs must come from the same flow (checked by md5, or --allow_offpolicy).
--metric may be a weighted sum of stored metrics, e.g. "mss=1,modspec_dist=0.5": candidates are ranked by the sum and
the render-noise margin uses the same weighted sum of the per-metric floors (only if every term has a floor).

Usage (desktop):
  python train_flow_dpo_v5.py --flow RUN/flow_ema_latest.pt --pairs dpo_pairs_a3p_s0 --out dpo_a3p_b100 \\
      --beta 100 --lr 1e-5 --steps 2000 --batch 64 --link_run RUN --link_kind ema --device cuda:0
"""
import argparse
import copy
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from layout_v6 import layout_for_flow  # noqa: E402
from models_v6 import flow_call  # noqa: E402


# ------------------------------------------------------------------------------------------------------ data
def parse_metric(spec, default):
    """"mss" -> {"mss": 1.0}; "mss=1,modspec_dist=0.5" -> weights. None -> {default: 1.0}."""
    if not spec:
        return {default: 1.0}
    out = {}
    for part in spec.split(","):
        name, _, w = part.strip().partition("=")
        out[name.strip()] = float(w) if w else 1.0
    return out


def combined(z, weights, prefix):
    """Weighted sum of stored arrays prefix + name; None if any term is missing."""
    keys = [f"{prefix}{n}" for n in weights]
    if not all(k in z for k in keys):
        return None
    return sum(w * np.asarray(z[k], dtype=np.float64) for k, w in zip(keys, weights.values()))


def load_pairs(dirs, metric=None, mode="best_worst", margin=0.0, noise_mult=1.0, max_per_cond=8):
    """-> dict: cond_audio int16 [C, N], midi [C], true [C, P] or None, and pair arrays cond [M], w [M, P], l [M, P],
    gap [M]; plus the manifests. metric: one stored metric or a weighted sum (see parse_metric)."""
    audio, midi, true, man = [], [], [], []
    cond, w, l, gap = [], [], [], []
    offset = 0
    for d in dirs:
        with open(os.path.join(d, "manifest.json")) as f:
            m = json.load(f)
        man.append(m)
        weights = parse_metric(metric, m["metrics"][0])
        missing = [n for n in weights if n not in m["metrics"]]
        if missing:
            raise SystemExit(f"{d} has no scores for {missing} (it has {m['metrics']})")
        for p in sorted(glob.glob(os.path.join(d, "shard_*.npz"))):
            with np.load(p) as z:
                zz = {k: z[k] for k in z.files}
                s, v = combined(zz, weights, "score__"), zz["cand_vecs"]
                floor = combined(zz, weights, "noise__")
                floor = np.zeros(len(s)) if floor is None else floor
                audio.append(zz["target_audio"])
                midi.append(zz["midi"])
                true.append(zz["true_vecs"] if "true_vecs" in zz else np.full(v[:, 0].shape, np.nan, np.float32))
            for i in range(len(s)):
                thr = margin + noise_mult * float(floor[i])
                order = np.argsort(s[i])
                if mode == "best_worst":
                    cands = [(order[0], order[-1])]
                else:
                    cands = sorted(((a, b) for ai, a in enumerate(order) for b in order[ai + 1:]),
                                   key=lambda ab: s[i][ab[0]] - s[i][ab[1]])[:max_per_cond]
                for a, b in cands:
                    if np.isfinite(s[i][a]) and np.isfinite(s[i][b]) and s[i][b] - s[i][a] > thr:
                        cond.append(offset + i)
                        w.append(v[i, a])
                        l.append(v[i, b])
                        gap.append(s[i][b] - s[i][a])
            offset += len(s)
    if not cond:
        raise SystemExit("no pair passes the margin: lower --margin / --noise_mult or check the pairs' headroom")
    t = np.concatenate(true)
    return {"cond_audio": np.concatenate(audio), "midi": np.concatenate(midi), "true": None if np.isnan(t).all() else t,
            "cond": np.array(cond), "w": np.stack(w).astype(np.float32), "l": np.stack(l).astype(np.float32),
            "gap": np.array(gap, np.float32), "manifests": man}


def split_conditions(n_cond, val_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_cond)
    n_val = int(round(val_frac * n_cond))
    return set(perm[n_val:].tolist()), set(perm[:n_val].tolist())


# ------------------------------------------------------------------------------------------------------ loss
def _err(v, u, w_enc):
    return ((w_enc * (v - u) ** 2).sum(dim=1)) / w_enc.sum()


def dpo_terms(policy, ref, mel, vec_w, vec_l, w_enc, beta, gen, tau=1.0, codec=None):
    """-> (loss, stats). mel [B, 128, 81]; vec_* [B, P] in the ordinal storage layout of the policy's layout."""
    codec = codec or layout_for_flow(policy).codec
    x1w, x1l = codec.encode(vec_w), codec.encode(vec_l)
    B, D = x1w.shape
    dev = x1w.device
    x0 = torch.randn(B, D, device=dev, generator=gen)
    t = torch.rand(B, device=dev, generator=gen)
    xtw, uw = policy.path(x0, x1w, t)
    xtl, ul = policy.path(x0, x1l, t)
    tau_t = None
    m = mel
    if getattr(policy, "cond_noise", False):
        tau_t = torch.full((2 * B,), float(tau), device=dev)
        if float(tau) < 1.0:
            m = policy.noisy(mel, tau_t[:B], generator=gen)      # one noise draw, shared by winner, loser, policy, ref
    xt, u, tt, mm = torch.cat([xtw, xtl]), torch.cat([uw, ul]), torch.cat([t, t]), torch.cat([m, m])
    note = torch.cat([vec_w[:, 0], vec_l[:, 0]])          # storage index 0; used by note-conditioned (v6) flows only
    ep = _err(flow_call(policy, xt, tt, mel=mm, tau=tau_t, note=note), u, w_enc)
    with torch.no_grad():
        er = _err(flow_call(ref, xt, tt, mel=mm, tau=tau_t, note=note), u, w_enc)
    dw, dl = ep[:B] - er[:B], ep[B:] - er[B:]
    delta = dw - dl
    loss = -F.logsigmoid(-beta * delta).mean()
    stats = {"loss": float(loss.detach()), "acc": float((delta < 0).float().mean().detach()),
             "delta": float(delta.detach().mean()), "dw": float(dw.detach().mean()), "dl": float(dl.detach().mean())}
    return loss, stats


def sft_term(policy, mel, vec, w_enc, gen, loss_type="mae", codec=None):
    codec = codec or layout_for_flow(policy).codec
    x1 = codec.encode(vec)
    x0 = torch.randn(x1.shape, device=x1.device, generator=gen)
    t = torch.rand(x1.shape[0], device=x1.device, generator=gen)
    xt, u = policy.path(x0, x1, t)
    tau_t = torch.ones(x1.shape[0], device=x1.device) if getattr(policy, "cond_noise", False) else None
    v = flow_call(policy, xt, t, mel=mel, tau=tau_t, note=vec[:, 0])
    err = (v - u).abs() if loss_type == "mae" else (v - u) ** 2
    return ((w_enc * err).sum(dim=1) / w_enc.sum()).mean()


# ---------------------------------------------------------------------------------------------------- export
def export(policy, src_ckpt, path, dpo_meta, step=None):
    codec = layout_for_flow(policy).codec
    obj = {"model_state": policy.state_dict(), "prior_bounds": src_ckpt.get("prior_bounds"),
           "args": {"model_type": "flow", "param_encoding": codec.ENCODING_V2, "time_scale": 1000.0,
                    "v4_config": getattr(policy, "v4_config", None)},
           "step": src_ckpt.get("step") if step is None else step, "prior": src_ckpt.get("prior"),
           "prior_md5": src_ckpt.get("prior_md5"), "dpo": dpo_meta}
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def link_run(out, run, kind, tag, policy, src_ckpt, dpo_meta):
    from heldout_eval_v4 import KIND_FILES
    jp = os.path.join(run, KIND_FILES[kind].format(m="jepa", tag=tag))
    jstep = torch.load(jp, map_location="cpu", weights_only=False).get("step")
    shutil.copyfile(jp, os.path.join(out, "jepa_online_latest.pt"))
    export(policy, src_ckpt, os.path.join(out, "flow_online_latest.pt"), dpo_meta, step=jstep)
    meta_path = os.path.join(run, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"source_run": run}
    meta["purpose"] = f"Flow-DPO of {os.path.basename(os.path.normpath(run))} {kind}/{tag}: {dpo_meta}"
    meta["run"] = os.path.basename(os.path.normpath(out))
    with open(os.path.join(out, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ------------------------------------------------------------------------------------------------------ main
def train(a, log=print):
    from flow_sampling_v5 import file_md5, mel_of
    from models_v4 import load_inverter_v4
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    src = torch.load(a.flow, map_location="cpu", weights_only=False)
    policy = load_inverter_v4(a.flow, device=a.device)
    layout = layout_for_flow(policy)
    codec = layout.codec
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.eval()                                                # BatchNorm frozen; weights still train
    for p in policy.parameters():
        p.requires_grad_(True)
    data = load_pairs(a.pairs, a.metric, a.pair_mode, a.margin, a.noise_mult, a.max_pairs_per_cond)
    md5 = file_md5(a.flow)
    for m in data["manifests"]:
        if m.get("flow_md5") != md5 and not a.allow_offpolicy:
            raise SystemExit(f"pairs were sampled from {m.get('flow')} (md5 {m.get('flow_md5')}), not from --flow "
                             f"(md5 {md5}); pass --allow_offpolicy to train on them anyway")
        if abs(float(m.get("tau", 1.0)) - a.tau) > 1e-9:
            log(f"warning: pairs sampled at tau {m.get('tau')}, DPO conditions at tau {a.tau}")
    n_cond = len(data["cond_audio"])
    tr_c, va_c = split_conditions(n_cond, a.val_frac, a.seed)
    is_tr = np.array([c in tr_c for c in data["cond"]])
    tr_idx, va_idx = np.where(is_tr)[0], np.where(~is_tr)[0]
    log(f"{len(data['cond'])} pairs from {n_cond} conditions ({len(tr_idx)} train / {len(va_idx)} validation pairs); "
        f"median score gap {np.median(data['gap']):.4f}")
    mel_all = mel_of(data["cond_audio"].astype(np.float32) / 32767.0, dev).to(dev)
    w_enc = codec.encoded_weights(torch.from_numpy(layout.spec.domain_weights()).to(dev))
    log(f"layout {layout.name} ({layout.encoded_dim}-d); ranking by "
        f"{parse_metric(a.metric, data['manifests'][0]['metrics'][0])}")
    W = torch.from_numpy(data["w"]).to(dev)
    L = torch.from_numpy(data["l"]).to(dev)
    C = torch.from_numpy(data["cond"]).long().to(dev)
    T = torch.from_numpy(data["true"]).to(dev) if data["true"] is not None else None
    if a.sft_weight > 0 and a.sft_target == "true" and T is None:
        raise SystemExit("--sft_target true needs synthetic pairs (they carry the true vectors)")
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    meta = {"flow": a.flow, "flow_md5": md5, "layout": layout.name, "metric": a.metric, "pairs": a.pairs, "beta": a.beta, "lr": a.lr, "batch": a.batch,
            "pair_mode": a.pair_mode, "margin": a.margin, "noise_mult": a.noise_mult, "sft_weight": a.sft_weight,
            "sft_target": a.sft_target, "tau": a.tau, "seed": a.seed}
    logf = open(os.path.join(a.out, "dpo_log.jsonl"), "a")

    def validate():
        if len(va_idx) == 0:
            return {}
        g = torch.Generator(device=dev).manual_seed(999)
        agg, n = {}, 0
        with torch.no_grad():
            for s in range(0, len(va_idx), a.batch):
                b = torch.from_numpy(va_idx[s:s + a.batch]).to(dev)
                _, st = dpo_terms(policy, ref, mel_all[C[b]], W[b], L[b], w_enc, a.beta, g, a.tau, codec)
                for k, v in st.items():
                    agg[k] = agg.get(k, 0.0) + v * len(b)
                n += len(b)
        return {f"val_{k}": v / n for k, v in agg.items()}

    v0 = validate()
    log(f"step 0 (policy = reference): {v0}")
    t0, run = time.time(), {}
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / max(1, a.warmup))
        for g_ in opt.param_groups:
            g_["lr"] = lr
        b = torch.from_numpy(rng.choice(tr_idx, size=min(a.batch, len(tr_idx)), replace=False)).to(dev)
        mel = mel_all[C[b]]
        loss, st = dpo_terms(policy, ref, mel, W[b], L[b], w_enc, a.beta, gen, a.tau, codec)
        if a.sft_weight > 0:
            tgt = W[b] if a.sft_target == "winner" else T[C[b]]
            sft = sft_term(policy, mel, tgt, w_enc, gen, a.sft_loss, codec)
            loss = loss + a.sft_weight * sft
            st["sft"] = float(sft.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), a.clip)
        if torch.isfinite(loss) and torch.isfinite(gn):
            opt.step()
        for k, v in st.items():
            run[k] = run.get(k, 0.0) + v
        if step % a.log_every == 0 or step == a.steps:
            rec = {"step": step, "lr": lr, **{k: v / a.log_every for k, v in run.items()}, "time_s": time.time() - t0}
            if step % a.val_every == 0 or step == a.steps:
                rec.update(validate())
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            log(" | ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
            run = {}
        if a.save_every and step % a.save_every == 0:
            export(policy, src, os.path.join(a.out, f"flow_dpo_step_{step:06d}.pt"), {**meta, "dpo_steps": step})
    meta["dpo_steps"] = a.steps
    export(policy, src, os.path.join(a.out, "flow_dpo_latest.pt"), meta)
    if a.link_run:
        link_run(a.out, a.link_run, a.link_kind, a.link_tag, policy, src, meta)
    logf.close()
    return policy, ref


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow", required=True, help="flow_*.pt the pairs were sampled from (policy init and reference)")
    ap.add_argument("--pairs", nargs="+", required=True, help="dpo_pairs_v5.py output directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default=None, help="score to rank candidates by (default: the pairs' first metric); "
                                                   "a weighted sum is written name=w,name=w")
    ap.add_argument("--pair_mode", choices=("best_worst", "all_margin"), default="best_worst")
    ap.add_argument("--margin", type=float, default=0.0, help="minimum score gap (metric units)")
    ap.add_argument("--noise_mult", type=float, default=1.0, help="plus this many render-noise floors (synthetic pairs)")
    ap.add_argument("--max_pairs_per_cond", type=int, default=8)
    ap.add_argument("--beta", type=float, default=100.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0, help="condition noise level during DPO (match the pairs)")
    ap.add_argument("--sft_weight", type=float, default=0.0)
    ap.add_argument("--sft_target", choices=("winner", "true"), default="winner")
    ap.add_argument("--sft_loss", choices=("mae", "mse"), default="mae")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--allow_offpolicy", action="store_true")
    ap.add_argument("--link_run", default="", help="source run dir: also write a heldout_eval_v4-compatible layout")
    ap.add_argument("--link_kind", default="ema", choices=("sf", "online", "ema"))
    ap.add_argument("--link_tag", default="latest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return ap


if __name__ == "__main__":
    train(build_parser().parse_args())
