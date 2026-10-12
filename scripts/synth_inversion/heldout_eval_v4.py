# Owner: KUANG (created 2026-10-12)
"""Per-item re-evaluation of trained Synth-JEPA + flow runs on fixed held-out sets (EVAL_PARADIGM_V4.md, tier T1).

Why: the trainer logs ONE number per metric per validation (val_kinds.jsonl), so two runs can only be compared by
eye, and the A1/A2/A3 report had to guess the noise. This script re-scores the exported weights on fixed sets and
writes PER-ITEM results: which held-out preset each note came from, the rank of the true match in both retrieval
directions, the flow loss, the per-parameter error of the flow's estimate, the aux-head prediction, and per-ladder
monotonicity. compare_runs_v4.py turns those into paired, preset-clustered comparisons with a replicate gauge.

Sets (all from the HELD-OUT presets, rendered by real Surge exactly as the trainer renders them):
  base     the trainer's own held-out set: run_meta's val_size notes, seed 1, the same RNG stream as
           train_realistic_bass_v3.render_fixed_set(). The recomputed retrieval and flow loss must therefore
           reproduce the logged numbers (parity check below).
  extra    --extra_n notes, --extra_seed: a larger set for statistical power, scored in galleries of --gallery
           notes so chance stays 1/gallery as in the logs.
  ladders  the trainer's held-out ladders (run_meta ladder_val_anchors per axis, seed 3) and an optional larger
           set (--extra_ladder_anchors, --extra_ladder_seed).
Kinds: sf = the Schedule-Free average x (jepa_<tag>.pt / flow_<tag>.pt), online = SF y (*_online_<tag>.pt),
ema (*_ema_<tag>.pt). Only rendering needs Surge; rendered sets are cached as mel spectrograms (--out/sets/).

Parity: when the exported step has a record in the run's val_kinds.jsonl, the base-set r_a2p / r_p2a / flow and
the ladder means are recomputed with the trainer's own chunking and generator seeds and printed beside the logged
values. They agree to float precision only on the device TYPE training used (torch's CPU and CUDA generators draw
different numbers). A mismatch on the same device means the evaluation does not see what training saw: stop and
find out why before comparing anything.

Usage (training machine; renders on CPU, scores on --device):
  python heldout_eval_v4.py --run A1=/run/media/kim/Mantu2/surge_200k_models/real_bass_v3_a1 \\
      --run A2=.../real_bass_v3_a2 --run A3=.../real_bass_v3_a3 --kinds sf online ema --tag latest \\
      --extra_n 2048 --out /run/media/kim/Mantu2/surge_200k_models/eval_v4 --device cuda:0
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import param_codec_v3 as codec  # noqa: E402
from surge_spec_v3 import CAT_INDICES, CONT_INDICES, NOTE_DUR_RANGE, PARAM_INDEX, PARAM_NAMES, domain_weights  # noqa: E402

import torch.nn.functional as F  # noqa: E402

KIND_FILES = {"sf": "{m}_{tag}.pt", "online": "{m}_online_{tag}.pt", "ema": "{m}_ema_{tag}.pt"}
VALIDATE_CHUNK = 128   # train_realistic_bass_v3.validate() default chunk; the generator stream depends on it
VALIDATE_SEED = 1234   # validate(): torch.Generator(device).manual_seed(1234), shared by SIGReg and the flow loss
LADDER_SEED = 4321     # training_controls_v3.ladder_validation()
FORMAT = 1


# ----------------------------------------------------------------------------------------------- item sampling
class PresetRecorder:
    """Proxy for np.random.RandomState. Every call passes through unchanged, so the RNG stream and every drawn
    vector are identical to the trainer's; additionally it records the archetype index each prior draw picks
    (RealisticBassPrior.sample_vector() calls rng.randint(0, N) first)."""

    def __init__(self, rng, n_presets):
        self._rng = rng
        self._n = int(n_presets)
        self._armed = False
        self.last = None

    def arm(self):
        self._armed, self.last = True, None

    def take(self):
        if self.last is None:
            raise RuntimeError("no preset index was recorded: RealisticBassPrior's sampling code changed")
        v, self.last, self._armed = self.last, None, False
        return v

    def randint(self, *args, **kwargs):
        v = self._rng.randint(*args, **kwargs)
        if (self._armed and self.last is None and len(args) >= 2 and args[0] == 0 and args[1] == self._n
                and kwargs.get("size") is None):
            self.last = int(v)
        return v

    def __getattr__(self, name):
        return getattr(self._rng, name)


def draw_items(prior, n, seed):
    """render_fixed_set()'s stream without rendering -> dict of patches, vecs [n, P], midi, dur, preset."""
    rec = PresetRecorder(np.random.RandomState(seed), prior.N)
    patches, vecs, midi, dur, preset = [], [], [], [], []
    for _ in range(n):
        rec.arm()
        patch, vec, m, d = prior.sample_patch_and_midi(rec)
        patches.append(patch)
        vecs.append(vec)
        midi.append(int(m))
        dur.append(float(d))
        preset.append(rec.take())
    return {"patches": patches, "vecs": np.stack(vecs).astype(np.float32), "midi": np.array(midi),
            "dur": np.array(dur, dtype=np.float32), "preset": np.array(preset)}


def verify_stream(prior, seed, vecs):
    """The recorder must not perturb the stream: replay with a plain RandomState and compare every vector."""
    rng = np.random.RandomState(seed)
    for i, v in enumerate(vecs):
        _, ref, _, _ = prior.sample_patch_and_midi(rng)
        if not np.array_equal(np.asarray(ref, dtype=np.float32), v):
            raise RuntimeError(f"recorded stream diverges from the trainer's at item {i}")


def draw_ladders(prior, n_anchors_per_axis, n_steps, seed):
    """training_controls_v3.render_ladder_set()'s stream without rendering, plus each ladder's anchor preset."""
    from surge_spec_v3 import vector_to_patch
    from training_controls_v3 import AXES, make_ladder
    rec = PresetRecorder(np.random.RandomState(seed), prior.N)
    patches, midi, dur, vecs, axis_ids, presets = [], [], [], [], [], []
    for ax, axis in enumerate(AXES):
        for _ in range(n_anchors_per_axis):
            rec.arm()
            _, base, midi_note, _ = prior.sample_patch_and_midi(rec)
            presets.append(rec.take())
            note_dur = float(rec.uniform(*NOTE_DUR_RANGE))
            for vec in make_ladder(base, axis, n_steps, rec):
                patches.append(vector_to_patch(vec))
                midi.append(int(midi_note))
                dur.append(note_dur)
                vecs.append(vec)
                axis_ids.append(ax)
    return {"patches": patches, "vecs": np.stack(vecs).astype(np.float32), "midi": np.array(midi),
            "dur": np.array(dur, dtype=np.float32), "axis": np.array(axis_ids), "preset": np.array(presets),
            "n_steps": int(n_steps)}


def render_items(patches, midi, dur, plugin):
    """Real Surge, exactly as the trainer's fixed sets (render_patch(synth, patch, midi_note, note_dur))."""
    from surge_spec_v3 import init_synth, render_patch
    synth = init_synth(plugin, verify=True)
    return np.stack([render_patch(synth, p, int(m), float(d)) for p, m, d in zip(patches, midi, dur)]).astype(np.float32)


def inharm_labels(audio, midi, config_path):
    """Pooled inharmonicity labels exactly as render_fixed_set() computes them (0 and valid=0 when invalid)."""
    from inharmonicity_target import InharmonicityConfig, InharmonicityExtractor
    ext = InharmonicityExtractor(InharmonicityConfig.from_json(config_path))
    vals, valid = [], []
    for a, m in zip(audio, midi):
        r = ext.process_phrase(a, int(m))
        vals.append(float(r["value"]) if r["valid"] else 0.0)
        valid.append(1.0 if r["valid"] else 0.0)
    return np.array(vals, dtype=np.float32), np.array(valid, dtype=np.float32)


def compute_mel(audio, device, chunk=VALIDATE_CHUNK):
    from audio_utils import ExactGpuMel
    mel_fn = ExactGpuMel(device=device)
    with torch.no_grad():
        return torch.cat([mel_fn(torch.from_numpy(audio[s:s + chunk]).to(device)).float().cpu()
                          for s in range(0, len(audio), chunk)])


def items_md5(vecs, midi, dur):
    h = hashlib.md5()
    for a in (vecs, midi, dur):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def file_md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------- scoring
def prepare_jepa_inputs(params):
    """Copy of train_realistic_bass_v3.prepare_jepa_inputs (kept here so this script imports without Essentia)."""
    cont = params[:, CONT_INDICES] * 2.0 - 1.0
    cat_onehots = [F.one_hot(torch.round(params[:, i] * (k - 1)).clamp(0, k - 1).long(), k).float()
                   for i, k in CAT_INDICES]
    return cont, cat_onehots


def flow_loss_per_item(flow, mel, params, w_enc, generator):
    """train_realistic_bass_v3.flow_loss() as validate() calls it (clean condition, no OT), per item, both
    squared (= the logged held-out 'flow') and absolute error (= the training loss with --flow_loss mae)."""
    x1 = codec.encode(params)
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, generator=generator)
    xt, target = flow.path(x0, x1, t)
    pred = flow(xt, t, mel)
    wsum = w_enc.sum()
    mse = (w_enc * (pred - target) ** 2).sum(dim=1) / wsum
    mae = (w_enc * (pred - target).abs()).sum(dim=1) / wsum
    return mse, mae


class _EvalMode:
    """Score in eval mode whatever the caller left behind (the flow's conv encoder has batch-dependent layers in
    train mode; validate() itself switches back to train mode on exit), then restore the previous modes."""

    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev = [m.training for m in self.models]
        for m in self.models:
            m.eval()

    def __exit__(self, *exc):
        for m, p in zip(self.models, self.prev):
            m.train(p)
        return False


def block_ranks(queries, gallery, block):
    """Rank of the true partner (same index) among the gallery items of its block: 0 = retrieved first.
    rank == 0 reproduces validate()'s argmin == target except on exact distance ties."""
    n = queries.shape[0]
    ranks = np.empty(n, dtype=np.int64)
    sizes = np.empty(n, dtype=np.int64)
    for s in range(0, n, block):
        d = torch.cdist(queries[s:s + block], gallery[s:s + block])
        true = d.diagonal().unsqueeze(1)
        ranks[s:s + block] = (d < true).sum(dim=1).cpu().numpy()
        sizes[s:s + block] = d.shape[1]
    return ranks, sizes


def score_items(jepa, normalizer, flow, mel, vecs, w_enc, device, **kw):
    """Per-item scores (see _score_items). Always in eval mode."""
    with _EvalMode(jepa, flow):
        return _score_items(jepa, normalizer, flow, mel, vecs, w_enc, device, **kw)


@torch.no_grad()
def _score_items(jepa, normalizer, flow, mel, vecs, w_enc, device, gallery=512, sample_draws=8, sample_steps=25,
                 sample_seed=0, chunk=VALIDATE_CHUNK):
    """The JEPA forward and the flow loss consume ONE generator per set in the same order as validate(), so their
    means reproduce the logged values; the flow sampling uses its own generator (sample_seed), identical across
    runs, so runs are compared on common random numbers."""
    n = len(vecs)
    vecs_t = torch.from_numpy(np.asarray(vecs, dtype=np.float32))
    gen = torch.Generator(device=device).manual_seed(VALIDATE_SEED)
    gen_s = torch.Generator(device=device).manual_seed(int(sample_seed))
    has_aux = getattr(jepa, "inharmonicity_head", None) is not None
    za_l, zp_l, mse_l, mae_l, aux_l, mean_l, d0_l, vote_l = [], [], [], [], [], [], [], []
    for s in range(0, n, chunk):
        m = mel[s:s + chunk].to(device)
        p = vecs_t[s:s + chunk].to(device)
        cont, cats = prepare_jepa_inputs(p)
        _, _, _, _, za, zp = jepa(normalizer.normalize(m), cont, cats, sigreg_generator=gen)
        mse, mae = flow_loss_per_item(flow, m, p, w_enc, gen)
        za_l.append(za)
        zp_l.append(zp)
        mse_l.append(mse.float().cpu())
        mae_l.append(mae.float().cpu())
        if has_aux:
            aux_l.append(jepa.inharmonicity_head(za).reshape(-1).float().cpu())
        draws = torch.stack([codec.decode(flow.sample(m, num_steps=sample_steps, generator=gen_s))
                             for _ in range(sample_draws)]).float().cpu()          # [K, B, P]
        mean_l.append(draws.mean(0))
        d0_l.append(draws[0])
        votes = []
        for i, k in CAT_INDICES:
            cls = torch.round(draws[:, :, i] * (k - 1)).clamp(0, k - 1).long()      # [K, B]
            votes.append(torch.mode(cls, dim=0).values)
        vote_l.append(torch.stack(votes, dim=1))                                     # [B, n_cat]
    za = torch.cat(za_l)
    zp = torch.cat(zp_l)
    rank_a2p, gal = block_ranks(jepa.f_a2p(za), zp, gallery)
    rank_p2a, _ = block_ranks(jepa.f_p2a(zp), za, gallery)
    est = torch.cat(mean_l).numpy()
    d0 = torch.cat(d0_l).numpy()
    true = vecs_t.numpy()
    true_cls = np.stack([np.clip(np.round(true[:, i] * (k - 1)), 0, k - 1) for i, k in CAT_INDICES], axis=1)
    out = {
        "rank_a2p": rank_a2p, "rank_p2a": rank_p2a, "gallery": gal,
        "flow_mse": torch.cat(mse_l).numpy(), "flow_mae": torch.cat(mae_l).numpy(),
        "cont_idx": np.array(CONT_INDICES), "cat_idx": np.array([i for i, _ in CAT_INDICES]),
        "cont_abs_err": np.abs(est[:, CONT_INDICES] - true[:, CONT_INDICES]).astype(np.float32),
        "cont_abs_err_draw0": np.abs(d0[:, CONT_INDICES] - true[:, CONT_INDICES]).astype(np.float32),
        "cat_correct": (torch.cat(vote_l).numpy() == true_cls).astype(np.float32),
    }
    out["aux_pred"] = torch.cat(aux_l).numpy() if has_aux else np.full(n, np.nan, dtype=np.float32)
    return out


def score_ladders(jepa, normalizer, flow, mel, vecs, axis_ids, n_steps, device, **kw):
    """Per-ladder scores (see _score_ladders). Always in eval mode."""
    with _EvalMode(jepa, flow):
        return _score_ladders(jepa, normalizer, flow, mel, vecs, axis_ids, n_steps, device, **kw)


@torch.no_grad()
def _score_ladders(jepa, normalizer, flow, mel, vecs, axis_ids, n_steps, device, flow_draws=8, flow_steps=12,
                   chunk=VALIDATE_CHUNK):
    """training_controls_v3.ladder_validation() without the averaging: one row per ladder (same chunking and
    generator, so the per-axis means reproduce the logged ladder numbers)."""
    from scipy.stats import spearmanr
    from training_controls_v3 import AXES, AXIS_PARAM
    gen = torch.Generator(device=device).manual_seed(LADDER_SEED)
    za_all, est_all = [], []
    for s in range(0, len(vecs), chunk):
        m = mel[s:s + chunk].to(device)
        za_all.append(jepa.encode_audio(normalizer.normalize(m)))
        draws = torch.stack([codec.decode(flow.sample(m, num_steps=flow_steps, generator=gen))
                             for _ in range(flow_draws)])
        est_all.append(draws.mean(0))
    za = torch.cat(za_all).view(-1, n_steps, za_all[0].shape[-1])
    est = torch.cat(est_all).float().cpu().numpy().reshape(-1, n_steps, vecs.shape[-1])
    true = np.asarray(vecs).reshape(-1, n_steps, vecs.shape[-1])
    ax_of = np.asarray(axis_ids).reshape(-1, n_steps)[:, 0]
    dist = (za[:, 1:] - za[:, :1]).norm(dim=-1).float().cpu().numpy()
    j_rho, f_rho, f_mae = [], [], []
    for r in range(len(ax_of)):
        k = PARAM_INDEX[AXIS_PARAM[AXES[int(ax_of[r])]]]
        j_rho.append(spearmanr(np.arange(1, n_steps), dist[r]).correlation)
        f_rho.append(spearmanr(true[r, :, k], est[r, :, k]).correlation)
        f_mae.append(float(np.mean(np.abs(true[r, :, k] - est[r, :, k]))))
    return {"axis": ax_of, "jepa_rho": np.array(j_rho, dtype=np.float64), "flow_rho": np.array(f_rho, dtype=np.float64),
            "flow_mae": np.array(f_mae, dtype=np.float64)}


def ladder_means(lad):
    """The logged summary: per-axis nanmean, then the nanmean over axes."""
    from training_controls_v3 import AXES
    out = {}
    for name in ("flow_rho", "jepa_rho"):
        per_axis = [np.nanmean(lad[name][lad["axis"] == a]) for a in range(len(AXES)) if np.any(lad["axis"] == a)]
        out[f"ladder_{name}_mean"] = float(np.nanmean(per_axis))
    return out


# --------------------------------------------------------------------------------------------- run handling
def read_run_meta(run_dir):
    path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: needed for the run's manifold, plugin and validation settings")
    with open(path) as f:
        return json.load(f)


def load_kind(run_dir, kind, tag, device):
    """(jepa, normalizer, flow, info) for one weight kind, or None if that export does not exist."""
    from models_v4 import load_inverter_v4
    from synth_jepa_search_v3 import load_synth_jepa
    jp = os.path.join(run_dir, KIND_FILES[kind].format(m="jepa", tag=tag))
    fp = os.path.join(run_dir, KIND_FILES[kind].format(m="flow", tag=tag))
    if not (os.path.exists(jp) and os.path.exists(fp)):
        return None
    jepa, normalizer = load_synth_jepa(jp, device=device)
    flow = load_inverter_v4(fp, device=device)
    meta_j = torch.load(jp, map_location="cpu", weights_only=False)
    meta_f = torch.load(fp, map_location="cpu", weights_only=False)
    if meta_j.get("step") != meta_f.get("step"):
        raise RuntimeError(f"{run_dir} {kind}/{tag}: JEPA step {meta_j.get('step')} != flow step {meta_f.get('step')}")
    info = {"jepa": jp, "flow": fp, "step": meta_j.get("step"), "prior_md5": meta_j.get("prior_md5"),
            "jepa_md5": file_md5(jp), "flow_md5": file_md5(fp), "flow_class": type(flow).__name__,
            "flow_encoder": getattr(flow, "v4_config", {}).get("encoder", "conv") if hasattr(flow, "v4_config") else "conv",
            "aux_head": getattr(jepa, "inharmonicity_head", None) is not None}
    return jepa, normalizer, flow, info


def logged_record(run_dir, step):
    """Last val_kinds.jsonl record at `step`, or None."""
    path = os.path.join(run_dir, "val_kinds.jsonl")
    if step is None or not os.path.exists(path):
        return None
    rec = None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("step") == step:
                rec = r
    return rec


def build_set(path, draw, verify, plugin, device, inharm_config):
    """Render (or load the cached) set; the cache is valid only if the drawn items are byte-identical."""
    key = items_md5(draw["vecs"], draw["midi"], draw["dur"])
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if str(z["items_md5"]) == key:
            return {k: z[k] for k in z.files}
        print(f"  cache {path} is for different items: re-rendering")
    verify()
    t0 = time.time()
    audio = render_items(draw["patches"], draw["midi"], draw["dur"], plugin)
    print(f"  rendered {len(audio)} notes in {time.time() - t0:.1f} s")
    rec = {"mel": compute_mel(audio, device).numpy(), "vecs": draw["vecs"], "midi": draw["midi"], "dur": draw["dur"],
           "preset": draw["preset"], "items_md5": np.array(key)}
    if "axis" in draw:
        rec["axis"] = draw["axis"]
        rec["n_steps"] = np.array(draw["n_steps"])
    if inharm_config:
        rec["inharm"], rec["inharm_valid"] = inharm_labels(audio, draw["midi"], inharm_config)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **rec)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (repeatable)")
    ap.add_argument("--kinds", nargs="+", default=["sf", "online", "ema"], choices=sorted(KIND_FILES))
    ap.add_argument("--tag", default="latest", help="export tag: latest | best (online_best / ema_best)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--manifold", default=None, help="default: from run_meta (must be the same for every run)")
    ap.add_argument("--plugin", default=None, help="default: from run_meta")
    ap.add_argument("--inharm_config", default=None, help="default: from the first run that used the aux head")
    ap.add_argument("--extra_n", type=int, default=2048, help="0 = skip the larger set")
    ap.add_argument("--extra_seed", type=int, default=101)
    ap.add_argument("--gallery", type=int, default=512)
    ap.add_argument("--extra_ladder_anchors", type=int, default=32, help="per axis; 0 = skip")
    ap.add_argument("--extra_ladder_seed", type=int, default=13)
    ap.add_argument("--sample_draws", type=int, default=8)
    ap.add_argument("--sample_steps", type=int, default=25)
    ap.add_argument("--sample_seed", type=int, default=0)
    a = ap.parse_args(argv)

    from realistic_bass_prior_v3 import RealisticBassPrior, manifold_md5

    runs = {}
    for spec in a.run:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--run {spec!r}: expected NAME=RUN_DIR")
        runs[name] = (path, read_run_meta(path))
    metas = [m for _, m in runs.values()]
    manifold = a.manifold or metas[0]["args"]["manifold"]
    plugin = a.plugin or metas[0]["args"]["plugin"]
    md5 = manifold_md5(manifold)
    for name, (_, m) in runs.items():
        if m["prior"]["md5"] != md5:
            raise SystemExit(f"{name} was trained on a different prior (md5 {m['prior']['md5'][:10]} vs {md5[:10]})")
    val_size = {m["args"].get("val_size", 512) for m in metas}
    lad_anchors = {m["args"].get("ladder_val_anchors", 0) for m in metas}
    lad_steps = {m["args"].get("ladder_steps", 16) for m in metas}
    if len(val_size) > 1 or len(lad_anchors) > 1 or len(lad_steps) > 1:
        raise SystemExit(f"runs differ in validation settings (val_size {val_size}, ladder anchors {lad_anchors}, "
                         f"steps {lad_steps}): their logged numbers are not on the same sets")
    val_size, lad_anchors, lad_steps = val_size.pop(), lad_anchors.pop(), lad_steps.pop()
    inharm_config = a.inharm_config or next((m["args"].get("inharm_config") for m in metas
                                             if float(m["args"].get("inharm_weight", 0) or 0) > 0), None)
    if inharm_config and not os.path.isabs(inharm_config):
        cand = [os.path.join(os.getcwd(), inharm_config), os.path.join(SCRIPT_DIR, os.path.basename(inharm_config))]
        inharm_config = next((c for c in cand if os.path.exists(c)), inharm_config)

    prior = RealisticBassPrior(manifold, split="val")
    print(f"prior {manifold} (md5 {md5[:10]}): {prior.N} held-out presets")
    sets_dir = os.path.join(a.out, "sets")
    sets = {}
    specs = [("base", "items", val_size, 1)]
    if a.extra_n > 0:
        specs.append(("extra", "items", a.extra_n, a.extra_seed))
    if lad_anchors > 0:
        specs.append(("ladders", "ladders", lad_anchors, 3))
    if a.extra_ladder_anchors > 0:
        specs.append(("ladders_extra", "ladders", a.extra_ladder_anchors, a.extra_ladder_seed))
    for name, kind, n, seed in specs:
        if kind == "items":
            draw = draw_items(prior, n, seed)

            def verify(draw=draw, seed=seed):
                verify_stream(prior, seed, draw["vecs"])
        else:
            draw = draw_ladders(prior, n, lad_steps, seed)

            def verify():
                pass
        path = os.path.join(sets_dir, f"{name}__md5{md5[:10]}__n{n}__seed{seed}.npz")
        print(f"set {name}: {len(draw['vecs'])} notes from {len(np.unique(draw['preset']))} held-out presets")
        sets[name] = (kind, build_set(path, draw, verify, plugin, a.device,
                                      inharm_config if kind == "items" else None))

    w_enc = codec.encoded_weights(torch.from_numpy(domain_weights()).to(a.device))
    os.makedirs(a.out, exist_ok=True)
    for rname, (rdir, meta) in runs.items():
        for kind in a.kinds:
            loaded = load_kind(rdir, kind, a.tag, a.device)
            if loaded is None:
                print(f"{rname}/{kind}: no {a.tag} export, skipped")
                continue
            jepa, normalizer, flow, info = loaded
            print(f"{rname}/{kind}: step {info['step']}, flow {info['flow_class']}, aux head {info['aux_head']}")
            parity = {}
            for sname, (skind, s) in sets.items():
                mel = torch.from_numpy(s["mel"])
                if skind == "items":
                    res = score_items(jepa, normalizer, flow, mel, s["vecs"], w_enc, a.device, gallery=a.gallery,
                                      sample_draws=a.sample_draws, sample_steps=a.sample_steps,
                                      sample_seed=a.sample_seed)
                    res.update({"preset": s["preset"], "midi": s["midi"]})
                    if "inharm" in s:
                        res.update({"inharm": s["inharm"], "inharm_valid": s["inharm_valid"]})
                    if sname == "base":
                        parity.update({"r_a2p": float(np.mean(res["rank_a2p"] == 0)),
                                       "r_p2a": float(np.mean(res["rank_p2a"] == 0)),
                                       "flow": float(np.mean(res["flow_mse"]))})
                else:
                    res = score_ladders(jepa, normalizer, flow, mel, s["vecs"], s["axis"], int(s["n_steps"]), a.device)
                    res["preset"] = s["preset"]
                    if sname == "ladders":
                        parity.update(ladder_means(res))
                res["items_md5"] = np.array(str(s["items_md5"]))
                np.savez(os.path.join(a.out, f"{sname}__{rname}__{kind}.npz"), **res)
            rec = logged_record(rdir, info["step"])
            check = {}
            if rec:
                for k, v in parity.items():
                    logged = rec.get(f"{kind}/heldout/{k}")
                    if logged is not None:
                        check[k] = {"logged": logged, "recomputed": v, "abs_diff": abs(logged - v)}
                for k, c in check.items():
                    print(f"  parity {k:24s} logged {c['logged']:.5f}  recomputed {c['recomputed']:.5f}")
            else:
                print(f"  parity: no val_kinds.jsonl record at step {info['step']} (export and validation steps differ)")
            with open(os.path.join(a.out, f"meta__{rname}__{kind}.json"), "w") as f:
                json.dump({"format": FORMAT, "run": rname, "kind": kind, "tag": a.tag, "run_dir": rdir,
                           "device": a.device, "info": info, "parity": check, "manifold_md5": md5,
                           "run_args": {k: meta["args"].get(k) for k in
                                        ("seed", "batch_size", "max_steps", "hours", "flow_encoder", "flow_film",
                                         "inharm_weight", "cond_noise", "flow_loss", "lr_jepa", "lr_flow",
                                         "lr_floor", "decay_frac", "num_workers", "ema_halflife_steps")},
                           "sets": {k: str(v[1]["items_md5"]) for k, v in sets.items()},
                           "sampling": {"draws": a.sample_draws, "steps": a.sample_steps, "seed": a.sample_seed},
                           "param_names": list(PARAM_NAMES)}, f, indent=2)
    print(f"wrote per-item results to {a.out}; next: compare_runs_v4.py --dir {a.out} --plan PLAN.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
