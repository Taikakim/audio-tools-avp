# Owner: KUANG (created 2026-10-13)
"""Pre-rendered training data shared by every arm of an experiment (V5_PROPOSAL_REVIEW.md, sec. 5.3).

Why: the trainer renders every training clip live in DataLoader workers (pedalboard + Surge VST3). Rendering the same
stream once and reading it back (1) removes Surge from the training loop, so a smaller model is not left waiting on
the CPU, (2) gives every arm the SAME data in the SAME order (common random numbers: a seed alone does not, because
arms build different modules before the DataLoader draws its base seed), and (3) frees the CPU for rendering
evaluation candidates while training runs.

What is stored (int16 audio, peak-normalised mono as rendered; the mel is computed on the GPU as in live training,
so the front end can still change):
  main_NNNNN_audio.npy  int16 [n, 35280]       main_NNNNN_meta.npz  vec [n, P], midi [n], inharm [n], inharm_valid [n],
                                                                     random_patch [n]
  ladder_NNNNN_audio.npy int16 [L, S, 35280]   ladder_NNNNN_meta.npz vec [L, S, P], midi [L], axis [L]
  manifest.json         prior md5, renderer, seeds, counts, the inharmonicity config md5
Item i of the main stream is drawn from its own RandomState(SeedSequence([seed, 0, i])), ladder j from
SeedSequence([seed, 1, j]), so a bank is reproducible, resumable shard by shard, and independent of worker scheduling.
Main items follow RealisticOnlineDataset (prior draw, or an unconstrained random patch with probability
random_patch_frac); ladders follow LadderDataset (one prior anchor, one random axis, make_ladder).

Usage (desktop):
  python render_bank_v5.py build --manifold M.npz --plugin "Surge XT.vst3" --out /data/bank_v5_s0 \\
      --n_items 786432 --n_ladders 12288 --workers 14 [--inharm_config inharm_config.json]
  python render_bank_v5.py verify --bank /data/bank_v5_s0 --n 32      # re-render a sample, compare with the bank
  train_realistic_bass_v3.py ... --bank /data/bank_v5_s0              # the trainer reads it instead of Surge
Sizing: 12 000 steps x 64 with a ladder batch every 4th step uses 9 000 x 64 = 576 000 main items (+ 16 000 for the
mel normaliser) and 3 000 ladder batches x 4 anchors = 12 000 ladders, without repeating anything. 786 432 items
(192 shards) are ~55 GB and 12 288 ladders of 16 steps ~14 GB. Longer runs cycle through the bank in new orders.

Status: CPU-tested with a fake renderer (reproducibility, resume, dataset shapes and order). NOT run on Surge here.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

FORMAT = 1


def item_rng(seed, stream, i):
    return np.random.RandomState(int(np.random.SeedSequence([int(seed), int(stream), int(i)]).generate_state(1)[0]))


def file_md5(path):
    if not path:
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def draw_main_item(prior, i, seed, random_patch_frac):
    rng = item_rng(seed, 0, i)
    if random_patch_frac > 0.0 and rng.rand() < random_patch_frac:
        from train_realistic_bass_v3 import sample_unconstrained_random_patch
        _, vec, midi, _ = sample_unconstrained_random_patch(rng)
        return np.asarray(vec, np.float32), int(midi), True
    _, vec, midi, _ = prior.sample_patch_and_midi(rng)
    return np.asarray(vec, np.float32), int(midi), False


def draw_ladder(prior, j, seed, n_steps):
    from surge_spec_v3 import NOTE_DUR_RANGE
    from training_controls_v3 import AXES, make_ladder
    rng = item_rng(seed, 1, j)
    _, base, midi, _ = prior.sample_patch_and_midi(rng)
    rng.uniform(*NOTE_DUR_RANGE)              # LadderDataset's draw order (the renderer ignores note_dur)
    ax = int(rng.randint(len(AXES)))
    return make_ladder(base, AXES[ax], n_steps, rng).astype(np.float32), int(midi), ax


def _paths(bank, kind, k):
    return (os.path.join(bank, f"{kind}_{k:05d}_audio.npy"), os.path.join(bank, f"{kind}_{k:05d}_meta.npz"))


def _save_shard(bank, kind, k, audio16, **meta):
    pa, pm = _paths(bank, kind, k)
    np.save(pa + ".tmp.npy", audio16)
    np.savez(pm + ".tmp.npz", **meta)
    os.replace(pm + ".tmp.npz", pm)          # meta first: a shard counts as done only when the audio is in place
    os.replace(pa + ".tmp.npy", pa)


def build(a, prior=None, pool=None, log=print):
    from render_pool_v5 import RenderPool, to_int16
    from surge_spec_v3 import AUDIO_LEN, FX_RENDER_ENABLED, SAMPLE_RATE
    os.makedirs(a.out, exist_ok=True)
    if prior is None:
        from realistic_bass_prior_v3 import RealisticBassPrior
        prior = RealisticBassPrior(a.manifold, split="train")
    from realistic_bass_prior_v3 import manifold_md5
    cfg = {"format": FORMAT, "prior_md5": manifold_md5(a.manifold) if a.manifold else None,
           "manifold": a.manifold, "split": "train", "seed": a.seed, "random_patch_frac": a.random_patch_frac,
           "n_items": a.n_items, "shard_size": a.shard_size, "n_ladders": a.n_ladders, "ladder_steps": a.ladder_steps,
           "ladder_shard": a.ladder_shard, "inharm_config": a.inharm_config, "inharm_config_md5": file_md5(a.inharm_config),
           "renderer": "pedalboard-vst3" if pool is None else "custom", "plugin": a.plugin,
           "fx_render_enabled": bool(FX_RENDER_ENABLED), "sample_rate": SAMPLE_RATE, "audio_len": AUDIO_LEN}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in cfg.items() if k not in ("created",) and old.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} holds a bank with a different configuration: {diff}")
        log(f"resuming the bank in {a.out}")
    else:
        with open(mpath, "w") as f:
            json.dump({**cfg, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    labels = bool(a.inharm_config)
    ctx = pool if pool is not None else RenderPool(a.plugin, workers=a.workers, inharm_config=a.inharm_config)
    n_main = math.ceil(a.n_items / a.shard_size)
    n_lad = math.ceil(a.n_ladders / a.ladder_shard) if a.n_ladders else 0
    t0, done = time.time(), 0
    with ctx as p:
        for k in range(n_main):
            if os.path.exists(_paths(a.out, "main", k)[0]):
                continue
            idx = range(k * a.shard_size, min((k + 1) * a.shard_size, a.n_items))
            draws = [draw_main_item(prior, i, a.seed, a.random_patch_frac) for i in idx]
            vecs = np.stack([d[0] for d in draws])
            midi = np.array([d[1] for d in draws], np.int16)
            if labels:
                audio, inh, val = p.render(vecs, midi, labels=True)
            else:
                audio, inh, val = p.render(vecs, midi), np.zeros(len(idx), np.float32), np.zeros(len(idx), np.float32)
            _save_shard(a.out, "main", k, to_int16(audio), vec=vecs, midi=midi, inharm=inh, inharm_valid=val,
                        random_patch=np.array([d[2] for d in draws]))
            done += len(idx)
            el = time.time() - t0
            log(f"main shard {k + 1}/{n_main}: {done} clips in {el:.0f} s ({done / max(el, 1e-9):.1f}/s)")
        for k in range(n_lad):
            if os.path.exists(_paths(a.out, "ladder", k)[0]):
                continue
            idx = range(k * a.ladder_shard, min((k + 1) * a.ladder_shard, a.n_ladders))
            draws = [draw_ladder(prior, j, a.seed, a.ladder_steps) for j in idx]
            vecs = np.stack([d[0] for d in draws])                       # [L, S, P]
            midi = np.array([d[1] for d in draws], np.int16)
            L, S, P = vecs.shape
            audio = p.render(vecs.reshape(L * S, P), np.repeat(midi, S)).reshape(L, S, -1)
            _save_shard(a.out, "ladder", k, to_int16(audio), vec=vecs, midi=midi, axis=np.array([d[2] for d in draws]))
            done += L * S
            el = time.time() - t0
            log(f"ladder shard {k + 1}/{n_lad}: {done} clips in total, {done / max(el, 1e-9):.1f}/s")
    return 0


def read_manifest(bank):
    with open(os.path.join(bank, "manifest.json")) as f:
        return json.load(f)


class _Shards:
    """Lazily memory-maps audio shards (per process: DataLoader workers open their own maps)."""

    def __init__(self, bank, kind, n_shards):
        self.bank, self.kind, self.n = bank, kind, n_shards
        self._audio, self._meta = {}, {}

    def get(self, k):
        if k not in self._audio:
            pa, pm = _paths(self.bank, self.kind, k)
            if not os.path.exists(pa):
                raise FileNotFoundError(f"{pa} missing: the bank is incomplete (run build again to resume)")
            self._audio[k] = np.load(pa, mmap_mode="r")
            with np.load(pm) as m:
                self._meta[k] = {key: m[key] for key in m.files}
        return self._audio[k], self._meta[k]


class BankMainDataset:
    """Map-style dataset with RealisticOnlineDataset's item format: (audio [N] float32, vec [P], inharm, valid)."""

    def __init__(self, bank):
        m = read_manifest(bank)
        self.n, self.shard = int(m["n_items"]), int(m["shard_size"])
        self.manifest = m
        self.sh = _Shards(bank, "main", math.ceil(self.n / self.shard))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        import torch
        k, j = divmod(int(i), self.shard)
        audio, meta = self.sh.get(k)
        a = np.asarray(audio[j], dtype=np.float32) / 32767.0
        return (torch.from_numpy(a), torch.from_numpy(np.asarray(meta["vec"][j], np.float32)),
                float(meta["inharm"][j]), float(meta["inharm_valid"][j]))


class BankLadderDataset:
    """Item b = ladders [b * n_anchors, (b + 1) * n_anchors) as one ladder batch, in LadderDataset's format:
    (audio [n_anchors * S, N], vecs [n_anchors * S, P], axis ids [n_anchors * S])."""

    def __init__(self, bank, n_anchors):
        m = read_manifest(bank)
        self.n_ladders, self.shard, self.n_steps = int(m["n_ladders"]), int(m["ladder_shard"]), int(m["ladder_steps"])
        self.n_anchors = int(n_anchors)
        self.sh = _Shards(bank, "ladder", math.ceil(self.n_ladders / self.shard) if self.n_ladders else 0)

    def __len__(self):
        return self.n_ladders // self.n_anchors

    def __getitem__(self, b):
        import torch
        audio, vecs, axis = [], [], []
        for j in range(b * self.n_anchors, (b + 1) * self.n_anchors):
            k, r = divmod(j, self.shard)
            au, meta = self.sh.get(k)
            audio.append(np.asarray(au[r], dtype=np.float32) / 32767.0)
            vecs.append(meta["vec"][r])
            axis.append(np.full(self.n_steps, int(meta["axis"][r])))
        return (torch.from_numpy(np.concatenate(audio)), torch.from_numpy(np.concatenate(vecs).astype(np.float32)),
                torch.from_numpy(np.concatenate(axis)))


def check_compatible(manifest, prior_md5, inharm_weight, inharm_config, ladder_steps, ladder_every, random_patch_frac):
    """Problems that make a bank the wrong data for this run (empty list = fine)."""
    probs = []
    if manifest.get("format") != FORMAT:
        probs.append(f"bank format {manifest.get('format')} != {FORMAT}")
    if manifest.get("prior_md5") != prior_md5:
        probs.append(f"bank prior md5 {manifest.get('prior_md5')} != run prior md5 {prior_md5}")
    if inharm_weight > 0 and not manifest.get("inharm_config_md5"):
        probs.append("the aux head needs inharmonicity labels; this bank was built without --inharm_config")
    elif inharm_weight > 0 and manifest.get("inharm_config_md5") != file_md5(inharm_config):
        probs.append("the aux head needs labels from the same inharmonicity config; the bank has "
                     f"{manifest.get('inharm_config')} (md5 {manifest.get('inharm_config_md5')})")
    if ladder_every > 0 and (manifest.get("n_ladders", 0) == 0 or manifest.get("ladder_steps") != ladder_steps):
        probs.append(f"ladder batches need ladders with {ladder_steps} steps; the bank has "
                     f"{manifest.get('n_ladders')} ladders of {manifest.get('ladder_steps')} steps")
    if abs(float(manifest.get("random_patch_frac", -1)) - float(random_patch_frac)) > 1e-9:
        probs.append(f"bank random_patch_frac {manifest.get('random_patch_frac')} != run {random_patch_frac}")
    return probs


def cycle(loader):
    """Endless iterator over a DataLoader; each pass draws a new (seeded) permutation from its sampler."""
    while True:
        for batch in loader:
            yield batch


def verify(a, prior=None, pool=None):
    """Re-render a sample of bank items from their seeds; report log-mel L1 to the stored audio (compare with Surge's
    render-to-render noise, ~0.06 on the 2026-10-06 measurement) and exact vector equality."""
    import audio_metrics_v5 as am
    from render_pool_v5 import RenderPool, from_int16
    m = read_manifest(a.bank)
    if prior is None:
        from realistic_bass_prior_v3 import RealisticBassPrior
        prior = RealisticBassPrior(m["manifold"], split="train")
    ds = BankMainDataset(a.bank)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
    vec_ok, stored, vecs, midi = True, [], [], []
    for i in idx:
        v, mi, _ = draw_main_item(prior, int(i), m["seed"], m["random_patch_frac"])
        k, j = divmod(int(i), ds.shard)
        au, meta = ds.sh.get(k)
        vec_ok &= bool(np.array_equal(meta["vec"][j], v))
        stored.append(from_int16(au[j]))
        vecs.append(v)
        midi.append(mi)
    ctx = pool if pool is not None else RenderPool(m["plugin"], workers=getattr(a, "workers", 0))
    with ctx as p:
        fresh = p.render(np.stack(vecs), np.array(midi))
    d = am.score_many(stored, list(fresh), ("logmel_l1",))["logmel_l1"]
    out = {"n": int(len(idx)), "vectors_identical": vec_ok, "logmel_l1_median": float(np.median(d)),
           "logmel_l1_max": float(np.max(d))}
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--manifold", required=True)
    b.add_argument("--plugin", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--n_items", type=int, required=True)
    b.add_argument("--n_ladders", type=int, default=0)
    b.add_argument("--ladder_steps", type=int, default=16)
    b.add_argument("--shard_size", type=int, default=4096)
    b.add_argument("--ladder_shard", type=int, default=256)
    b.add_argument("--random_patch_frac", type=float, default=0.05)
    b.add_argument("--inharm_config", default=None)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=8)
    v = sub.add_parser("verify")
    v.add_argument("--bank", required=True)
    v.add_argument("--n", type=int, default=32)
    v.add_argument("--workers", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        if a.plugin is None:
            from surge_spec_v3 import DEFAULT_PLUGIN_PATH
            a.plugin = DEFAULT_PLUGIN_PATH
        return build(a)
    verify(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
