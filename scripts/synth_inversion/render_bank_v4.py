"""Pre-renders training data for v4 soundset (60-d params, 108-d onehot).

Composition:
- 30% prior-bass-bracketed (Templates + Factory Basses)
- 10% prior-all (All factory + 3rdparty presets)
- 60% pure-random (Unconstrained subtractive synth)
- 10% with waveshaping (>50% loud > 18 dB)
- Unison 1-6 voices
- Both Classic and Modern subtractive oscs
- Filter feedback across [-1.0, 1.0]
- Full resonance [0.0, 1.0]
- Anti-glitch spectral dominance gate: (peak_power_ratio > 0.88) or (total_energy < 1e-4)

Inharmonicity labels (KUANG 2026-10-19): with --inharm_config the bank stores real labels (the A1 / A3 aux head);
without it the labels are stored as 0 with valid = 0, and a trainer run with --inharm_weight > 0 refuses the bank
(check_compatible) instead of silently training the aux head on nothing. Banks built before this change have no
"inharm_labels" field and count as unlabelled. This builder renders no ladders: runs with --ladder_every > 0 cannot
use a bank (BankLadderDataset is empty) and are refused too.

Usage:
  python render_bank_v4.py build --out /home/kim/sao_runs/bank_v5_expanded --n_items 786432 --workers 14 \
      --inharm_config scripts/synth_inversion/inharm_config.json
  python render_bank_v4.py verify --bank /home/kim/sao_runs/bank_v5_expanded --n 32
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

from realistic_prior_v4 import UnifiedSoundsetPrior, BASS_MANIFOLD_PATH, ALL_MANIFOLD_PATH
from render_pool_v4 import RenderPoolV4, to_int16
from surge_spec_v4 import AUDIO_LEN, DEFAULT_PLUGIN_PATH, NUM_PARAMS, PARAM_NAMES, SAMPLE_RATE

FORMAT = 4


def item_rng(seed, i):
    return np.random.RandomState(int(np.random.SeedSequence([int(seed), int(i)]).generate_state(1)[0]))


def _paths(bank, kind, k):
    return (os.path.join(bank, f"{kind}_{k:05d}_audio.npy"), os.path.join(bank, f"{kind}_{k:05d}_meta.npz"))


def _save_shard(bank, kind, k, audio16, **meta):
    pa, pm = _paths(bank, kind, k)
    np.save(pa + ".tmp.npy", audio16)
    np.savez(pm + ".tmp.npz", **meta)
    os.replace(pm + ".tmp.npz", pm)
    os.replace(pa + ".tmp.npy", pa)


def build(a, log=print):
    os.makedirs(a.out, exist_ok=True)
    prior = UnifiedSoundsetPrior(a.bass_manifold, a.all_manifold, split="train")

    cfg = {
        "format": FORMAT,
        "bass_manifold": a.bass_manifold,
        "all_manifold": a.all_manifold,
        "seed": a.seed,
        "n_items": a.n_items,
        "shard_size": a.shard_size,
        "plugin": a.plugin,
        "sample_rate": SAMPLE_RATE,
        "audio_len": AUDIO_LEN,
        "num_params": NUM_PARAMS,
        "bass_manifold_md5": file_md5(a.bass_manifold),
        "all_manifold_md5": file_md5(a.all_manifold),
        "inharm_labels": bool(getattr(a, "inharm_config", None)),
        "inharm_config_md5": file_md5(getattr(a, "inharm_config", None)),
        "distribution": "60% random, 30% bass-manifold, 10% all-manifold; 10% waveshaped (>50% >18dB)",
    }
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in cfg.items() if k not in ("created",) and old.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} holds a bank with a different configuration: {diff}")
        log(f"Resuming bank generation in {a.out}")
    else:
        with open(mpath, "w") as f:
            json.dump({**cfg, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    n_main = math.ceil(a.n_items / a.shard_size)
    t0, done = time.time(), 0

    labels = bool(getattr(a, "inharm_config", None))
    with RenderPoolV4(a.plugin, workers=a.workers, inharm_config=getattr(a, "inharm_config", None)) as pool:
        for k in range(n_main):
            if os.path.exists(_paths(a.out, "main", k)[0]):
                done += min((k + 1) * a.shard_size, a.n_items) - k * a.shard_size
                continue

            idx = range(k * a.shard_size, min((k + 1) * a.shard_size, a.n_items))
            draws = [prior.sample_vector(item_rng(a.seed, i)) for i in idx]
            vecs = np.stack([d[0] for d in draws])
            midi = np.array([d[1] for d in draws], np.int16)
            sources = np.array([d[2] for d in draws])

            # Render through pool with anti-glitch gate (a glitching draw is replaced by a random patch, possibly on
            # another note: store the note each clip was actually rendered at)
            if labels:
                audio, inh, val, final_vecs = pool.render(vecs, midi, labels=True)
            else:
                audio, final_vecs = pool.render(vecs, midi)
                inh = np.zeros(len(idx), dtype=np.float32)
                val = np.zeros(len(idx), dtype=np.float32)
            redrawn = np.any(final_vecs != vecs, axis=1)
            sources = np.where(redrawn, "redraw", sources)

            _save_shard(a.out, "main", k, to_int16(audio), vec=final_vecs, midi=pool.last_midi, source=sources,
                        inharm=inh, inharm_valid=val)
            done += len(idx)
            el = time.time() - t0
            speed = done / max(el, 1e-9)
            eta_m = (a.n_items - done) / max(speed, 1e-9) / 60.0
            log(f"Main shard {k + 1}/{n_main}: {done}/{a.n_items} clips in {el:.0f}s ({speed:.1f}/s, ETA: {eta_m:.1f}m)")

    log(f"Bank build complete: {done} clips in {time.time() - t0:.1f}s -> {a.out}")


def file_md5(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


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
    """Map-style dataset with (audio [N] float32, vec [P], inharm, valid)."""
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
        inharm = float(meta["inharm"][j]) if "inharm" in meta else 0.0
        inharm_valid = float(meta["inharm_valid"][j]) if "inharm_valid" in meta else 0.0
        return (torch.from_numpy(a), torch.from_numpy(np.asarray(meta["vec"][j], np.float32)), inharm, inharm_valid)


class BankLadderDataset:
    def __init__(self, *args, **kwargs):
        pass
    def __len__(self):
        return 0


def cycle(loader):
    """Endless iterator over a DataLoader; each pass draws a new (seeded) permutation from its sampler."""
    while True:
        for batch in loader:
            yield batch


def check_compatible(manifest, bass_manifold=None, all_manifold=None, inharm_weight=0.0, inharm_config=None,
                     ladder_every=0, fmt=FORMAT, num_params=NUM_PARAMS):
    """-> (hard, soft) problem lists. hard: the bank cannot serve the run (wrong format or layout, aux labels or
    ladders it does not have). soft: it was built from other manifolds (overridable with --allow_prior_change).
    fmt / num_params: the bank format and storage size the run expects (render_bank_v6 passes 6 / 59)."""
    hard, soft = [], []
    if manifest.get("format") != fmt:
        hard.append(f"bank format {manifest.get('format')} != {fmt}")
    if int(manifest.get("num_params", num_params)) != num_params:
        hard.append(f"bank has {manifest.get('num_params')} params, this layout {num_params}")
    if inharm_weight > 0:
        if not manifest.get("inharm_labels"):
            hard.append("--inharm_weight > 0 but the bank has no inharmonicity labels (rebuild with --inharm_config)")
        elif inharm_config and manifest.get("inharm_config_md5") != file_md5(inharm_config):
            hard.append("the bank's inharmonicity labels were made with another --inharm_config")
    if ladder_every and ladder_every > 0:
        hard.append("--ladder_every > 0 but banks hold no ladders (use --ladder_every 0 or live mode)")
    for key, path in (("bass_manifold", bass_manifold), ("all_manifold", all_manifold)):
        if path is None:
            continue
        want, have = file_md5(path), manifest.get(f"{key}_md5")
        if have is None:
            if os.path.abspath(str(manifest.get(key))) != os.path.abspath(str(path)):
                soft.append(f"{key}: bank {manifest.get(key)} (no md5 stored) vs run {path}")
        elif want != have:
            soft.append(f"{key}: bank md5 {str(have)[:10]} vs run {str(want)[:10]} ({path})")
    return hard, soft


def verify(a, log=print):
    mpath = os.path.join(a.bank, "manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"No manifest in {a.bank}")
    with open(mpath) as f:
        cfg = json.load(f)
    log(f"Verifying bank {a.bank} (manifest: {cfg.get('distribution', 'unknown')})")

    p0_a, p0_m = _paths(a.bank, "main", 0)
    if not os.path.exists(p0_a):
        raise SystemExit("Shard 0 not found")
    audio = np.load(p0_a)
    meta = np.load(p0_m)
    log(f"Shard 0 audio shape: {audio.shape}, dtype: {audio.dtype}")
    log(f"Shard 0 vec shape: {meta['vec'].shape}, midi shape: {meta['midi'].shape}")
    assert meta["vec"].shape[1] == NUM_PARAMS, f"Expected {NUM_PARAMS} params, got {meta['vec'].shape[1]}"
    log("Verification PASSED!")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--bass_manifold", default=BASS_MANIFOLD_PATH)
    b.add_argument("--all_manifold", default=ALL_MANIFOLD_PATH)
    b.add_argument("--seed", type=int, default=1337)
    b.add_argument("--n_items", type=int, default=786432)
    b.add_argument("--shard_size", type=int, default=4096)
    b.add_argument("--workers", type=int, default=14)
    b.add_argument("--plugin", default=DEFAULT_PLUGIN_PATH)
    b.add_argument("--inharm_config", default=None, help="store real inharmonicity labels (needed for --inharm_weight)")

    v = sub.add_parser("verify")
    v.add_argument("--bank", required=True)
    v.add_argument("--n", type=int, default=32)

    args = ap.parse_args()
    if args.cmd == "build":
        build(args)
    elif args.cmd == "verify":
        verify(args)


if __name__ == "__main__":
    main()
