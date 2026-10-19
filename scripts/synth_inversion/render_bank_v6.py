# Owner: KUANG (created 2026-10-19)
"""Pre-rendered training bank for the v6 layout (surge_spec_v6, 59 stored values, note at index 0) drawn from the
simplicity prior (soundset_prior_v6), with real inharmonicity labels when --inharm_config is given.

Same shard layout as render_bank_v4 (main_NNNNN_audio.npy int16 + main_NNNNN_meta.npz with vec, midi, source,
inharm, inharm_valid), so render_bank_v4.BankMainDataset reads it; manifest format 6, num_params 59.
Glitching renders (render_bank_v4's gate: near-silent or one dominant frequency) are redrawn from the same prior with
a seed taken from the item's own vector, up to 5 times; the stored vector, note and source are the ones rendered.

  python render_bank_v6.py build --out ~/sao_runs/bank_v6 --bass_manifold B.npz --all_manifold A.npz \\
      --n_items 100000 --workers 14 --inharm_config scripts/synth_inversion/inharm_config.json
  python render_bank_v6.py verify --bank ~/sao_runs/bank_v6 --n 32
"""
import argparse
import json
import math
import os
import sys
import time
import zlib

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import surge_spec_v6 as S  # noqa: E402
from render_bank_v4 import (BankMainDataset, _paths, _save_shard, check_compatible as _check_v4,  # noqa: E402,F401
                            cycle, file_md5, item_rng, read_manifest)

FORMAT = 6
MAX_REDRAWS = 5


def check_compatible(manifest, bass_manifold=None, all_manifold=None, inharm_weight=0.0, inharm_config=None,
                     ladder_every=0):
    return _check_v4(manifest, bass_manifold, all_manifold, inharm_weight, inharm_config, ladder_every,
                     fmt=FORMAT, num_params=S.NUM_PARAMS)


def _render(pool, vecs, midi, labels):
    if labels:
        return pool.render(vecs, midi, labels=True)
    a = pool.render(vecs, midi)
    return a, np.zeros(len(a), np.float32), np.zeros(len(a), np.float32)


def render_items(pool, prior, vecs, midi, sources, labels):
    """Render, redraw glitches from the prior (seed from each item's vector), re-render, up to MAX_REDRAWS times."""
    vecs, midi, sources = vecs.copy(), midi.copy(), sources.copy()
    audio, inh, val = _render(pool, vecs, midi, labels)
    for attempt in range(MAX_REDRAWS):
        bad = np.array([S.check_glitch(a) for a in audio])
        if not bad.any():
            break
        idx = np.where(bad)[0]
        for i in idx:
            rng = np.random.RandomState((zlib.crc32(vecs[i].tobytes()) + 7919 * (attempt + 1)) % (2 ** 32))
            v, m, _ = prior.sample_vector(rng)
            vecs[i], midi[i], sources[i] = v, m, "redraw"
        a2, i2, v2 = _render(pool, vecs[idx], midi[idx], labels)
        audio[idx], inh[idx], val[idx] = a2, i2, v2
    return audio, inh, val, vecs, midi, sources


def build(a, log=print, prior=None, pool=None):
    from render_pool_v5 import RenderPool, to_int16
    os.makedirs(a.out, exist_ok=True)
    if prior is None:
        from soundset_prior_v6 import SoundsetPriorV6
        prior = SoundsetPriorV6(a.bass_manifold, a.all_manifold, split="train")
    labels = bool(getattr(a, "inharm_config", None))
    cfg = {"format": FORMAT, "layout": "v6", "bass_manifold": a.bass_manifold, "all_manifold": a.all_manifold,
           "bass_manifold_md5": file_md5(a.bass_manifold), "all_manifold_md5": file_md5(a.all_manifold),
           "seed": a.seed, "n_items": a.n_items, "shard_size": a.shard_size, "plugin": a.plugin,
           "sample_rate": S.SAMPLE_RATE, "audio_len": S.AUDIO_LEN, "num_params": S.NUM_PARAMS,
           "inharm_labels": labels, "inharm_config_md5": file_md5(getattr(a, "inharm_config", None)),
           "prior": {"mix": getattr(prior, "mix", None), "sparse": getattr(prior, "sparse", None)}}
    mpath = os.path.join(a.out, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            old = json.load(f)
        diff = {k: (old.get(k), v) for k, v in cfg.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} holds a bank with a different configuration: {diff}")
        log(f"Resuming bank generation in {a.out}")
    else:
        with open(mpath, "w") as f:
            json.dump({**cfg, "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    n_main = math.ceil(a.n_items / a.shard_size)
    t0, done = time.time(), 0
    ctx = pool if pool is not None else RenderPool(a.plugin, workers=a.workers, inharm_config=a.inharm_config, spec="v6")
    with ctx as pl:
        for k in range(n_main):
            lo, hi = k * a.shard_size, min((k + 1) * a.shard_size, a.n_items)
            if os.path.exists(_paths(a.out, "main", k)[0]):
                done += hi - lo
                continue
            draws = [prior.sample_vector(item_rng(a.seed, i)) for i in range(lo, hi)]
            vecs = np.stack([d[0] for d in draws]).astype(np.float32)
            midi = np.array([d[1] for d in draws], np.int16)
            sources = np.array([d[2] for d in draws], dtype="<U12")
            audio, inh, val, vecs, midi, sources = render_items(pl, prior, vecs, midi, sources, labels)
            _save_shard(a.out, "main", k, to_int16(audio), vec=vecs, midi=midi, source=sources, inharm=inh,
                        inharm_valid=val)
            done += hi - lo
            el = time.time() - t0
            log(f"Main shard {k + 1}/{n_main}: {done}/{a.n_items} clips in {el:.0f}s ({done / max(el, 1e-9):.1f}/s)")
    log(f"Bank build complete: {done} clips -> {a.out}")


def verify(a, log=print, pool=None):
    """Re-render n stored items and compare: vectors must be valid v6 storage, audio within render noise."""
    import audio_metrics_v5 as am
    from render_pool_v5 import RenderPool, from_int16
    m = read_manifest(a.bank)
    if m.get("format") != FORMAT:
        raise SystemExit(f"{a.bank}: format {m.get('format')}, not a v6 bank")
    audio = np.load(_paths(a.bank, "main", 0)[0])
    with np.load(_paths(a.bank, "main", 0)[1]) as z:
        vecs, midi = z["vec"][:a.n], z["midi"][:a.n]
    if vecs.shape[1] != S.NUM_PARAMS:
        raise SystemExit(f"stored vectors have {vecs.shape[1]} values, v6 has {S.NUM_PARAMS}")
    ctx = pool if pool is not None else RenderPool(m["plugin"], workers=0, spec="v6")
    with ctx as pl:
        again = pl.render(vecs, midi)
    d = am.score_many(list(from_int16(audio[:a.n])), list(again), ["logmel_l1"])["logmel_l1"]
    log(f"re-render logmel_l1 median {np.median(d):.4f} (Surge render noise is ~0.06 or less)")
    return float(np.median(d))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--bass_manifold", required=True)
    b.add_argument("--all_manifold", required=True)
    b.add_argument("--seed", type=int, default=1337)
    b.add_argument("--n_items", type=int, default=100_000)
    b.add_argument("--shard_size", type=int, default=4096)
    b.add_argument("--workers", type=int, default=14)
    b.add_argument("--plugin", default=S.DEFAULT_PLUGIN_PATH)
    b.add_argument("--inharm_config", default=None, help="store real inharmonicity labels (needed for --inharm_weight)")
    v = sub.add_parser("verify")
    v.add_argument("--bank", required=True)
    v.add_argument("--n", type=int, default=32)
    a = ap.parse_args(argv)
    return build(a) if a.cmd == "build" else verify(a)


if __name__ == "__main__":
    main()
