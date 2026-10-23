# Owner: KUANG (created 2026-10-13)
"""Audio-space evaluation of inversion systems on held-out synthetic notes, plus a listening pack
(V5_PROPOSAL_REVIEW.md, sec. 4 and 6, stage 1).

The v4 per-item evaluation never renders a prediction. This does: for N held-out notes (the prior's 'val' split,
heldout_eval_v4.draw_items, so every item carries its archetype preset for clustering), each system proposes
vectors, Surge renders them, and every candidate is scored against the target audio with audio_metrics_v5.

Systems:
  --flow NAME=flow_*.pt    K candidates (flow_sampling_v5, MIDI pinned, an independent condition-noise draw per
                           candidate at tau < 1). Reported as NAME/draw0 (one draw: what a single sample sounds like)
                           and NAME/bestK (best of K by --select, default log-mel L1; its select-metric score is
                           optimistic by construction, the other metrics are not selected on).
  --jepa NAME=jepa_*.pt    the renderer-free JADE + Adam search of synth_jepa_search_v3 (budget --jepa_budget).
Statistics: per system and metric the item mean with a preset-clustered 95 % CI, and paired deltas between systems
(eval_stats_v4.paired_delta, preset clusters). The headroom NAME/draw0 - NAME/bestK is what DPO can at most distill.

Listening pack (--listen N): the first N items as listen/<item>__<system>.wav plus listen/refs/<item>.wav and an
index.html with the scores beside each clip; the layout listening_test_v5.py make reads (--clips listen --refs
listen/refs).

Layouts (KUANG 2026-10-19): all --flow systems must share one layout (layout_v6: 92 -> v3, 108 -> v4, 126 -> v5);
targets and candidates are rendered with that layout's spec. v3 draws held-out items from --manifold (val split,
clustered by archetype preset). v4 / v5 draw from the soundset prior of their trainer (--bass_manifold, --all_manifold,
val split): its draws are 60 % unconstrained random patches, 30 % bass presets, 10 % all presets, so every random
item is its own cluster and preset items cluster by "bass:<i>" / "all:<i>". The report gives the source share.
--jepa systems are v3-only (synth_jepa_search_v3) and are refused for other layouts.

Usage (desktop):
  python audio_eval_v5.py --manifold M.npz --plugin "Surge XT.vst3" --n 256 --k 8 --workers 14 \\
      --flow A3p=RUN_A3P/flow_ema_latest.pt --flow A3pDPO=dpo_a3p_b100/flow_dpo_latest.pt \\
      --jepa A3pJEPA=RUN_A3P/jepa_ema_latest.pt --listen 16 --out audio_eval_s1 --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import audio_metrics_v5 as am  # noqa: E402
import eval_stats_v4 as st  # noqa: E402


class SoundsetRecorder:
    """RandomState proxy that records the preset index a SubManifoldPrior draw picks (its first randint(0, N)).
    Every call passes through, so the stream equals a plain RandomState's."""

    def __init__(self, rng, sizes):
        self._rng, self._sizes, self.last = rng, set(int(s) for s in sizes), None

    def randint(self, *args, **kwargs):
        v = self._rng.randint(*args, **kwargs)
        if self.last is None and len(args) >= 2 and args[0] == 0 and int(args[1]) in self._sizes \
                and kwargs.get("size") is None:
            self.last = int(v)
        return v

    def __getattr__(self, name):
        return getattr(self._rng, name)


def draw_items_soundset(prior, n, seed):
    """Held-out items from realistic_prior_v4.UnifiedSoundsetPrior -> dict like heldout_eval_v4.draw_items, with
    preset = cluster label and source = random / bass_prior / all_prior."""
    rng = np.random.RandomState(seed)
    vecs, midi, preset, source = [], [], [], []
    for i in range(n):
        rec = SoundsetRecorder(rng, (prior.bass_prior.N, prior.all_prior.N))
        v, m, src = prior.sample_vector(rec)
        vecs.append(np.asarray(v, np.float32))
        midi.append(int(m))
        source.append(src)
        if src == "random" or rec.last is None:
            preset.append(f"r{i}")
        else:
            preset.append(f"{src.split('_')[0]}:{rec.last}")
    return {"vecs": np.stack(vecs), "midi": np.array(midi), "preset": np.array(preset), "source": np.array(source)}


def flows_layout(specs, device):
    """All --flow checkpoints must share one layout; -> Layout (v3 if there are no flows)."""
    import torch
    from layout_v6 import layout_for_dim, layout_named
    names = set()
    for _, path in specs:
        state = torch.load(path, map_location="cpu", weights_only=False)["model_state"]
        names.add(layout_for_dim(state["head.weight"].shape[0]).name)
    if len(names) > 1:
        raise SystemExit(f"--flow checkpoints mix layouts {sorted(names)}; evaluate each layout separately")
    return layout_named(names.pop() if names else "v3")


def parse_specs(specs):
    out = []
    for s in specs or []:
        if "=" not in s:
            raise SystemExit(f"expected NAME=PATH, got {s!r}")
        out.append(tuple(s.split("=", 1)))
    return out


def flow_system(name, path, targets_mel, midi, pool, k, steps, tau, seed, device):
    from flow_sampling_v5 import sample_candidates
    from models_v4 import load_inverter_v4
    flow = load_inverter_v4(path, device=device)
    vecs = sample_candidates(flow, targets_mel, midi, k=k, steps=steps, seed=seed, tau=tau)
    n, K, P = vecs.shape
    audio = pool.render(vecs.reshape(n * K, P), np.repeat(midi, K)).reshape(n, K, -1)
    return vecs, audio


def jepa_system(path, targets_mel, midi, pool, budget, seed, device):
    from synth_jepa_search_v3 import SynthJEPASearcher, checkpoint_bounds, load_synth_jepa
    model, norm = load_synth_jepa(path, device=device)
    searcher = SynthJEPASearcher(model, norm, device=device, seed=seed, bounds=checkpoint_bounds(path))
    vecs = np.stack([np.asarray(searcher.search(targets_mel[i], midi_note=int(midi[i]), total_eval_budget=budget)["vector"],
                                np.float32) for i in range(len(midi))])
    return vecs[:, None, :], pool.render(vecs, midi)[:, None, :]


def score_system(targets, audio, metrics, workers=0):
    """audio [n, K, N] -> {metric: [n, K]}"""
    n, K, _ = audio.shape
    sc = am.score_many([targets[i] for i in range(n) for _ in range(K)], [audio[i, j] for i in range(n) for j in range(K)],
                       metrics, workers=workers)
    return {m: v.reshape(n, K) for m, v in sc.items()}


def views(name, scores, select, kind):
    """-> {view name: (per-metric arrays [n], chosen candidate index [n])}"""
    if kind == "jepa":
        return {name: ({m: v[:, 0] for m, v in scores.items()}, np.zeros(len(scores[select]), int))}
    best = np.nanargmin(np.where(np.isfinite(scores[select]), scores[select], np.inf), axis=1)
    rows = np.arange(len(best))
    return {f"{name}/draw0": ({m: v[:, 0] for m, v in scores.items()}, np.zeros(len(best), int)),
            f"{name}/best{scores[select].shape[1]}": ({m: v[rows, best] for m, v in scores.items()}, best)}


def summarize(per_view, clusters, metrics, n_boot=10000):
    out = {"systems": {}, "paired": []}
    names = list(per_view)
    for v in names:
        out["systems"][v] = {m: st.mean_ci(per_view[v][m], clusters, n_boot=n_boot) for m in metrics}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for m in metrics:
                r = st.paired_delta(per_view[a][m], per_view[b][m], clusters, n_boot=n_boot)
                out["paired"].append({"a": a, "b": b, "metric": m, **r})
    return out


def report_md(summary, metrics):
    lines = ["| system | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for v, s in summary["systems"].items():
        lines.append(f"| {v} | " + " | ".join(f"{s[m]['estimate']:.4f} [{s[m]['lo']:.4f}, {s[m]['hi']:.4f}]" for m in metrics) + " |")
    lines += ["", "Paired deltas a - b (lower is better for every metric here), preset-clustered 95 % CI:", "",
              "| a | b | metric | delta | CI | p |", "|---|---|---|---|---|---|"]
    for r in summary["paired"]:
        lines.append(f"| {r['a']} | {r['b']} | {r['metric']} | {r['estimate']:+.4f} | [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['p']:.3f} |")
    return "\n".join(lines) + "\n"


def write_listen_pack(out, targets, chosen_audio, per_view, metrics, n_items, sr=44100):
    import soundfile as sf
    d = os.path.join(out, "listen")
    os.makedirs(os.path.join(d, "refs"), exist_ok=True)
    rows = []
    for i in range(min(n_items, len(targets))):
        item = f"syn{i:03d}"
        sf.write(os.path.join(d, "refs", f"{item}.wav"), targets[i], sr, subtype="PCM_16")
        cells = [f'<td>target<br><audio controls preload="none" src="refs/{item}.wav"></audio></td>']
        for v, audio in chosen_audio.items():
            safe = v.replace("/", "_")
            sf.write(os.path.join(d, f"{item}__{safe}.wav"), audio[i], sr, subtype="PCM_16")
            sc = ", ".join(f"{m} {per_view[v][m][i]:.3f}" for m in metrics[:3])
            cells.append(f'<td>{v}<br><audio controls preload="none" src="{item}__{safe}.wav"></audio><br><small>{sc}</small></td>')
        rows.append(f"<tr><th>{item}</th>{''.join(cells)}</tr>")
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Listening pack</title><style>"
            "body{font-family:sans-serif;margin:16px}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}"
            "audio{width:220px}table{border-collapse:collapse}</style></head><body><h1>Listening pack</h1>"
            "<p>Same held-out items for every system and checkpoint. Not blind: for decisions use listening_test_v5.py."
            "</p><table>" + "".join(rows) + "</table></body></html>")
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main(argv=None, pool=None, prior=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifold", default=None, help="v3 flows: archetype manifold (val split)")
    ap.add_argument("--plugin", default=None)
    ap.add_argument("--bass_manifold", default=None, help="v4 / v5 flows: soundset prior (val split)")
    ap.add_argument("--all_manifold", default=None, help="v4 / v5 flows: soundset prior (val split)")
    ap.add_argument("--flow", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--jepa", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=11, help="item draw seed (val split); the trainer's sets use 1, 2, 3")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--jepa_budget", type=int, default=2048)
    ap.add_argument("--select", default="logmel_l1")
    ap.add_argument("--metrics", nargs="+", default=list(am.DEFAULT_EVAL), choices=sorted(am.METRICS))
    ap.add_argument("--clap_ckpt", default=None, help="add CLAP audio-to-audio distance (laion_clap + checkpoint)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--listen", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.select not in a.metrics:
        raise SystemExit("--select must be one of --metrics")
    from flow_sampling_v5 import mel_of
    from heldout_eval_v4 import draw_items
    from render_pool_v5 import RenderPool, to_int16
    os.makedirs(a.out, exist_ok=True)
    layout = flows_layout(parse_specs(a.flow), a.device) if a.flow else None
    lname = layout.name if layout is not None else "v3"
    if lname != "v3" and a.jepa:
        raise SystemExit("--jepa (synth_jepa_search_v3) only covers v3 layouts")
    if prior is None:
        if lname == "v3":
            from realistic_bass_prior_v3 import RealisticBassPrior
            if not a.manifold:
                raise SystemExit("v3 flows need --manifold")
            prior = RealisticBassPrior(a.manifold, split="val")
        else:
            from realistic_prior_v4 import UnifiedSoundsetPrior
            from soundset_prior_v6 import SoundsetPriorV6
            if not (a.bass_manifold and a.all_manifold):
                raise SystemExit(f"{lname} flows need --bass_manifold and --all_manifold")
            prior = (SoundsetPriorV6 if lname == "v6" else UnifiedSoundsetPrior)(a.bass_manifold, a.all_manifold,
                                                                                  split="val")
    items = draw_items(prior, a.n, a.seed) if lname == "v3" else draw_items_soundset(prior, a.n, a.seed)
    midi, clusters = items["midi"], items["preset"]
    t0 = time.time()
    ctx = pool if pool is not None else RenderPool(a.plugin, workers=a.workers, spec=lname)
    per_view, chosen_audio, raw = {}, {}, {}
    with ctx as p:
        targets = p.render(items["vecs"], midi)
        mel = mel_of(targets, a.device)
        systems = [(n, path, "flow") for n, path in parse_specs(a.flow)] + [(n, path, "jepa") for n, path in parse_specs(a.jepa)]
        for name, path, kind in systems:
            if kind == "flow":
                vecs, audio = flow_system(name, path, mel, midi, p, a.k, a.steps, a.tau, a.seed, a.device)
            else:
                vecs, audio = jepa_system(path, mel, midi, p, a.jepa_budget, a.seed, a.device)
            sc = score_system(targets, audio, a.metrics, workers=a.workers)
            if a.clap_ckpt:
                n, K, _ = audio.shape
                sc["clap_dist"] = am.clap_distances([targets[i] for i in range(n) for _ in range(K)],
                                                    list(audio.reshape(n * K, -1)), a.clap_ckpt).reshape(n, K)
            raw[name] = {"vecs": vecs, **{f"score__{m}": v for m, v in sc.items()}}
            for v, (arrs, idx) in views(name, sc, a.select, kind).items():
                per_view[v] = arrs
                chosen_audio[v] = audio[np.arange(len(idx)), idx]
            print(f"{name}: done at {time.time() - t0:.0f} s", flush=True)
    metrics = list(a.metrics) + (["clap_dist"] if a.clap_ckpt else [])
    summary = summarize(per_view, clusters, metrics)
    if "source" in items:
        summary["source_share"] = {s: float(np.mean(items["source"] == s)) for s in sorted(set(items["source"]))}
    summary["meta"] = {"layout": lname, "n": a.n, "seed": a.seed, "k": a.k, "steps": a.steps, "tau": a.tau, "select": a.select,
                       "systems": systems, "time_s": time.time() - t0, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md(summary, metrics))
    np.savez_compressed(os.path.join(a.out, "items.npz"), target_audio=to_int16(targets), midi=midi, preset=clusters,
                        true_vecs=items["vecs"],
                        **{f"{s}__{k}": v for s, d in raw.items() for k, v in d.items()})
    if a.listen:
        write_listen_pack(a.out, targets, chosen_audio, per_view, metrics, a.listen)
    print(report_md(summary, metrics))
    return summary


if __name__ == "__main__":
    import os
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)     # skip interpreter teardown (the Surge plugin can hang it); flush first, os._exit does not
