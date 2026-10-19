# Owner: KUANG / KIM (created 2026-10-19)
"""V6 Guided Preference Data Generator for Flow-DPO (V6.1 / Phase 4).
Generates pairs (winner, loser) by combining:
1. Passive candidate sampling from V6_C flow model (Euler integration)
2. Active guided cutoff & waveshape polish:
   - Shifts filter 1 & 2 cutoff in the direction of the target spectral centroid / shoulder error.
   - Damps artificial filter 2 resonance if cutoff was over-choked.
   - Guides oscillator shape to --square_shape (symmetric square) if the target exhibits strong odd-harmonic parity (<0.12).
3. Renders all candidates through Surge XT in parallel (RenderPool).
4. Scores candidates with logmel_l1, mss, and mel_ssim_dist for Flow-DPO training.

KUANG 2026-10-19 (V6_REVIEW.md section 9):
- The guided candidate is built in the stored v6 vector (guide_candidate_vector), from candidate 0's RENDERED
  centroid against the target's (V6_1_PLAN.md 2.3: shift only if the error exceeds 120 Hz); the first version
  compared the target centroid with fixed cutoff thresholds and never looked at the candidate's sound.
- Parity is measured on the phrase's first note only (am6.phrase_parity): the second phrase note is an octave up and
  puts its odd harmonics on the first note's even ones.
- The guided candidate is only an extra candidate; pairs are still chosen by the rendered score, so a heuristic
  that is wrong simply loses. The manifest reports how often it wins (guided_best_share) and the usual headroom,
  and is written before the first shard, so train_flow_dpo_v5 can read a partial run.
- --square_shape: the osc shape value of a symmetric square (check_v6_spec.py --shape_scan). KUANG 2026-10-20: no
  default any more. 0.5 is a saw in the classic oscillator (Antigravity's scan, parity 0.289); without the value the
  parity rule does not fire. Parity also needs harmonics to read (am6.phrase_parity is nan on a near sine).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import audio_metrics_v6 as am6
import surge_spec_v6 as S6
from flow_sampling_v5 import file_md5, mel_of, sample_candidates
from layout_v6 import layout_for_flow
from models_v4 import load_inverter_v4
from render_pool_v5 import RenderPool, to_int16
from realistic_prior_v4 import manifold_md5
from soundset_prior_v6 import SoundsetPriorV6

N_AUDIO = 35280

def synth_conditions(prior, start, n, seed):
    from render_bank_v4 import item_rng
    vecs, midi = [], []
    for i in range(start, start + n):
        v, m, _ = prior.sample_vector(item_rng(seed, i))
        vecs.append(np.asarray(v, np.float32))
        midi.append(int(m))
    return np.stack(vecs), np.array(midi)

def guide_candidate_vector(cand_vec, cand_audio, target_audio, midi_note, sr=44100, square_shape=None,
                           centroid_tol_hz=120.0):
    """Guided variant of a flow candidate (V6_1_PLAN.md 2.3), in the v6 stored vector:
      1. parity: target's first note has odd harmonics only (< 0.12) -> shape = square_shape, ring and noise off,
         filter 2 resonance off;
      2. cutoff: candidate centroid more than 120 Hz from the target's -> both cutoffs shift by
         clip(error / 2500, +-0.2) (am6.guided_cutoff_shift); resonance -0.25 when opening a resonant filter;
      3. filter envelope: bright target (> 1000 Hz) with little envelope (< 0.40) -> +0.25 / +0.20; dark target
         (< 400 Hz) with a lot (> 0.70) -> -0.30 (floor 0.20);
      4. keytrack: note above MIDI 45 with ~0 % keytrack -> 100 % (filter 1 raw 0.75, filter 2 raw 0.75).
    -> (vector, list of the rules that fired)."""
    I = S6.PARAM_INDEX
    v = np.array(cand_vec, np.float32, copy=True)
    fired = []
    c_tgt = am6.get_centroid(target_audio, sr)
    c_cand = am6.get_centroid(cand_audio, sr)
    if square_shape is not None and am6.phrase_parity(target_audio, midi_note, sr) < 0.12:
        v[I["shape"]] = square_shape
        v[I["ring_1x2_volume"]] = v[I["ring_2x3_volume"]] = v[I["noise_volume"]] = 0.0
        v[I["filter_2_resonance"]] = 0.0
        fired.append("parity")
    if abs(c_tgt - c_cand) > centroid_tol_hz:
        shift = am6.guided_cutoff_shift(c_tgt, c_cand, max_shift=0.2)
        v[I["cutoff"]] = float(np.clip(v[I["cutoff"]] + shift, *S6.CONT_BOUNDS["cutoff"]))
        v[I["filter_2_cutoff"]] = float(np.clip(v[I["filter_2_cutoff"]] + shift, *S6.CONT_BOUNDS["filter_2_cutoff"]))
        if shift > 0:
            for r in ("resonance", "filter_2_resonance"):
                if v[I[r]] > 0.35:
                    v[I[r]] = max(0.0, float(v[I[r]]) - 0.25)
        fired.append("cutoff")
    if c_tgt > 1000.0 and v[I["feg_amount"]] < 0.40:
        v[I["feg_amount"]] = float(np.clip(v[I["feg_amount"]] + 0.25, *S6.CONT_BOUNDS["feg_amount"]))
        v[I["filter_2_feg_amount"]] = float(np.clip(v[I["filter_2_feg_amount"]] + 0.20, 0.0, 1.0))
        fired.append("feg_up")
    elif c_tgt < 400.0 and v[I["feg_amount"]] > 0.70:
        v[I["feg_amount"]] = max(0.20, float(v[I["feg_amount"]]) - 0.30)
        v[I["filter_2_feg_amount"]] = max(0.20, float(v[I["filter_2_feg_amount"]]) - 0.30)
        fired.append("feg_down")
    if midi_note > 45 and abs(float(v[I["keytrack"]])) < 0.16:      # stored p: raw = 0.5 + 0.5 p; p < 0.16 ~ 0 %
        v[I["keytrack"]] = 0.5                                          # raw 0.75 = 100 %
        v[I["filter_2_keytrack"]] = 0.75
        fired.append("keytrack")
    return S6.canonicalize_vector(v), fired


def build_shard(flow, pool, device, targets, midi, k_passive=4, steps=25, tau=1.0, seed=0, true_vecs=None,
                metrics=("logmel_l1", "mss", "mel_ssim_dist"), square_shape=None, sample_fn=None):
    """Passive flow candidates + one guided candidate (from passive candidate 0 and its render) -> [n, K, P],
    scores {metric: [n, K]}, guided_rules [n] (which rules fired). The guided candidate is the last column."""
    mel = mel_of(targets, device)
    sample_fn = sample_fn or sample_candidates
    passive_cands = sample_fn(flow, mel, midi, k=k_passive, steps=steps, seed=seed, tau=tau)
    n, K_p, P = passive_cands.shape
    passive_audio = pool.render(passive_cands.reshape(n * K_p, P), np.repeat(midi, K_p)).reshape(n, K_p, -1)
    guided, rules = [], []
    for i in range(n):
        g, fired = guide_candidate_vector(passive_cands[i, 0], passive_audio[i, 0], targets[i], int(midi[i]),
                                          square_shape=square_shape)
        guided.append(g)
        rules.append(",".join(fired))
    guided = np.stack(guided)
    guided_audio = pool.render(guided, midi)
    all_cands = np.concatenate([passive_cands, guided[:, None, :]], axis=1)          # [n, K_p + 1, P]
    all_audio = np.concatenate([passive_audio, guided_audio[:, None, :]], axis=1)
    K_tot = K_p + 1
    tgt_expanded = [targets[i] for i in range(n) for _ in range(K_tot)]
    sc = am6.score_many(tgt_expanded, list(all_audio.reshape(n * K_tot, -1)), list(metrics))
    out = {"cand_vecs": all_cands, "guided_rules": np.array(rules),
           **{f"score__{m}": v.reshape(n, K_tot) for m, v in sc.items()}}
    
    if true_vecs is not None:
        again = pool.render(true_vecs, midi)
        fl = am6.score_many(list(targets), list(again), list(metrics))
        out.update({f"noise__{m}": v for m, v in fl.items()})
        
    return out

def guided_best_share(shards, metrics):
    """Share of conditions where the guided candidate (last column) has the best score, per metric."""
    out = {}
    for m in metrics:
        s = np.concatenate([sh[f"score__{m}"] for sh in shards])
        out[m] = float(np.mean(np.nanargmin(np.where(np.isfinite(s), s, np.inf), axis=1) == s.shape[1] - 1))
    return out


def main(argv=None, pool=None, flow=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow", required=True, help="V6_C flow model checkpoint (flow_ema_latest.pt)")
    ap.add_argument("--bass_manifold", required=True, help="real_manifold_v4_bass.npz")
    ap.add_argument("--all_manifold", required=True, help="real_manifold_v4_all.npz")
    ap.add_argument("--plugin", default=None)
    ap.add_argument("--n", type=int, default=8192, help="number of conditions")
    ap.add_argument("--k_passive", type=int, default=4, help="passive flow candidates per condition")
    ap.add_argument("--steps", type=int, default=25, help="Euler sampling steps")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", type=int, default=512)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--square_shape", type=float, default=None,
                    help="osc shape value of a symmetric square (check_v6_spec.py --shape_scan); parity rule off without it")
    ap.add_argument("--out", required=True, help="output directory for DPO preference shards")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    flow = flow if flow is not None else load_inverter_v4(a.flow, device=a.device)
    layout = layout_for_flow(flow)
    if layout.name != "v6":
        raise SystemExit(f"{a.flow} is a {layout.name} flow; render_pairs_dpo_v6 guides v6 vectors only")
    
    prior = SoundsetPriorV6(a.bass_manifold, a.all_manifold, split="train")
    metrics = ["logmel_l1", "mss", "mel_ssim_dist"]
    
    manifest = {
        "flow": a.flow,
        "flow_md5": file_md5(a.flow),
        "source": "synth_guided",
        "layout": layout.name,
        "n": a.n,
        "k": a.k_passive + 1,
        "k_passive": a.k_passive,
        "guided": True,
        "steps": a.steps,
        "tau": a.tau,
        "metrics": metrics,
        "seed": a.seed,
        "shard": a.shard,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "square_shape": a.square_shape,
        "prior_md5": manifold_md5(a.bass_manifold) + "+" + manifold_md5(a.all_manifold),
    }
    mpath = os.path.join(a.out, "manifest.json")
    with open(mpath, "w") as f:                                 # early: a partial run is already usable
        json.dump(manifest, f, indent=2)
    
    shards = []
    t0 = time.time()
    n_shards = math.ceil(a.n / a.shard)
    
    print(f"Generating {a.n} guided DPO conditions ({n_shards} shards) to {a.out}...")
    with (pool if pool is not None else RenderPool(a.plugin, workers=a.workers, spec="v6")) as pool:
        for s in range(n_shards):
            path = os.path.join(a.out, f"shard_{s:05d}.npz")
            if os.path.exists(path):
                print(f"Shard {s + 1}/{n_shards} already exists, loading...")
                with np.load(path) as z:
                    shards.append({key: z[key] for key in z.files})
                continue
                
            lo, hi = s * a.shard, min((s + 1) * a.shard, a.n)
            true_vecs, midi = synth_conditions(prior, lo, hi - lo, a.seed)
            targets = pool.render(true_vecs, midi)
            
            sh = build_shard(flow, pool, a.device, targets, midi, k_passive=a.k_passive,
                             steps=a.steps, tau=a.tau, seed=a.seed * 100_003 + s, true_vecs=true_vecs,
                             metrics=metrics, square_shape=a.square_shape)
            sh.update({"target_audio": to_int16(targets), "midi": midi, "cond_index": np.arange(lo, hi), "true_vecs": true_vecs})
            
            np.savez(path + ".tmp.npz", **sh)
            os.replace(path + ".tmp.npz", path)
            shards.append(sh)
            el = time.time() - t0
            print(f"Shard {s + 1}/{n_shards}: {hi}/{a.n} conditions in {el:.0f}s ({hi / max(el, 1e-9):.1f} cond/s)", flush=True)

    from dpo_pairs_v5 import headroom
    manifest["headroom"] = {m: headroom(shards, m) for m in metrics}
    manifest["guided_best_share"] = guided_best_share(shards, metrics)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    for m, h in manifest["headroom"].items():
        print(f"{m}: candidate 0 {h['draw0']:.4f} | best of {manifest['k']} {h['best']:.4f} | "
              f"guided candidate best in {100 * manifest['guided_best_share'][m]:.0f}% of conditions")
        
    print(f"\nGuided DPO Preference Generation Complete! Stored in {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
