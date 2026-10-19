# Owner: KUANG / KIM (created 2026-10-19)
"""V6 Guided Preference Data Generator for Flow-DPO (V6.1 / Phase 4).
Generates pairs (winner, loser) by combining:
1. Passive candidate sampling from V6_C flow model (Euler integration)
2. Active guided cutoff & waveshape polish:
   - Shifts filter 1 & 2 cutoff in the direction of the target spectral centroid / shoulder error.
   - Damps artificial filter 2 resonance if cutoff was over-choked.
   - Guides oscillator shape to 0.50 (symmetric square) if the target exhibits strong odd-harmonic parity (<0.12).
3. Renders all candidates through Surge XT in parallel (RenderPool).
4. Scores candidates with logmel_l1, mss, and mel_ssim_dist for Flow-DPO training.
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
import surge_spec_v5 as S5
from flow_sampling_v5 import file_md5, mel_of, sample_candidates
from layout_v6 import layout_for_flow
from models_v4 import load_inverter_v4
from render_pool_v5 import RenderPool, to_int16
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

def guide_candidate_patch(cand_vec, target_audio, midi_note, sr=44100):
    """Generate an actively guided patch from a predicted vector by comparing to target audio."""
    v5_cand = S6.to_v5(cand_vec)
    p = S5.vector_to_patch(v5_cand)
    
    # Target centroid
    c_tgt = am6.get_centroid(target_audio, sr)
    
    # Estimate candidate centroid from target note fundamental & cutoff heuristic
    # Shift cutoff in unison based on spectral energy mismatch
    f0 = float(440.0 * (2.0 ** ((midi_note - 69.0) / 12.0)))
    parity = am6.compute_harmonic_parity(target_audio, f0, sr)
    
    guided_p = p.copy()
    
    # 1. Harmonic parity guidance (Square Wave)
    if parity < 0.12:
        guided_p["shape"] = 0.50 # set to symmetric square
        guided_p["ring_1x2_volume"] = 0.0
        guided_p["ring_2x3_volume"] = 0.0
        guided_p["noise_volume"] = 0.0
        guided_p["filter_2_resonance"] = 0.0
        
    # 2. Directed Cutoff shift
    # Quick probe: if target has high centroid (> 1200 Hz) but candidate has low cutoff (< 0.50)
    if c_tgt > 1200.0 and p["cutoff"] < 0.55:
        guided_p["cutoff"] = float(np.clip(p["cutoff"] + 0.20, 0.0, 1.0))
        guided_p["filter_2_cutoff"] = float(np.clip(p["filter_2_cutoff"] + 0.15, 0.0, 1.0))
        if p["resonance"] > 0.35:
            guided_p["resonance"] = float(max(0.0, p["resonance"] - 0.25))
        if p["filter_2_resonance"] > 0.35:
            guided_p["filter_2_resonance"] = float(max(0.0, p["filter_2_resonance"] - 0.25))
    elif c_tgt < 600.0 and p["cutoff"] > 0.55:
        guided_p["cutoff"] = float(np.clip(p["cutoff"] - 0.20, 0.0, 1.0))
        guided_p["filter_2_cutoff"] = float(np.clip(p["filter_2_cutoff"] - 0.15, 0.0, 1.0))

    # 3. FEG mod amount guidance
    if c_tgt > 1000.0 and p.get("feg_amount", 0.5) < 0.40:
        guided_p["feg_amount"] = float(np.clip(p.get("feg_amount", 0.5) + 0.25, 0.0, 1.0))
        guided_p["filter_2_feg_amount"] = float(np.clip(p.get("filter_2_feg_amount", 0.3) + 0.20, 0.0, 1.0))
    elif c_tgt < 400.0 and p.get("feg_amount", 0.5) > 0.70:
        guided_p["feg_amount"] = float(max(0.20, p.get("feg_amount", 0.5) - 0.30))
        guided_p["filter_2_feg_amount"] = float(max(0.20, p.get("filter_2_feg_amount", 0.5) - 0.30))

    # 4. Filter Keytracking guidance (for higher register notes)
    if midi_note > 45 and abs(p.get("keytrack_raw", 0.5) - 0.50) < 0.08:
        guided_p["keytrack_raw"] = 0.75
        guided_p["filter_2_keytrack"] = 0.75
        
    v5_guided = S5.patch_to_vector(guided_p)
    v6_guided, _ = S6.from_v5(v5_guided, midi=midi_note)
    return v6_guided

def build_shard(flow, pool, device, targets, midi, k_passive=4, steps=25, tau=1.0, seed=0, true_vecs=None):
    """Sample passive flow candidates + inject active guided candidates -> [n, K, P] and scores."""
    mel = mel_of(targets, device)
    # 1. Sample passive flow candidates
    passive_cands = sample_candidates(flow, mel, midi, k=k_passive, steps=steps, seed=seed, tau=tau)
    n, K_p, P = passive_cands.shape
    
    # 2. Generate active guided candidate from passive candidate 0
    guided_cands = []
    for i in range(n):
        g = guide_candidate_patch(passive_cands[i, 0], targets[i], midi[i])
        guided_cands.append(g)
    guided_cands = np.stack(guided_cands)[:, None, :] # [n, 1, P]
    
    # Combine candidates: total K = K_p + 1 (e.g. 4 passive + 1 guided = 5)
    all_cands = np.concatenate([passive_cands, guided_cands], axis=1) # [n, K_tot, P]
    n, K_tot, P = all_cands.shape
    
    # 3. Render all candidates in parallel
    cands_flat = all_cands.reshape(n * K_tot, P)
    midi_flat = np.repeat(midi, K_tot)
    rendered_audio = pool.render(cands_flat, midi_flat)
    
    # 4. Score all candidates against target
    tgt_expanded = [targets[i] for i in range(n) for _ in range(K_tot)]
    metrics = ["logmel_l1", "mss", "mel_ssim_dist"]
    sc = am6.score_many(tgt_expanded, list(rendered_audio), metrics)
    
    out = {"cand_vecs": all_cands, **{f"score__{m}": v.reshape(n, K_tot) for m, v in sc.items()}}
    
    if true_vecs is not None:
        again = pool.render(true_vecs, midi)
        fl = am6.score_many(list(targets), list(again), metrics)
        out.update({f"noise__{m}": v for m, v in fl.items()})
        
    return out

def main():
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
    ap.add_argument("--out", required=True, help="output directory for DPO preference shards")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    flow = load_inverter_v4(a.flow, device=a.device)
    layout = layout_for_flow(flow)
    
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
    }
    
    shards = []
    t0 = time.time()
    n_shards = math.ceil(a.n / a.shard)
    
    print(f"Generating {a.n} guided DPO conditions ({n_shards} shards) to {a.out}...")
    with RenderPool(a.plugin, workers=a.workers, spec="v6") as pool:
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
                             steps=a.steps, tau=a.tau, seed=a.seed * 100_003 + s, true_vecs=true_vecs)
            sh.update({"target_audio": to_int16(targets), "midi": midi, "cond_index": np.arange(lo, hi), "true_vecs": true_vecs})
            
            np.savez(path + ".tmp.npz", **sh)
            os.replace(path + ".tmp.npz", path)
            shards.append(sh)
            el = time.time() - t0
            print(f"Shard {s + 1}/{n_shards}: {hi}/{a.n} conditions in {el:.0f}s ({hi / max(el, 1e-9):.1f} cond/s)", flush=True)

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nGuided DPO Preference Generation Complete! Stored in {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
