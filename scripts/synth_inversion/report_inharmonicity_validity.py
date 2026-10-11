import argparse
import json
import numpy as np
from tqdm import tqdm
from realistic_bass_prior_v3 import MANIFOLD_PATH, RealisticBassPrior
import surge_spec_v3
from inharmonicity_target import InharmonicityExtractor, InharmonicityConfig, Reason, RejectionStats

def analyze_split(prior, split_name, n_samples, plugin):
    print(f"\nAnalyzing split: {split_name} ({n_samples} samples)")
    with open('scripts/synth_inversion/inharm_config.json', 'r') as cf: cfg_dict = json.load(cf)
    cfg = InharmonicityConfig(**cfg_dict)
    extractor = InharmonicityExtractor(cfg)
    
    # We will track all windows and the pooled label
    pooled_values = []
    window_valid = {1: [], 2: []}
    rejection_reasons = {1: {}, 2: {}}
    all_excluded = []
    max_peaks_hits = 0
    valid_by_note = {}
    labels_out = []
    
    for i in tqdm(range(n_samples)):
        # One draw per sample (prior returns patch dict)
        patch, _, midi_note, note_dur = prior.sample_patch_and_midi()
        
        # Pick a random midi note 24-48, or we can just sweep notes.
        # But wait, 'One draw per sample... valid rate by MIDI note'.
        # Let's draw a random note for the patch to keep it realistic.
        
        audio = surge_spec_v3.render_patch(plugin, patch, midi_note, note_dur)
        
        result = extractor.process_phrase(audio, midi_note)
        
        # record valid by note
        if midi_note not in valid_by_note:
            valid_by_note[midi_note] = {"total": 0, "valid": 0}
        valid_by_note[midi_note]["total"] += 1
        if result["valid"]:
            valid_by_note[midi_note]["valid"] += 1
            pooled_values.append(result["value"])
            labels_out.append({"index": i, "midi_note": midi_note, "value": result["value"]})
            
        for w_idx, w_res in enumerate(result["window_results"]):
            wid = w_idx + 1
            if wid > 2:
                break
            if w_res["valid"]:
                window_valid[wid].append(1)
            else:
                window_valid[wid].append(0)
                r = w_res["rejection_reason"]
                rejection_reasons[wid][r] = rejection_reasons[wid].get(r, 0) + 1
                
            if "coverage_diagnostics" in w_res:
                cd = w_res["coverage_diagnostics"]
                if cd.get("excluded_energy_fraction") is not None:
                    all_excluded.append(cd["excluded_energy_fraction"])
                if cd.get("input_peak_count") == 100:
                    max_peaks_hits += 1
    
    # Compute metrics
    pooled_valid_rate = len(pooled_values) / n_samples if n_samples > 0 else 0
    win_rates = {wid: float(np.mean(window_valid[wid])) if window_valid[wid] else 0.0 for wid in [1, 2]}
    
    if pooled_values:
        qs = np.quantile(pooled_values, [0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0])
        mean_val = float(np.mean(pooled_values))
        std_val = float(np.std(pooled_values))
        variance = float(np.var(pooled_values))
    else:
        qs = [0]*7
        mean_val = std_val = variance = 0.0
        
    ex_arr = np.array(all_excluded) if all_excluded else np.array([0.0])
    ex_shares = {
        "0.05": float(np.mean(ex_arr > 0.05)),
        "0.10": float(np.mean(ex_arr > 0.10)),
        "0.15": float(np.mean(ex_arr > 0.15)),
        "0.25": float(np.mean(ex_arr > 0.25)),
        "0.35": float(np.mean(ex_arr > 0.35))
    }
    
    note_rates = {note: valid_by_note[note]["valid"] / valid_by_note[note]["total"] 
                  for note in sorted(valid_by_note.keys())}
                  
    return {
        "pooled_valid_rate": pooled_valid_rate,
        "window_1_valid_rate": win_rates[1],
        "window_2_valid_rate": win_rates[2],
        "window_1_rejections": rejection_reasons[1],
        "window_2_rejections": rejection_reasons[2],
        "pooled_quantiles": {
            "min": float(qs[0]), "5%": float(qs[1]), "25%": float(qs[2]),
            "50%": float(qs[3]), "75%": float(qs[4]), "95%": float(qs[5]), "max": float(qs[6])
        },
        "pooled_mean": mean_val,
        "pooled_std": std_val,
        "pooled_variance_MSE": variance,
        "excluded_energy_fraction_shares": ex_shares,
        "max_peaks_100_cap_share": max_peaks_hits / len(all_excluded) if all_excluded else 0.0,
        "valid_rate_by_note": note_rates,
        "labels": labels_out
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifold", default=MANIFOLD_PATH)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="label_stats.json")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    plugin = surge_spec_v3.init_synth()
    
    prior_train = RealisticBassPrior(args.manifold, split="train", jitter=0.0)
    stats_train = analyze_split(prior_train, "train", args.n, plugin)
    
    prior_val = RealisticBassPrior(args.manifold, split="val", jitter=0.0)
    stats_val = analyze_split(prior_val, "val", args.n, plugin)
    
    out = {"train": stats_train, "val": stats_val}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        
    print(f"\nConstant-prediction MSE (Train): {stats_train['pooled_variance_MSE']:.5f}")
    print(f"Constant-prediction MSE (Val):   {stats_val['pooled_variance_MSE']:.5f}")

if __name__ == "__main__":
    main()
