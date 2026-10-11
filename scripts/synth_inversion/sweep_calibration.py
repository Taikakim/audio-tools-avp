import numpy as np
import surge_spec_v3
from inharmonicity_target import InharmonicityConfig, InharmonicityExtractor
import itertools
from scipy.stats import spearmanr
import json

def get_clean_saw():
    p = surge_spec_v3.draw_patch(42)
    p = surge_spec_v3.vector_to_patch(surge_spec_v3.patch_to_vector(p)) # reset FX etc
    # Reset FX and mods but keep tuning
    p["shape"] = 0.0
    p["width"] = 0.5
    p["cutoff"] = 1.0
    p["resonance"] = 0.0
    p["aeg_sustain"] = 1.0
    p["aeg_decay"] = 0.8
    p["aeg_release"] = 0.3
    p["feg_sustain"] = 1.0
    p["drive_raw"] = 0.0
    p["fm_depth"] = 0.0
    p["unison"] = False
    p["unison_detune"] = 0.0
    p["sync"] = 0.0
    p["chorus_mix"] = 0.0
    p["delay_mix"] = 0.0
    p["sub_mix"] = 0.0
    p["noise_volume"] = 0.0
    p["ring_1x2_volume"] = 0.0
    p["ring_2x3_volume"] = 0.0
    p["osc_2_volume"] = 0.0
    p["osc_3_volume"] = 0.0
    p["osc_1_volume"] = 1.0
    p["osc_1_octave"] = 0.5
    return p

def main():
    plugin = surge_spec_v3.init_synth()
    clean = get_clean_saw()
    
    factors = {
        "fm_depth": np.linspace(0, 1, 9),
        "unison_detune": np.linspace(0, 1, 9),
        "osc_1_octave": np.linspace(0, 1, 9),
        "sync": np.linspace(0, 1, 9),
        "noise_color": np.linspace(0, 1, 9),
        "drive": np.linspace(0, 1, 9)
    }
    notes = [24, 28, 32, 36, 40, 44, 48]
    
    grid = {
        "tolerance": [0.10, 0.15, 0.20, 0.25],
        "max_harmonics": [20, 40, 80],
        "max_excluded_energy_fraction": [0.05, 0.10, 0.15, 0.25]
    }
    
    # Prerender audio
    print("Rendering audio...")
    audio_data = {}
    for factor, values in factors.items():
        for val in values:
            for note in notes:
                p = dict(clean)
                p[factor] = float(val)
                audio_data[(factor, val, note)] = surge_spec_v3.render_patch(plugin, p, note, 1.0)
    
    # Also render clean class
    clean_audio = {note: surge_spec_v3.render_patch(plugin, clean, note, 1.0) for note in notes}
    
    cells = list(itertools.product(grid["tolerance"], grid["max_harmonics"], grid["max_excluded_energy_fraction"]))
    results = []
    
    for tol, mh, meef in cells:
        import json
        with open('scripts/synth_inversion/inharm_config.json', 'r') as cf: cfg_dict = json.load(cf)
        cfg_dict.update({'tolerance': tol, 'max_harmonics': mh, 'max_excluded_energy_fraction': meef})
        cfg = InharmonicityConfig(**cfg_dict)
        ext = InharmonicityExtractor(cfg)
        
        # Clean class
        clean_valid = []
        clean_vals = []
        peaks_100 = 0
        total_wins = 0
        
        for note in notes:
            r = ext.process_phrase(clean_audio[note], note)
            for w in r["window_results"]:
                if w is None: continue
                total_wins += 1
                clean_valid.append(w["valid"])
                if w["valid"]:
                    clean_vals.append(w["value"])
                if "coverage_diagnostics" in w and w["coverage_diagnostics"].get("input_peak_count") == 100:
                    peaks_100 += 1
                    
        clean_vr = np.mean(clean_valid)
        clean_p95 = np.percentile(clean_vals, 95) if clean_vals else 0.0
        
        # False harmonics
        fh_count = 0
        fh_total = 0
        
        # Tracking
        spearman_pass = True
        for factor, values in factors.items():
            if factor not in ["fm_depth", "unison_detune"]: continue
            for note in notes:
                f_vals = []
                f_true = []
                for val in values:
                    r = ext.process_phrase(audio_data[(factor, val, note)], note)
                    for w in r["window_results"]:
                        if w is None: continue
                        if w["valid"]:
                            f_vals.append(w["value"])
                            f_true.append(val)
                            
                            if w["value"] < clean_p95 and val > 0:
                                fh_count += 1
                        fh_total += 1
                if len(f_vals) > 2:
                    rho, _ = spearmanr(f_vals, f_true)
                    if abs(rho) < 0.7:
                        spearman_pass = False
                else:
                    spearman_pass = False
        
        fh_rate = fh_count / fh_total if fh_total > 0 else 1.0
        
        # Pitch neutrality
        note_vrs = []
        for note in notes:
            vrs = [w["valid"] for w in ext.process_phrase(clean_audio[note], note)["window_results"] if w is not None]
            note_vrs.append(np.mean(vrs) if vrs else 0.0)
        pn_spread = max(note_vrs) - min(note_vrs)
        pn_rho, _ = spearmanr(clean_vals, [notes[i//2] for i in range(len(clean_vals))]) if len(clean_vals)>2 else (1.0, 1.0)
        
        # Gates
        g1 = fh_rate <= 0.05
        g2 = clean_vr >= 0.90
        g3 = spearman_pass
        g4 = pn_spread <= 0.20 and abs(pn_rho) <= 0.3
        
        pass_all = g1 and g2 and g3 and g4
        
        results.append({
            "tol": tol, "mh": mh, "meef": meef,
            "clean_vr": clean_vr,
            "fh_rate": fh_rate,
            "spearman_pass": spearman_pass,
            "pn_spread": pn_spread,
            "pn_rho": pn_rho,
            "pass": pass_all,
            "peaks_100_share": peaks_100 / total_wins if total_wins else 0.0
        })
        
        print(f"Tol {tol:.2f} MH {mh:2d} MEEF {meef:.2f} | Pass: {pass_all} | CleanVR: {clean_vr:.2f} | FH: {fh_rate:.2f} | Sprm: {spearman_pass} | PN_s: {pn_spread:.2f} | PN_r: {pn_rho:.2f} | P100: {peaks_100/total_wins:.2f}")

if __name__ == "__main__":
    main()
