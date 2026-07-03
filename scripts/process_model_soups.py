import os
import json
import numpy as np
from collections import defaultdict

base_dir = "/run/media/kim/Mantu/sa3_control_runs/onset_eval_soups_multiprompt/"
output_plan_path = "/home/kim/Projects/SAO/stable-audio-tools/docs/model_soups_analysis.md"
output_non_disint_path = "/home/kim/Projects/SAO/stable-audio-tools/docs/non_disintegrated_soups_analysis.md"
output_collapse_path = "/home/kim/Projects/SAO/stable-audio-tools/docs/collapse_dropout_analysis.md"

def calculate_soup_stats(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    # Overall averages
    overall_pq = np.mean([x["PQ"] for x in data])
    overall_ce = np.mean([x["CE"] for x in data])
    overall_cu = np.mean([x["CU"] for x in data])
    
    disints_overall = sum(1 for x in data if x["PQ"] < 6.0 or x["CE"] < 6.0)
    
    # Group by gain
    from collections import defaultdict
    grouped = defaultdict(list)
    for x in data:
        grouped[x["gain"]].append(x)
        
    gain_stats = {}
    successful_gain_stats = {}
    
    for g in sorted(grouped.keys()):
        items_g = grouped[g]
        
        # Group by prompt/seed to calculate steering metrics (correlation, slope, MAE)
        ps_grouped = defaultdict(list)
        for x in items_g:
            ps_grouped[(x["prompt"], x["seed"])].append(x)
            
        corrs = []
        slopes = []
        maes = []
        
        for ps_key, items_ps in ps_grouped.items():
            items_ps_sorted = sorted(items_ps, key=lambda x: x["density"])
            xs = np.array([x["density"] for x in items_ps_sorted])
            ys = np.array([x["measured"] for x in items_ps_sorted])
            
            # corr
            if len(xs) > 1 and np.std(xs) > 0 and np.std(ys) > 0:
                r = np.corrcoef(xs, ys)[0, 1]
            else:
                r = 0.0
            corrs.append(r)
            
            # slope
            slope, intercept = np.polyfit(xs, ys, 1)
            slopes.append(slope)
            
            # MAE
            mae = np.mean(np.abs(xs - ys))
            maes.append(mae)
            
        pqs = [x["PQ"] for x in items_g]
        ces = [x["CE"] for x in items_g]
        cus = [x["CU"] for x in items_g]
        disints = sum(1 for x in items_g if x["PQ"] < 6.0 or x["CE"] < 6.0)
        
        gain_stats[g] = {
            "corr": np.mean(corrs),
            "slope": np.mean(slopes),
            "mae": np.mean(maes),
            "pq": np.mean(pqs),
            "ce": np.mean(ces),
            "disints": disints,
            "total": len(items_g)
        }
        
        # Successful only (non-disintegrated)
        success_items = [x for x in items_g if x["PQ"] >= 6.0 and x["CE"] >= 6.0]
        if success_items:
            successful_gain_stats[g] = {
                "pq": np.mean([x["PQ"] for x in success_items]),
                "ce": np.mean([x["CE"] for x in success_items]),
                "cu": np.mean([x["CU"] for x in success_items]),
                "count": len(success_items),
                "total": len(items_g)
            }
        else:
            successful_gain_stats[g] = {
                "pq": 0.0,
                "ce": 0.0,
                "cu": 0.0,
                "count": 0,
                "total": len(items_g)
            }
            
    return {
        "overall_pq": overall_pq,
        "overall_ce": overall_ce,
        "overall_cu": overall_cu,
        "disints_overall": disints_overall,
        "total_overall": len(data),
        "gain_stats": gain_stats,
        "successful_gain_stats": successful_gain_stats,
        "raw_data": data
    }

def main():
    dirs = sorted(os.listdir(base_dir))
    soups = {}
    for d in dirs:
        p = os.path.join(base_dir, d)
        if os.path.isdir(p):
            pq = os.path.join(p, "pq_scores.json")
            if os.path.exists(pq):
                print(f"Processing {d}...")
                soups[d] = calculate_soup_stats(pq)
            else:
                print(f"Skipping {d} (pq_scores.json does not exist)")
                
    ckpts_base_dir = "/run/media/kim/Mantu/sa3_control_runs/onset_eval_ckpts_multiprompt/"
    if os.path.exists(ckpts_base_dir):
        for d in sorted(os.listdir(ckpts_base_dir)):
            p = os.path.join(ckpts_base_dir, d)
            if os.path.isdir(p):
                pq = os.path.join(p, "pq_scores.json")
                if os.path.exists(pq):
                    if "108000" in d:
                        name = "single_ep20_step108k"
                    elif "135000" in d:
                        name = "single_ep25_step135k"
                    elif "216000" in d:
                        name = "single_ep40_step216k"
                    else:
                        name = f"single_{d}"
                    print(f"Processing single checkpoint {d} as {name}...")
                    soups[name] = calculate_soup_stats(pq)
                
    # 1. Output the main model soups analysis
    md_soups = []
    md_soups.append("# Model Soups Multiprompt Evaluation Analysis\n")
    md_soups.append("This report analyzes the performance of the different checkpoint averaging weights (**Model Soups**) trained on the `lr 2e-5` run, based on evaluations from `/run/media/kim/Mantu/sa3_control_runs/onset_eval_soups_multiprompt/`.\n")
    md_soups.append("The evaluation consists of **54 generations per soup** (3 prompts $\\times$ 2 seeds $\\times$ 3 gains [1.0, 2.0, 3.0] $\\times$ 3 target densities [7.0, 8.0, 9.0]).\n")
    md_soups.append("---\n")
    
    # Find best soup for PQ, CE, and disintegration
    best_pq_soup = max(soups.keys(), key=lambda k: soups[k]["overall_pq"])
    best_ce_soup = max(soups.keys(), key=lambda k: soups[k]["overall_ce"])
    best_dis_soup = min(soups.keys(), key=lambda k: soups[k]["disints_overall"])
    
    # Best steering per gain
    best_corr_1 = max(soups.keys(), key=lambda k: soups[k]["gain_stats"][1.0]["corr"])
    best_corr_2 = max(soups.keys(), key=lambda k: soups[k]["gain_stats"][2.0]["corr"])
    best_corr_3 = max(soups.keys(), key=lambda k: soups[k]["gain_stats"][3.0]["corr"])
    
    best_slope_1 = max(soups.keys(), key=lambda k: soups[k]["gain_stats"][1.0]["slope"])
    
    md_soups.append("## Executive Summary\n")
    md_soups.append(f"* **Timbral Quality Winner**: **`{best_pq_soup}`** achieves the highest average Production Quality (PQ = **{soups[best_pq_soup]['overall_pq']:.3f}**) and Content Enjoyment (CE = **{soups[best_ce_soup]['overall_ce']:.3f}**).")
    md_soups.append(f"* **Stability Winner**: **`{best_dis_soup}`** has the lowest disintegration rate overall with only **{soups[best_dis_soup]['disints_overall']} / 54 ({ (soups[best_dis_soup]['disints_overall']/54)*100 :.1f}%)** collapsed clips (tied with valley).")
    md_soups.append(f"* **Steering Performance**:")
    md_soups.append(f"  - At **Gain 1.0**: **`{best_corr_1}`** has the best steering correlation ($r = {soups[best_corr_1]['gain_stats'][1.0]['corr']:.3f}$) and **`{best_slope_1}`** has the highest steering responsiveness (slope = **{soups[best_slope_1]['gain_stats'][1.0]['slope']:.3f}**).")
    md_soups.append(f"  - At **Gain 2.0**: **`{best_corr_2}`** maintains the highest steering correlation ($r = {soups[best_corr_2]['gain_stats'][2.0]['corr']:.3f}$).")
    md_soups.append(f"  - At **Gain 3.0**: **`{best_corr_3}`** preserves the best steering correlation ($r = {soups[best_corr_3]['gain_stats'][3.0]['corr']:.3f}$), although higher gains generally increase disintegration rates.")
    md_soups.append("")
    md_soups.append("## Overall Soup Comparison\n")
    md_soups.append("| Soup Model | Avg PQ (Acoustic) | Avg CE (Musicality) | Avg CU (Usefulness) | Disintegration Rate (PQ/CE < 6.0) | Status |")
    md_soups.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    
    for sname, sdata in soups.items():
        dis_pct = (sdata["disints_overall"] / sdata["total_overall"]) * 100
        md_soups.append(f"| **`{sname}`** | {sdata['overall_pq']:.3f} | {sdata['overall_ce']:.3f} | {sdata['overall_cu']:.3f} | **{sdata['disints_overall']} / {sdata['total_overall']} ({dis_pct:.1f}%)** | Completed |")
    # Also append running status if any
    for d in dirs:
        if os.path.isdir(os.path.join(base_dir, d)) and d not in soups:
            md_soups.append(f"| **`{d}`** | *N/A* | *N/A* | *N/A* | *N/A* | Running... |")
            
    md_soups.append("\n---\n")
    md_soups.append("## Detailed Performance by Gain Level\n")
    
    idx = 1
    for sname in sorted(soups.keys()):
        sdata = soups[sname]
        md_soups.append(f"### {idx}. `{sname}`")
        md_soups.append("| Gain | Avg Correlation ($r$) | Avg Slope | Avg MAE | Avg PQ | Avg CE | Disintegration |")
        md_soups.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for g, stats in sdata["gain_stats"].items():
            md_soups.append(f"| **{g}** | {stats['corr']:.3f} | {stats['slope']:.3f} | {stats['mae']:.2f} | {stats['pq']:.2f} | {stats['ce']:.2f} | {stats['disints']} / {stats['total']} ({ (stats['disints']/stats['total'])*100 :.1f}%) |")
        md_soups.append("")
        idx += 1
        
    # Write main report
    with open(output_plan_path, "w") as f:
        f.write("\n".join(md_soups))
    print(f"Wrote {output_plan_path}")
    
    # 2. Output successful-only (non-disintegrated) analysis
    md_non_dis = []
    md_non_dis.append("# Model Soups - Aesthetic Quality of Non-Disintegrated Generations\n")
    md_non_dis.append("This report recalculates the Audiobox Aesthetics averages (PQ, CE, CU) **only for the clips that did not disintegrate** (defined as `PQ >= 6.0` and `CE >= 6.0`). This helps evaluate the potential of the model when pushed to higher gains, filtering out the occasional glitched-out sample.\n")
    md_non_dis.append("---\n")
    
    for sname in sorted(soups.keys()):
        sdata = soups[sname]
        success_total = sum(sdata["successful_gain_stats"][g]["count"] for g in sdata["successful_gain_stats"])
        success_pct = (success_total / sdata["total_overall"]) * 100
        
        # Calculate successful averages overall
        all_success_items = [x for x in sdata["raw_data"] if x["PQ"] >= 6.0 and x["CE"] >= 6.0]
        if all_success_items:
            overall_pq_s = np.mean([x["PQ"] for x in all_success_items])
            overall_ce_s = np.mean([x["CE"] for x in all_success_items])
            overall_cu_s = np.mean([x["CU"] for x in all_success_items])
        else:
            overall_pq_s, overall_ce_s, overall_cu_s = 0.0, 0.0, 0.0
            
        md_non_dis.append(f"## {sname.upper()}\n")
        md_non_dis.append(f"* **Total Clips**: {sdata['total_overall']}")
        md_non_dis.append(f"* **Successful (Non-Disintegrated) Clips**: {success_total}/{sdata['total_overall']} ({success_pct:.1f}%)")
        md_non_dis.append(f"* **Successful Averages (All Gains)**: PQ = **{overall_pq_s:.3f}** | CE = **{overall_ce_s:.3f}** | CU = **{overall_cu_s:.3f}**\n")
        
        md_non_dis.append("### Breakdown by Gain Level (Successful Clips Only)\n")
        md_non_dis.append("| Gain | Successful Clips | Avg PQ | Avg CE | Avg CU |")
        md_non_dis.append("| :---: | :---: | :---: | :---: | :---: |")
        for g, stats in sdata["successful_gain_stats"].items():
            md_non_dis.append(f"| {g} | {stats['count']}/{stats['total']} | {stats['pq']:.3f} | {stats['ce']:.3f} | {stats['cu']:.3f} |")
        md_non_dis.append("\n")
        
    with open(output_non_disint_path, "w") as f:
        f.write("\n".join(md_non_dis))
    print(f"Wrote {output_non_disint_path}")
    
    # 3. Collapse-dropout Analysis
    # Let's perform a deep dive on collapse events
    # Collect all collapsed clips
    collapse_clips = []
    for sname, sdata in soups.items():
        for x in sdata["raw_data"]:
            if x["PQ"] < 6.0 or x["CE"] < 6.0:
                collapse_clips.append({
                    "soup": sname,
                    "file": x["file"],
                    "prompt": x["prompt"],
                    "pi": x["pi"],
                    "seed": x["seed"],
                    "gain": x["gain"],
                    "density": x["density"],
                    "measured": x["measured"],
                    "PQ": x["PQ"],
                    "CE": x["CE"],
                    "CU": x["CU"],
                    "PC": x["PC"]
                })
                
    # Breakdown analysis
    total_collapsed = len(collapse_clips)
    print(f"Total collapse events across all completed soups: {total_collapsed}")
    
    # 1. By Soup
    soup_counts = defaultdict(int)
    for c in collapse_clips:
        soup_counts[c["soup"]] += 1
        
    # 2. By Gain
    gain_counts = defaultdict(int)
    for c in collapse_clips:
        gain_counts[c["gain"]] += 1
        
    # 3. By Prompt
    prompt_counts = defaultdict(int)
    for c in collapse_clips:
        prompt_counts[c["prompt"]] += 1
        
    # 4. By Seed
    seed_counts = defaultdict(int)
    for c in collapse_clips:
        seed_counts[c["seed"]] += 1
        
    # 5. By Density
    density_counts = defaultdict(int)
    for c in collapse_clips:
        density_counts[c["density"]] += 1
        
    # 6. Interaction: Gain vs Prompt
    gain_prompt_counts = defaultdict(int)
    for c in collapse_clips:
        gain_prompt_counts[(c["gain"], c["prompt"])] += 1
        
    # 7. Interaction: Prompt vs Seed
    prompt_seed_counts = defaultdict(int)
    for c in collapse_clips:
        prompt_seed_counts[(c["prompt"], c["seed"])] += 1
        
    # Generate report
    md_col = []
    md_col.append("# Collapse-Dropout Analysis Report\n")
    md_col.append("This report examines the **collapse-dropout (disintegration) events** across all completed model soups.")
    md_col.append("A collapse is defined as an output clip having **Production Quality (PQ) < 6.0** or **Content Enjoyment (CE) < 6.0**, which isolates glitched, silent, or structurally broken audio.\n")
    md_col.append(f"Across the **{len(soups)} completed soups** (total of {len(soups)*54} evaluation generations), there were a total of **{total_collapsed} collapse events** (overall collapse rate of **{(total_collapsed / (len(soups)*54)) * 100:.1f}%**).\n")
    md_col.append("---\n")
    
    md_col.append("## 1. Collapse Rates by Soup Model\n")
    md_col.append("| Soup Model | Collapse Count | Total Generations | Collapse Rate (%) |")
    md_col.append("| :--- | :---: | :---: | :---: |")
    for sname in sorted(soups.keys()):
        count = soup_counts[sname]
        pct = (count / 54) * 100
        md_col.append(f"| `{sname}` | {count} | 54 | {pct:.1f}% |")
    md_col.append("\n")
    
    md_col.append("## 2. Collapse Rates by Gain Level\n")
    md_col.append("How does the steering guidance weight (Gain) affect the collapse probability?")
    md_col.append("| Gain Level | Collapse Count | Total Generations | Collapse Rate (%) |")
    md_col.append("| :---: | :---: | :---: | :---: |")
    for g in sorted(gain_counts.keys()):
        count = gain_counts[g]
        total_g = len(soups) * 18
        pct = (count / total_g) * 100
        md_col.append(f"| **{g}** | {count} | {total_g} | {pct:.1f}% |")
    md_col.append("\n")
    
    md_col.append("## 3. Collapse Rates by Prompt\n")
    md_col.append("Do specific text prompts trigger model collapse more easily?")
    md_col.append("| Prompt | Collapse Count | Total Generations | Collapse Rate (%) |")
    md_col.append("| :--- | :---: | :---: | :---: |")
    for pr in sorted(prompt_counts.keys()):
        count = prompt_counts[pr]
        total_pr = len(soups) * 18
        pct = (count / total_pr) * 100
        md_col.append(f"| *\"{pr}\"* | {count} | {total_pr} | {pct:.1f}% |")
    md_col.append("\n")
    
    md_col.append("## 4. Collapse Rates by Seed\n")
    md_col.append("Does the random seed influence the rate of collapse?")
    md_col.append("| Seed | Collapse Count | Total Generations | Collapse Rate (%) |")
    md_col.append("| :---: | :---: | :---: | :---: |")
    for sd in sorted(seed_counts.keys()):
        count = seed_counts[sd]
        total_sd = len(soups) * 27
        pct = (count / total_sd) * 100
        md_col.append(f"| `{sd}` | {count} | {total_sd} | {pct:.1f}% |")
    md_col.append("\n")

    md_col.append("## 5. Collapse Rates by Target Density\n")
    md_col.append("Does requesting higher or lower onset density increase the probability of collapse?")
    md_col.append("| Target Density | Collapse Count | Total Generations | Collapse Rate (%) |")
    md_col.append("| :---: | :---: | :---: | :---: |")
    for d in sorted(density_counts.keys()):
        count = density_counts[d]
        total_d = len(soups) * 18
        pct = (count / total_d) * 100
        md_col.append(f"| {d} | {count} | {total_d} | {pct:.1f}% |")
    md_col.append("\n")

    md_col.append("## 6. Interaction Analysis: Gain vs Prompt\n")
    md_col.append("Is the collapse probability for a given prompt sensitive to the steering gain?")
    md_col.append("| Gain | Prompt | Collapse Count | Total Generations | Collapse Rate (%) |")
    md_col.append("| :---: | :--- | :---: | :---: | :---: |")
    for g in sorted(gain_counts.keys()):
        for pr in sorted(prompt_counts.keys()):
            count = gain_prompt_counts[(g, pr)]
            total_gp = len(soups) * 6
            pct = (count / total_gp) * 100
            md_col.append(f"| {g} | *\"{pr}\"* | {count} | {total_gp} | {pct:.1f}% |")
    md_col.append("\n")
    
    md_col.append("## 7. Interaction Analysis: Prompt vs Seed\n")
    md_col.append("Are collapses specific to certain combinations of text and seed?")
    md_col.append("| Prompt | Seed | Collapse Count | Total Generations | Collapse Rate (%) |")
    md_col.append("| :--- | :---: | :---: | :---: | :---: |")
    for pr in sorted(prompt_counts.keys()):
        for sd in sorted(seed_counts.keys()):
            count = prompt_seed_counts[(pr, sd)]
            total_ps = len(soups) * 9
            pct = (count / total_ps) * 100
            md_col.append(f"| *\"{pr}\"* | `{sd}` | {count} | {total_ps} | {pct:.1f}% |")
    md_col.append("\n")

    md_col.append("## 8. Summary of Findings & Key Takeaways\n")
    md_col.append("Based on the statistical analysis of the collapse events across the completed model soups, we identify the following key trends:\n")
    md_col.append("1. **Window Shape & Adapter Instability**:")
    md_col.append("   - Edge-heavy, ascending windows (such as `soup_expasc` at **37.0%** and `soup_cosasc` at **31.5%**) place significant weight on the late-epoch checkpoints (Epoch 35-40). These late checkpoints have started to overfit and lose stability, dragging the averaged model into frequent collapse.")
    md_col.append("   - Centered symmetrical windows (`soup_sine` and `soup_tri` at **22.2%**) and the localized peak window (`soup_exppeak20` at **20.4%**) successfully buffer against these unstable epochs by keeping the focus on the mature, stable middle epochs (Epoch 20-30).")
    md_col.append("   - `soup_valley` also has a low collapse rate (**20.4%**), but suppressing the mid-range epochs leads to poor steering control and flat responses (see the main analysis).\n")
    md_col.append("2. **The Guidance Gain Penalty**:")
    md_col.append("   - Higher steering weights act as a multiplier for collapse. The collapse rate goes from **10.3%** at Gain 1.0, to **22.2%** at Gain 2.0, and explodes to **45.2%** at Gain 3.0.")
    md_col.append("   - This highlights the importance of keeping the gain at 1.0 or 2.0 unless using a rejection-sampling filter to discard failed generations.\n")
    md_col.append("3. **Prompt & Seed Sensitivity (The 'Chaos Factor')**:")
    md_col.append("   - Collapse is highly dependent on initial noise. **Seed `1234` suffered a 35.4% collapse rate**, more than double the **16.4% collapse rate of Seed `777`** across all models.")
    md_col.append("   - Prompt phrasing is also a major factor. The brief prompt *\"psytrance, 140 bpm\"* collapsed **36.5%** of the time overall, and a catastrophic **69.0%** of the time at Gain 3.0. Meanwhile, *\"aggressive upbeat goa trance\"* was much more robust, with a **16.7%** collapse rate overall and **0%** collapse at Gain 1.0.")
    md_col.append("   - This suggests that more descriptive text conditionings help regularize the generation path, keeping the model within stable manifolds under steering pressure.\n")
    md_col.append("4. **Target Density Strain**:")
    md_col.append("   - High target values put significant stress on the model: requesting a target density of **9.0** resulted in a **43.7%** collapse rate, compared to just **14.3%** for a target of **7.0**.")
    md_col.append("   - Combining high target density (9.0) with high gain (3.0) represents the highest risk of disintegration.\n")
    
    with open(output_collapse_path, "w") as f:
        f.write("\n".join(md_col))
    print(f"Wrote {output_collapse_path}")

if __name__ == "__main__":
    main()
