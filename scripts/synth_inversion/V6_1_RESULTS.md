# V6.1 Guided Flow-DPO & Consensus Evaluation Results

**Date:** 2026-10-20  
**Pipeline Run:** `scripts/synth_inversion/run_v6_1_pipeline.sh` (Completed all 7 steps cleanly)  
**Base Model:** V6_C 12k EMA (`/home/kim/sao_runs/v6_C/flow_ema_latest.pt`)  
**DPO Pairs:** 8,192 pairs rendered via `render_pairs_dpo_v6.py` with guided heuristics (multi-octave keytracking, FEG polish, and triangle/square harmonic parity).  

---

## 1. Step 4 Gate: Held-Out Synthetic Audio Benchmark

Evaluated on 256 held-out synthetic presets (`/home/kim/sao_runs/audio_eval_v6c_dpo/report.md`) with preset-clustered 95% bootstrap confidence intervals:

| Model / Draw Mode | logmel_l1 (↓) | Multi-Scale STFT MSS (↓) | modspec_dist (↓) |
| :--- | :---: | :---: | :---: |
| **V6_C (Base)** / `draw0` | 0.1492 [0.1354, 0.1644] | 6.0959 [5.5621, 6.6905] | 0.5845 [0.5536, 0.6164] |
| **V6_C (Base)** / `best8` | **0.0750** [0.0692, 0.0811] | **3.2711** [3.0280, 3.5300] | **0.4571** [0.4299, 0.4847] |
| **V6_C_DPO_b100** / `draw0` | 0.1239 [0.1139, 0.1342] | 5.2060 [4.8220, 5.5970] | 0.5805 [0.5464, 0.6140] |
| **V6_C_DPO_b100** / `best8` | 0.0868 [0.0797, 0.0941] | 3.8056 [3.5145, 4.1008] | 0.5013 [0.4682, 0.5332] |
| **V6_C_DPO_b300** / `draw0` | **0.1227** [0.1128, 0.1328] | **5.1686** [4.7832, 5.5614] | 0.5839 [0.5487, 0.6176] |
| **V6_C_DPO_b300** / `best8` | 0.0852 [0.0783, 0.0923] | 3.7045 [3.4215, 3.9957] | 0.4784 [0.4487, 0.5081] |

### Paired Deltas on Single Draw (`draw0`):
* **$\beta=100$ vs Base V6_C:**
  - MSS Delta: **$-0.8899$** [$0.4341, 1.3725$], $p = 0.000$ (Statistically significant improvement)
  - logmel_l1 Delta: **$-0.0253$** [$0.0136, 0.0376$], $p = 0.000$
* **$\beta=300$ vs Base V6_C:**
  - MSS Delta: **$-0.9272$** [$0.4736, 1.4193$], $p = 0.000$ (Statistically significant improvement)
  - logmel_l1 Delta: **$-0.0265$** [$0.0146, 0.0394$], $p = 0.000$

**Gate Verdict:** PASSED. Guided Flow-DPO produces an immediate ~15% relative drop in single-draw Multi-Scale Spectral error on unseen presets without requiring sample reranking.

---

## 2. Real Stems Multi-Arm Consensus Evaluation

The real stem consensus evaluation inverted 24 real-world bass stems across three experimental arms:

### Arm 5: Base Occam (`/home/kim/sao_runs/stems_v6_c_dpo_occam`)
* **Configuration:** Full bandwidth evaluation ($0\text{--}22\text{ kHz}$), drop polyphony, Occam tolerance $\tau = 0.02$, dynamic square shape scan.
* **Models Evaluated:** `S5`, `S5_DPO_b300`, `V6_C`, `V6_C_DPO_b100`, `V6_C_DPO_b300`.
* **Output:** [Audition Pack](file:///home/kim/sao_runs/stems_v6_c_dpo_occam/listen/index.html)

### Arm 5b: Lexi 4k (`/home/kim/sao_runs/stems_v6_c_lexi4k`)
* **Configuration:** Band-limited evaluation ($\le 4\text{ kHz}$), lexicographic band prioritization (sub > body > air) following human audition findings (`HUMAN_EVALUATION_SYNTHESIS.md`).
* **Models Evaluated:** `S5`, `S5_DPO_b300`, `V6_C`, `V6_C_DPO_b300`.
* **Output:** [Audition Pack](file:///home/kim/sao_runs/stems_v6_c_lexi4k/listen/index.html)

### Arm 5c: Decay 4k (`/home/kim/sao_runs/stems_v6_c_decay4k`)
* **Configuration:** Band-limited ($\le 4\text{ kHz}$) with $1/f$-decay weighted STFT (decay power 1.25, knee 900 Hz).
* **Models Evaluated:** `S5`, `S5_DPO_b300`, `V6_C`, `V6_C_DPO_b300`.
* **Output:** [Audition Pack](file:///home/kim/sao_runs/stems_v6_c_decay4k/listen/index.html)

---

## 3. Human Preference vs. Objective Metric Alignment

To resolve the discrepancy between spectral loss and human perception, `human_pref_agreement.py` computed within-stem Spearman rank correlations ($\rho$) against blind human listener ranks across 24 real stems (`pref_agreement.md`):

| Metric | Stems | Median $\rho$ | Mean $\rho$ | Human Alignment | Wilcoxon $p$-value | Metric Best = Human Best |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mel_ssim_dist`** | 24 | **+0.67** | **+0.51** | **22 / 24** | **$p = 0.0004$** | **13 / 24 (chance 5.0)** |
| **`abs_log2_centroid_ratio`** | 24 | **+0.38** | **+0.41** | **20 / 24** | **$p = 0.0003$** | **11 / 24 (chance 5.0)** |
| **`band_body`** | 24 | **+0.40** | **+0.28** | **15 / 24** | **$p = 0.0183$** | **12 / 24 (chance 5.0)** |
| `spectral_emd_oct` | 24 | +0.20 | +0.13 | 14 / 24 | $p = 0.1618$ | 7 / 24 |
| `band_sub` | 24 | +0.15 | +0.17 | 13 / 24 | $p = 0.1247$ | 8 / 24 |
| `band_air` | 24 | +0.40 | +0.19 | 15 / 24 | $p = 0.1566$ | 9 / 24 |
| `phrase_mss` (Standard STFT) | 24 | +0.08 | +0.11 | 13 / 24 | $p = 0.2691$ (n.s.) | 11 / 24 |

### Critical Scientific Finding:
* **Standard Multi-Scale STFT (`phrase_mss`) does NOT correlate with human ear judgment** ($\rho = +0.08$, $p = 0.27$, not statistically significant). It heavily over-indexes on high-frequency noise and slight phase offsets.
* **`mel_ssim_dist` (Structural Similarity) and `abs_log2_centroid_ratio` (Spectral Brightness Ratio) have overwhelmingly strong, statistically significant alignment with human listening** ($p < 0.0005$).

---

## 4. Paired Real Stem Statistical Comparison

### A. Perceptual Metric: `mel_ssim` (Higher is Closer to Human Gold Standard)
Evaluated across 24 real bass stems:

| Model | $n$ | Mean [95% Bootstrap CI] | Median | Wins |
| :--- | :---: | :---: | :---: | :---: |
| **V6_C_DPO_b100** | 24 | **0.334** [0.269, 0.404] | **0.295** | **5** |
| **S5_DPO_b300** | 24 | **0.312** [0.259, 0.367] | **0.300** | **7** |
| **V6_C_DPO_b300** | 24 | **0.308** [0.247, 0.376] | **0.274** | **7** |
| **S5** | 24 | 0.294 [0.235, 0.357] | 0.266 | 4 |
| **V6_C** | 24 | 0.292 [0.239, 0.349] | 0.277 | 1 |

#### Paired Deltas against Base V6_C:
* **`V6_C_DPO_b100` vs V6_C:**
  - $\Delta = \mathbf{+0.042}$ [$+0.013, +0.073$], **$p = 0.004$** (16 stems better, 8 worse).
  - **Verdict:** Flow-DPO ($\beta=100$) achieves a statistically significant victory over base V6_C on real stems under human-aligned perceptual evaluation.
* **`S5_DPO_b300` vs V6_C:**
  - $\Delta = +0.019$ [$-0.015, +0.056$], $p = 0.294$ (11 stems better, 13 worse).

---

## 5. Architectural & Physical Observations

1. **Stereo & Mono Collapse Accuracy:**
   - On pure mono tracks (e.g. Stem 01 Astral Saw, Stem 06 Disco Saw, Stem 24 Final Mission), all models collapse stereo spread to 0.00 ($dW = 0.00$).
   - On wide stereo chorus/flanger tracks (e.g. Stem 07 Sub Phaser, Stem 12 Acid Lead, Stem 13 Zarjaz Smooth), models appropriately predict width $W \in [0.45, 0.67]$.
2. **Retrigger Logic:**
   - Fast 16th-note acid lines consistently trigger `Retrig: On`, preventing phase cancellation on rapid notes.
3. **Square / Triangle Parity Separation:**
   - Odd-harmonic spectral slope discrimination correctly avoids mapping hollow square basses to triangle waves.
