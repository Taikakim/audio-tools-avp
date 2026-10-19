# V6.1 / Phase 4: Guided Flow-DPO with Multi-Octave Keytracking, FEG Polish, and Spectral Knee Guidance

**Owner:** Kuang / Kim  
**Base Model:** `/home/kim/sao_runs/v6_C/flow_ema_latest.pt` (V6_C 12k EMA; synthetic held-out draw0 MSS 6.07 / best8 MSS 3.30, real stems consensus mel_ssim_dist 0.284 where dist = 1 - SSIM)  
**Status:** Approved & Implemented; Unit-tested; Ready for GPU execution when available (~04:30)

---

## 1. Executive Summary & Problem Diagnosis

During rigorous stem-by-stem human auditioning of the 24 real bass stems across `S5`, `DPO_b300`, `V6_A`, `V6_B`, and `V6_C`, four fundamental physical degeneracies were identified:

1. **Cutoff vs. Resonance Degeneracy:**
   - Standard spectral centroid $\mu = \frac{\sum f |X|}{\sum |X|}$ calculates the center-of-mass of the spectrum.
   - A narrow, high-resonance ladder filter at 600 Hz with +18 dB $Q$ produces the exact same centroid as a wide-open 1.5 kHz filter with 0.0 resonance.
   - Consequently, the model often hallucinated screaming resonance peaks to reach target brightness instead of opening the cutoff shoulder.

2. **Multi-Octave Register Imbalance (Filter Keytracking):**
   - Transcribed phrases spanning multiple octaves (e.g. disco octave jumps, electro bass lines) require correct filter keytracking.
   - With 0% keytracking (`keytrack_raw = 0.50`), higher octave notes get choked and muffled because their fundamental rises toward or above the stationary cutoff.
   - With 100% keytracking (`keytrack_raw = 0.75`), filter cutoffs track pitch 1:1 (12 semitones per octave), preserving brightness across registers.
   - Evaluating keytracking is subtle because single isolated notes cannot reveal keytracking errors—it is only audible and measurable when playing notes across differing octaves in the full musical phrase.

3. **Envelope Sweep Depth (FEG Mod Amount):**
   - In stems like Stem 10 and 11, static filters or near-instantaneous 30 ms decay drops gave the perception of "lacking an FEG" or sounding overly bright.
   - Local polish must fine-tune `feg_amount` and `filter_2_feg_amount` around the predicted baseline to achieve proper dynamic attack snap and filter sweep depth.

4. **Gate Normalization (Decay vs. Release Disentanglement):**
   - Real stem note crops of 80–110 ms with 5 ms fades caused models trained on fixed 125 ms note gates to mistake short gate cuts for zero release or choked decays.
   - 16th-grid musical quantization maps note boundaries to whole rhythmic subdivisions, guaranteeing conditioning buffers conform to the model's 125 ms template.

5. **Harmonic Parity & Oscillator Geometry (Triangle vs. Saw vs. Square):**
   - In Surge XT's Classic Oscillator:
     * `shape = 0.00`: **Triangle wave** (odd harmonics only, amplitude decays as $1/k^2$, slope $\approx -2.39$).
     * `shape = 0.50`: **Saw wave** (all harmonics present, parity $\approx 0.289$, slope $\approx -2.39$).
     * `shape = 1.00`, width 0.50: **Symmetric square wave** (odd harmonics only, amplitude decays as $1/k$).
   - Parity alone ($P_{\text{harm}} < 0.12$) cannot differentiate triangle from square (both lack even harmonics). Parity is now coupled with odd-harmonic spectral slope ($1/k$ vs $1/k^2$) and requires harmonic richness $\ge 0.05$ (preventing near-sine low-passed signals from false-triggering parity penalties).
   - `--square_shape` is measured dynamically by `check_v6_spec.py --shape_scan` and passed to pair rendering; hardcoded defaults are disabled.

---

### Synthetic Held-Out Benchmark Reference (`audio_eval_v5.py`)

Held-out 256-preset synthetic test evaluation (preset-clustered 95% CI):

| System | Draw Mode | logmel_l1 (↓) | MultiScaleSTFT MSS (↓) | modspec_dist (↓) | bass_energy_dist (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **S5** | `draw0` | 0.1400 [0.128, 0.153] | 5.6860 [5.196, 6.204] | 0.6248 [0.587, 0.668] | 0.5023 [0.420, 0.603] |
| **S5** | `best8` | 0.0800 [0.073, 0.087] | 3.3800 [3.088, 3.699] | 0.5112 [0.476, 0.551] | 0.4108 [0.356, 0.474] |
| **V6_A** | `draw0` | 0.1723 [0.156, 0.190] | 7.0264 [6.396, 7.712] | 0.6179 [0.586, 0.651] | 0.4812 [0.424, 0.545] |
| **V6_A** | `best8` | 0.0820 [0.076, 0.089] | 3.5586 [3.303, 3.832] | 0.4887 [0.460, 0.519] | 0.4260 [0.375, 0.481] |
| **V6_B** | `draw0` | 0.1724 [0.157, 0.190] | 7.0539 [6.428, 7.736] | 0.6144 [0.582, 0.647] | 0.4804 [0.419, 0.546] |
| **V6_B** | `best8` | 0.0822 [0.076, 0.089] | 3.5447 [3.292, 3.812] | 0.4907 [0.461, 0.520] | 0.4340 [0.378, 0.496] |
| **V6_C** | `draw0` | 0.1485 [0.135, 0.163] | 6.0739 [5.551, 6.653] | 0.5887 [0.556, 0.621] | 0.4652 [0.408, 0.527] |
| **V6_C** | `best8` | **0.0753** [0.070, 0.081] | **3.3044** [3.057, 3.560] | **0.4713** [0.443, 0.500] | **0.4155** [0.365, 0.472] |

*Key Findings on Synthetic Held-Out:*
- Single draw (`draw0`): S5 leads on raw MSS (5.69 vs 6.07) due to smaller unconstrained parameter surface, but V6_C decisively outperforms V6_A (7.03) and V6_B (7.05).
- Reranked draw (`best8`): **V6_C outperforms all baselines**, achieving lowest MSS (3.30 vs 3.38 for S5, 3.56 for V6_A, 3.54 for V6_B) and lowest spectral distance (0.0753 vs 0.0800 for S5).
- Metric note: In real-stem consensus reporting, `mel_ssim_dist` is structural dissimilarity ($1 - \text{SSIM}$, lower is better, range $[0, 1]$). V6_C achieved 0.284 on real stems.

---

## 2. Core Algorithmic Architecture

### 2.1 Spectral Shoulder Detection (`audio_metrics_v6.py`)
- Computes Welch-averaged log-magnitude spectral envelope across the full audio clip.
- Note: Pure spectral shoulder on steep saw waves primarily follows the $1/k$ roll-off; local polish uses spectral centroid delta and band distances for robust guidance.

### 2.2 Harmonic Parity & Richness Metric
$$P_{\text{harm}} = \frac{\sum_{\text{even}} |X(k \cdot f_0)|}{\sum_{\text{odd}} |X(k \cdot f_0)|}$$
Evaluated only when harmonic richness $\frac{\sum_{k=2}^{10} |X(k \cdot f_0)|}{|X(f_0)|} \ge 0.05$. Square targets are confirmed when $P_{\text{harm}} < 0.12$ and odd-harmonic amplitude roll-off follows $1/k$ rather than $1/k^2$.

### 2.3 Directed Guidance in DPO Candidate Generation (`render_pairs_dpo_v6.py`)
During preference pair generation, passive candidate 0 is actively perturbed to generate a guided candidate:
1. **Cutoff Shift:** Steers cutoffs in unison toward target centroid/shoulder if error $> 120\text{ Hz}$.
2. **Parity Alignment:** When $P_{\text{harm}} < 0.12$ with $1/k$ roll-off and a measured `--square_shape` is available, biases oscillator toward that shape, mutes ring modulation, and damps secondary filter resonance.
3. **FEG Modulation:** Boosts `feg_amount` (+0.25) when bright transients require dynamic sweeps; damps FEG when low damped basses (<400 Hz) are over-modulated.
4. **Keytrack Guidance:** Biases higher register notes (MIDI > 45) toward standard 100% keytracking (`0.75`).

### 2.4 Multi-Parameter Consensus & Local Polish (`eval_real_stems_consensus_v6.py`)
Prior to Occam pruning, the winning candidate undergoes 3-stage local polish on the full audio playback:
1. **Cutoff Polish:** Steps cutoffs $\pm 5–15\%$ in the direction of spectral centroid delta.
2. **FEG Mod Amount Polish:** Explores $\Delta \text{FEG} \in [-0.15, -0.08, +0.08, +0.15]$ for primary and secondary filters.
3. **Multi-Octave Keytrack Polish:** When the phrase exhibits a pitch span $\ge 7$ semitones, evaluates keytrack options (`0.50` = 0%, `0.75` = 100%, `0.625` = 50%, `0.25` = -100%, `0.875` = 150%) to maximize cross-register phrase coherence.

---

## 4. Cumulative Spectral Prefix Prediction & Frequency-Decay Hierarchy

Formulated during the Stem 24 audit (*Chakra & Edi Mis – The Final Mission*), this paradigm resolves the fundamental vulnerability of neural synth inversion to source-separation artifacts.

### 4.1 The Core Problem: Linear FFT Dominance
Standard STFT loss treats all frequency bins with equal weight. In a 2048-point STFT at 44.1 kHz (~21.5 Hz per bin):
- **Sub-Bass (20–120 Hz)**: Only ~5 bins (0.5% of total).
- **Body (120–2500 Hz)**: ~110 bins (10.7% of total).
- **Air (> 2500 Hz)**: ~910 bins (88.8% of total).

When a real stem is separated from a full mix via Demucs or BS-Roformer, the upper 90% of bins are contaminated with residual cymbals, hi-hat bleed, and phase-cancellation noise (-40 to -60 dB). Unweighted loss forces optimizers to open filters to 0.95 and add FM/unison to fit this high-frequency noise, sacrificing the fundamental and filter shoulder.

### 4.2 Cumulative Spectral Energy Distribution (1D Wasserstein / EMD)
Instead of pointwise bin error, evaluate the spectrum iteratively from bin 0 upwards via the **Cumulative Energy Distribution (CDF)**:

$$\text{CDF}(f) = \frac{\sum_{j=0}^{k(f)} |X_j|^2}{\sum_{j=0}^{K} |X_j|^2}, \qquad \mathcal{L}_{\text{EMD}} = \int_0^{f_{\text{max}}} \left| \text{CDF}_{\text{pred}}(f) - \text{CDF}_{\text{real}}(f) \right| df$$

- **Mathematical Invariant**: Bin 0 (Sub/Fundamental) is included in every prefix term from $1 \dots K$. Higher bins only influence the tail.
- **Empirical Validation (Stem 24)**: Raw unweighted STFT favored `V6_A` (3.51) over `S5` (4.58) due to high-frequency noise matching Demucs hiss. Cumulative Spectral EMD penalized `V6_A` by **26×** (`0.2671` vs `0.0096` for S5), correctly matching human listening preference for the closed analog saw bass.

### 4.3 $1/f^\gamma$ Frequency-Decay STFT Loss
Scales the multi-scale spectral convergence and log-magnitude L1 by a frequency-decay envelope:

$$w(f) = \frac{1}{\left(1 + \frac{f}{f_{\text{knee}}}\right)^\gamma}, \qquad f_{\text{knee}} \approx 800\text{–}1000\text{ Hz}, \quad \gamma \in [1.0, 1.5]$$

- Fundamental ($50\text{ Hz}$): $w(f) \approx 1.00$
- Filter Shoulder ($800\text{ Hz}$): $w(f) \approx 0.50$
- Air / Separation Noise ($8000\text{ Hz}$): $w(f) \approx 0.03$ ($30\times$ down-weighted)

### 4.4 Coarse-to-Fine Frequency Hierarchy in Neural Architecture (V6.2 Roadmap)
Structured as a 3-stage frequency cascade:
1. **Stage 1 (Sub Band: 20–150 Hz)**: Predicts strictly fundamental parameters (`root_midi`, `osc_1_octave`, `sub_mix`) from sub-bins only. Completely blind to high frequencies.
2. **Stage 2 (Body Band: 150–2500 Hz)**: Conditioned on Stage 1, predicts `filter_idx`, `cutoff`, `resonance`, `shape` (saw vs. square), and `aeg_decay`.
3. **Stage 3 (Air Band: > 2500 Hz)**: Conditioned on Stages 1 & 2, refines fine edge parameters (noise, chorus, transient click). Crucially, Stage 3 has no gradient path to Stage 1 or Stage 2 outputs—it is mathematically forbidden from altering cutoff or octave to fit air.

---

## 5. Verification & Benchmark Plan

1. **Unit Verification:**
   - Run `audio_metrics_v6.py` unit tests for knee detection and parity computation.
   - Run `test_polish_keytrack_feg.py` to confirm bounds and parameter mapping.
2. **Training & Real Stems Pipeline:**
   - Launch `scripts/synth_inversion/run_v6_1_pipeline.sh` when GPU 0 is free:
     * Render 8,192 guided preference pairs via 14 CPU workers.
     * Train Flow-DPO on `V6_C` ($\beta=100$ and $\beta=300$).
     * Run 24 Real Stems Benchmark with `--max_eval_freq 4000` and frequency-decay weighting.
     * Build interactive audition page at `/home/kim/sao_runs/stems_v6_c_dpo_occam/listen/index.html`.

