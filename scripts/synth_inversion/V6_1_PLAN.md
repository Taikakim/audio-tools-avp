# V6.1 / Phase 4: Guided Flow-DPO with Multi-Octave Keytracking, FEG Polish, and Spectral Knee Guidance

**Owner:** Kuang / Kim  
**Base Model:** `/home/kim/sao_runs/v6_C/flow_ema_latest.pt` (V6_C 12k EMA, held-out MSS 6.07 / SSIM 0.284)  
**Status:** Approved & Implemented; Scheduled for GPU execution when available (~04:30)

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

5. **Harmonic Parity Degeneracy (Square vs. Saw):**
   - Even harmonic energy is zero for symmetric square waves ($P_{\text{harm}} \approx 0.0$) and ~0.33 for saw waves.
   - Odd-harmonic dominance ($P_{\text{harm}} < 0.12$) must explicitly penalize asymmetric saw shapes (`shape ≠ 0.50`), dual-ring mod bleed, and excessive filter resonance.

---

## 2. Core Algorithmic Architecture

### 2.1 Spectral Shoulder Detection (`audio_metrics_v6.py`)
- Computes running-median smoothed log-magnitude spectral envelope (stripping individual harmonic spikes).
- Identifies the $-6\text{ dB}$ knee frequency relative to the passband shelf.
- Invariant to filter resonance $Q$ up to +18 dB.

### 2.2 Harmonic Parity Metric
$$P_{\text{harm}} = \frac{\sum_{\text{even}} |X(k \cdot f_0)|}{\sum_{\text{odd}} |X(k \cdot f_0)|}$$
When $P_{\text{harm}} < 0.12$, the target is identified as an odd-harmonic square wave.

### 2.3 Directed Guidance in DPO Candidate Generation (`render_pairs_dpo_v6.py`)
During preference pair generation, passive candidate 0 is actively perturbed to generate a guided candidate:
1. **Cutoff Shift:** Steers cutoffs in unison toward target centroid/shoulder if error $> 120\text{ Hz}$.
2. **Parity Alignment:** Forces `shape = 0.50`, mutes ring modulation, and damps secondary filter resonance when $P_{\text{harm}} < 0.12$.
3. **FEG Modulation:** Boosts `feg_amount` (+0.25) when bright transients require dynamic sweeps; damps FEG when low damped basses (<400 Hz) are over-modulated.
4. **Keytrack Guidance:** Biases higher register notes (MIDI > 45) toward standard 100% keytracking (`0.75`).

### 2.4 Multi-Parameter Consensus & Local Polish (`eval_real_stems_consensus_v6.py`)
Prior to Occam pruning, the winning candidate undergoes 3-stage local polish on the full audio playback:
1. **Cutoff Polish:** Steps cutoffs $\pm 5–15\%$ in the direction of spectral centroid delta.
2. **FEG Mod Amount Polish:** Explores $\Delta \text{FEG} \in [-0.15, -0.08, +0.08, +0.15]$ for primary and secondary filters.
3. **Multi-Octave Keytrack Polish:** When the phrase exhibits a pitch span $\ge 7$ semitones, evaluates keytrack options (`0.50` = 0%, `0.75` = 100%, `0.625` = 50%, `0.25` = -100%, `0.875` = 150%) to maximize cross-register phrase coherence.

---

## 3. Verification & Benchmark Plan

1. **Unit Verification:**
   - Run `audio_metrics_v6.py` unit tests for knee detection and parity computation.
   - Run `test_polish_keytrack_feg.py` to confirm bounds and parameter mapping.
2. **Training & Real Stems Pipeline:**
   - Launch `scratch/run_v6_dpo_pipeline.sh` when GPU 0 is free:
     * Render 8,192 guided preference pairs via 14 CPU workers.
     * Train Flow-DPO on `V6_C` ($\beta=100$ and $\beta=300$).
     * Run 24 Real Stems Benchmark comparing `S5`, `DPO_b300`, `V6_C`, `V6_C_DPO_b100`, `V6_C_DPO_b300`.
     * Build interactive audition page at `/home/kim/sao_runs/stems_v6_c_dpo_occam/listen/index.html`.
