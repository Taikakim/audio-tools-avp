# Drum Separation & Transcription Test Plan (DSS & ADT)

**Author:** KUANG / Kim  
**Created:** 2026-10-20  
**Target Repositories:** `/home/kim/Projects/bs-roformer-infer`, `/home/kim/Projects/inverse-drum-machine`, `/home/kim/Projects/Separate-and-detect`  
**Downstream Coupling:** SAO Synthesizer Inversion & Percussion Sound Matching (Surge XT / Neural Sampler)  
**Status:** PROPOSED & READY FOR BENCHMARKING  

---

## 1. Executive Summary & Problem Formulation

In modern electronic and acoustic music production, drums form the rhythmic backbone. Decomposing polyphonic drum recordings into isolated, editable tracks and symbolic representations (MIDI) is essential for:
1. **Acoustic Remixing & Rebalancing:** Processing individual kick, snare, and cymbal stems independently with dedicated EQ, compression, and spatialization.
2. **Analysis-by-Synthesis & Inversion:** Deriving exact transient profiles, decay times, and subharmonic clicks to invert drum hits into synthesizer parameters (e.g., Surge XT percussion patches, analog kick/snare models).
3. **Symbolic Transcription (ADT):** Extracting sample-accurate timestamped onsets, velocities, and kit labels for sequencing and pattern analysis.

This test plan defines the evaluation methodology, datasets, metrics, and execution steps to benchmark the available drum separation and transcription tooling in our local workspace:
* **Tier 1 (Macro Mix Isolation):** `bs-roformer-infer` (BS-RoFormer 4-stem, 6-stem, and MVSep 53-stem mega models).
* **Tier 2A (Sub-Stem Decomposition via Analysis-by-Synthesis):** `inverse-drum-machine` (IDM; Torres et al., IEEE TASLP 2025; 9 classes @ 44.1 kHz).
* **Tier 2B (Sub-Stem Demixing via Latent Diffusion):** `Separate-and-detect` (MSG-LD; Hsu et al., ISMIR 2026; 5 classes @ 16.0 kHz mono).
* **Tier 3 (Percussion Inversion):** Synthesizing isolated drum one-shots in Surge XT and matching them against isolated stems using multi-scale spectral loss.

---

## 2. Comparative System Taxonomy

| Dimension | Tier 1: BS-RoFormer (`bs-roformer-infer`) | Tier 2A: Inverse Drum Machine (`idm`) | Tier 2B: Separate-and-Detect (`MSG-LD`) |
| :--- | :--- | :--- | :--- |
| **Model Type** | Band-Split RoPE Transformer | ConvNeXt + TCN Synthesizer + Wiener Mask | Latent Diffusion (MusicLDM U-Net) + HiFi-GAN |
| **Input** | Polyphonic Song Mixture | Isolated Drum Mix / Drum Loop | Isolated Drum Mix / Demucs Drum Stem |
| **Output Audio** | Full Drum Stem (44.1 kHz stereo) | **9 Kit Sub-Stems (44.1 kHz)** | **5 Kit Sub-Stems (16.0 kHz mono)** |
| **Classes** | `drums` (or 53 individual instruments) | `KD`, `SD`, `HH_CHH`, `HH_OHH`, `CY_CR`, `CY_RD`, `TT_HFT`, `TT_HMT`, `TT_LMT` | `Kick`, `Snare`, `Toms`, `Hi_Hats`, `Cymbals` |
| **Transcription** | None (Audio separation only) | Native (Joint onset activations & velocities) | Downstream backend (madmom SuperFlux / CNN) |
| **Computational Cost** | ~100M+ params, GPU chunked | **~1M params (ultra-lightweight)**, single pass | Heavy (Multi-step diffusion reverse process) |
| **Checkpoints on Disk** | `BS-Rofo-SW-Fixed.ckpt` (Ready in `mir/models`) | `val-epoch=518-global_step=0.ckpt` (Ready) | Remote HF (`ddman1101/Separate-and-Detect`) |
| **Phase / Bleed Behavior** | Linear STFT masking; occasional musical noise | Synth-guided Wiener filter; sharp transients | Generative synthesis; zero bleed in silence |

---

## 3. Evaluation Benchmarks & Test Datasets

### A. Test Corpus
1. **StemGMD Test Set (Synthesized Multitracks):**
   - Clean ground-truth multitracks with known MIDI onsets and velocities.
   - Evaluates separation SDR without acoustic microphone bleed.
2. **IDMT-SMT-Drums (Real Acoustic Kits):**
   - Realistic acoustic drum recordings spanning diverse snare tunings, cymbals, and room ambiences.
3. **MDB-Drums & ENST-Drums (Standard MIREX Evaluation):**
   - Standard benchmark for Automatic Drum Transcription (ADT) and bleed resilience.
4. **Electronic Production Loops (Psytrance / Synthwave / Techno):**
   - 24 in-house electronic drum loops featuring synthesized 909 kicks, acoustic/synthetic layered snares, rolling 16th closed hats, and open splash cymbals.

### B. Evaluation Metrics

#### 1. Audio Separation Fidelity
* **BSS-Eval v4 / `mir_eval.separation`:**
  - Signal-to-Distortion Ratio (SDR, dB)
  - Signal-to-Interference Ratio (SIR, dB — bleed rejection)
  - Signal-to-Artifacts Ratio (SAR, dB — musical noise / phase distortion)
* **Spectral Distance:**
  - Multi-Scale Spectral Distance (MSS, lower is closer) across STFT resolutions $[2048, 1024, 512, 256]$.
  - Mel-SSIM Structural Similarity ($[0, 1]$, higher is closer).

#### 2. Symbolic Drum Transcription (ADT)
* **Onset Detection Metrics (`mir_eval.onset`):**
  - Precision ($P$), Recall ($R$), and F1-score ($F_1$) evaluated per class at standard tolerance windows:
    - Tight: $\pm 20$ ms
    - Standard MIREX: $\pm 50$ ms
* **Velocity Estimation Correlation:**
  - Pearson $r$ between estimated stem peak energy / velocity conditioning and ground-truth MIDI velocity $[0, 127]$.

#### 3. Percussion Inversion Fidelity
* **Surge XT Parameter Inversion:**
  - Inverting isolated Kick hits into Surge XT analog kick patches (sine pitch drop, suboscillator click, saturation drive).
  - Inverting isolated Snare hits into dual-band noise + tone models.
  - Evaluation: Reconstruction MSS between rendered synth one-shot and target isolated stem.

---

## 4. Test Execution Phases

### Phase 1: Local IDM 44.1 kHz Verification (Tier 2A)
* **Objective:** Validate the local pretrained checkpoint `pretrained/idm-44-train-kits/checkpoints/val-epoch=518-global_step=0.ckpt`.
* **Execution Script:** `scripts/synth_inversion/test_idm_separation.py`
* **Test Steps:**
  1. Load model in `/home/kim/Projects/SAO/.venv` (or IDM dedicated environment).
  2. Ingest 5 reference drum loops (acoustic rock, electronic 909, hip-hop break, psytrance rolling groove).
  3. Run forward pass with Wiener filter masking (`alpha=1.0`).
  4. Inspect extracted stems (`KD.wav`, `SD.wav`, `HH_CHH.wav`, `CY_CR.wav`, etc.).
  5. Check transcription onset dictionary against ground-truth grid.

### Phase 2: BS-RoFormer Macro Separation & 53-Stem Verification (Tier 1)
* **Objective:** Test end-to-end full mix isolation and verify the MVSep 53-stem mega model.
* **Test Steps:**
  1. Benchmark 6-stem `BS-Rofo-SW-Fixed.ckpt` against a full electronic mix track: extract `drums.wav`.
  2. Test `mvsep_mega_model_bs_roformer_53_stems` on a short (60s) test clip with `--output_format flac16`.
  3. Validate VRAM stability and output track consistency for `kick`, `snare`, `toms`, `hh`, and `percussion`.
  4. Measure cross-talk from bass into the isolated kick stem.

### Phase 3: Cascade Pipeline Integration (Tier 1 $\rightarrow$ Tier 2A)
* **Objective:** Connect the two systems: Full Mix $\rightarrow$ BS-RoFormer $\rightarrow$ Drum Stem $\rightarrow$ IDM $\rightarrow$ 9 Kit Pieces + MIDI.
* **Test Steps:**
  1. Pipe BS-RoFormer's isolated drum stem directly into IDM.
  2. Compare separation SDR of the cascaded output against running IDM directly on the full mix.
  3. Verify that the intermediate macro separation eliminates vocal and bass spill from triggering false-positive drum onsets.

### Phase 4: Downstream Surge XT Percussion Inversion (Tier 3)
* **Objective:** Connect isolated drum stems to synthesizer parameter inversion.
* **Test Steps:**
  1. Extract isolated Kick hits from IDM stem `KD.wav`.
  2. Fit pitch envelope start frequency ($f_0 \approx 200\text{--}400\text{ Hz}$), pitch decay ($20\text{--}60\text{ ms}$), and resting sub frequency ($45\text{--}65\text{ Hz}$).
  3. Render matched Kick in Surge XT and compute Multi-Scale Spectral distance against the target hit.

---

## 5. Deliverables & Acceptance Criteria

1. **Test Runner Script:** Automated script `scripts/synth_inversion/run_drum_eval.sh` executing the test suite across CPU/GPU with minimal manual intervention.
2. **Audio Audition Page:** Web-based HTML listening interface (similar to `stems_v6_c_dpo_occam/listen/index.html`) presenting:
   - Original mixture
   - Macro drum stem (BS-RoFormer)
   - 9 separated kit pieces (IDM)
   - Synthesized/reconstructed drum mix
   - Interactive pianoroll visualization of transcribed onsets.
3. **Benchmark Report:** Markdown summary documenting per-class SDR, Onset F1, and spectral distance across acoustic and electronic test sets.
