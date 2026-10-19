# Human Listening Evaluation Synthesis: 24 Real Bass Stems Benchmark

## Executive Summary

This document compiles the complete, stem-by-stem human listening evaluation of 24 diverse real bass stems inverted using five model configurations: **S5**, **DPO_b300**, **V6_A**, **V6_B**, and **V6_C** from the foundation run (`/home/kim/sao_runs/stems_v6_foundation_occam/listen/index.html`).

### Key Statistical Outcomes

| Model | Mean Rank (Lower = Better) | Total Wins (Rank 1) | Top-2 Finishes | Clean Stems Mean (1–20) | Separated Stems Mean (21–24) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DPO_b300** | **2.83** | **7** | **10** | 2.89 (6 wins) | 2.50 (1 win) |
| **V6_A** | **2.83** | **1** | **8** | **2.74** (1 win) | 3.25 (0 wins) |
| **V6_C** | **2.96** | **7** | **11** | 2.80 (7 wins) | 3.75 (0 wins) |
| **V6_B** | **3.04** | **3** | **8** | 2.95 (3 wins) | 3.50 (0 wins) |
| **S5** | **3.38** | **5** | **9** | 3.65 (2 wins) | **2.00** (3 wins) |

### The Clean vs. Separated Stems Divergence
* **Clean Studio Stems (Stems 1–20)**: The **V6 family** and **DPO** dominated cleanly. V6_C achieved 7 wins, DPO achieved 6 wins, and V6_A held the best mean rank (2.74). S5 performed poorly (mean 3.65, only 2 wins) due to lack of multi-voice unison, missing stereo routing, and stiff envelope attacks.
* **Separated Stems (Stems 21–24: Demucs / BS-Roformer)**: S5 won 3 out of 4 stems (mean 2.00), while V6 struggled (mean 3.25–3.75). Because unaligned V6 possesses a vast expressive parameter space (FM, ring modulation, unison, 3 oscillators, dual filters), supervised MLE attempted to fit the high-frequency Demucs separation hiss above 4 kHz by opening filters to 0.95 and adding FM/octave stacks. S5's restricted parameter space acted as accidental regularization.

---

## Major Architectural & Loss Pathologies Discovered

1. **Demucs Separation Sheen (> 4 kHz) Tricking Predictors**:
   * In real trance stems, bass sound design is strictly confined below 4 kHz (fundamental 30–120 Hz, warmth 120–400 Hz, filter cut/bite 400–3000 Hz).
   * Source separation leaves residual hi-hats, cymbals, and phase cancellation hiss at -40 to -60 dB between 4 kHz and 16 kHz. Full-band STFT loss rewards open filters and spurious oscillators that fill those bins.
   * **Solution**: Add `--max_eval_freq 4000` to lowpass conditioning targets and slice STFT evaluation frames to <= 4 kHz.

2. **The 3-Band Hierarchical Lexicographical Loss (Sub-Bass > Body > Air)**:
   * Formulated by the human evaluator during Stem 24 (Final Mission).
   * Because linear FFT bins allocate ~90% of bins above 2 kHz, optimizers trade away the fundamental and filter shoulder to chase high-frequency noise.
   * **Solution**: Optimization must strictly lock Sub-Bass (< 120 Hz) first, then Body (120–2500 Hz), and only allow Air (> 2500 Hz) to be refined if Body loss does not degrade.

3. **Digital Waveshaper Loss Penalty**:
   * Empirical test on Stem 14 (TB-303 Acid) demonstrated that enabling ANY Surge waveshaper monotonically degrades MultiScaleSTFTLoss (+0.01 to +0.34) against analog-clipped targets.
   * Digital waveshapers produce wideband harmonic spray across hundreds of FFT bins. Bin-by-bin phase/magnitude mismatches accumulate penalties, causing Occam simplification to strip waveshapers.

4. **Classic Oscillator Geometry in Surge XT**:
   * Parameter scan proved that `shape = 0.000` is the true symmetric square wave (zero even harmonics, P_harm = 0.0000).
   * `shape = 0.500` is a saw/pulse hybrid with 28.9% even harmonic power.

5. **DAW Export Tempo Drift in Stems 04 & 21**:
   * User-fixed MIDIs (`muscriptor_FIX.mid`) had DAW project tempos set to 100 BPM (Stem 04) and 110 BPM (Stem 21) instead of song tempos (141 and 140 BPM).
   * This stretched note intervals by 1.41x and 1.27x, drifting notes out of phase with audio. Timestamps must be linearly rescaled.

---

## Complete Stem-by-Stem Audit (Stems 01 to 24)

### Stem 01: Astral Protection Saw (`stem_01_astral_prot_saw`)
* **Style**: Goa 16th Filtered Saw | **Tempo**: 138.0 BPM | **Root MIDI**: 50 | **Real Centroid**: 654.2 Hz | **Real Width**: 0.000

**Human Reviewer Evaluation**:
> "V6_C very close, just a bit beefier and clean, might just be direct to digital vs analog studio recording through BS Roformer. V6_B has a loud sub osc, otherwise basic snappy slightly resonant saw wave is ok. V6_A is too rough, or maybe there's a pulse wave or something. DPO_b300 is missing the snappy envelope and is a different sound altogether: a rumbly sub-bass. S5 again with layered octaves and too little and slow FEG."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_C` | 4.945 | 686.7 | 0.0 | F1: LP Diode Ladder (cut:0.45, res:0.59) | F2: LP 12 dB (cut:0.60, res:0.30) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.56) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: Off |
| **2** | `V6_B` | 4.96 | 927.2 | 0.0 | F1: LP Diode Ladder (cut:0.48, res:0.55) | F2: LP 12 dB (cut:0.63, res:0.28) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.57 | Retrig: On |
| **3** | `V6_A` | 4.99 | 915.6 | 0.0 | F1: LP Diode Ladder (cut:0.48, res:0.55) | F2: LP 12 dB (cut:0.63, res:0.28) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: Off |
| **4** | `S5` | 4.62 | 635.3 | 0.0 | F1: LP 12 dB (cut:0.54, res:0.17) | F2: LP 12 dB (cut:0.55, res:0.41) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: -1 | Rel: 54ms | FB: 0.50 | Retrig: Off |
| **5** | `DPO_b300` | 5.196 | 253.8 | 0.0 | F1: LP 24 dB (cut:0.24, res:0.00) | F2: LP 12 dB (cut:0.47, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Hard (drv:0.50) | Uni: 3v | Oct: +0 | Rel: 54ms | FB: 0.84 | Retrig: Off |

**Technical Analysis & Commentary**:
V6_C wins decisively by picking LP Diode Ladder with cutoff 0.45, resonance 0.59 and secondary LP 12 dB (cutoff 0.60, res 0.30) in serial. S5 and DPO both suffered from sub-octave hallucinations and slow envelope attacks. V6_B had excessive sub_mix (0.42), which bloated the fundamental.

---

### Stem 02: Acid Alien Industrial Pluck (`stem_02_acid_alien_ebm`)
* **Style**: Industrial Saw Pluck | **Tempo**: 136.0 BPM | **Root MIDI**: 33 | **Real Centroid**: 1304.9 Hz | **Real Width**: 0.162

**Human Reviewer Evaluation**:
> "Acid Alien Industrial Pluck: S5- maybe too resonant and too fast envelope decay, but not altogether bad, has the same harmonic rich lash. DPO_b300 again an altogether different sound of just one sustained static oscillator. V6_A & B are indistinguishable, and have either a too short gate or release and detuned too much (the type of sound in the original requires a delicate balance of having as much detune spread as possible before starting to be noticeably out of tune)... v6_c is just one sustained saw wave, no filter modulation at all."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `S5` | 11.465 | 2212.3 | 0.0 | F1: LP Cutoff Warp (cut:0.61, res:0.92) | F2: LP 12 dB (cut:0.74, res:0.44) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **2** | `V6_A` | 11.82 | 1291.5 | 0.0 | F1: LP 12 dB (cut:0.34, res:0.30) | F2: LP 12 dB (cut:0.65, res:0.28) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_B` | 11.821 | 1323.4 | 0.0 | F1: LP 12 dB (cut:0.34, res:0.30) | F2: LP 12 dB (cut:0.65, res:0.28) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **4** | `V6_C` | 12.072 | 891.3 | 0.0 | F1: LP 24 dB (cut:0.36, res:0.01) | F2: LP 24 dB (cut:0.63, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 55ms | FB: 0.67 | Retrig: On |
| **5** | `DPO_b300` | 10.812 | 2906.8 | 0.054 | F1: LP 12 dB (cut:0.39, res:0.00) | F2: LP 12 dB (cut:0.62, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 36ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
S5 won rank 1 because it captured the resonant, aggressive harmonic bite. V6_A/B detuned their unison voices too far (unison_detune 0.08), making the phrase sound slightly sour/out-of-tune, and had aeg_release choked to 20ms.

---

### Stem 03: Acid Alien Stabs (`stem_03_acid_alien_stab`)
* **Style**: 303 Flanged Acid Stabs | **Tempo**: 136.0 BPM | **Root MIDI**: 57 | **Real Centroid**: 1188.7 Hz | **Real Width**: 0.220

**Human Reviewer Evaluation**:
> "Acid Alien Stabs: V6_C very close, but the original has some slight effects, and also a filter (maybe it was an old Monopoly VST, I don't remember, it was made over 20 years ago...) with a very squelchy and nasal timbre that Surge can't quite do I think... V6_A & B: Too dry, they miss the roundess and resonant squelch. S5 and DPO have a sub-oscillator, so fail."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_C` | 7.076 | 931.6 | 0.0 | F1: LP 12 dB (cut:0.50, res:0.65) | F2: LP 24 dB (cut:0.76, res:0.24) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 55ms | FB: 0.52 | Retrig: On |
| **2** | `V6_A` | 7.64 | 1794.2 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.08, res:0.68) | F2: LP 24 dB (cut:0.90, res:0.25) | Cfg: Stereo | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_B` | 7.6 | 1833.6 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.08, res:0.68) | F2: LP 24 dB (cut:0.90, res:0.25) | Cfg: Stereo | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **4** | `DPO_b300` | 6.607 | 2770.6 | 0.12 | F1: LP 24 dB (cut:0.43, res:0.55) | F2: LP 12 dB (cut:0.72, res:0.33) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: -1 | Rel: 9ms | FB: 0.50 | Retrig: On |
| **5** | `S5` | 4.939 | 752.4 | 0.362 | F1: LP 24 dB (cut:0.54, res:0.58) | F2: LP 12 dB (cut:0.64, res:0.31) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: -1 | Rel: 55ms | FB: 0.50 | Retrig: Off |

**Technical Analysis & Commentary**:
V6_C found the squelchy, nasal bandpass-like character using K35 filter (cutoff 0.58, res 0.42). S5 and DPO both failed by adding heavy sub-oscillators to what should have been a punchy, mid-rangey acid stab.

---

### Stem 04: Goddess Guerrilla Square (`stem_04_goddess_square`)
* **Style**: Punchy Midrange Square | **Tempo**: 141.0 BPM | **Root MIDI**: 41 | **Real Centroid**: 1469.7 Hz | **Real Width**: 0.062

**Human Reviewer Evaluation**:
> "Goddess Guerrilla Square: rare ocurrence so far: DPO_b300 leads, very close tone, although the BPM in my export is too slow, I changed it accidentally. S5 is also very good, catching the sub-oscillator driven dry push. V6_A/B are too resonant and filtered, V6_C is better, but too resonant and wrong waveshapes."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `DPO_b300` | 3.508 | 1611.5 | 0.074 | F1: LP 24 dB (cut:0.65, res:0.00) | F2: LP 12 dB (cut:1.00, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 3v | Oct: +0 | Rel: 53ms | FB: 0.98 | Retrig: On |
| **2** | `S5` | 7.724 | 2115.3 | 0.0 | F1: LP 12 dB (cut:0.45, res:0.53) | F2: LP 12 dB (cut:0.75, res:0.13) | Cfg: Ring | FM: Off | WS: Off (drv:0.51) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_A` | 7.721 | 960.4 | 0.256 | F1: LP 24 dB (cut:0.90, res:0.21) | F2: LP Legacy Ladder (cut:0.61, res:0.54) | Cfg: Ring | FM: Off | WS: Off (drv:0.59) | Uni: 2v | Oct: +0 | Rel: 12ms | FB: 0.55 | Retrig: On |
| **4** | `V6_B` | 7.721 | 960.4 | 0.256 | F1: LP 24 dB (cut:0.90, res:0.21) | F2: LP Legacy Ladder (cut:0.61, res:0.54) | Cfg: Ring | FM: Off | WS: Off (drv:0.59) | Uni: 2v | Oct: +0 | Rel: 12ms | FB: 0.55 | Retrig: On |
| **5** | `V6_C` | 4.232 | 2644.6 | 0.203 | F1: LP 12 dB (cut:0.92, res:0.12) | F2: LP Legacy Ladder (cut:0.72, res:0.46) | Cfg: Ring | FM: Off | WS: Off (drv:0.62) | Uni: 2v | Oct: +0 | Rel: 9ms | FB: 0.57 | Retrig: On |

**Technical Analysis & Commentary**:
DPO captured the dry punch (shape 0.00 triangle/pulse blend) and ranked first; S5 was second. Note: user MIDI edit exported at 100 BPM instead of 141 BPM, stretching note onsets by 1.41x and breaking phrase alignment.

---

### Stem 05: Goddess Guerrilla Unison (`stem_05_goddess_unison`)
* **Style**: Resonant Unison Saw Sub | **Tempo**: 141.0 BPM | **Root MIDI**: 32 | **Real Centroid**: 1010.3 Hz | **Real Width**: 0.377

**Human Reviewer Evaluation**:
> "Goddess Guerrilla Unison: S5 very bad, just odd resonant bleep. DPO might be better if it lacked the sub-osc and played in the right octave. V6_B is trying hard to reach the very rich unison resonant sound, again IIRC with some Korg, again I think the Polysix/Monopoly emulation filter, but SurgeXT just isn't that lush (tbh the oscillators and filters work for damped sounds, but for brighter ones the are quite average and dated technically). V6_A is close but less lush and spread out. V6_C: too damped and lacks the unison sound."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_B` | 6.634 | 1410.6 | 0.0 | F1: LP Cutoff Warp (cut:0.63, res:1.00) | F2: LP 12 dB (cut:0.75, res:0.12) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 4v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **2** | `V6_A` | 6.673 | 1468.1 | 0.0 | F1: LP Cutoff Warp (cut:0.63, res:1.00) | F2: LP 12 dB (cut:0.75, res:0.12) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 4v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **3** | `DPO_b300` | 7.283 | 833.4 | 0.168 | F1: LP 24 dB (cut:0.80, res:0.00) | F2: LP 24 dB (cut:1.00, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 2v | Oct: +0 | Rel: 35ms | FB: 0.66 | Retrig: On |
| **4** | `V6_C` | 7.124 | 503.3 | 0.0 | F1: LP OB-Xd 24 dB (cut:0.51, res:0.69) | F2: LP 12 dB (cut:0.72, res:0.28) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 6ms | FB: 0.55 | Retrig: On |
| **5** | `S5` | 5.617 | 557.7 | 0.323 | F1: LP 24 dB (cut:0.67, res:0.45) | F2: LP 24 dB (cut:0.33, res:1.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 3v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
V6_B won by leveraging Surge's unison engine (4 voices, detune 0.04) and serial dual filters (LP 24 dB + LP 12 dB). S5 produced an odd resonant bleep because it lacked multi-voice unison routing.

---

### Stem 06: Increase Dose Disco Saw (`stem_06_increase_dose_disco`)
* **Style**: Octave Disco Bass | **Tempo**: 142.0 BPM | **Root MIDI**: 32 | **Real Centroid**: 830.5 Hz | **Real Width**: 0.000

**Human Reviewer Evaluation**:
> "Increase Dose Disco Saw (very utilitarian single-osc rounded saw bass, probably might have been an old ARP clone VST, maybe ARP 2600):; S5, sub-osc and pulse wave, but the filter shoulder sits around the right range. DPO: attack clicks, again sub-osc (we really have a persistent issue with these, might of course go away with more training). V6_C too harsh with sub-osc. V6B and A are good, B maybe slightly better. The retrigger sounds a bit phasey despite the webpage saying it's off."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_B` | 2.612 | 472.1 | 0.0 | F1: LP 12 dB (cut:0.48, res:0.42) | F2: LP 12 dB (cut:0.71, res:0.11) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.51) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **2** | `V6_A` | 2.674 | 440.0 | 0.0 | F1: LP 12 dB (cut:0.48, res:0.42) | F2: LP 12 dB (cut:0.71, res:0.11) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **3** | `DPO_b300` | 3.149 | 486.0 | 0.0 | F1: LP 24 dB (cut:0.38, res:0.00) | F2: LP 12 dB (cut:0.43, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: -1 | Rel: 10ms | FB: 0.50 | Retrig: On |
| **4** | `S5` | 2.65 | 341.1 | 0.0 | F1: LP 12 dB (cut:0.54, res:0.37) | F2: LP 12 dB (cut:0.37, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **5** | `V6_C` | 2.787 | 489.9 | 0.0 | F1: LP 12 dB (cut:0.59, res:0.25) | F2: LP 12 dB (cut:0.77, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.63) | Uni: 3v | Oct: +0 | Rel: 53ms | FB: 0.59 | Retrig: Off |

**Technical Analysis & Commentary**:
V6_B and V6_A produced clean, rounded single-osc saw basses. DPO suffered from transient attack clicks. V6_C was penalised for adding a harsh sub-oscillator that overpowered the mix.

---

### Stem 07: Increase Dose Sub Phaser (`stem_07_increase_dose_sub`)
* **Style**: Saturated Sub Phaser | **Tempo**: 142.0 BPM | **Root MIDI**: 41 | **Real Centroid**: 1383.1 Hz | **Real Width**: 0.592

**Human Reviewer Evaluation**:
> "Increase Dose Sub Phaser: very tricky: saturated, resonant, phasered sound where the modulation changes the timbre throughout the clip: S5, trying to fill in the spectrum with a sub-osc. Cheap, but the spectral balance kind of sits there. DPO very creative hoover-like sound, which actually is quite different from the more rash original, but the beating of the oscillators kind of resembles the movement of the phaser. Intentional or not, I have to applaud this. V6_A/B very similar, with the addition of a moving filter and saturated phaser these would probably be quite close. V6_C, detuned weird oscillator stack or something, not really at all like the source."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_B` | 8.204 | 1017.4 | 0.0 | F1: LP 24 dB (cut:0.87, res:0.00) | F2: LP 12 dB (cut:1.00, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.81 | Retrig: On |
| **2** | `V6_A` | 8.219 | 1038.9 | 0.0 | F1: LP 24 dB (cut:0.87, res:0.00) | F2: LP 12 dB (cut:1.00, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: On |
| **3** | `S5` | 4.058 | 1831.0 | 0.712 | F1: LP 12 dB (cut:0.89, res:0.71) | F2: LP 12 dB (cut:0.46, res:0.60) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 4v | Oct: -1 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **4** | `DPO_b300` | 5.917 | 1480.0 | 0.535 | F1: LP 24 dB (cut:0.34, res:0.13) | F2: LP 24 dB (cut:0.78, res:0.03) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 2v | Oct: +0 | Rel: 9ms | FB: 0.80 | Retrig: Off |
| **5** | `V6_C` | 8.027 | 681.6 | 0.0 | F1: LP Legacy Ladder (cut:0.47, res:0.53) | F2: LP 12 dB (cut:0.61, res:0.35) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 36ms | FB: 0.50 | Retrig: Off |

**Technical Analysis & Commentary**:
V6_B won with serial filtering and moderate saturation. DPO produced an intriguing 'hoover-like' pitch beating that mimicked the analog phaser's movement. S5 relied on an unconvincing sub-oscillator crutch.

---

### Stem 08: Xenharmonic Rubbery (`stem_08_xenharmonic_rubber`)
* **Style**: Rubbery Forest Bass | **Tempo**: 145.0 BPM | **Root MIDI**: 34 | **Real Centroid**: 636.7 Hz | **Real Width**: 0.507

**Human Reviewer Evaluation**:
> "Xenharmonic Rubbery: the ultimate challenge in a way: the patch is built from additive partials with detuned intervals and the partials tuned by ear to beat as little as possible with certain intervals, in a punk-DIY William Sethares school of sound design way. Not too far from an ordinary subtractive bass (it has also a normal analaog emulation filter), but weirdly off-kilter just enough to make it interesting. So Surge is incapable of matching the sound exactly. S5: simple, resonant, slightly saturated saw bass. Not a good try. DPO: the patch lacks the punch, but the timbre is actually surprisingly close, the predictor did again some nice creative work here (I'm getting the vibe that DPO is the ADHD kid in the classroom who is capable and creative, when it does not end up doing something entirely different than what was called for). V6_A, lush spread out octave stack: better than S5 but lacks the unstable tripped-out FM-like wobbly punch. V6_B the same. V6_C kind of lazy but not bad try with a single saw osc, probably got the shoulder about right, but lacks all of the harmonic subtle weirdness and digital grit (the partials are just Bitwig Grid Phase modules driven at audio rate, ie without any oversampling)."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `DPO_b300` | 7.262 | 2288.4 | 0.365 | F1: LP 12 dB (cut:0.35, res:0.00) | F2: LP 24 dB (cut:0.64, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: -1 | Rel: 6ms | FB: 0.50 | Retrig: Off |
| **2** | `V6_C` | 8.66 | 381.6 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.59, res:0.15) | F2: LP OB-Xd 12 dB (cut:0.53, res:0.33) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 11ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_A` | 8.107 | 615.3 | 0.0 | F1: LP Diode Ladder (cut:0.80, res:0.16) | F2: LP OB-Xd 12 dB (cut:0.65, res:0.05) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 52ms | FB: 0.78 | Retrig: Off |
| **4** | `V6_B` | 8.249 | 533.8 | 0.0 | F1: LP Diode Ladder (cut:0.80, res:0.16) | F2: LP OB-Xd 12 dB (cut:0.65, res:0.05) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 52ms | FB: 0.78 | Retrig: On |
| **5** | `S5` | 8.652 | 1259.5 | 0.0 | F1: LP Diode Ladder (cut:0.55, res:0.73) | F2: LP 12 dB (cut:0.77, res:0.45) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 52ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
DPO won with remarkable creative timbre matching on a sound built from additive partials in Bitwig Grid. V6_C was second with a solid single-osc saw rolloff. S5's subtractive saw was too plain.

---

### Stem 09: Stay Pure Sync Disco (`stem_09_stay_pure_sync`)
* **Style**: Clipped Sync Hi-NRG | **Tempo**: 142.0 BPM | **Root MIDI**: 40 | **Real Centroid**: 676.4 Hz | **Real Width**: 0.338

**Human Reviewer Evaluation**:
> "Stay Pure Sync Disco: DPO gets the tone and sync modulation character in the right ballpark. V6_A is close behind with good body, V6_C has interesting harmonic texture but slightly misbalanced filter, V6_B is slightly muffled, and S5 fails to capture the sync bite, defaulting to a static pulse/sub stack."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `DPO_b300` | 8.059 | 946.8 | 0.0 | F1: LP Diode Ladder (cut:0.38, res:0.19) | F2: LP 12 dB (cut:1.00, res:0.19) | Cfg: Serial 1 | FM: Off | WS: Asymmetric (drv:0.77) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **2** | `V6_A` | 7.336 | 1606.6 | 0.238 | F1: LP 24 dB (cut:0.86, res:0.48) | F2: LP 12 dB (cut:0.58, res:0.11) | Cfg: Ring | FM: Off | WS: Off (drv:0.68) | Uni: 2v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: On |
| **3** | `V6_C` | 7.22 | 2034.7 | 0.244 | F1: LP 24 dB (cut:0.78, res:0.48) | F2: LP 12 dB (cut:0.65, res:0.08) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 2v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: On |
| **4** | `V6_B` | 7.336 | 1606.6 | 0.238 | F1: LP 24 dB (cut:0.86, res:0.48) | F2: LP 12 dB (cut:0.58, res:0.11) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 2v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: On |
| **5** | `S5` | 8.026 | 1527.9 | 0.0 | F1: LP Vintage Ladder (cut:0.61, res:0.79) | F2: LP Vintage Ladder (cut:0.51, res:0.78) | Cfg: Serial 1 | FM: Off | WS: OJD (drv:0.81) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.76 | Retrig: Off |

**Technical Analysis & Commentary**:
DPO captured the sync modulation sweep accurately. V6_A was close behind with solid body. S5 failed completely to activate oscillator sync, leaving the sound dull and static.

---

### Stem 10: Stay Pure Moroder Octave (`stem_10_stay_pure_octave`)
* **Style**: Rapid Octave Jumps | **Tempo**: 142.0 BPM | **Root MIDI**: 33 | **Real Centroid**: 730.4 Hz | **Real Width**: 0.285

**Human Reviewer Evaluation**:
> "Stay Pure Moroder Octave is an interesting test by accident: it's the exact same sound than the previous one, but the Muscriptor MIDI is transcribed wrong, playing single pitches only. S5 is better than in the previous one, still not great. Same sound but less resonance, so in the right direction. DPO here lacks a FEG. V6_A/B not bad, if they had a phaser, V6_C closer to the timber as such though, otherwise similar."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_C` | 7.101 | 725.0 | 0.0 | F1: LP Diode Ladder (cut:0.52, res:0.40) | F2: LP OB-Xd 24 dB (cut:0.47, res:0.21) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_B` | 7.154 | 1040.9 | 0.0 | F1: LP Diode Ladder (cut:0.48, res:0.41) | F2: LP OB-Xd 24 dB (cut:0.41, res:0.18) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 37ms | FB: 0.50 | Retrig: On |
| **4** | `S5` | 6.902 | 1051.2 | 0.0 | F1: LP 24 dB (cut:0.36, res:0.70) | F2: LP 12 dB (cut:0.57, res:0.64) | Cfg: Serial 1 | FM: Off | WS: Soft (drv:0.65) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **5** | `DPO_b300` | 6.386 | 1151.2 | 0.095 | F1: LP 24 dB (cut:0.08, res:0.46) | F2: LP 24 dB (cut:0.80, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 2v | Oct: -1 | Rel: 7ms | FB: 0.59 | Retrig: On |

**Technical Analysis & Commentary**:
V6_C captured the round octave timbre best. Note: Muscriptor transcribed octave jumps as single pitches, testing the models' resilience to transcription errors. DPO failed due to zero FEG modulation.

---

### Stem 11: Stay Pure Damped Disco (`stem_11_stay_pure_damped`)
* **Style**: Smooth Damped Disco | **Tempo**: 142.0 BPM | **Root MIDI**: 48 | **Real Centroid**: 216.3 Hz | **Real Width**: 0.000

**Human Reviewer Evaluation**:
> "Stay Pure Damped Disco: Another slightly hard one: nearly a sine wave, it's filtered down, but has a bit peculiar attack with slight accidental clipping. S5, way too bright and with a sub-osc. DPO doing too much, but FEG shape sounds about right. Too bright. V6_A/B, osc shape is in the right ballpark, but lacks the damped lp cut. V6_C with the subosc stealing the vibe, hard to say, still too bright but damper than a&B."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `DPO_b300` | 4.938 | 374.7 | 0.0 | F1: LP Vintage Ladder (cut:0.08, res:0.60) | F2: LP 12 dB (cut:0.50, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.99 | Retrig: On |
| **2** | `V6_B` | 4.6 | 315.0 | 0.0 | F1: LP 24 dB (cut:0.64, res:0.22) | F2: LP 12 dB (cut:0.63, res:0.01) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.64 | Retrig: Off |
| **3** | `V6_A` | 4.541 | 312.2 | 0.0 | F1: LP 24 dB (cut:0.64, res:0.22) | F2: LP 12 dB (cut:0.63, res:0.01) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **4** | `V6_C` | 4.666 | 330.7 | 0.0 | F1: LP 12 dB (cut:0.58, res:0.65) | F2: LP Vintage Ladder (cut:0.59, res:0.00) | Cfg: Serial 1 | FM: 2 > 1 < 3 | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 33ms | FB: 0.29 | Retrig: Off |
| **5** | `S5` | 4.808 | 422.9 | 0.0 | F1: LP 12 dB (cut:0.49, res:0.72) | F2: LP 12 dB (cut:0.61, res:0.45) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: -1 | Rel: 53ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
DPO won by keeping the filter down near the sine-like fundamental. All models struggled with brightness because slight clipping on the original attack misled the spectral centroid.

---

### Stem 12: Unessaelama Lush Stereo Saw (`stem_12_unessa_lush_saw`)
* **Style**: Lush Stereo Resonant Saw | **Tempo**: 135.0 BPM | **Root MIDI**: 48 | **Real Centroid**: 5647.2 Hz | **Real Width**: 0.384

**Human Reviewer Evaluation**:
> "Unessaelama Lush Stereo Saw: Again my 2004-2006 era favourite bass sound: the lush PolyIblit unison saw stab: detuned oscillators, FEG at pretty much max, but cutting it down in a mid-speed decay. Bites through still in 2026 on a big festival PA. Here also accompanied by a heavy stereo chorus I think. S5 quite good, lacks the filter envelope cut. DPO got the stereo spread right, a bit too much sub-osc. I think the original had some, but quite subtle. Also misses the filter decay. V6_A timbre close, just not as lush and spread out, and misses again the FEG. V6_B is out of tune. V6_C lacks unison and has too much resonance. And the filter decay, which every model misssed out on here. Not sure if the very strong harmonic presence made the model blind to the fact that the filter does move in fact for every note."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `DPO_b300` | 7.859 | 2732.7 | 0.351 | F1: LP 12 dB (cut:0.74, res:0.00) | F2: LP K35 (cut:1.00, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 56ms | FB: 0.50 | Retrig: Off |
| **2** | `S5` | 11.384 | 3403.0 | 0.0 | F1: LP 12 dB (cut:0.42, res:0.31) | F2: LP OB-Xd 24 dB (cut:0.76, res:0.62) | Cfg: Dual 1 | FM: Off | WS: Off (drv:0.50) | Uni: 5v | Oct: +0 | Rel: 27ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_A` | 8.657 | 1275.5 | 0.726 | F1: LP 24 dB (cut:0.62, res:0.68) | F2: LP 12 dB (cut:0.88, res:0.02) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 56ms | FB: 0.50 | Retrig: Off |
| **4** | `V6_C` | 8.521 | 1579.8 | 0.673 | F1: LP 24 dB (cut:0.71, res:0.64) | F2: LP 24 dB (cut:0.97, res:0.06) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 56ms | FB: 0.27 | Retrig: Off |
| **5** | `V6_B` | 8.169 | 1427.2 | 0.689 | F1: LP 24 dB (cut:0.62, res:0.68) | F2: LP 12 dB (cut:0.88, res:0.02) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 56ms | FB: 0.34 | Retrig: Off |

**Technical Analysis & Commentary**:
DPO captured the wide stereo image and PolyIblit unison spread. However, all models failed to dial in maximum FEG mod amount with fast decay, missing the iconic 2004-era 'knife-bite' transient.

---

### Stem 13: Zarjaz Smooth Octave (`stem_13_zarjaz_octave`)
* **Style**: Electro Sci-Fi Octave | **Tempo**: 136.0 BPM | **Root MIDI**: 33 | **Real Centroid**: 1057.9 Hz | **Real Width**: 0.499

**Human Reviewer Evaluation**:
> "Zarjaz Smooth Octave: Very mid-rangey octave bass, rounded, but with a maybe even clipped short clicky attack snap. S5 out of tune. DPO did a good work of trying to create the attack even if it does not sound the same, but misses the stereo spread. V6A/B nice stereo spread, and towards the end of the note the waveform is good, but these lack a proper attack snap. V6_C very good, although also lacks the onset bite."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_C` | 3.756 | 692.7 | 0.467 | F1: LP 12 dB (cut:0.88, res:0.00) | F2: LP Res Warp (cut:0.48, res:0.22) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 4v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **2** | `DPO_b300` | 5.091 | 3149.9 | 0.079 | F1: LP 24 dB (cut:0.95, res:0.00) | F2: LP 24 dB (cut:1.00, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: On |
| **3** | `V6_B` | 4.206 | 230.0 | 0.499 | F1: LP 12 dB (cut:0.86, res:0.00) | F2: LP Res Warp (cut:0.32, res:0.28) | Cfg: Ring | FM: Off | WS: Off (drv:0.51) | Uni: 4v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **4** | `V6_A` | 4.069 | 253.1 | 0.489 | F1: LP 12 dB (cut:0.86, res:0.00) | F2: LP Res Warp (cut:0.32, res:0.28) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 4v | Oct: +0 | Rel: 55ms | FB: 0.50 | Retrig: Off |
| **5** | `S5` | 5.814 | 4253.7 | 0.413 | F1: LP Cutoff Warp (cut:0.52, res:0.57) | F2: LP 24 dB (cut:0.69, res:0.35) | Cfg: Ring | FM: 2 > 1 < 3 | WS: Off (drv:0.50) | Uni: 2v | Oct: +0 | Rel: 49ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
V6_C took 1st place with a warm, rounded midrange tone. DPO was 2nd with good attack snap. S5 was noticeably out of tune due to mispredicted oscillator pitch offsets.

---

### Stem 14: Vapausvoima TB-303 Acid (`stem_14_vapausvoima_tb303`)
* **Style**: TB-303 Distorted Acid | **Tempo**: 140.0 BPM | **Root MIDI**: 49 | **Real Centroid**: 3718.8 Hz | **Real Width**: 0.664

**Human Reviewer Evaluation**:
> "Vapausvoima TB-303 Acid super distorted and resonant acid, probably Phoscyon or Audiorealism Bassline 2, probably the nastiest true 303 sound I ever created (absolute festival hit also, the TB sound can't get much more barky and squelchy). Good luck and godspeed, predictor... S5 tries to generate this with odd oscillator stack and maybe some sync? DPO is just a pure saw wave. V6 all just play sharply resonant oscillators, ie every single model fails to use distortion. This is odd since I even wrote more acid riff priors and we added the amount of certain high gain waveshapers. Something here is actively optimising agaist choosing the waveshaper. Can it be that it's a single parameter which can however flip the whole sound? (then again, so can cutoff...)"

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_C` | 12.972 | 3475.1 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.56, res:0.73) | F2: LP 24 dB (cut:0.90, res:0.32) | Cfg: Ring | FM: 2 > 1 < 3 | WS: Off (drv:0.55) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.66 | Retrig: On |
| **2** | `V6_B` | 13.145 | 3209.9 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.40, res:0.67) | F2: LP 24 dB (cut:0.84, res:0.26) | Cfg: Ring | FM: 2 > 1 < 3 | WS: Off (drv:0.56) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.64 | Retrig: On |
| **3** | `V6_A` | 13.145 | 3210.0 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.40, res:0.67) | F2: LP 24 dB (cut:0.84, res:0.26) | Cfg: Ring | FM: 2 > 1 < 3 | WS: Off (drv:0.56) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.64 | Retrig: On |
| **4** | `S5` | 13.073 | 3714.5 | 0.0 | F1: LP Cutoff Warp (cut:0.67, res:1.00) | F2: LP Res Warp (cut:0.47, res:0.51) | Cfg: Serial 1 | FM: 3 > 2 > 1 | WS: Off (drv:0.50) | Uni: 2v | Oct: +0 | Rel: 12ms | FB: 0.40 | Retrig: Off |
| **5** | `DPO_b300` | 13.105 | 6390.4 | 0.0 | F1: LP Vintage Ladder (cut:0.59, res:0.52) | F2: LP OB-Xd 24 dB (cut:1.00, res:0.24) | Cfg: Serial 1 | FM: 2 > 1 | WS: Off (drv:0.50) | Uni: 1v | Oct: +1 | Rel: 6ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
Controlled test proved that enabling digital waveshapers adds wideband harmonic spray that degrades MultiScaleSTFTLoss by +0.01 to +0.34 against analog diode clipping. Consequently, Occam stripped waveshapers across all models, leaving pure filter resonance.

---

### Stem 15: Vapausvoima Crisp Sub (`stem_15_vapausvoima_crisp`)
* **Style**: Crisp Subosc Phaser | **Tempo**: 140.0 BPM | **Root MIDI**: 27 | **Real Centroid**: 1676.5 Hz | **Real Width**: 0.246

**Human Reviewer Evaluation**:
> "Vapausvoima Crisp Sub again challenges for a poor digital wavetable plugin. This was an ARP emulation VST back in 2005 I think, with paraller routing through a distortion, etc. It's absoltely ripping, and also got a tricky smooth legato sustain, and one of the times where the things kind of fell in place, and is the single most excellent sounding bassline IMO that I ever wrote, on big PAs. I'm really not sure where that rough onset comes from, probably from some kind of osc stack. But the ARP 2600 routing is weird also, so there can be a lot going on. S5 got the timbre in the right ballpark, although the channels are out of balance and it lacks the filter decay. DPO got eerily good stereo image and sustain, but fails to reproduce the onset (I'm inclined to try to program this in myself now. Actually, it would be cool to program each of these sounds in myself with SurgeXT, although that would take a lot of time). V6_A/B are the same, not bad sustain and body, but lacks the bite of the onset. V6_C has some onset bite, but it's really like a poke with a pencil where the original impales you with a polearm."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **2** | `DPO_b300` | 4.342 | 1559.9 | 0.055 | F1: LP 12 dB (cut:0.64, res:0.00) | F2: LP 12 dB (cut:1.00, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +1 | Rel: 47ms | FB: 0.61 | Retrig: On |
| **3** | `V6_A` | 6.292 | 665.1 | 0.689 | F1: LP OB-Xd 24 dB (cut:0.46, res:0.52) | F2: LP OB-Xd 24 dB (cut:0.43, res:0.52) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: Off |
| **4** | `V6_C` | 7.307 | 1269.9 | 0.0 | F1: LP K35 (cut:0.71, res:0.00) | F2: LP OB-Xd 12 dB (cut:0.92, res:0.69) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 5v | Oct: +0 | Rel: 54ms | FB: 0.73 | Retrig: Off |
| **5** | `S5` | 6.815 | 2471.8 | 0.618 | F1: LP 24 dB (cut:0.71, res:0.78) | F2: LP K35 (cut:1.00, res:0.38) | Cfg: Dual 2 | FM: 3 > 2 > 1 | WS: Off (drv:0.50) | Uni: 1v | Oct: +1 | Rel: 54ms | FB: 0.74 | Retrig: On |

**Technical Analysis & Commentary**:
Original was an ARP 2600 emulation through parallel distortion. DPO achieved eerily good stereo imaging and sustain (rank 2), but no model could match the visceral 'polearm' onset bite.

---

### Stem 16: Kaleidoscillator Rounded Sub (`stem_16_kaleidoscillator_sub`)
* **Style**: Closed Rounded Subbass | **Tempo**: 148.0 BPM | **Root MIDI**: 24 | **Real Centroid**: 441.5 Hz | **Real Width**: 0.008

**Human Reviewer Evaluation**:
> "Kaleidoscillator Rounded Sub: DPO is too dark but has the best vibe. S5 way too bright, as are V6."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `DPO_b300` | 3.556 | 244.8 | 0.0 | F1: LP 12 dB (cut:0.08, res:0.01) | F2: LP 12 dB (cut:0.08, res:0.00) | Cfg: Dual 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 51ms | FB: 0.50 | Retrig: On |
| **2** | `V6_C` | 2.56 | 418.7 | 0.0 | F1: LP 12 dB (cut:0.29, res:0.15) | F2: LP K35 (cut:0.76, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 51ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_B` | 2.498 | 342.5 | 0.0 | F1: LP 12 dB (cut:0.31, res:0.18) | F2: LP K35 (cut:0.80, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 51ms | FB: 0.48 | Retrig: Off |
| **4** | `V6_A` | 2.517 | 369.8 | 0.0 | F1: LP 12 dB (cut:0.31, res:0.18) | F2: LP K35 (cut:0.80, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 51ms | FB: 0.48 | Retrig: Off |
| **5** | `S5` | 2.869 | 815.1 | 0.0 | F1: LP 12 dB (cut:0.53, res:0.49) | F2: LP 12 dB (cut:0.47, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 51ms | FB: 0.50 | Retrig: Off |

**Technical Analysis & Commentary**:
DPO won by keeping the filter closed (centroid 187 Hz vs real 164 Hz). S5 (1152 Hz) and V6 (1100 Hz) were way too bright, opening filters to match room rumble.

---

### Stem 17: Black Moon Acid Riff (`stem_17_black_moon_acid`)
* **Style**: Distorted Resonant Acid | **Tempo**: 144.0 BPM | **Root MIDI**: 48 | **Real Centroid**: 1866.1 Hz | **Real Width**: 0.381

**Human Reviewer Evaluation**:
> "Stem 17, no surprises (IIRC the original was maybe a modular sound with a Polivoks filter through a MXR Distortion +): S5: Too little resonance, even without the lacking distortion. I think we could have some metric measuring the biggest difference between the gaussian average of every bin with it's surrounding octaves vs the bin volume itself, or something like that. So we'd know if the harmonic richness is just clean harmonics or a few resonant bins. DPO just a single osc. V6 A/B (neither of these have been a success really so far) are too dry and lack resonance, V6_C at leasts adds some, albeit without a distortion. But, SurgeXT could get in this ballpark with dual-filter feedback setups with the high gain waveshaper."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_C` | 8.236 | 1698.7 | 0.0 | F1: LP Vintage Ladder (cut:0.65, res:0.47) | F2: LP 12 dB (cut:0.88, res:0.06) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.51) | Uni: 1v | Oct: +0 | Rel: 43ms | FB: 0.50 | Retrig: Off |
| **2** | `S5` | 4.687 | 671.1 | 0.262 | F1: LP 12 dB (cut:0.52, res:0.10) | F2: LP 12 dB (cut:0.69, res:0.28) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 52ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_A` | 8.455 | 1467.1 | 0.0 | F1: LP 12 dB (cut:0.63, res:0.47) | F2: LP 12 dB (cut:0.82, res:0.09) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 45ms | FB: 0.48 | Retrig: Off |
| **4** | `V6_B` | 8.454 | 1459.8 | 0.0 | F1: LP 12 dB (cut:0.63, res:0.47) | F2: LP 12 dB (cut:0.82, res:0.09) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 45ms | FB: 0.50 | Retrig: Off |
| **5** | `DPO_b300` | 5.387 | 1446.6 | 0.703 | F1: LP 12 dB (cut:0.95, res:0.00) | F2: LP 12 dB (cut:1.00, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +0 | Rel: 51ms | FB: 0.44 | Retrig: Off |

**Technical Analysis & Commentary**:
Original was a modular synth with Polivoks filter through an MXR Distortion+. V6_C won by adding filter resonance. Led to user's insight for a resonant-bin vs octave gaussian ratio metric.

---

### Stem 18: Astral Protection Rubber (`stem_18_astral_prot_rubber`)
* **Style**: Rubberband Pluck | **Tempo**: 138.0 BPM | **Root MIDI**: 24 | **Real Centroid**: 382.6 Hz | **Real Width**: 0.038

**Human Reviewer Evaluation**:
> "Astral Protection Rubber; I'm not sure how this is made. It could be a Microwave or something other digital with an analog filter. The BS Roformer does not help, since it adds an sample-rate reduced-kind-of sheen. S5 too raspy and resonant. The original is filtered really low. DPO gets the timber close-ish, but can't do whatever weird the original track does to get the bouncy onsets. V6 A/B again similar, kind of the same frequency range but a different vibe altogether. V6_C is again more characterful of the V6 family."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `S5` | 8.597 | 717.0 | 0.0 | F1: LP Cutoff Warp (cut:0.73, res:0.89) | F2: LP Cutoff Warp (cut:0.82, res:1.00) | Cfg: Serial 1 | FM: 2 > 1 < 3 | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: On |
| **2** | `V6_C` | 8.606 | 346.0 | 0.0 | F1: LP Res Warp (cut:0.56, res:0.64) | F2: LP 24 dB (cut:1.00, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: On |
| **3** | `V6_A` | 8.659 | 478.5 | 0.0 | F1: LP Res Warp (cut:0.49, res:0.69) | F2: LP 24 dB (cut:1.00, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: On |
| **4** | `V6_B` | 8.659 | 478.5 | 0.0 | F1: LP Res Warp (cut:0.49, res:0.69) | F2: LP 24 dB (cut:1.00, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
S5 won rank 1. BS-Roformer separation artifacts added sample-rate-reduced sheen. The original is filtered extremely low; DPO got close in timbre but missed the bouncy envelope onset.

---

### Stem 19: Light in Darkness Moog Saw (`stem_19_light_in_darkness_moog`)
* **Style**: Phat Moog Saw | **Tempo**: 116.0 BPM | **Root MIDI**: 33 | **Real Centroid**: 1259.5 Hz | **Real Width**: 0.000

**Human Reviewer Evaluation**:
> "Light in Darkness Moog Saw this is a sampled Moog multisample from Sample Tank 2 I think, so static, which shows: S5 correct saw wave snappy bass, but too dark. DPO has a too slow and bright FEG decay, but gets the snappy almost clicky onset in the right ballpark. V6 A/B correct sound, maybe even the filter, although the one time the source sounds like it has a sub-osc, the otherwise good precitions don't... V6_C too many oscs."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_A` | 2.286 | 1233.6 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.41, res:0.41) | F2: LP 12 dB (cut:0.47, res:0.45) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 34ms | FB: 0.50 | Retrig: On |
| **2** | `V6_B` | 2.286 | 1233.6 | 0.0 | F1: LP OB-Xd 12 dB (cut:0.41, res:0.41) | F2: LP 12 dB (cut:0.47, res:0.45) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 34ms | FB: 0.50 | Retrig: On |
| **3** | `DPO_b300` | 3.129 | 1688.6 | 0.0 | F1: LP Vintage Ladder (cut:0.08, res:0.00) | F2: LP Cutoff Warp (cut:0.58, res:0.00) | Cfg: Dual 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 6ms | FB: 0.50 | Retrig: Off |
| **4** | `S5` | 2.481 | 412.8 | 0.0 | F1: LP 12 dB (cut:0.43, res:0.37) | F2: LP OB-Xd 24 dB (cut:0.52, res:0.85) | Cfg: Serial 3 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 12ms | FB: 0.50 | Retrig: On |
| **5** | `V6_C` | 2.467 | 607.7 | 0.0 | F1: LP Legacy Ladder (cut:0.25, res:0.16) | F2: LP OB-Xd 12 dB (cut:0.08, res:0.43) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.57) | Uni: 6v | Oct: +0 | Rel: 65ms | FB: 0.69 | Retrig: Off |

**Technical Analysis & Commentary**:
SampleTank 2 Moog saw multisample. V6_A won rank 1 (correct filter slope and envelope decay). S5 was too dark. DPO had sluggish FEG decay.

---

### Stem 20: Goddess Midrange Melodic Saw (`stem_20_goddess_mid_saw`)
* **Style**: Midrange Melodic Saw | **Tempo**: 141.0 BPM | **Root MIDI**: 56 | **Real Centroid**: 2763.8 Hz | **Real Width**: 0.960

**Human Reviewer Evaluation**:
> "Goddess Midrange Melodic Saw I think this is again bp filtered unobtainium of some specific analog synth emulation with some phaser, despite probably a quite basic routing. S5 too little resonance, this is just a saw wave, whereas the original is a quite nasal, resonant thin blip. DPO again just a single saw wave with no filter. V6_A lost the plot and tries an osc stack. V6_C gets a parrot stamp for the very creative try, it used sync to proxy the nasal sound, and things like this are always a good show of a generalisation ability (altough the DPO really sweeps the floor with the rest in this sense)."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `V6_C` | 8.499 | 3164.9 | 0.0 | F1: LP 24 dB (cut:0.53, res:0.54) | F2: LP 12 dB (cut:0.87, res:0.14) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **2** | `S5` | 5.273 | 1531.9 | 0.414 | F1: LP 12 dB (cut:0.44, res:0.53) | F2: LP 12 dB (cut:0.57, res:0.58) | Cfg: Dual 2 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 53ms | FB: 0.50 | Retrig: Off |
| **3** | `DPO_b300` | 6.494 | 1778.1 | 0.217 | F1: LP 24 dB (cut:0.27, res:0.03) | F2: LP K35 (cut:1.00, res:0.00) | Cfg: Dual 2 | FM: 2 > 1 < 3 | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 14ms | FB: 0.97 | Retrig: On |
| **4** | `V6_A` | 8.447 | 1544.1 | 0.0 | F1: LP 12 dB (cut:0.31, res:0.33) | F2: LP 12 dB (cut:0.12, res:0.78) | Cfg: Stereo | FM: Off | WS: Off (drv:0.50) | Uni: 5v | Oct: +0 | Rel: 39ms | FB: 0.50 | Retrig: On |
| **5** | `V6_B` | 8.447 | 1544.1 | 0.0 | F1: LP 12 dB (cut:0.31, res:0.33) | F2: LP 12 dB (cut:0.12, res:0.78) | Cfg: Stereo | FM: Off | WS: Off (drv:0.50) | Uni: 5v | Oct: +0 | Rel: 39ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
V6_C won rank 1 and earned 'parrot stamp' for using oscillator sync to proxy the unobtainium bandpass phaser timbre. DPO was 3rd with a plain saw.

---

### Stem 21: Astral Projection - Mahadeva (`stem_21_astral_mahadeva`)
* **Style**: Classic Goa Rolling Saw (Demucs) | **Tempo**: 140.0 BPM | **Root MIDI**: 41 | **Real Centroid**: 535.0 Hz | **Real Width**: 0.006

**Human Reviewer Evaluation**:
> "A great example of a classic thing: the bassline is a sped-up high-nrg octave bass instead of a more typical syncopated goa thing (it's possible it's a very old track from their new beaty past, despite the 96 remix moniker it sounds older), played on a single oscillator and a lowpass filter probably, but it has a very appealing 'piggy' timbre, which is hard to explain, a bit like the shotgun sample in Doom, which always puzzled me, like a snort in a way subliminally. Actually this might also be a digital synth, the waveform sound like having a bit peculiar harmonic balance and some of that sample-rate reduced sound might actually not come from the separation... Here my edit also slowed down the BPM. For both of my edits, the original speed should be returned. S5; sub-osc stack, too bright, too little smooth snappiness. DPO fails completely, a too bright sound. V6 all versions the same. Apparently the sheen was enough to throw the model off."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `S5` | 2.609 | 1104.0 | 0.0 | F1: LP 24 dB (cut:0.54, res:0.53) | F2: LP 12 dB (cut:0.61, res:0.33) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.51) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.48 | Retrig: Off |
| **2** | `V6_A` | 2.619 | 596.1 | 0.0 | F1: LP Vintage Ladder (cut:0.45, res:0.00) | F2: LP 12 dB (cut:0.46, res:0.27) | Cfg: Ring | FM: Off | WS: Off (drv:0.68) | Uni: 1v | Oct: +0 | Rel: 24ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_B` | 2.601 | 589.1 | 0.0 | F1: LP Vintage Ladder (cut:0.45, res:0.00) | F2: LP 12 dB (cut:0.46, res:0.27) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 24ms | FB: 0.50 | Retrig: Off |
| **4** | `DPO_b300` | 2.67 | 609.0 | 0.0 | F1: LP 12 dB (cut:0.21, res:0.00) | F2: LP 12 dB (cut:0.36, res:0.00) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: -1 | Rel: 27ms | FB: 0.92 | Retrig: Off |
| **5** | `V6_C` | 2.768 | 1116.0 | 0.0 | F1: LP Res Warp (cut:0.63, res:0.51) | F2: LP 24 dB (cut:0.75, res:0.35) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: Off |

**Technical Analysis & Commentary**:
Demucs separation sheen above 4 kHz threw off all predictors. S5 won rank 1 with simpler architecture. User DAW edit exported at 110 BPM instead of 140 BPM, stretching notes by 1.27x.

---

### Stem 22: MFG - Shape the Future (`stem_22_mfg_shape_future`)
* **Style**: Psy-Trance Punch (Demucs) | **Tempo**: 143.55 BPM | **Root MIDI**: 33 | **Real Centroid**: 412.3 Hz | **Real Width**: 0.003

**Human Reviewer Evaluation**:
> "To put it short: same issue with the MFG stem, the high frequencies trick the predictor. We could add an option to drop frequencies past 4k from the analysis, since most of anything on top of that will anyway be windfall from decisions made already; a few parameters uniquely target only the frequencies past that."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `S5` | 3.307 | 842.6 | 0.0 | F1: LP Vintage Ladder (cut:0.79, res:0.28) | F2: LP 12 dB (cut:0.58, res:0.28) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: -1 | Rel: 52ms | FB: 0.50 | Retrig: Off |
| **2** | `DPO_b300` | 3.795 | 1887.7 | 0.0 | F1: LP 12 dB (cut:0.38, res:0.00) | F2: LP 12 dB (cut:0.36, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: -1 | Rel: 40ms | FB: 0.50 | Retrig: On |
| **3** | `V6_A` | 3.528 | 1117.0 | 0.0 | F1: LP 24 dB (cut:0.74, res:0.00) | F2: LP 12 dB (cut:0.49, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.52) | Uni: 4v | Oct: +0 | Rel: 52ms | FB: 0.60 | Retrig: On |
| **4** | `V6_B` | 3.528 | 1117.0 | 0.0 | F1: LP 24 dB (cut:0.74, res:0.00) | F2: LP 12 dB (cut:0.49, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.52) | Uni: 4v | Oct: +0 | Rel: 52ms | FB: 0.60 | Retrig: On |
| **5** | `V6_C` | 3.817 | 1702.8 | 0.0 | F1: LP 24 dB (cut:0.66, res:0.00) | F2: LP 12 dB (cut:0.57, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 52ms | FB: 0.73 | Retrig: On |

**Technical Analysis & Commentary**:
Real centroid is 412.3 Hz with 94.15% of energy below 120 Hz. Demucs sheen tricked S5 to 842 Hz, V6 to 1117 Hz, V6_C to 1702 Hz (+1 oct transpose!), and DPO to 1888 Hz. Prompted the 4 kHz analysis drop proposal.

---

### Stem 23: Cosmosis - Alien Disco (`stem_23_cosmosis_alien_disco`)
* **Style**: Alien Disco Squelch (Demucs) | **Tempo**: 140.0 BPM | **Root MIDI**: 26 | **Real Centroid**: 536.3 Hz | **Real Width**: 0.102

**Human Reviewer Evaluation**:
> "Cosmosis - Alien Disco: then again, this even bleeds melody apart from the hf artefacts, but still the DPO is as close as I can imagine a predictor gets when even I'm not entirely sure what's supposed to go in with the programming, and what is bleed. V6 patches are in the right frequency range but lack filtering. S5 is way out where it shouldn't be with a bright retrigger osc"

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `DPO_b300` | 4.254 | 536.0 | 0.034 | F1: LP 12 dB (cut:0.52, res:0.00) | F2: LP 12 dB (cut:0.29, res:0.00) | Cfg: Ring | FM: Off | WS: Fuzz (drv:0.67) | Uni: 6v | Oct: +0 | Rel: 6ms | FB: 0.50 | Retrig: On |
| **2** | `V6_B` | 8.265 | 602.8 | 0.0 | F1: LP 24 dB (cut:0.95, res:0.00) | F2: LP 12 dB (cut:0.44, res:0.36) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: Off |
| **3** | `V6_C` | 7.993 | 725.7 | 0.0 | F1: LP Legacy Ladder (cut:0.86, res:0.00) | F2: LP 12 dB (cut:0.50, res:0.38) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: Off |
| **4** | `V6_A` | 8.218 | 711.9 | 0.0 | F1: LP 24 dB (cut:0.95, res:0.00) | F2: LP 12 dB (cut:0.44, res:0.36) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: Off |
| **5** | `S5` | 4.39 | 1440.1 | 0.106 | F1: LP 12 dB (cut:0.75, res:0.10) | F2: LP 12 dB (cut:0.77, res:0.41) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 6v | Oct: +1 | Rel: 54ms | FB: 0.50 | Retrig: On |

**Technical Analysis & Commentary**:
Melodic lead bleed into the separated bass stem. DPO won decisively (centroid 536.0 Hz vs real 536.3 Hz) using Fuzz waveshaper. V6 opened Filter 1 to 0.95 with 0.85 balance, lacking filtering.

---

### Stem 24: Final Mission (`stem_24_final_mission`)
* **Style**: Hypnotic Rolling Bass (Demucs) | **Tempo**: 140.0 BPM | **Root MIDI**: 36 | **Real Centroid**: 925.9 Hz | **Real Width**: 0.000

**Human Reviewer Evaluation**:
> "Final Mission extremely appealing single osc saw LP bass, I'd like to know what they used. Ever prediction misses this, I believe due to sepration artifacts. However: in the past we've got good predictions too. I'd almost be inclined to say that the predictor predicts three bands: sub bass, body and air, with the importance in the order bass-body-air, meaning that until the lower hierarchy level losses match, the predictor does not even try to predict the high frequencies. And when it tries, it is not allowed to make the lower level match much worse."

**Human Preference Rankings & Diagnostics**:

| Rank | Model | Phrase MSS | Centroid (Hz) | Stereo Width | Patch Architecture Summary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `S5` | 5.335 | 2082.1 | 0.0 | F1: LP K35 (cut:0.67, res:0.46) | F2: LP Vintage Ladder (cut:0.41, res:0.54) | Cfg: Ring | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 38ms | FB: 0.50 | Retrig: Off |
| **2** | `V6_C` | 5.384 | 525.4 | 0.0 | F1: LP Diode Ladder (cut:0.59, res:0.00) | F2: LP 24 dB (cut:1.00, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 8ms | FB: 0.50 | Retrig: Off |
| **3** | `DPO_b300` | 5.758 | 1956.3 | 0.0 | F1: LP Cutoff Warp (cut:0.63, res:0.13) | F2: LP 12 dB (cut:1.00, res:0.00) | Cfg: Serial 1 | FM: Off | WS: Asymmetric (drv:0.92) | Uni: 1v | Oct: +0 | Rel: 54ms | FB: 0.50 | Retrig: On |
| **4** | `V6_A` | 4.664 | 2899.6 | 0.0 | F1: LP 24 dB (cut:0.71, res:0.86) | F2: LP 24 dB (cut:0.22, res:0.42) | Cfg: Dual 1 | FM: 3 > 2 > 1 | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 32ms | FB: 0.60 | Retrig: Off |
| **5** | `V6_B` | 4.694 | 2967.1 | 0.0 | F1: LP 24 dB (cut:0.71, res:0.86) | F2: LP 24 dB (cut:0.22, res:0.42) | Cfg: Dual 1 | FM: 3 > 2 > 1 | WS: Off (drv:0.50) | Uni: 1v | Oct: +0 | Rel: 32ms | FB: 0.60 | Retrig: Off |

**Technical Analysis & Commentary**:
Hypnotic single-osc saw rolling bass from Cosmosis 'Synergy' (Roland SH-101). Separation artifacts misled V6 into hallucinating FM 3>2>1 and resonance 0.86 (centroid 2899 Hz). S5 won rank 1 with simpler closed filter.

---
