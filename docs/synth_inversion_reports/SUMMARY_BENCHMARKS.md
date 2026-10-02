# Comprehensive Benchmark & Ground-Truth Analysis Summary

> **⚠ Status (2026-10-02 code review): several results below are NOT valid evidence. Re-measure before relying on them.**
> The code is fixed in v2 (`scripts/synth_inversion/`, see its README §3 for the re-run commands); the numbers here are unchanged from v1.
> 1. **ResMLP vs flow "Best Val Loss" (0.0327 vs 0.0542) is not a model comparison.** ResMLP was scored on its point estimate (the MSE-optimal conditional mean), the flow on one fresh random draw per epoch (~2x the posterior variance by construction), with different Muon LRs (0.01 vs 0.03). Re-measure with `evaluate_holdout_audio.py` (audio-domain, with oracle and trivial baselines).
> 2. **The DE "ground truth" on `untitled.wav` inherited six parameters it never searched** (width, sync, FM depth, unison, filter-EG sustain, amp sustain) from the ResMLP render that ran just before it, including the pulse width behind the "narrow pulse (shape = 0.919)" conclusion. It was also a small search (8 iterations) optimising the very metric it is scored on: a reference, not ground truth.
> 3. **The neural numbers on `untitled.wav` are out of distribution**: a 99 ms clip with an 80 ms note, against training notes of 180-450 ms in 0.8 s windows.
> 4. **STFT losses are not comparable across sections**: `evaluate_200k_inversion.py` and `match_untitled_note.py` each used their own loss with different FFT sizes.
> 5. **Real-stem inputs were not resampled to 44.1 kHz or onset-aligned** in v1, so a non-44.1 kHz stem was fed to the model mis-scaled.
> 6. **v1 filter-circuit and waveshaper predictions were ordinal regressions rounded to the nearest index**, so uncertain predictions landed on whatever sat mid-list. The reported circuits partly reflect list order.
> 7. **The `.vstpreset` files are pedalboard raw-state blobs**, not VST3 preset files.
> 8. **The 544.9 Hz centroid is listed for both the 99 ms note and the 30 s stem**, which looks like a copy error.
> 9. **The "FEG_release >= AMP_release" coupling is not implemented** in the dataset generator (the filter-EG release is never set).
> 10. **The overnight suite ran `train_bracket.py` from an agent scratch directory**, so the committed code may not be the code that produced these numbers.

This report aggregates the sound matching experiments, 200k scaling benchmarks, and the single-note ground truth results on classic Goa/Psytrance basslines inverted into Surge XT 1.3.4.

---

## 1. Ground Truth Discovery: Waveform & Harmonics on `untitled.wav`

When matching the isolated single 16th-note sample from Chakra & Edi Mis - Final Mission (`untitled.wav`, 99.0 ms, C2 / 65.4 Hz):

### Harmonic Structure:
- **Fundamental (H1, 65.4 Hz)**: 205.2
- **Second Harmonic (H2, 130.8 Hz)**: **411.4** (+6.0 dB above fundamental!)
- **Third Harmonic (H3, 196.2 Hz)**: 61.8
- **Fourth Harmonic (H4, 261.6 Hz)**: 138.3

Because H2 is dominant over H1 and even harmonics are elevated relative to odd harmonics, unconstrained numerical solvers converged towards a narrow pulse wave (`shape = 0.919`) or rounded morph (`shape = 0.38`). 

When strictly constrained to a **pure Sawtooth (`shape = 0.0`)**:
- **Optimal Filter Topology**: `LP OB-Xd 24 dB`
- **Cutoff**: `0.832` (~2.8 kHz peak)
- **Resonance**: `0.490` (moderate analog Q shelf)
- **Sub-Oscillator**: `0.413` (compensates fundamental under the 24dB slope)
- **FEG Modulation**: Depth `0.371`, snappy decay `0.409` (~180 ms)
- **Multi-Scale STFT Loss**: **2.591**

---

## 2. Benchmark Comparison Table

| Target Audio | Method / Model | Osc Shape | Filter Circuit | Spectral Centroid | Multi-Scale STFT Loss | Inversion Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Isolated Note (`untitled.wav`)** | **Target Audio** | — | — | **544.9 Hz** | `0.000` | — |
| | **Closed-Loop DE (Pulse)** | `0.919` (Pulse) | LP OB-Xd 12 dB | 480.2 Hz | **2.168** | 51.1 s |
| | **Closed-Loop DE (Pure Saw)** | `0.000` (Saw) | LP OB-Xd 24 dB | 512.4 Hz | **2.591** | 42.0 s |
| | **ResMLP 200k (30 ep)** | `0.377` (Morph) | LP OB-Xd 24 dB | 390.1 Hz | **3.430** | **1.2 ms** |
| | **DeepFlow 200k (30 ep)** | `0.310` (Morph) | LP Legacy Ladder | 210.5 Hz | **6.301** | **18.2 ms** |
| **Full Track (30s, `bass.mid`)** | **Original Stem** | — | — | **544.9 Hz** | `0.000` | — |
| | **DE Pulse Patch** | `0.919` (Pulse) | LP OB-Xd 12 dB | 432.9 Hz | **2.722** | — |
| | **DE Pure Saw Patch** | `0.000` (Saw) | LP OB-Xd 24 dB | 991.4 Hz | **3.617** | — |
| | **DeepFlow 200k** | `0.310` (Morph) | LP Legacy Ladder | 131.3 Hz | **4.771** | — |

---

## 3. Physical Envelope Constraints & Coupling Rules

1. **AMP ADSR Sweet Spots for 16th-Note Psy Bass (140–145 BPM)**:
   - Attack: 0.1 – 25 ms (clickless transient)
   - Decay: 100 – 1000 ms
   - Sustain: 0 – 100%
   - Release: 5 – 500 ms (most useful range: **30 – 250 ms**, note duration boundary at 105 ms)
2. **FEG ADSR**:
   - Attack: 0 – 50 ms
   - Decay: 250 – 6000 ms
   - Sustain: 0 – 100%
   - Release: 0 – 3000 ms
3. **Physical Coupling Rule**:
   $$\text{FEG}_{\text{release}} \ge \text{AMP}_{\text{release}}$$
   When filter envelope modulation is positive, the filter release must outlast or match the amp release. If filter release drops below amp release, the cutoff drops while the VCA is still open, creating muddy, muffled sub-thuds instead of snappy 16th gating.

---

## 4. Overtraining & Step Telemetry Monitoring

To investigate representation grokking on waveform boundaries and filter circuits, an overnight suite is currently running:
- **`G03_resmlp_200ep_normuon_sf`**: 200 epochs of NorMuon (lr=0.01, Radial Brake=0.85) + Schedule-Free AdamW (lr=0.001, c_warmup=200 steps).
- **`G04_resmlp_100ep_adamw`**: 100 epochs of pure Decoupled AdamW with Cosine Annealing.
- **Real-Time Step Monitoring**: Logs Weight Velocity ($\|\Delta W\| \times \frac{1000}{\text{steps}}$), Direction Cosine ($\cos \theta$), Gradient Norms, and Weight Norms every 25 steps to `G03_resmlp_200ep_normuon_sf_step_telemetry.tsv`.
