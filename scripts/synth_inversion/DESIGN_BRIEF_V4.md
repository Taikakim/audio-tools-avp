# Design Brief: Surge XT Neural Inversion v4 (Attack Dynamics, Psychoacoustics & Dual Resonance)

**Author:** Antigravity / Kim  
**Date:** 2026-10-11  
**Target Audience:** Autonomous Implementation Agent & Audio ML Engineers  
**Target Repository:** `scripts/synth_inversion/`  

---

## 1. Executive Summary & Problem Definition

The v3 system achieved major breakthroughs: 110M JEPA parameter capacity, stable Shampoo matrix whitening, ordinal ladder regularization ($\rho > 0.90$), and fast flow inversion. However, auditioning the latest batch across 24 real bass stems (including classic Goa, Acid, and Industrial lines like Astral Projection's *Mahadeva* and *Vapausvoima*) identified three fundamental physical and perceptual failure modes:

1. **"Averagey" Timbral Hedging (Sluggish Attack, Sluggish Decay):**  
   Real electronic bass patches feature razor-sharp attack transients ($<30\,\text{ms}$) that plunge rapidly into damped, filtered sustain bodies. The v3 model hedges toward an "average" static filter cutoff that is slightly too dark on the attack and too bright on the tail.
2. **Resonance Underestimation (The Discrete STFT Double-Penalty):**  
   High-Q resonant filters produce narrow spectral spikes. In discrete STFT bins, even a 20 Hz frequency misalignment incurs a severe double penalty (error at target bin + error at synth bin). The optimizer minimizes loss by flattening the peak (dropping resonance $Q$).
3. **Static Sync & FM Artifacts:**  
   In classic dance production, hard sync and FM are transient modulations driven by fast envelope decays. Because v3 parameterizes sync and FM as static continuous constants, the model uses them to mimic attack brightness, leaving a sterile, glassy metallic comb filter ringing through the sustain.
4. **Architectural Blindness to Temporal Decay:**  
   `AudioEncoder` in `models_v3.py` collapses all 81 time frames into a single vector via `AdaptiveAvgPool2d((1, 1))`. The neural latent space literally cannot perceive $\frac{\partial \text{Spectrum}}{\partial t}$ (decay slope).

**v4 Goal:** Overhaul the temporal representation, loss formulation, and synth parameterization to achieve human-level transient bite, accurate resonant peak prominence, clean harmonic coherence, and dynamic sync/FM modulation.

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Two-Anchor Boundary Loss ($L_{\text{anchor}}$)
Rather than treating a note as a uniform time series where the 30 ms attack contributes only 3.7% of the $L_1$ loss, we formulate envelope modeling as a **two-anchor boundary-value problem**:

* **Anchor A (Attack Transient, $t \in [0, 40\,\text{ms}]$):**  
  Captures peak filter opening ($f_{\text{cutoff}} + f_{\text{feg\_amt}}$), initial pitch envelope, attack click/thump, and transient sync/FM.
* **Anchor B (Resting Sustain Tail, $t \in [0.25\,T, T]$):**  
  Captures resting filter cutoff ($f_{\text{cutoff}}$), baseline oscillator mix, harmonic ratio stack, and amplitude sustain level.

The loss function applies weighted multi-scale STFT across these distinct temporal regimes:
$$\mathcal{L}_{\text{time}} = \lambda_{\text{att}} \cdot \mathcal{L}_{\text{STFT}}(x_{\text{att}}, y_{\text{att}}) + \lambda_{\text{sus}} \cdot \mathcal{L}_{\text{STFT}}(x_{\text{sus}}, y_{\text{sus}}) + \lambda_{\text{global}} \cdot \mathcal{L}_{\text{STFT}}(x, y)$$
*Recommended weights:* $\lambda_{\text{att}} = 4.0$, $\lambda_{\text{sus}} = 2.0$, $\lambda_{\text{global}} = 1.0$.

*Result:* Because points A and B are firmly pinned, `feg_decay` and `aeg_decay` are mathematically forced to fit the true exponential decay rate connecting them.

---

### 2.2 Neural-Psychoacoustic Hearing Loss ($\mathcal{L}_{\text{psycho}}$)
Human audition does not compute fixed-window Fourier magnitudes:
1. **Auditory Onset Exaggeration:** The cochlea and auditory cortex amplify transients via rapid inner hair cell adaptation loops (Dau model).
2. **Phase & Chirp Sensitivity:** Downward sweeps (resonant plucks) sound punchy; upward sweeps or static noise sound completely different even if their time-averaged magnitude spectrum is identical.

**Implementation in v4:**
* Integrate **CDPAM** (Contrastive Differentiable Perceptual Audio Metric) or **DPAM** (Manocha et al., Stanford/Adobe) as a secondary loss:
  $$\mathcal{L}_{\text{psycho}} = \text{CDPAM}(x, y)$$
* *Fallback / Lightweight option:* A differentiable multi-scale Gammatone / ERB filterbank with logarithmic frequency warping and temporal finite-difference derivative loss ($\Delta x / \Delta t$).

---

### 2.3 Spectral Prominence & Bandwidth Resonance Matching ($\mathcal{L}_{\text{reso}}$)
To decouple resonance estimation from exact center-frequency bin alignment and eliminate the discrete STFT double-penalty:

1. **Log Spectrum:** $S(f) = 20 \log_{10} |X(f)|$ in dB.
2. **Rolling Median Background Baseline ($B(f)$):**  
   Computed with frequency window $W_{\text{freq}} \in [400, 600\,\text{Hz}]$:
   $$B(f) = \text{median}_{f' \in [f - W/2, f + W/2]} S(f')$$
   $B(f)$ traces the smooth, underlying filter roll-off curve without being distorted by sharp resonance spikes.
3. **Prominence Spectrum:**
   $$P(f) = \max\big(0,\, S(f) - B(f)\big) \quad \text{for } f > 800\,\text{Hz}$$
4. **Peak Feature Extraction:**
   Extract the Top-$K$ prominent peaks ($K=2$ to supervise Surge XT Filter 1 and Filter 2):
   $$\text{Peak}_k = (f_k, H_k, W_k)$$
   where $H_k = P(f_k)$ is the peak prominence height in dB, and $W_k$ is the Full Width at Half Maximum (FWHM) bandwidth.
5. **Prominence Matching Loss:**
   $$\mathcal{L}_{\text{reso}} = \sum_{k=1}^K \Big( \alpha_H |H_{k,\text{synth}} - H_{k,\text{target}}| + \alpha_W |W_{k,\text{synth}} - W_{k,\text{target}}| + \alpha_f \log \frac{f_{k,\text{synth}}}{f_{k,\text{target}}} \Big)$$

---

### 2.4 Harmonic Ratio Snapping & Dissonance Penalty
Sub-bass and melodic bass foundations require phase and harmonic stability. Non-harmonic continuous oscillator detuning creates unpleasant beating.
* Enforce musical ratio attraction during loss computation:
  $$\mathcal{L}_{\text{harm\_ratio}} = \min_{r \in \{2:1, 1:1, 1:2, 1:3\}} |\text{ratio}(\text{osc}_1, \text{osc}_2) - r|$$
* Invert continuous `osc_2_pitch` by snapping candidate patches or adding a heavy penalty against unharmonic irrational intervals unless specifically indicated by high inharmonic spectral energy.

---

### 2.5 Dynamic Surge XT Parameterization: Enveloped Sync & FM
Update `surge_spec_v4.py`:
* **Modulated Sync:** Expose `sync_feg_depth` (Filter/Mod envelope to oscillator sync). Allows a sharp 20 ms laser-like attack bite that drops to 0 during sustain.
* **Modulated FM:** Expose `fm_feg_depth` (Filter/Mod envelope to FM depth). Allows an acoustic knock/click attack without ringing inharmonic sidebands into the sustain.
* **Dual Filter Routing:** Expose `filter_1_type`, `filter_1_cutoff`, `filter_1_reso`, `filter_2_type`, `filter_2_cutoff`, `filter_2_reso`, and `filter_configuration` (Series = 0, Parallel = 1).

---

## 3. Architecture Specification

### 3.1 AudioEncoder v4: Temporal-Aware Sequence Aggregator
Replace the global pooling bottleneck in `models_v3.py`:

```python
# CURRENT v3 (Bottleneck):
self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 81 frames squashed to 1

# v4 ARCHITECTURE:
class TemporalAudioEncoderV4(nn.Module):
    def __init__(self, in_channels=1, embed_dim=768):
        super().__init__()
        # Conv2D backbone (spectrogram feature extractor)
        self.backbone = ConvBackbone(...)  # outputs [B, C, F, T]
        
        # Frequency pooling -> temporal frame sequence [B, C, T]
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Dual Temporal Segment Pooling
        # Frame indices: T_att = frames 0..8 (0-40ms), T_sus = frames 25..80 (200-800ms)
        self.proj_attack = nn.Sequential(nn.Linear(C, embed_dim // 2), nn.GELU())
        self.proj_sustain = nn.Sequential(nn.Linear(C, embed_dim // 2), nn.GELU())
        
        # Temporal Cross-Attention / Self-Attention Layer
        self.temporal_transformer = nn.TransformerEncoderLayer(d_model=C, nhead=8)
        
        # Final fused embedding
        self.out_proj = nn.Linear(C + embed_dim, embed_dim)
```
*Output:* A 768-D representation containing explicit information on both transient onset spectrum, steady-state sustain spectrum, and the temporal decay rate connecting them.

---

## 4. File Structure & Implementation Contracts

The external agent should construct the following files under `scripts/synth_inversion/`:

| File | Purpose / Deliverable |
|---|---|
| `surge_spec_v4.py` | Updated Surge XT wrapper exposing modulated sync/FM, dual filter parameters, and harmonic ratio enums. |
| `models_v4.py` | `TemporalAudioEncoderV4` with split attack/sustain pooling + updated `SynthFlowInverterV4`. |
| `realistic_loss_v4.py` | Modular loss module containing: `TwoAnchorSTFTLoss`, `SpectralProminenceResonanceLoss`, `HarmonicRatioLoss`, and optional `CDPAMLoss`. |
| `train_realistic_bass_v4.py` | Training loop incorporating the v4 losses, maintaining Shampoo whitening and ordinal ladder regularization. |
| `inference_v4.py` | Flow inference with two-anchor candidate reranking, peak prominence scoring, and automatic harmonic ratio locking. |
| `render_v4_eval.py` | Benchmark harness testing the 24 curated real bass stems with MuScriptor MIDIs, comparing v3 vs v4 metrics. |

---

## 5. Verification & Acceptance Criteria

1. **Transient Attack Fidelity:**  
   On sharp plucks (e.g. *Acid Alien Industrial Pluck*, *Stay Pure Damped Disco*), the initial 40 ms spectral centroid and energy contour must match target attack within 5% error.
2. **Resonance Prominence Score:**  
   On resonant acid stems (*Vapausvoima TB-303*, *Astral Protection Rubber*), the extracted peak prominence $H$ must match ground truth within $\pm 2.0\,\text{dB}$, eliminating the zero-resonance hedging bias.
3. **Absence of Static Comb Artifacts:**  
   Auditioning of steady-state sustains must confirm zero unmodulated sync/FM buzz on pure basslines (e.g. *Mahadeva*, *Increase Dose Disco Saw*).
4. **MuScriptor Full-Phrase STFT:**  
   Overall 24-stem phrase STFT loss must improve or match v3 while perceptually delivering punchier transients and authentic resonance.

---

## 6. Execution Guidelines for Implementing Agent

* **Preserve GPU Safety:** The existing A0/A1 training run holds GPU 0 under `gpu_guard.sh`. Do NOT execute GPU jobs until the training queue finishes or an isolated GPU device is assigned.
* **Keep Dependencies Modular:** The CDPAM / neural psychoacoustic package should be wrapped with a graceful fallback (if `cdpam` is not installed in the environment, fallback cleanly to `TwoAnchorSTFTLoss + SpectralProminenceResonanceLoss`).
* **Unit Tests First:** Implement test suite `test_v4_components.py` verifying:
  - Top-Hat rolling median baseline extracts peak prominence accurately on synthetic 12 dB resonant filters.
  - Two-anchor STFT produces high loss when a synth patch has sluggish decay.
  - `surge_spec_v4.py` round-trips all new parameters through the Surge XT native C++ engine without assertions.
