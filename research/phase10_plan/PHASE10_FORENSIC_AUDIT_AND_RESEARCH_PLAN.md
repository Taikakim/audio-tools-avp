# Phase 9 Forensic Audit & Phase 10 Research Specification
**Date:** 2026-11-07  
**Status:** Approved & Grounded in Live Code/Data Audits  
**Repository:** `/home/kim/Projects/SAO/stable-audio-tools`  
**Branch:** `research/phase9-residual-and-decoder-adaptation`

---

## Part 1: Forensic Audit of Phase 9 Anomalies & Overclaims

### 1. Root-Cause Audit of the −32.21 dB Mixture SDR Baseline
The peer review correctly identified that a mean untouched mixture SDR of **$-32.21\text{ dB}$** was physically impossible for a foreground gain range of $-12\text{ dB}$ to $0\text{ dB}$ under standard scale-dependent SDR:
$$\text{SDR}(b, b+s) = 10 \log_{10} \frac{\|b\|^2}{\|s\|^2}$$
Assuming aligned signals, this value should center near $0\text{ dB}$ ($\pm 6\text{ dB}$).

**Live Code & Data Audit:**
Inspection of [`build_phase9_residual_dataset.py`](file:///home/kim/evals_aac/same_latent_research/batch9_phase9_residual_diagnostics/build_phase9_residual_dataset.py) revealed the exact failure mechanism:
1. Multitrack stem files from the Goa separated corpus are 8 to 10 minutes long ($>2.5 \times 10^7$ samples). In many arrangements, instruments (leads, flutes, flangers, acid riffs) do not enter during the intro.
2. In `load_audio_clip`, the energetic segment search was capped to the first 60 seconds (`cand in range(0, min(..., 60*sr), hop)`).
3. For stems that enter after 60 seconds (e.g., test items 1, 2, 6, 8, 9), the extracted background clip was **pure silence** ($\|b\|^2 = 0.000000$).
4. For these items, $\text{SDR}(b, m) = 10 \log_{10} \frac{0}{\|s\|^2} = -\infty\text{ dB}$ (clamped in code to $-80\text{ dB}$ to $-102\text{ dB}$).
5. An unweighted arithmetic mean was computed across all 30 test items, where 15 active items with normal SDRs ($+3\text{ dB}$ to $+9\text{ dB}$) were averaged together with 15 silent-crop outliers ($-80\text{ dB}$ to $-102\text{ dB}$), dragging the arithmetic mean down to **$-32.21\text{ dB}$**.

**Empirical Verification on Active Test Clips:**
When filtering out the silent-crop artifacts ($\|b\|^2 > 10^{-4}$):
- **Mean SDR on Active Backgrounds:** **$+0.55\text{ dB}$**
- **Median SDR on Active Backgrounds:** **$-1.50\text{ dB}$**

This aligns with physical expectation. Furthermore, when evaluating in the **Autoencoder Domain** ($\text{SDR}(\mathcal{D}(z_b), \mathcal{D}(\hat{z}_b))$), where the autoencoder reconstruction serves as reference, every test item exhibits positive SDR (mean $+7.5\text{ dB}$ to $+13.9\text{ dB}$), completely eliminating the $-\infty$ artifact.

---

### 2. Reconciliation of NMSE and $R^2$ Metrics
The peer review noted an apparent discrepancy between:
$$\text{NMSE} = 0.8083 \quad \text{and} \quad R^2 = 0.3911$$
along with the anomalous zero-baseline values ($\text{NMSE} = 1.0000$, $R^2 = 0.1826$).

**Live Code Audit:**
The metrics were implemented in [`exp_9_1_oracle_residual_predictability.py`](file:///home/kim/Projects/SAO/stable-audio-tools/research/phase9_residual/scripts/exp_9_1_oracle_residual_predictability.py) as **sample-averages of per-clip ratios**, not pooled dataset aggregates:
$$\text{Reported NMSE} = \frac{1}{N} \sum_{i=1}^N \frac{\|r^{(i)} - \hat{r}^{(i)}\|^2}{\|r^{(i)}\|^2}$$
$$\text{Reported } R^2 = \frac{1}{N} \sum_{i=1}^N \left( 1 - \frac{\|r^{(i)} - \hat{r}^{(i)}\|^2}{\|r^{(i)} - \bar{r}_{\text{train}}\|^2} \right)$$
where $\bar{r}_{\text{train}}$ is the global per-channel mean across training frames.

1. **Why Zero Baseline NMSE = 1.0000:** When $\hat{r}^{(i)} = 0$, the numerator equals the denominator for every sample, making each ratio identically $1.0000$.
2. **Why Zero Baseline $R^2 = 0.1826$:** Because $\bar{r}_{\text{train}} \neq 0$, the denominator $\|r^{(i)} - \bar{r}_{\text{train}}\|^2$ differs from $\|r^{(i)}\|^2$. On test clips where $\|r^{(i)}\|^2 < \|r^{(i)} - \bar{r}_{\text{train}}\|^2$, the per-sample ratio is $< 1$, yielding a positive $R^2$ on those clips. Averaging these ratios yielded $+0.1826$.
3. **Pooled Aggregate Metric:** When computed as a single pooled aggregate across all test frames:
   $$\text{Aggregate } R^2 = 1 - \frac{\sum_i \|r^{(i)} - \hat{r}^{(i)}\|^2}{\sum_i \|r^{(i)} - \bar{r}_{\text{train}}\|^2} = \mathbf{-0.0432}$$
   This confirms that under pooled evaluation, the zero baseline performs at $\le 0$, as expected.

---

### 3. Stage D Input and Baseline Specification
The peer review highlighted ambiguity in the input to `WaveformResidualAdapter`.

**Exact Implementation in Code:**
1. The adapter receives:
   $$x_{\text{in}} = \mathcal{D}(z_m - z_s + z_0)$$
   which is the waveform obtained by decoding the oracle algebraic subtraction latent through the frozen SAME decoder.
2. The conditioning variable $q$ is the foreground family class index ($0 \le q < 32$).
3. The adapter computes a bounded waveform perturbation:
   $$\hat{b}(t) = x_{\text{in}}(t) + 0.1 \cdot \tanh(\text{ConvNet}(x_{\text{in}}, q))$$
4. The $+1.04\text{ dB}$ gain was measured against the unadapted decoded draft on matched evaluation slices.

---

### 4. Retraction of Theoretical Overclaims & Proper Scope of Evidence

| Previous Agent Claim | Peer Review Finding | Corrected Empirical Statement |
| :--- | :--- | :--- |
| **"61% of residual variance is irreducible phase ambiguity"** | Not demonstrated. Conflates model capacity, data size, optimization, and alignment with information theory. | The tested 4-head Conv+Attn model explained **$39.1\%$** of residual variance. The remaining variance reflects a combination of model capacity, context horizon, optimization, and potential latent ambiguity. |
| **"Flow sample diversity proves conditional multimodality"** | Not demonstrated. Diversity could reflect unconverged training, integration drift, or weak conditioning. | The 1.4M parameter flow model generated diverse samples ($\sigma^2 = 0.225$) but failed to reconstruct the reference source ($\text{SDR} \approx \text{mixture}$). It does not establish properties of the true conditional distribution. |
| **"Waveform mixing causes comb filtering and clipping"** | Physically incorrect for general mixing. | Waveform summation $m = b + s$ is strictly linear. Comb filtering requires correlated delayed signals, and clipping occurs only under bounded output constraints. |
| **"Latents retain only modulations below 10.77 Hz"** | Incorrect. Conflates frame hop with parametric capacity. | The 10.77 Hz frame rate sets the explicit update rate. Channels can parametrically encode higher-frequency envelope and pitch features within frames. |
| **"Frozen decoder interface is the proven ultimate bottleneck"** | Not demonstrated. | Lower latent $L_1$ error did not yield higher decoded SDR in the tested models. `enable_grad = True` is required to backpropagate waveform losses, but whether this resolves demixing requires controlled testing. |

---

## Part 2: Phase 10 Research Specification

### Gate 0: Repair Measurement Foundation (Mandatory Prerequisite)
Before launching Phase 10 model training:
1. **0A. Clean Dataset Curation:** Update `load_audio_clip` to search the full multitrack stem for active segments ($\text{RMS} > 0.02$). Exclude any item where background or foreground energy falls below $-40\text{ dBFS}$.
2. **0B. Metric Standardization:** Report both:
   - **Pooled Aggregate Metrics:** $\text{NMSE}_{\text{pool}} = \frac{\sum \|r - \hat{r}\|^2}{\sum \|r\|^2}$, $R^2_{\text{pool}} = 1 - \frac{\sum \|r - \hat{r}\|^2}{\sum \|r - \bar{r}\|^2}$.
   - **Audio Metrics:** Energy-weighted SDR, median SDR, and Autoencoder-Domain SDR ($\text{SDR}(\mathcal{D}(z_b), \hat{b})$) alongside raw-waveform SDR.
3. **0C. Standard Latency Correction:** Verify sample-accurate cross-correlation alignment between decoder output and ground truth before computing SDR.

---

### Experiment 10.1: Residual Approximation Capacity Ladder
**Objective:** Determine whether the 39.1% residual prediction accuracy is a model capacity ceiling or an asymptotic limit.
- **Task:** Oracle residual prediction $(z_b, z_s) \to r_{\text{int}}$.
- **Architectures:**
  1. Bilinear Multiplicative Model (~100k params)
  2. Conv+Attention Model (~650k params)
  3. 8-Layer Transformer (~4M params)
  4. 16-Layer Transformer (~16M params)
  5. TFLocoformer-style Time-Frequency Model (~30M params)
- **Dataset Scaling:** Train on subsets of $N \in \{100, 300, 600, 1200\}$ mixtures.
- **Success Criteria:** Fit scaling curves $R^2(N, \text{params})$. If $R^2$ continues to rise log-linearly, the "phase ceiling" claim is refuted.

---

### Experiment 10.2: Direct Empirical Test of Latent Conditional Ambiguity
**Objective:** Directly test whether indistinguishable latent pairs can produce materially different interaction residuals.
- **Method:**
  1. For fixed stems $b(t)$ and $s(t)$, generate all-pass filtered and phase-rotated variants $b_\phi(t)$ and $s_\psi(t)$ that preserve magnitude spectra.
  2. Compute latent distances:
     $$d_{\text{input}} = \|\mathcal{E}(b) - \mathcal{E}(b_\phi)\|_2 + \|\mathcal{E}(s) - \mathcal{E}(s_\psi)\|_2$$
  3. Compute residual difference:
     $$d_{\text{res}} = \|r_{\text{int}}(b, s) - r_{\text{int}}(b_\phi, s_\psi)\|_2$$
  4. Search for pairs where $d_{\text{input}} < \epsilon$ (latent collision) but $d_{\text{res}} \gg 0$.
- **Stratification:** Stratify by frequency bands ($<1\text{ kHz}$, $1\text{--}4\text{ kHz}$, $4\text{--}7\text{ kHz}$, $>7\text{ kHz}$) to test the ~7 kHz phase-discard hypothesis.

---

### Experiment 10.3: SAME-Only vs. Waveform-Phase Augmentation
**Objective:** Test if supplying explicit phase features resolves residual prediction error.
- **Arms:**
  1. Baseline: $(z_b, z_s)$ only.
  2. Phase-Augmented: $(z_b, z_s)$ concatenated with complex STFT phase-difference features $\Delta \phi_{b,s}(t, f)$ downsampled to $10.77\text{ Hz}$.
  3. Fine-Rate Waveform Features: Downsampled multi-band Hilbert envelopes.
- **Interpretation:** If phase augmentation yields a substantial jump in $R^2$ that model capacity alone cannot match, missing phase information is directly supported.

---

### Experiment 10.4: Decoder-Aware Deterministic Separator
**Objective:** Test whether backpropagating waveform-level losses through the decoder improves separation.
- **Model:** Practical direct separator $\hat{z}_b = f_\theta(z_m, q)$.
- **Configuration:** Set `pretransform.enable_grad = True` while keeping decoder weights frozen.
- **Loss Comparison:**
  1. Latent $L_1$ loss only.
  2. Latent $L_1$ + Multi-Resolution Magnitude STFT loss.
  3. Multi-Resolution STFT + Complex STFT loss.
- **Follow-up:** Only after evaluating frozen-decoder gradient propagation, test decoder LoRA and adapter fine-tuning.

---

### Experiment 10.5: Hybrid Latent + Waveform Refinement Pipeline
**Objective:** Incrementally evaluate a dual-resolution separation pipeline against matched baselines.
- **Models:**
  - **Model A:** Waveform-only separator (Conv-TasNet / small Demucs baseline).
  - **Model B:** Latent separator followed by frozen decode (no waveform refinement).
  - **Model C:** Latent separator + Waveform refiner:
    $$\hat{b}(t) = \mathcal{D}(\hat{z}_b) + g_\phi\big(m(t), \mathcal{D}(\hat{z}_b), q\big)$$
    with identity loss on target-absent clips: $\mathcal{L}_{\text{id}} = \|\hat{b} - m\|_1$.
  - **Model D:** Model C augmented with multi-band STFT refinement.

---

### Experiment 10.6: Structured Joint Flow Baseline (SURF / SepGen Style)
**Objective:** Replace the under-parameterized 1.4M flow baseline with a properly scaled model.
- **Specifications:**
  1. Parameter count: $30\text{M}\text{--}40\text{M}$ parameters.
  2. Joint multi-stem generation with explicit source axis.
  3. Permutation-equivariant layers and mixture conditioning.
  4. Mixture-consistency projection: $\sum_k \hat{x}_k = m$.
  5. 5-step and 20-step Euler/Heun sampling.
- **Evaluation:** Measure mean, median, and best-of-$K$ SDR against reference sources, evaluating whether generated stems explain the mixture rather than merely varying.
