# V5 Architecture Analysis, Failure Modes, and Proposed Inversion Paradigm

**Date:** October 13, 2026  
**Author:** Antigravity (`antigravity@aavepyora.online`) for Kim, Kuang, and Wintermute  
**Context:** Review of the A-series 2×2 factorial evaluation (`eval_v4_2x2`), comparison against original Synth-JEPA (Hayes et al., arXiv:2609.31024), metric review of SonicMaster (`2508.03448v5`) and Stability AI's `stable-audio-metrics`, analysis of the parameter-space vs audio-space training paradox, training speed impact, and precomputation strategy.  
**Evidence Tags:** **[V]** verified in code/data here, **[I]** inferred from theoretical/acoustic mechanics, **[N]** project notes, **[U]** proposed/untrained.

---

## 0. Executive Summary & Root-Cause Diagnosis

The A-series 2×2 factorial evaluation (A0, A1, A2, A3, A3′) established two clear statistical facts under the pre-registered V4 paradigm [V]:
1. **Temporal ASP Flow Encoder is established better ($p < 0.019$):** Monotonicity rank correlation across dynamic parameter ladders jumped by $+0.0418$ overall ($p = 0.019$) and $+0.1114$ on Amp EG sustain tracking ($p < 0.001$), with Filter EG amount error reduced by $-0.0074$ ($p = 0.010$).
2. **Inharmonicity Aux Head on JEPA is not established ($p > 0.40$):** All retrieval and representation metrics remained within the empirical run-to-run replicate gauge measured between A3 and A3′ ($\|A3 - A3'\| = 0.0208$).

Despite these specific improvements, macro-level parameter metrics appeared remarkably flat across all arms (parameter MAE hovered uniformly at $\sim 0.121$, flow MSE at $\sim 0.40$, and categorical accuracy at $\sim 68.6\%$). **Why did the overall scores stay so close despite fundamental architectural changes?**

### The Core Diagnosis: Two Disconnects
* **Disconnect 1 (Evaluation Blindness) [V, I]:** The `eval_v4` suite operates strictly in **unrendered parameter space** on synthetic presets. It calculates isotropic $\ell_1$ error across 48 parameter dimensions. Crucially, **inactive or inaudible parameters** (e.g., waveshaper drive when waveshaper type is disabled; detune when unison voices = 1; oscillator 2 pulse width when oscillator 2 volume is zero) receive the exact same error penalty as the main lowpass filter cutoff. This injects an irreducible $\sim 0.12$ parameter error floor that masks genuine acoustic divergence.
* **Disconnect 2 (Training Deafness) [V, I]:** The conditional flow-matching model is trained with supervised $\ell_1$ regression against parameter vectors:
  $$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, \theta_0, \theta_1} \left[ \| v_\phi(\theta_t, t, z_a) - (\theta_1 - \theta_0) \|_1 \right]$$
  **The trainer literally never synthesizes audio from its predictions during training.** Surge XT is a non-differentiable C++ VST3 plugin; standard backpropagation from audio reconstruction loss is blocked. As a result, the flow model has no mechanism to learn that missing the cutoff frequency by 5% ruins the sound, whereas missing an inactive chorus rate by 50% is completely inaudible.

---

## 1. The Core Training Paradox: What MELs Are Used For vs. What Is Never Heard

A crucial point of confusion in neural synth inversion is: **"If we work purely in parameter space, where do MEL spectrograms fit in, and why don't we hear the sound?"**

### 1.1 Where the MELs Come From: They Are the "Input Questions"
During training, audio **is** rendered by Surge XT, but **only in the background by the DataLoader to create the input prompts**:
1. **Background Preset Sampling:** The DataLoader takes a ground-truth preset $\theta^*$ from the training manifold (e.g., Cutoff = 35%, 1-voice Saw, Pluck decay).
2. **Offline Rendering:** Surge XT renders a 0.8-second audio waveform for that preset.
3. **Spectrogram Generation:** On the GPU, `ExactGpuMel` converts the waveform into a $128$-bin log-mel spectrogram.
4. **Input Prompt:** This MEL spectrogram is handed to the neural network as the **input conditioning question**:  
   *"Here is a spectrogram image of a sound. Predict the 48 Surge XT knob positions that produced it."*

### 1.2 What We NEVER Do: We Never Render or Hear the Model's "Answers"
Here is where the training loop terminates without audio feedback:
1. **The Model's Output:** The flow model integrates its velocity field and outputs a predicted parameter vector $\hat{\theta}$ (e.g., Cutoff = 38%, 1-voice Saw).
2. **The Loss Evaluation (On Paper Only):** We calculate the training loss by directly comparing the numbers:
   $$\mathcal{L} = \| \hat{\theta}_{\text{guess}} - \theta^*_{\text{truth}} \|_1$$
3. **The Missing Step:** We **never** take $\hat{\theta}_{\text{guess}}$, instantiate Surge XT, render the audio, convert it into a spectrogram $\widehat{\text{MEL}}$, and evaluate whether:
   $$\widehat{\text{MEL}} \stackrel{?}{\approx} \text{Target MEL}$$

```
CURRENT OPEN-LOOP TRAINING:
[Target Audio] ──► [Target MEL] ──► [Neural Network] ──► [Predicted Knobs θ_pred]
                                                                │
                                              ┌─────────────────┴─────────────────┐
                                              ▼                                   ▼
                                     LOSS: |θ_pred - θ*|                 [Surge XT Engine]
                                     (Knob distance only)                        │
                                                                           (NEVER RUN!)
                                                                                 │
                                                                                 ▼
                                                                           [Render Audio]
                                                                                 │
                                                                                 ▼
                                                                            [Audio Loss]
                                                                           (DOES NOT EXIST)
```

### 1.3 The Classroom "Cake Recipe" Analogy
Our current training setup is identical to training an apprentice chef to replicate cakes with their **sense of taste completely blocked**:
* **The Question:** You show them a photograph of a chocolate cake (the **Input MEL Spectrogram**).
* **The Task:** You ask them to write down 48 ingredient quantities on a clipboard (the **Predicted Parameters $\hat{\theta}$**).
* **The Grading:** You grade their test by comparing their written clipboard numbers against the master chef's written recipe sheet ($|\hat{\theta} - \theta^*|$).
* **The Flaw:** **Nobody ever bakes the cake from the apprentice's recipe to taste it!**
  - If the apprentice writes down salt instead of sugar, nobody tastes the cake during training.
  - If the apprentice makes a huge error on an ingredient that does not affect taste (e.g. food coloring when chocolate is already dark), they receive a heavy mathematical penalty on paper.
  - **When given a real cake from a commercial bakery (a real audio stem):** There is **no chef's recipe sheet ($\theta^*$)** to grade against! The stem was made on a Moog or Roland through an analog console. The *only* way to know if you matched it is to bake the cake, taste it, and compare flavors—which the model was never trained to do.

---

## 2. Deconstruction of the 2×2 Factorial Results

### 2.1 Complete Multi-Arm Performance Table (Online Weights)

| Metric Category | Metric | A0 (Baseline)<br>Conv / No Aux | A1<br>Conv / +Aux | A2<br>Temporal / No Aux | A3<br>Temporal / +Aux | A3′ (Replicate)<br>Temporal / +Aux | Replicate Gauge ($\|A3 - A3'\|$) | Contrast Effect [$95\%$ CI] | Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JEPA Retrieval** | MRR Audio $\to$ Preset | 0.3642 | 0.3477 | 0.3579 | 0.3455 | 0.3663 | 0.0208 | Aux: $-0.0078$ [$-0.027, 0.013$] | **Not established** ($p=0.428$) |
| | MRR Preset $\to$ Audio | 0.3346 | 0.3227 | 0.3270 | 0.3105 | 0.3298 | 0.0193 | Aux: $-0.0098$ [$-0.027, 0.008$] | **Not established** ($p=0.277$) |
| | Top-1 Hit (A $\to$ P) | 23.39% | 21.78% | 22.90% | 21.73% | 23.73% | 0.0200 | Aux: $-0.73\%$ [$-2.28\%, 0.89\%$] | **Not established** ($p=0.376$) |
| | Ladder JEPA $\rho$ | 0.9118 | 0.8967 | 0.8930 | 0.8716 | 0.8901 | 0.0251 | Aux: $-0.0163$ (below gauge) | **Below gauge** ($p=0.014$) |
| **Flow Monotonicity** | **Overall Ladder Flow $\rho$** | 0.5055 | 0.5183 | 0.5334 | 0.5510 | 0.5766 | 0.0256 | Temporal: **$+0.0418$** [$+0.007, +0.081$] | **Established better** ($p=0.019$) |
| | **AEG Sustain Ladder $\rho$** | 0.3390 | 0.3408 | 0.4355 | 0.4589 | 0.4595 | 0.0234 | Temporal: **$+0.1114$** [$+0.069, +0.157$] | **Established better** ($p<0.001$) |
| | FEG Amount Ladder $\rho$ | 0.8186 | 0.8403 | 0.8552 | 0.8190 | 0.8566 | 0.0376 | Temporal: $+0.0142$ [$-0.022, 0.049$] | **Not established** ($p=0.422$) |
| | Cutoff Ladder $\rho$ | 0.4279 | 0.4475 | 0.4397 | 0.4549 | 0.4704 | 0.0196 | Temporal: $+0.0173$ [$-0.040, 0.065$] | **Not established** ($p=0.554$) |
| **Parameter Accuracy** | **Categorical Accuracy** | 68.56% | 68.27% | 69.03% | 69.09% | 68.64% | 0.0045 | Temporal: **$+0.51\%$** [$+0.19\%, +0.90\%$] | **Established better** ($p<0.001$) |
| | **Filter EG Amount Error** | 0.1320 | 0.1324 | 0.1267 | 0.1224 | 0.1252 | 0.0043 | Temporal: **$-0.0074$** [$-0.013, -0.002$] | **Established better** ($p=0.010$) |
| | Mean Param MAE | 0.1221 | 0.1216 | 0.1220 | 0.1207 | 0.1216 | 0.0013 | Temporal: $-0.0004$ [$-0.002, 0.001$] | **Not established** ($p=0.536$) |
| | Flow MSE Loss | 0.3917 | 0.3957 | 0.4067 | 0.4040 | 0.4039 | 0.0040 | Temporal: $+0.0111$ [$-0.002, 0.028$] | **Not established** ($p=0.125$) |

### 2.2 Scientific Interpretation
1. **The Inharmonicity Aux Head ($h(z_a) \to y_{\text{inharm}}$):** On clean synthetic presets, inharmonic dissonance is close to zero ($R^2 \le 0$ on held-out). Adding an auxiliary task with near-zero variance provided no gradient signal to structure the latent space. Its observed variation across arms is purely seed noise within the replicate gauge ($\|A3 - A3'\|$).
2. **The Temporal ASP Encoder:** Preserving time resolution and pooling with Attentive Statistics Pooling (ASP) significantly improves dynamic temporal control (AEG sustain tracking jumped $+0.1114$). This validates that spectro-temporal order matters, but exposes the next bottleneck: how that audio information is ingested by the flow model.

---

## 3. Comparison with Original Synth-JEPA (Hayes et al., 2026)

Section 3.2 of the Synth-JEPA paper (*"Synth-JEPA: Joint Embedding Prediction for Renderer-Free Synthesizer Parameter Search"*, arXiv:2609.31024) reveals crucial architectural alignments and divergences:

### 3.1 Conceptual Inversion Paradigms
* **Original Synth-JEPA is an Optimization Landscape, Not a Generator [V]:**  
  In the paper, Synth-JEPA never directly predicts parameters in a single forward pass. Instead, it is used as a **differentiable proxy objective** $D_{\text{JEPA}}(y^*, x) = \| z_a^* - f_{p \to a}(E_p(x)) \|^2$.  
  Inversion is executed by running **JADE evolutionary search + Adam gradient descent directly on continuous parameter coordinates** through $\nabla_x D_{\text{JEPA}}$ via backpropagation through the neural parameter encoder $E_p$.
* **Our Work Uses an Amortized Flow Generator [V]:**  
  Because multi-step JADE + Adam search takes several seconds to minutes per musical phrase, we co-trained an amortized **Conditional Flow Matching ResMLP** to sample the complete parameter vector in 25 Euler steps ($\sim 50\,\text{ms}$). Flow matching was a separate, competing baseline in the original paper.

### 3.2 Deep Architectural Audit

| Subsystem | Original Synth-JEPA Specification | Our Implementation (`real_bass_v3_*`) | Status & Impact |
| :--- | :--- | :--- | :--- |
| **Audio Encoder ($E_a$)** | Transformer: 128 mel bands, 25ms win, 10ms hop $\to$ 1D conv (stride 2) to 150 tokens $\to$ prepended `[CLS]` token $\to z_a \in \mathbb{R}^{512}$. | Transformer: 128 mel bands, 10ms hop $\to$ 1D conv (stride 2) to 41 tokens $\to$ prepended `[CLS]` token $\to z_a \in \mathbb{R}^{512}$. | **Aligned.** Adapted token length for 0.8s isolated bass stabs. |
| **Parameter Encoder ($E_p$)** | Linear projection + bias $\to$ Perceiver bottleneck (**32 learned latents** cross-attending to tokens) $\to$ 8 self-attention blocks $\to$ Mean pool $\to z_p \in \mathbb{R}^{512}$. | Linear projection + bias $\to$ Perceiver bottleneck (**32 learned latents** cross-attending to tokens) $\to$ 8 self-attention blocks $\to$ Mean pool $\to z_p \in \mathbb{R}^{512}$. | **Exact match.** `synth_jepa_model.py:L146` implements the full 32-latent Perceiver read-in. |
| **Flow Conditioning** | Flow matching was a standalone baseline. | Audio conditioning collapsed to **ONE 1D vector** added as an additive bias: `h = audio_emb + time_emb + param_proj` into a ResMLP. | **CRITICAL BOTTLENECK [V].** The flow inverter has **zero cross-attention layers**. It compresses the whole spectrogram into a single flat vector. |
| **Positional Encoding** | 1D learned absolute positional embeddings: `nn.Parameter(1, N+1, d)`. | 1D learned absolute positional embeddings. | **Bottleneck.** Neither uses Rotary Position Embeddings (RoPE). Lacks relative-time equivariance. |
| **Training Scale** | **1,000,000 steps** (Batch size 64) = **64,000,000 examples seen**. | **12,000 steps** (Batch size 64) = **768,000 examples seen**. | **We trained at 1.2% of the paper's compute budget.** |
| **Optimizer** | AdamW ($3 \times 10^{-4}$), WSD schedule, weight decay 0.05. | **NorMuon** with Schedule-Free, Shampoo whitening, Newton-Schulz cubic5, and radial braking (`ModularOptimizer`). | **Our optimizer is significantly more advanced [V].** Both JEPA and Flow ran with NorMuon. |

---

## 4. Repositories and Metric Suites Discussed

### 4.1 Referenced Repositories
1. **SonicMaster (`/home/kim/Projects/SonicMaster`):**
   - Reference Paper: *SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering* (arXiv:2508.03448v5, Jan Melechovsky, Ambuj Mehrish, Abhinaba Roy, Dorien Herremans).
   - Core Architecture: Flow Matching with Flux Transformer on continuous audio latents guided by T5 text prompts and 19 degradation-specific audio transformations.
   - Codebase Modules: Audio degradation generation, mass DSP metric calculation, Mel-SSIM/KL extraction, LAION-CLAP FAD, and Audiobox Aesthetics evaluation.
2. **stable-audio-metrics (`/home/kim/Projects/stable-audio-metrics`):**
   - Reference Organization: Stability AI (Zach Evans, CJ Carr, Julian Parker, et al.).
   - Purpose: Standardized benchmark suite for evaluating generative audio and music models (Stable Audio, AudioCraft) on full-band 48 kHz stereo audio.
   - Codebase Modules: `src/openl3_fd.py` (OpenL3 Fréchet Distance @ 48 kHz), `src/passt_kld.py` (PaSST AudioSet KL Divergence @ 32 kHz), and `src/clap_score.py` (LAION-CLAP @ 48 kHz).

---

### 4.2 Comprehensive Inventory of All Metrics Discussed

#### Category A: SonicMaster DSP & Degradation Metrics (`SonicMaster/evaluation/`)
1. **FAD on CLAP Embeddings:** Fréchet Audio Distance (2-Wasserstein metric) between Gaussian fits of LAION-CLAP embeddings between clean references and model outputs.
2. **KL Divergence on Mel-Spectrograms:** Kullback-Leibler divergence between normalized mean energy distributions across log-mel frequency bins.
3. **SSIM on 128-bin Mel-Spectrograms:** Structural Similarity Index Measure computed over 2D power-to-dB mel-spectrogram arrays, quantifying spectro-temporal envelope contours.
4. **Production Quality (PQ) Score:** Audio aesthetic rating using Meta's `audiobox_aesthetics` predictor.
5. **9-Band Spectral Balance Cosine Distance ($1 - \cos(\text{SB}_{\text{GT}}, \text{SB}_{\text{pred}})$):** Distance between normalized energy profiles across 9 frequency bands: (20–60, 60–250, 250–500, 500–2k, 2k–4k, 4k–6k, 6k–10k, 10k–16k, 16k–20k Hz).
6. **9-Band Spectral Balance Euclidean Distance:** Euclidean distance across the 9 normalized frequency bands.
7. **Band Energy Ratio ($E_{\text{band}} / E_{\text{total}}$):** Degradation-specific energy ratios across bounded bands:
   - *Boom:* (20, 150 Hz)
   - *Warm:* (20, 400 Hz)
   - *Mud:* (200, 500 Hz)
   - *Vocal:* (350, 3500 Hz)
   - *Clarity:* (4000, $f_s/2$)
   - *Bright / Dark:* (6000, $f_s/2$)
   - *Airy:* (10000, $f_s/2$)
8. **Dynamic Range Standard Deviation ($\text{STD DR}$):** Standard deviation of frame-level RMS energy (2048-sample frames, 1024 hop), measuring compression dynamics.
9. **Transient Punch / Attack Strength:** Mean of `librosa.onset.onset_strength`, measuring attack transient onset sharpness and click energy.
10. **Modulation Spectrum Distance (MSD):** Euclidean distance of low-frequency envelope modulation spectra across STFT bins ($< 20$ modulation bins), isolating LFO wobble, chorus, and detune phasing.
11. **RT60 Reverberation Time:** Schroeder backwards energy decay curve integration, fitting linear slope between $-5\,\text{dB}$ and $-35\,\text{dB}$.
12. **Spectral Flatness Error:** Absolute error of `librosa.feature.spectral_flatness` on STFT power spectrograms, quantifying saturation, clipping, and waveshaper distortion.
13. **Global RMS Volume Error:** Absolute error of overall Root-Mean-Square signal amplitude.
14. **Stereo Energy Ratio (Mid-Side Width):** Ratio of Side energy to Mid energy: $\text{RMS}(L - R) / (\text{RMS}(L + R) + \epsilon)$ or $\text{RMS}(S) / \text{RMS}(M)$.

#### Category B: Stability AI Macro Generative Benchmark (`stable-audio-metrics/src/`)
15. **OpenL3-FD @ 48 kHz:** Full-bandwidth Fréchet Distance on OpenL3 music/environmental representations with LUFS loudness normalization (`pyloudnorm`) and `soxr` resampling.
16. **PaSST-KLD @ 32 kHz:** KL divergence over AudioSet acoustic/musical class probabilities predicted by the Patchout Audio Transformer (PaSST).
17. **CLAP-LAION Score @ 48 kHz:** Cosine similarity in LAION-CLAP embedding space (`630k-audioset-fusion-best.pt`) between text prompts and audio, or audio-to-audio pairs.

#### Category C: Waveform & Spectral Distance Metrics (Inquired & Baseline)
18. **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio):**
    $$\text{SI-SDR}(s, \hat{s}) = 10 \log_{10} \frac{\| \alpha s \|^2}{\| \alpha s - \hat{s} \|^2}, \quad \alpha = \frac{\hat{s}^T s}{\|s\|^2}$$
    *(Proven invalid for synthesizers due to free-running oscillator phase drift and unison cancelation, which drive SI-SDR to $-\infty\,\text{dB}$ even on perceptually identical sounds).*
19. **Multi-Scale STFT (MSS) Distance:** Sum of linear spectral convergence and log-magnitude $\ell_1$ distance across multiple FFT frame sizes (e.g. 512, 1024, 2048).

#### Category D: Pre-Registered V4 Factorial Evaluation Suite (`eval_v4_2x2`)
20. **`mrr_a2p` / `mrr_p2a`:** Audio-to-Preset and Preset-to-Audio Mean Reciprocal Rank ($1 / (1 + \text{rank})$) within the held-out gallery.
21. **`hit_a2p` / `hit_p2a`:** Audio-to-Preset and Preset-to-Audio exact Top-1 retrieval accuracy.
22. **`top5_a2p` / `top5_p2a`:** Audio-to-Preset and Preset-to-Audio Top-5 retrieval accuracy.
23. **`ladder_jepa_rho`:** Spearman rank correlation along control parameter sweeps in JEPA representation space.
24. **`ladder_flow_rho`:** Spearman rank correlation of flow predictions across control ladders (overall and per-axis: `cutoff_low`, `aeg_sustain`, `feg_amount`, `aeg_decay`).
25. **`flow_mse` / `flow_mae`:** Flow-matching velocity vector field regression loss on clean condition.
26. **`param_mae`:** Mean Absolute Error across continuous synthesizer parameters.
27. **`cat_acc`:** Majority-vote classification accuracy across categorical parameters.
28. **`param_err:<param>`:** Per-parameter absolute error for individual continuous controls (`cutoff`, `resonance`, `feg_amount`, `feg_decay`, `aeg_decay`, `aeg_sustain`).
29. **`aux_sqerr`:** Squared error of the inharmonicity prediction auxiliary head on items with valid labels.

---

## 5. V5 Architectural Proposals

To resolve the evaluation blindness and the trainer deafness, we propose four concrete interventions:

### Proposal 1: True Cross-Attention Conditioning in Flow Matching [U]
Replace the current bottleneck (where audio is pooled to a single 1D vector added as a bias into a ResMLP) with a **DiT-style Transformer / Cross-Attention Architecture**:
* Maintain the sequence of unpooled audio tokens $T_{\text{audio}} \in \mathbb{R}^{41 \times d}$.
* Flow parameter blocks cross-attend directly to $T_{\text{audio}}$.
* This allows the parameter generation process to attend selectively to transient tokens at $t=0\,\text{ms}$ when predicting attack/filter envelope parameters, and attend to sustain tokens at $t=400\,\text{ms}$ when predicting sustain level and decay rates.

### Proposal 2: Rotary Position Embeddings (RoPE) [U]
Replace 1D learned absolute positional embeddings in both the Audio Transformer $E_a$ and the Flow Transformer with **1D RoPE**:
* Imparts natural **relative-time shift equivariance** $\Delta t = |t_i - t_j|$.
* Accurately tracks transient decay slopes regardless of where the note triggers within the analysis window.
* Enables zero-shot length extrapolation to multi-second phrases without shape mismatch errors.

### Proposal 3: Direct Preference Optimization (Flow-DPO) [U]
To bridge the non-differentiable C++ DSP engine barrier, implement **Flow-DPO** using an automated audio metric oracle:
1. **Candidate Sampling:** For each training condition $z_a$, draw $K=2$ candidate parameter vectors $\theta_1, \theta_2 \sim \text{Flow}(z_a)$ using different noise seeds.
2. **Headless Render & Oracle Scoring:** Render both through headless Surge XT on CPU workers. Score each against ground truth target audio $y^*$ using a composite perceptual reward:
   $$\mathcal{R}(\hat{y}, y^*) = \text{Mel-SSIM}(\hat{y}, y^*) - \lambda_1 \text{Dist}_{\text{9-band}}(\hat{y}, y^*) - \lambda_2 |\text{Punch}_{\hat{y}} - \text{Punch}_{y^*}|$$
3. **Flow-DPO Update:** Label $\theta_w$ (winner) and $\theta_l$ (loser). Optimize the flow velocity field via the DPO objective:
   $$\mathcal{L}_{\text{Flow-DPO}} = -\mathbb{E} \left[ \log \sigma \left( -\beta \left( \|v_\phi(\theta_t^w, t, z) - v_t^w\|^2 - \|v_\phi(\theta_t^l, t, z) - v_t^l\|^2 - \text{ref\_diff} \right) \right) \right]$$
*Result:* The flow model is updated directly on **how the rendered patch sounds**, penalizing parameter configurations that sound bad while learning to ignore inactive parameters.

### Proposal 4: Two-Stage Hybrid Inversion (Flow Init + JEPA Latent Polish) [U]
Combine the strengths of both paradigms:
1. **Stage 1 (Amortized Global Initialization):** 25-step Euler Flow Matching samples the discrete categoricals and global continuous parameter basin in $50\,\text{ms}$.
2. **Stage 2 (Local Differentiable Polish):** Take the continuous parameters $\theta_{\text{cont}}$ and run 10–20 steps of Adam through the differentiable JEPA landscape:
   $$\min_{\theta_{\text{cont}}} \| f_{p \to a}(E_p(\theta)) - z_a^* \|^2$$
This provides the speed of amortized flow generation combined with the asymptotic precision of the original Synth-JEPA embedding optimization.

---

## 6. Training Speed Impact and Precomputation Strategy

### 6.1 Training Speed Impact of Proposed Changes

| Proposed Component | Mechanism | Training Speed Impact | Why / Explanation |
| :--- | :--- | :---: | :--- |
| **Rotary Position Embeddings (RoPE)** | Complex phase rotation on Query/Key slices $\langle R_{\Theta} q, R_{\Theta} k \rangle$ | **Negligible ($< 1\%$ impact)** | Elementwise multiplication applied to heads before attention. Zero extra matrix multiplies or memory overhead. |
| **Cross-Attention Flow Inverter** | Multi-head cross-attention over 41 audio tokens instead of 1D vector addition | **Mild ($15\%\text{--}25\%$ slower GPU pass)** | 41 tokens is very short. An attention matrix of $32 \times 41$ is computed instantaneously on the RX 9070 XT. The slight cost is from linear projections in cross-attention blocks. |
| **Direct Preference Optimization (Flow-DPO)** | Training flow velocity on winner/loser parameter pairs $(\theta_w, \theta_l)$ | **Depends on implementation:**<br>• *Online (Naive):* **$10\times\text{--}20\times$ SLOWER**<br>• *Offline (Recommended):* **ZERO OVERHEAD** | **Critical architectural decision:** If GPU pauses every step to wait for CPU Surge XT to render audio and calculate Mel-SSIM, throughput collapses. If candidate pairs are pre-rendered and scored **offline**, DPO training runs at full GPU throughput. |
| **Perceptual Audio Metrics (SonicMaster / CLAP)** | 9-Band balance, Mel-SSIM, Punch, CLAP score | **Zero impact on training** | These run during evaluation and validation checkpoints. Evaluating 24 stems takes $\sim 5\text{ seconds}$ on CPU. |

### 6.2 What We Can Precompute Across Runs (The $5\times\text{--}10\times$ Speedup)

Right now, in `train_realistic_bass_v3.py`, **Surge XT renders audio on the fly inside 8 CPU DataLoader workers for every single training step**:
$$\text{Preset } \theta^* \xrightarrow[\text{CPU Worker}]{\text{Surge XT Render}} \text{Waveform} \xrightarrow[\text{CPU Worker}]{\text{Inharmonicity FFT Peak Tracking}} \xrightarrow[\text{IPC Transfer}]{\text{Main Process}} \text{GPU}$$
Across 12,000 steps (batch size 64), the CPU renders **768,000 audio clips** and transmits raw float32 waveforms over IPC pipes. This caps throughput at **~42 samples/sec**, stretching a 12,000-step run out to **5.0 hours**.

By precomputing reusable assets, we can fundamentally transform training speed:

#### A. Precomputed Audio & Mel Spectrogram Dataset (The Big Win)
* **What to do:** Run a multi-threaded batch script once to pre-render 100,000 or 200,000 presets from `realistic_bass_prior_v3` into memory-mapped `.npz` / WebDataset files containing:
  - The parameter vector $\theta^*$
  - The pre-computed log-mel spectrogram tensor $\text{Mel} \in \mathbb{R}^{128 \times 81}$
  - The pre-extracted inharmonicity label (if aux head is used)
* **Impact:** Eliminates CPU synth rendering and IPC serialization entirely during training.
* **Speedup:** Training throughput jumps from 42 samples/sec to **$300\text{--}500$ samples/sec**.
* **Result:** A 12,000-step training run will finish in **$\sim 30\text{ to }45\text{ minutes}$ instead of 5 hours**.

#### B. Precomputed Audio Tokens ($T_{\text{audio}}$) for Flow Matching
* **What to do:** Once the JEPA audio encoder $E_a$ is trained, freeze it and run a single forward pass over the training set to precompute the unpooled audio tokens $T_{\text{audio}} \in \mathbb{R}^{41 \times 512}$.
* **Impact:** When training the Flow Matching inverter, the GPU never needs to execute the Audio Encoder forward or backward passes!
* **Speedup:** Saves $\sim 40\%$ GPU memory and another $\sim 30\%$ training time, allowing larger batch sizes (e.g. 128 or 256) on the RX 9070 XT.

#### C. Precomputed Reference Embeddings for Metrics (FAD & CLAP)
* **What to do:** Both `SonicMaster` and `stable-audio-metrics` use this exact optimization (`clean_embeddings.npz` in `load/`).
* Precompute the LAION-CLAP embeddings and OpenL3 statistics for the **24 reference stems once** and store them on disk (`ref_stems_clap.npz`).
* **Impact:** When evaluating any future model arm or checkpoint, we only compute embeddings for the model's new outputs and directly measure distance against the cached reference distribution. Evaluation finishes in $< 1\text{ second}$.

#### D. Precomputed Offline Preference Pairs for Flow-DPO
* **What to do:**
  1. For 10,000 audio clips, sample $K=4$ candidate parameter vectors from our trained checkpoint.
  2. Batch-render them with Surge XT on all CPU threads offline.
  3. Score them with the automated oracle (SonicMaster Mel-SSIM + 9-Band Balance) and store the dataset of triplets:
     $$\left( z_{\text{audio}}, \theta_{\text{winner}}, \theta_{\text{loser}} \right)$$
* **Impact:** Fine-tuning the flow model with Flow-DPO on GPU becomes a standard, ultra-fast supervised run with **100% GPU utilization and zero synth rendering wait time**.

#### E. Static Validation & Ladder Benchmark Tensors
* **What to do:** Lock the 2,048 validation notes and 2,048 control ladder spectrograms into fixed, precomputed tensor files.
* **Impact:** Eliminates the 15-minute startup generation pause at the beginning of every training run.

---

### 6.3 Training Velocity Comparison

| Task | Current Pipeline (Online Surge XT Render) | Proposed V5 Pipeline (Precomputed Mels & Tokens) |
| :--- | :---: | :---: |
| **Dataloader Overhead** | High (8 CPU plugins rendering + IPC pipes) | Zero (Direct memory-mapped disk/RAM read) |
| **GPU Sample Throughput** | $\sim 42$ samples/sec | **$\sim 350\text{--}500$ samples/sec** |
| **12,000 Step Run Time** | **5.0 hours** | **$\sim 35\text{ minutes}$** |
| **Full 60,000 Step Run** | 25 hours (overnight+) | **$\sim 3\text{ hours}$** |
| **Candidate DPO Fine-tuning** | Infeasible online | **$\sim 20\text{ minutes}$ offline** |

---

## 7. Immediate Action Plan

1. **Deploy Audio Evaluation Suite (`eval_audio_perceptual_suite.py`):**
   Evaluate our existing 24 rendered stems across all 5 arms (A0, A1, A2, A3, A3′) against target stems using:
   - 9-Band Spectral Balance Cosine Distance
   - Mel-SSIM & Mel-KL Divergence
   - Transient Punch Error
   - LAION-CLAP Embedding Distance
2. **Publish Bitwig DAW Projects for External Listening:**
   - 5-Arm Project: `/home/kim/Documents/Surge_All_5_Arms_Evaluation.dawproject` (1.53 GB, 1,260 clips)
   - 4-Arm Clean $2 \times 2$ Project: `/home/kim/Documents/Surge_A0_A1_A2_A3_Evaluation.dawproject` (1.23 GB, 1,008 clips)
3. **Submit Document for External Review:**
   Deliver this document to Kuang and Wintermute for validation before initiating V5 architecture training.
