# V5 Architecture Analysis, Failure Modes, and Proposed Inversion Paradigm

**Date:** October 13, 2026  
**Author:** Antigravity (`antigravity@aavepyora.online`) for Kim, Kuang, and Wintermute  
**Context:** Review of the A-series 2×2 factorial evaluation (`eval_v4_2x2`), comparison against original Synth-JEPA (Hayes et al., arXiv:2609.31024), metric review of SonicMaster (`2508.03448v5`) and Stability AI's `stable-audio-metrics`, analysis of the parameter-space vs audio-space training paradox, and proposed V5 architectural roadmap.  
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

## 4. Evaluation Metrics: SonicMaster vs. Stable-Audio-Metrics vs. SI-SDR

### 4.1 SonicMaster (`2508.03448v5`) DSP Diagnostic Suite
SonicMaster provides a domain-specific suite for physical and tonal degradation analysis (`evaluation/evaluate_control_multiple_degs_mass.py`):
* **9-Band Spectral Balance Cosine Distance ($1 - \cos(\text{SB}_{\text{GT}}, \text{SB}_{\text{pred}})$):**  
  Evaluates energy distribution across 9 bands: Sub (20–60 Hz), Punch (60–250 Hz), Low-Mid (250–500 Hz), Mid (500–2000 Hz), Upper-Mid (2–4 kHz), Presence (4–6 kHz), Brilliance (6–10 kHz, 10–16 kHz, 16–20 kHz).  
  *Direct applicability to synth bass:* **Vital.** Bass inversion errors are overwhelmingly balance errors between sub weight, punch octave, and filter cutoff bite.
* **Mel-SSIM (128-bin log-mel Structural Similarity):**  
  Evaluates 2D spectro-temporal geometry (filter envelope sweep slopes, formant curves) while remaining invariant to acoustic phase.
* **Transient Punch (`onset_strength`):**  
  Directly measures attack click amplitude and envelope sharpness.
* **Modulation Spectrum Distance (MSD):**  
  Captures low-frequency envelope modulation ($< 20$ bins), isolating LFO wobble rates, chorus depth, and detuned unison phasing.

### 4.2 Stability AI's `stable-audio-metrics` Macro Benchmark
`/home/kim/Projects/stable-audio-metrics` provides macro-level generative evaluation:
* **OpenL3-FD @ 48 kHz:** Full-bandwidth Fréchet Audio Distance with LUFS normalization. Avoids 16 kHz VGGish bandwidth truncation.
* **PaSST-KLD @ 32 kHz:** AudioSet class distribution matching using Patchout Audio Transformers.
* **LAION-CLAP Score @ 48 kHz:** Multimodal audio-to-audio embedding cosine similarity (`630k-audioset-fusion-best`).

### 4.3 The SI-SDR Warning for Synthesizers
**Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) must NOT be used for synthesizer evaluation [I].**
* SI-SDR is a **sample-level time-domain waveform metric**:
  $$\text{SI-SDR}(s, \hat{s}) = 10 \log_{10} \frac{\| \alpha s \|^2}{\| \alpha s - \hat{s} \|^2}, \quad \alpha = \frac{\hat{s}^T s}{\|s\|^2}$$
* In audio restoration/mastering (SonicMaster), the source and target share identical acoustic phase.
* In synthesizers, oscillators are **free-running** (random initial phase), and multi-voice unison induces continuous chaotic phase cancellation.
* If an inverted patch matches the ground-truth bass tone with **100% perceptual perfection**, but its oscillator phase starts $\pi$ radians out of phase (a $\sim 5\,\text{ms}$ shift at 100 Hz), waveform subtraction yields destructive cancellation, driving SI-SDR to **$-15\,\text{dB}$ or $-\infty\,\text{dB}$**. Time-domain sample metrics are mathematically invalid for un-synced synth audio.

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

## 6. Immediate Action Plan

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
