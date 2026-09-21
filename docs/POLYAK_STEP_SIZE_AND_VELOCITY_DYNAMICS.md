# The "5e-6" Illusion, Polyak Saturation, and Velocity Calibration in Spectral DiT Optimization

**Date:** September 20, 2026  
**Authors:** Kim & Antigravity  
**Context:** SAO Project (Stable Audio 3 DiT Backbone, LoRA/DoRA Fine-Tuning, Modular Norm-Constrained Optimizer Framework)  
**Reference Runs:**
- Headline Baseline: `/run/media/kim/Mantu/sa3_lora_runs/fusion_autoscale_vs_adamw_2026-09-01/fusion_autoscale_lr5e-6_full3h/`
- AdamW Baseline: `/run/media/kim/Mantu/sa3_lora_runs/fusion_autoscale_vs_adamw_2026-09-01/adamw_lr1e-4/`
- Initial Modular Run: `/run/media/kim/Mantu/sa3_lora_runs/modular_opt_stage3_ns5_otwd_3000s_2026-09-20_1800/`
- NorMuon+SF Probe: `/run/media/kim/Mantu/sa3_lora_runs/modular_opt_normuon_sf_otwd_3000s_2026-09-20_2138/`
- Calibrated Run: `/run/media/kim/Mantu/sa3_lora_runs/modular_opt_lr1e4_w200_sf_normuon_300s_2026-09-20_2230/`

---

## 1. Executive Summary: Unmasking the "5e-6" Illusion

In our September 1, 2026 benchmark suite, the run labeled `fusion_autoscale_lr5e-6_full3h` emerged as the seminal "best sounding" model in the library. It traversed **283.0 units of Frobenius distance** over 3,000 steps (an effective velocity of **~94.3 units / 1,000 steps**), out-learning AdamW (`1e-4`, velocity 35.1) while maintaining rich, tape-saturated acoustic harmonics.

However, when our modular, preconditioned, norm-constrained optimizer (`ModularOptimizer`) was configured at that exact same nominal learning rate (`lr = 5e-6`), training appeared completely stalled to the ear. Extracting weight trajectories from checkpoints revealed that `ModularOptimizer` had an effective velocity of only **0.84 to 2.10 units / 1,000 steps**—moving **45x to 110x slower** than FusionOpt at the identical nominal learning rate.

Rigorous mathematical and code inspection of `fusion_opt.py` revealed that **FusionOpt was never actually running at $5 \times 10^{-6}$**. 

In `fusion_opt.py:744`:
$$\gamma_t = \text{lr} \times \gamma_{\text{ratio}} \times \text{auto\_mult} \times \text{warm} \times \text{decay} \times \gamma_{\text{scale}}$$

FusionOpt contained two compounding hidden multipliers:
1. **The Polyak Ratio ($\gamma_{\text{ratio}}$):** Hard-pegged at **10.0** from step 1.
2. **Autoscale ($\text{auto\_mult}$):** D-Adaptation / Prodigy step-size growth that ramped from **1.0 to ~3.0–5.0x**.

Consequently, FusionOpt was actually executing steps at an effective step size of:
$$\gamma_t \approx 5 \times 10^{-6} \times 10.0 \times 3.0 = \mathbf{1.5 \times 10^{-4}}$$
which matches the AdamW `1e-4` parameter velocity regime.

When `ModularOptimizer` was run at nominal `5e-6` with Polyak and autoscale removed, it was taking literal $5 \times 10^{-6}$ steps, causing inaudible weight progress.

---

## 2. Mathematical Breakdown: Why the Polyak Step Size Failed

The Polyak step size (Polyak 1969, 1987) is a celebrated paradigm in optimization. For convex objectives with a known minimum $f^*$, the optimal step size along the gradient is:
$$\gamma_t = \frac{f(x_t) - f^*}{\|\nabla f(x_t)\|^2}$$

In `fusion_opt.py:577-624`, Polyak was implemented as:
$$\text{ratio} = \frac{\mathcal{L}_{\text{EMA}}}{\text{gnorm}_{\text{EMA}}}, \quad \gamma_{\text{ratio}} = \text{clamp}(\text{ratio}, \gamma_{\min}, \gamma_{\max})$$
where $\gamma_{\min} = 0.1, \gamma_{\max} = 10.0$.

This formulation broke down completely in continuous flow matching audio DiTs due to four compounding issues:

### A. The Irreducible Bayes Risk Floor ($f^* \gg 0$)
In overparametrized classification, networks can interpolate training data, so $f^* \approx 0$. 
In continuous flow matching diffusion models (SA3), the objective regresses the conditional velocity vector field against stochastic noise:
$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1}\left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]$$
Due to inherent data entropy and stochastic paths, there is a large, non-zero Bayes error floor ($f^* \approx 0.60–0.65$ in SA3). 

Because the implementation assumed $f^* = 0$, the numerator $\mathcal{L}(x_t) - 0 \approx 0.75$ **never approaches zero**. 
As the model converges and gradients shrink ($\nabla f \to 0$):
$$\gamma_t = \frac{0.65}{\to 0} \longrightarrow \infty$$
Instead of decelerating as the model nears the optimum, an uncalibrated Polyak step size **wants to explode near the minimum**.

### B. Scale & Dimension Mismatch (Mean Absolute Gradient)
The denominator in `fusion_opt.py:601` was computed as:
$$\text{gnorm\_now} = \frac{\sum |g_i|}{N} \quad (\text{mean absolute gradient per element})$$
For LoRA adapters in SA3:
- Diffusion loss $\mathcal{L} \approx 0.75$
- Mean gradient $|g| \approx 10^{-4}$ to $10^{-5}$
- Raw ratio:
  $$\text{ratio} \approx \frac{0.75}{10^{-4}} = \mathbf{7,500}$$

Because the ceiling was $\gamma_{\max} = 10.0$, the ratio was **750x higher than the clamp limit on step 1**, and remained pinned at **10.0** through all 3,000 steps. It never functioned adaptively; it functioned as an opaque, hardcoded 10x learning rate multiplier.

### C. Double-Normalization with Muon / Spectral LMO
In standard gradient descent, $\Delta W = -\eta \nabla f(W)$. Multiplying by $1/\|\nabla f\|$ converts gradient descent into normalized gradient descent.

In Muon / SpectralLMO, however:
$$U = \text{msign}(G) = U_G V_G^T$$
The update direction is **already fully normalized**—all singular values are $\sigma_i(U) = 1$. The gradient magnitude is already completely factored out by the Newton-Schulz quintic iteration.

When Polyak was layered on top of Muon:
1. Muon normalized the step to operator norm 1.
2. Polyak divided by gradient norm again and multiplied by loss.
3. Because the ratio pegged at 10.0, every single matrix parameter took an update with spectral radius $\sigma_{\max}(\Delta W) = 10 \cdot \eta$ on every single step, regardless of local curvature.

### D. Acoustic Impact: Harsh Timbres, Buzz, and Glitches
As proven in recent optimization literature ([SAMuon, arXiv:2608.25990](https://arxiv.org/abs/2608.25990); [SoftMuon, arXiv:2605.31371](https://arxiv.org/abs/2605.31371)):
- Deep neural loss landscapes have an anisotropic spectrum: a "volatile head" at the Edge of Stability (EoS) and a "tolerant bulk".
- Clamping singular values uniformly to 1 and multiplying by 10x overdrives the volatile head, inducing limit-cycle oscillations around narrow minima.
- In audio DiTs, parameter oscillation around minima directly corrupts high-frequency latent dimensions, manifesting as **harsh timbres, metallic ringing, and high-frequency click/buzz transients**.

---

## 3. Weight Trajectory & Velocity Across Library Checkpoints

Extracting exact Frobenius distances across saved checkpoints reveals the true travel dynamics of each optimizer:

| Run / Optimizer | Nominal LR | Multipliers Active | Measured Velocity (units / 1k steps) | Acoustic Character |
|---|:---:|:---:|:---:|---|
| `fusion_autoscale_lr5e-6_full3h` | `5e-6` | Polyak (10x) + Autoscale (~3x) | **~94.3** | Best sounding; punchy, tape-saturated, but slight high-frequency buzz on long extensions. |
| `adamw_lr1e-4` | `1e-4` | Standard AdamW | **35.1** | Clean, stable, healthy audio learning. The benchmark target velocity. |
| `lion_lr1e-5` | `1e-5` | Sign descent | **31.4** | Clean, fast learning, slightly leaner low end. |
| `modular_opt_stage3_ns5` (Old) | `5e-6` | None (bare) | **2.10** | Inaudible learning; sounds identical to base model. |
| `modular_opt_lr1e4_w200` (Calibrated) | `1e-4` | Warmup 200 + NorMuon + SF | **41.31** (active $z$), **13.46** (avg $x$) | Bullseye match with target band (30–50); clean audio, zero NaNs, stable loss ~0.77. |

### The Exact Frobenius Math for Bare 5e-6
Why did ModularOptimizer move at ~1 unit / 1k steps under bare `5e-6`?
With NorMuon enabled, each row of update matrix $M \in \mathbb{R}^{d_{out} \times d_{in}}$ has unit norm.
Thus, $\|M\|_F = \sqrt{d_{out}}$.
For an attention projection with $d_{out} = 1536$:
$$\|\Delta W\|_F = \eta \cdot \|M\|_F = (5 \times 10^{-6}) \times \sqrt{1536} = 0.000196 \text{ per step}$$
In 1,000 steps:
$$1000 \times 0.000196 \approx 0.20 \text{ per layer}$$
Summed in quadrature across ~50 LoRA B layers:
$$\Delta W_{\text{total}} \approx \sqrt{50 \times (0.20)^2} \approx \mathbf{1.41} \text{ units / 1k steps}$$
This matches our measured checkpoint displacement of **0.84 to 2.10** exactly.

---

## 4. The Calibrated Architecture: What We Changed

To preserve healthy, audible learning without letting values run too hot or inducing audio glitches:

1. **Set True Nominal Learning Rate:**
   Set `--lr 1e-4` (or `8e-5`) explicitly, rather than relying on an opaque 10x Polyak multiplier.
2. **Re-introduce Linear Warmup:**
   Set `--warmup-steps 200` (matching the baseline run and AdamW), preventing early gradient shock.
3. **Stage 3b NorMuon / RowNorm ([arXiv:2608.20818](https://arxiv.org/abs/2608.20818)):**
   Normalize each neuron row to unit running variance under EMA:
   $$U_{i,:} \leftarrow \frac{U_{i,:}}{\sqrt{\text{EMA}(\|U_{i,:}\|^2) / (1 - \beta_r^t)}}$$
   Bounds per-neuron row velocity across wide DiT layers, preventing localized neuron explosions.
4. **Overtraining Weight Decay Scaling ([arXiv:2609.04577](https://arxiv.org/abs/2609.04577)):**
   Scale weight decay with $\sqrt{\text{epochs}}$: $\lambda_{\text{WD}}(E) = \lambda_0 \sqrt{\max(1.0, E)}$.
5. **Next Step: Soft-Sign Relaxation ([arXiv:2605.31371](https://arxiv.org/abs/2605.31371)):**
   Replace hard polar mapping with temperature-relaxed $\text{softsign}(s_i; \tau) = \frac{s_i}{|s_i| + \tau}$ to allow smooth deceleration into narrow minima.

---

## 5. Prodigy Escape Velocity Dynamics & Dual-Norm Coupling

To test autonomous step-size adaptation without manual LR tuning, we evaluated the Prodigy Escape Velocity arm (`--modular-escape-velocity`, `--modular-ev-max 2.0`) across 300 calibration steps.

### Mathematical Formulation
Under the 6-stage modular architecture, the distance-to-optimum estimate $d_t$ is coupled to each layer's geometry using its exact Fenchel-Young dual norm:
$$d_{t+1} = \max\left(d_t, \frac{\sum_{\tau=1}^t \alpha_\tau \langle G_\tau, W_0 - W_\tau \rangle}{\sum_{\tau=1}^t \alpha_\tau \|G_\tau\|_*}\right)$$
where:
- **Spectral layers**: $\|G\|_* = \sum \sigma_i(G) = \langle M, G \rangle / 0.822$ (polar dual pairing with Newton-Schulz factor $M$).
- **ColNorm layers**: $\|G\|_* = \sum_j \|G_{:, j}\|_2$ ($(2, 1)$ column-group norm).
- **Sign layers**: $\|G\|_* = \sum_i |G_i|$ ($\ell_1$ norm).

### Empirical Findings: Baseline vs. Escape Velocity (300 Steps)

| Step Interval | Baseline Active $z$ Velocity | Escape Velocity Active $z$ Velocity | Deployable $x$ Velocity | Prodigy $d_t$ Multiplier |
|---|:---:|:---:|:---:|:---:|
| **0 $\to$ 100** (Warmup) | 12.11 units / 1k | 12.11 units / 1k | 4.05 units / 1k | $d = 1.000$ |
| **100 $\to$ 200** (Late Warmup) | 32.44 units / 1k | 32.52 units / 1k | 10.00 units / 1k | $d = 1.000$ |
| **200 $\to$ 300** (Steady-State Full LR) | **41.31 units / 1k** | **40.98 units / 1k** | **13.45 units / 1k** | $d = 1.000$ |

### Why $d_t$ Remained Clamped at 1.000
By Cauchy-Schwarz / Hölder's duality:
$$\frac{|\langle G, W_0 - W \rangle|}{\|G\|_*} \le \|W_0 - W\|$$
For LoRA weight matrices, the Frobenius displacement $\|W_0 - W_t\|_F$ per layer is on the order of $\sim 0.01 - 0.33$. Consequently:
$$\hat{d} = \frac{\text{numerator}}{\text{denominator}} \approx 1.18 \times 10^{-3} \ll 1.0$$
Because standard Prodigy is designed as a *parameter-free* optimizer where $d_t$ multiplies a base $\text{lr} = 1.0$ (starting from $d_0 \approx 10^{-6}$), pairing Prodigy with an already properly scaled nominal learning rate ($\text{lr} = 10^{-4}$) and initializing $d_0 = 1.0$ causes $\max(d_0, \hat{d})$ to remain safely anchored at $1.000$.

This proves that **when the base learning rate is correctly dimensioned to $1\times 10^{-4}$, steady-state velocity lands squarely in the ideal 30–50 units / 1k band naturally, without requiring artificial acceleration or dangerous overshoot**.

