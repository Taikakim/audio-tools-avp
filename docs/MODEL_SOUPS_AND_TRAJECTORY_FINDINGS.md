# Onset Density Adapter: Weight-Space Trajectories and Model Soups Evaluation Report

This document synthesizes our empirical evaluations of checkpoint averaging (**Model Soups**) and weight-space **Trajectory Statistics** for the onset density steering adapter trained on the `lr 2e-5` run. 

---

## Executive Summary

1. **The Core Problem**: The rectified-flow training loss stays flat throughout training, yet the adapter's steering authority peaks around Epoch 24 and then declines, ending in late-epoch collapse (50% collapse rate by Epoch 40). The flat loss indicates the optimizer is drifting in an under-determined landscape rather than classic overfitting.
2. **The Solution**: Weight-space trajectory analysis shows the run has low path efficiency (**0.698**), confirming the model is wandering in a flat basin. **Post-hoc model soups (weight-averaging) pull the model back to the stable center of this basin, rescuing the late-epoch collapse and yielding superior steering authority and timbral quality.**
3. **The Winners**: 
   * **`soup_exppeak25_ep10-40`** (exponential peak centered at Epoch 25) is the **overall winner**. It achieves the lowest collapse rate (**18.5%**) and maintains high steering correlation ($r \approx 0.90$) across all gains, alongside the highest steering responsiveness (slope = **0.421**) at Gain 1.0.
   * **`soup_exppeak20_ep10-40`** achieves the **highest timbral quality** ($PQ = 7.935$, $CE = 6.493$), matching the mathematical centroid of the training run.

---

## Key Findings & Analysis

### 1. The Mathematical Centroid Matches Empirical Peak Quality
Our trajectory statistics compute the mathematical **centroid (soup-center)** of the entire training run to be exactly **Epoch 20.0** (step 108,000). 
Empirically, the model that places its exponential peaking weight at Epoch 20 (**`soup_exppeak20`**) achieves the absolute highest overall timbral quality (Acoustic Quality $PQ = 7.935$, Musicality $CE = 6.493$). This indicates that the geometric mean of the optimization path corresponds directly to the point of maximum timbral stability.

### 2. The Souping Regularization Effect (Soups vs. Single Checkpoints)
Averaging checkpoint weights smooths out local instabilities, resulting in models that outperform their individual checkpoint counterparts:
* **Epoch 25 Comparison**:
  * **`single_ep25_step135k`**: $PQ = 7.876$, $CE = 6.394$, Collapse Rate = **24.1%** (4/18 at Gain 2.0, 8/18 at Gain 3.0).
  * **`soup_exppeak25_ep10-40`**: $PQ = 7.854$, $CE = 6.420$, Collapse Rate = **18.5%** (2/18 at Gain 2.0, 7/18 at Gain 3.0).
  * The averaged model (`soup_exppeak25`) achieves **better steering correlation** at Gain 2.0 ($r = 0.915$ vs. $0.889$) and Gain 3.0 ($r = 0.891$ vs. $0.860$), while reducing the collapse rate significantly.
* **Epoch 40 Rescue**:
  * **`single_ep40_step216k`** collapses massively (**50% overall collapse**, jumping to **72.2% at Gain 3.0**).
  * Averaging it with early epochs in **`soup_sine_ep10-40`** reduces the collapse rate to **22.2%** while restoring musicality ($CE = 6.433$ vs. $5.744$).

### 3. Early Epochs act as a Crucial Regularizer
Omitting the early epochs (10–19) in **`soup_ep20_ep40`** led to a collapse rate of **27.8%** and caused the steering correlation at Gain 2.0 to drop to a poor **$r = 0.208$**. This proves that early-run checkpoints—despite having lower individual steering authority—act as an essential stabilizer. A wide window starting from Epoch 10 is necessary to anchor the averaged model to stable latent regions.

### 4. Path Efficiency and Optimizer Drift
The path efficiency (net displacement / total path length) for the `lr2e5` run is **0.698**. This is lower than the higher learning rate runs (`lr8e5` is 0.761; `lr1e4` is 0.825). A lower path efficiency indicates a highly curved, oscillating trajectory in a flat valley. The optimizer keeps moving (velocity stays high at $\approx 0.65$ per epoch late in the run), but it is drifting. Weight-averaging (or EMA) acts as a damping filter that stabilizes this drift.

### 5. The Collapse-Dropout Profile (The "Chaos Factor")
Our deep dive into the 174 collapse events across all 12 models revealed major sensitivities:
* **Guidance Gain Penalty**: Guidance gain acts as an instability multiplier. The collapse rate across all models is **9.9% at Gain 1.0**, doubles to **20.8% at Gain 2.0**, and explodes to **45.1% at Gain 3.0**.
* **Seed Dependence**: Initial noise layouts play a massive role. Seed `1234` suffered a **34.7% collapse rate** across all models, compared to just **15.7%** for Seed `777` (a 2.2x difference).
* **Prompt Phrasing Regularization**: The short prompt *\"psytrance, 140 bpm\"* collapsed **35.2%** of the time overall (and **68.1%** at Gain 3.0). Meanwhile, the more descriptive *\"aggressive upbeat goa trance\"* was much more stable, collapsing only **16.2%** of the time overall, and **0%** of the time at Gain 1.0. Detailed text conditionings help restrict the model to stable generation manifolds.
* **Target Density Strain**: Requesting high target densities stretches the model. Target density **9.0** resulted in a **43.1% collapse rate**, compared to just **14.8%** for target 7.0.

---

## Actionable Recommendations for Inference & Training

### 1. Optimal Inference Configurations
* For general use, select **`soup_exppeak25_ep10-40`** or **`soup_sine_ep10-40`**.
* Keep steering weight at **Gain 1.0 or 2.0** for stable, high-quality audio.

### 2. High-Gain Rejection Sampling (Best-of-N)
If high steering authority is required (Gain 3.0), implement a **rejection sampling filter** at inference time:
1. Generate 3 or 4 candidates at Gain 3.0.
2. Discard any candidate with an Audiobox timbral score below the collapse threshold ($PQ < 6.0$ or $CE < 6.0$).
3. The successful high-gain clips will have intense steering responsiveness while preserving pristine, aesthetic quality ($PQ \approx 7.97$ and $CE \approx 6.68$).

### 3. Training: EMA is Required
Because the optimizer drifts in a flat loss landscape, running individual late checkpoints directly is counterproductive. The recipe lesson for training future adapters is to **employ Exponential Moving Average (EMA) or checkpoint weight-averaging during training**, rather than relying on a simple learning rate decay.
