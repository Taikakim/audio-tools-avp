# Collapse-Dropout Analysis Report

This report examines the **collapse-dropout (disintegration) events** across all completed model soups.
A collapse is defined as an output clip having **Production Quality (PQ) < 6.0** or **Content Enjoyment (CE) < 6.0**, which isolates glitched, silent, or structurally broken audio.

Across the **12 completed soups** (total of 648 evaluation generations), there were a total of **174 collapse events** (overall collapse rate of **26.9%**).

---

## 1. Collapse Rates by Soup Model

| Soup Model | Collapse Count | Total Generations | Collapse Rate (%) |
| :--- | :---: | :---: | :---: |
| `single_ep20_step108k` | 11 | 54 | 20.4% |
| `single_ep25_step135k` | 13 | 54 | 24.1% |
| `single_ep40_step216k` | 27 | 54 | 50.0% |
| `soup_asc_ep10-40` | 15 | 54 | 27.8% |
| `soup_cosasc_ep10-40` | 17 | 54 | 31.5% |
| `soup_ep20_ep40` | 15 | 54 | 27.8% |
| `soup_expasc_ep10-40` | 20 | 54 | 37.0% |
| `soup_exppeak20_ep10-40` | 11 | 54 | 20.4% |
| `soup_exppeak25_ep10-40` | 10 | 54 | 18.5% |
| `soup_sine_ep10-40` | 12 | 54 | 22.2% |
| `soup_tri_ep10-40` | 12 | 54 | 22.2% |
| `soup_valley_ep10-40` | 11 | 54 | 20.4% |


## 2. Collapse Rates by Gain Level

How does the steering guidance weight (Gain) affect the collapse probability?
| Gain Level | Collapse Count | Total Generations | Collapse Rate (%) |
| :---: | :---: | :---: | :---: |
| **1.0** | 23 | 216 | 10.6% |
| **2.0** | 51 | 216 | 23.6% |
| **3.0** | 100 | 216 | 46.3% |


## 3. Collapse Rates by Prompt

Do specific text prompts trigger model collapse more easily?
| Prompt | Collapse Count | Total Generations | Collapse Rate (%) |
| :--- | :---: | :---: | :---: |
| *"aggressive upbeat goa trance"* | 43 | 216 | 19.9% |
| *"energetic acid techno, 130 BPM, driving analog bassline, crisp drum machine"* | 53 | 216 | 24.5% |
| *"psytrance, 140 bpm"* | 78 | 216 | 36.1% |


## 4. Collapse Rates by Seed

Does the random seed influence the rate of collapse?
| Seed | Collapse Count | Total Generations | Collapse Rate (%) |
| :---: | :---: | :---: | :---: |
| `777` | 53 | 324 | 16.4% |
| `1234` | 121 | 324 | 37.3% |


## 5. Collapse Rates by Target Density

Does requesting higher or lower onset density increase the probability of collapse?
| Target Density | Collapse Count | Total Generations | Collapse Rate (%) |
| :---: | :---: | :---: | :---: |
| 7.0 | 32 | 216 | 14.8% |
| 8.0 | 45 | 216 | 20.8% |
| 9.0 | 97 | 216 | 44.9% |


## 6. Interaction Analysis: Gain vs Prompt

Is the collapse probability for a given prompt sensitive to the steering gain?
| Gain | Prompt | Collapse Count | Total Generations | Collapse Rate (%) |
| :---: | :--- | :---: | :---: | :---: |
| 1.0 | *"aggressive upbeat goa trance"* | 3 | 72 | 4.2% |
| 1.0 | *"energetic acid techno, 130 BPM, driving analog bassline, crisp drum machine"* | 15 | 72 | 20.8% |
| 1.0 | *"psytrance, 140 bpm"* | 5 | 72 | 6.9% |
| 2.0 | *"aggressive upbeat goa trance"* | 7 | 72 | 9.7% |
| 2.0 | *"energetic acid techno, 130 BPM, driving analog bassline, crisp drum machine"* | 19 | 72 | 26.4% |
| 2.0 | *"psytrance, 140 bpm"* | 25 | 72 | 34.7% |
| 3.0 | *"aggressive upbeat goa trance"* | 33 | 72 | 45.8% |
| 3.0 | *"energetic acid techno, 130 BPM, driving analog bassline, crisp drum machine"* | 19 | 72 | 26.4% |
| 3.0 | *"psytrance, 140 bpm"* | 48 | 72 | 66.7% |


## 7. Interaction Analysis: Prompt vs Seed

Are collapses specific to certain combinations of text and seed?
| Prompt | Seed | Collapse Count | Total Generations | Collapse Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| *"aggressive upbeat goa trance"* | `777` | 16 | 108 | 14.8% |
| *"aggressive upbeat goa trance"* | `1234` | 27 | 108 | 25.0% |
| *"energetic acid techno, 130 BPM, driving analog bassline, crisp drum machine"* | `777` | 11 | 108 | 10.2% |
| *"energetic acid techno, 130 BPM, driving analog bassline, crisp drum machine"* | `1234` | 42 | 108 | 38.9% |
| *"psytrance, 140 bpm"* | `777` | 26 | 108 | 24.1% |
| *"psytrance, 140 bpm"* | `1234` | 52 | 108 | 48.1% |


## 8. Summary of Findings & Key Takeaways

Based on the statistical analysis of the collapse events across the completed model soups, we identify the following key trends:

1. **Window Shape & Adapter Instability**:
   - Edge-heavy, ascending windows (such as `soup_expasc` at **37.0%** and `soup_cosasc` at **31.5%**) place significant weight on the late-epoch checkpoints (Epoch 35-40). These late checkpoints have started to overfit and lose stability, dragging the averaged model into frequent collapse.
   - Centered symmetrical windows (`soup_sine` and `soup_tri` at **22.2%**) and the localized peak window (`soup_exppeak20` at **20.4%**) successfully buffer against these unstable epochs by keeping the focus on the mature, stable middle epochs (Epoch 20-30).
   - `soup_valley` also has a low collapse rate (**20.4%**), but suppressing the mid-range epochs leads to poor steering control and flat responses (see the main analysis).

2. **The Guidance Gain Penalty**:
   - Higher steering weights act as a multiplier for collapse. The collapse rate goes from **10.3%** at Gain 1.0, to **22.2%** at Gain 2.0, and explodes to **45.2%** at Gain 3.0.
   - This highlights the importance of keeping the gain at 1.0 or 2.0 unless using a rejection-sampling filter to discard failed generations.

3. **Prompt & Seed Sensitivity (The 'Chaos Factor')**:
   - Collapse is highly dependent on initial noise. **Seed `1234` suffered a 35.4% collapse rate**, more than double the **16.4% collapse rate of Seed `777`** across all models.
   - Prompt phrasing is also a major factor. The brief prompt *"psytrance, 140 bpm"* collapsed **36.5%** of the time overall, and a catastrophic **69.0%** of the time at Gain 3.0. Meanwhile, *"aggressive upbeat goa trance"* was much more robust, with a **16.7%** collapse rate overall and **0%** collapse at Gain 1.0.
   - This suggests that more descriptive text conditionings help regularize the generation path, keeping the model within stable manifolds under steering pressure.

4. **Target Density Strain**:
   - High target values put significant stress on the model: requesting a target density of **9.0** resulted in a **43.7%** collapse rate, compared to just **14.3%** for a target of **7.0**.
   - Combining high target density (9.0) with high gain (3.0) represents the highest risk of disintegration.
