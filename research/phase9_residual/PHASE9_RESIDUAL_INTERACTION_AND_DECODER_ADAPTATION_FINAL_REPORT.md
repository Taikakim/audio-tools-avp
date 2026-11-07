# Phase 9: Nonlinear Latent Interaction, Decoder Sensitivity, and Separation Interface Experiments
## Final Scientific Report

**Research programme:** SAME latent analysis and source manipulation  
**Evaluated dataset:** Dataset B controlled synthetic mixtures  
**Dataset size:** 1,000 mixtures, divided into 600 training, 200 validation, and 200 test examples  
**Test categories:** Kick, snare, hi-hat, percussion, bass, and synthesizer lead  
**Target gain range:** Reported as −12 to 0 dB  
**Hardware:** AMD Radeon RX 9070 XT, ROCm 7.2.4  
**Repository branch:** `research/phase9-residual-and-decoder-adaptation`  

---

### 1. Executive Summary

Phase 9 investigated four related questions:
1. Whether SAME’s non-additive interaction residual contains predictable structure.
2. Whether latent-coordinate errors have direction-dependent effects after decoding.
3. Whether explicitly modelling the target and interaction residual improves practical separation.
4. Whether waveform-domain refinement improves the output of a frozen-latent separator.

The principal findings are:
- **Oracle Interaction Structure:** A contextual model supplied with the separately encoded oracle background and target predicted useful structure in the nonlinear interaction residual. It achieved a reported test $R^2$ of 0.3911 and improved decoded waveform SDR by 0.79 dB over zero interaction correction. This establishes partial predictability under the tested oracle information contract. It does not establish a theoretical predictability ceiling.
- **Directional Decoder Sensitivity:** The SAME decoder exhibited direction-dependent sensitivity to controlled latent perturbations. Small perturbations aligned with the measured interaction residual degraded decoded SDR approximately 2.59 times faster than the tested isotropic-random direction under the reported slope calculation. This demonstrates local anisotropy under the selected perturbation protocol. It does not establish that the tested directions correspond to a formally characterized data manifold.
- **Direct vs. Structured Separation:** A direct latent background predictor outperformed a capacity-matched structured model that separately predicted the target latent and interaction residual. The direct model achieved −23.69 dB SDR, compared with −26.65 dB for the structured model. The structured model nevertheless obtained a lower latent $L_1$ loss, showing that lower error in the chosen latent-coordinate metric did not guarantee better decoded waveform fidelity.
- **Conditional Flow Matching Baseline:** The tested 1.4-million-parameter conditional flow model generated variable outputs but did not improve practical source recovery. Its reported SDR was −32.09 dB, close to the reported untouched-mixture baseline of −32.21 dB. This is a negative result for that particular model, conditioning scheme, solver, and training setup. It does not resolve the general suitability of conditional flow matching for latent separation.
- **Waveform Refinement:** A waveform residual adapter achieved −22.65 dB SDR. If its input was the output of the direct separator, this represents an improvement of approximately 1.04 dB over that practical latent baseline. The exact adapter input and no-adapter comparator must be stated explicitly in the final experiment record.
- **Decoder Gradient Path:** Code inspection identified that the relevant pretransform decode path is wrapped in `torch.no_grad()` unless `pretransform.enable_grad` is enabled. Thus, decoder-aware training requires explicitly enabling gradient propagation through the decode operation, even if decoder weights themselves remain frozen. This is an implementation requirement, not evidence that decoder adaptation will necessarily solve separation.

The overall result is therefore narrower than the original report claimed:
> SAME’s nonlinear interaction residual contains partially predictable structure, the decoder is sensitive to the direction of latent errors, and decoder-aware or waveform-domain objectives merit further investigation. However, the experiments do not establish an information-theoretic phase ceiling, irreducible conditional ambiguity, latent-manifold membership, or a proven frozen-interface bottleneck.

---

### 2. Definitions and Information Contracts

Let the frozen SAME encoder and decoder be:
$$E: x \mapsto z \quad \text{and} \quad D: z \mapsto \hat{x}$$
For an additive waveform mixture $m = b + s$, define:
$$z_m = E(b + s), \quad z_b = E(b), \quad z_s = E(s), \quad \text{and the encoded-silence reference } z_0 = E(0)$$

#### 2.1 Non-additive Interaction Residual
The interaction residual is:
$$\boxed{ r_{\mathrm{int}} = z_m - z_b - z_s + z_0 }$$
or equivalently:
$$z_m = z_b + z_s - z_0 + r_{\mathrm{int}}$$
This residual quantifies the departure of the encoder from the centered additive model:
$$E(b + s) \approx E(b) + E(s) - E(0)$$
For fixed waveforms $b$ and $s$, a deterministic encoder produces a fixed $r_{\mathrm{int}}$. However, this does not imply that $r_{\mathrm{int}}$ is uniquely recoverable from the compressed pair $(z_b, z_s)$. Nor does failure by a particular predictor prove that the remaining error is irreducible.

#### 2.2 Removal Displacement
The practical latent displacement required to recover the encoded background is:
$$r_{\mathrm{remove}} = z_m - z_b$$
Using the interaction identity:
$$r_{\mathrm{remove}} = z_s - z_0 + r_{\mathrm{int}}$$
The interaction residual and the removal displacement are therefore related but are not the same target.

#### 2.3 Oracle and Deployable Conditions
- **Oracle information:** $z_b, z_s, r_{\mathrm{int}}$. Experiments using these inputs diagnose representation geometry and approximation difficulty. They do not constitute deployable source separation.
- **Practical models:** Receive only $z_m$ plus the declared legitimate query representation $q$.

---

### 3. Measurement Qualifications

#### 3.1 SDR Definition
For true background $b$ and estimate $\hat{b}$, the scale-dependent SDR used in this programme is:
$$\mathrm{SDR}(b, \hat{b}) = 10 \log_{10} \frac{\|b\|_2^2}{\|b - \hat{b}\|_2^2 + \epsilon}$$
All signals must have equal lengths, consistent channel treatment, verified temporal alignment, consistent scaling, and no independent normalization of the reference and estimate.

The reported untouched-mixture SDR of −32.21 dB is unexpectedly low relative to the stated target-gain range and should be audited before being treated as a definitive benchmark. The report should include actual rendered target-to-background energy ratios, alignment offsets, and separate original-domain and autoencoder-domain measurements. Until that audit is complete, the Phase 9 SDR values should be treated as internally comparative results under the implemented metric pipeline.

#### 3.2 NMSE and $R^2$
A conventional uncentered NMSE is:
$$\mathrm{NMSE} = \frac{\sum_i |r_i - \hat{r}_i|_2^2}{\sum_i |r_i|_2^2 + \epsilon}$$
A conventional centered coefficient of determination is:
$$R^2 = 1 - \frac{\sum_i |r_i - \hat{r}_i|_2^2}{\sum_i |r_i - \bar{r}|_2^2 + \epsilon}$$
These quantities need not satisfy $R^2 = 1 - \mathrm{NMSE}$ because their denominators differ. The reported zero and global-mean baselines indicate that the Phase 9 calculations use different centering or normalization conventions. The formulas and aggregation level across examples, frames, and channels must therefore be published alongside the metrics.

#### 3.3 Clean Autoencoder Reconstruction
The clean reconstruction $D(E(b))$ is an oracle encoded-background reference under the existing encoder and decoder. It is not a strict theoretical upper bound on all possible decoder outputs, because another latent could theoretically reconstruct $b$ more faithfully than $E(b)$. Accordingly, this report calls it the **clean autoencoder reference**, not the absolute separation ceiling.

---

### 4. Experiment A1: Oracle Interaction-Residual Prediction

#### 4.1 Research Question
Given oracle isolated-component latents $(z_b, z_s)$, how well can models predict:
$$r_{\mathrm{int}} = z_m - z_b - z_s + z_0?$$
This experiment evaluates the approximation of interaction structure from compressed isolated-component representations. It does not test practical source separation.

#### 4.2 Evaluated Models
The tested models included:
- Zero interaction prediction.
- Global mean prediction.
- Class-conditional mean prediction.
- Bilinear interaction model: $\hat{r}_{\mathrm{int},t} = W_o \left[ (W_b z_{b,t}) \odot (W_s z_{s,t}) \right]$.
- Local temporal CNN.
- Contextual convolution-plus-attention model.

#### 4.3 Reported Results

| Model | Reported NMSE | Reported $R^2$ | Decoded SDR | Improvement over zero interaction |
| :--- | :---: | :---: | :---: | :---: |
| **Zero interaction** | 1.0000 | 0.1826 | −12.95 dB | 0.00 dB |
| **Global mean** | 1.3331 | 0.0000 | −12.96 dB | −0.01 dB |
| **Class mean** | 1.2994 | 0.0221 | −12.92 dB | +0.03 dB |
| **Bilinear model** | 0.9489 | 0.2960 | −12.45 dB | +0.50 dB |
| **Local temporal CNN** | 1.7006 | −0.1454 | −12.85 dB | +0.10 dB |
| **Contextual sequence model** | 0.8083 | 0.3911 | −12.16 dB | +0.79 dB |

#### 4.4 Defensible Finding
The contextual model captured useful interaction-residual structure not captured by the zero, mean, class-mean, bilinear, or tested local-CNN baselines. Its predicted correction improved decoded SDR by 0.79 dB relative to zero interaction correction. The bilinear model’s improvement indicates that simple multiplicative component interactions explain some useful structure, while the additional improvement from the contextual model suggests that temporal context contributes under the tested dataset and architecture.

#### 4.5 What This Experiment Does Not Establish
The experiment does not establish that:
- 39.1% is the maximum predictable fraction;
- the remaining 60.9% is irreducible;
- the remaining error is predominantly high-frequency phase ambiguity;
- the encoder has discarded all information needed to predict the remaining residual;
- the contextual model is close to the Bayes-optimal predictor;
- or the residual possesses a uniquely defined deterministic and stochastic decomposition.

The test result is a model-dependent lower bound on recoverable predictive structure, not an information-theoretic ceiling.

---

### 5. Experiment A2: Directional Decoder Sensitivity

#### 5.1 Research Question
Do equal-norm latent perturbations in different directions have different effects on decoded waveform quality?
For unit direction $u$ and perturbation magnitude $\epsilon$:
$$z' = z_b + \epsilon u \implies \hat{b}_\epsilon = D(z') \implies Q(\epsilon, u) = \mathrm{SDR}\left(b, D(z_b + \epsilon u)\right)$$

#### 5.2 Tested Directions
- Isotropic random direction
- Top principal-component direction
- Bottom principal-component direction
- Interaction-residual direction
- Removal-displacement direction

#### 5.3 Reported Local Slopes

| Direction | Reported initial slope | Ratio to random |
| :--- | :---: | :---: |
| **Isotropic random** | −0.029 dB/unit | 1.000 |
| **PCA top-1** | −0.141 dB/unit | 4.892 |
| **PCA bottom-1** | −0.087 dB/unit | 3.003 |
| **Interaction residual** | −0.075 dB/unit | 2.591 |
| **Removal displacement** | −0.025 dB/unit | 0.861 |

The reported average norms were approximately 8.448 for $r_{\mathrm{int}}$ and 28.554 for $r_{\mathrm{remove}}$.

#### 5.4 Defensible Finding
The decoder’s response to latent perturbations is anisotropic under the selected reference points, directions, normalization, and metric. This helps explain why a model with lower unweighted latent-coordinate error may nevertheless produce worse decoded audio. Latent errors should not be assumed to have uniform acoustic consequences.

#### 5.5 Limitations
This experiment does not establish that:
- interaction-residual directions lie on a formally identified data manifold;
- random directions are necessarily off-manifold;
- one local slope characterizes decoder behaviour at realistic perturbation magnitudes;
- or the tested PCA and random controls span the relevant error geometry.

The phrase “manifold aligned” should therefore be avoided unless manifold membership is operationally defined and independently tested. A more cautious interpretation is: *At matched norms, the tested interaction-residual perturbations caused less waveform degradation than the tested isotropic-random perturbations at some larger magnitudes, even though their local sensitivity was higher.*

---

### 6. Experiment A3: Residual Variation by Sound Family and Time

#### 6.1 Reported Family Ratios
The programme calculated:
$$\rho = \frac{\|r_{\mathrm{int}}\|_F}{\|z_s - z_0\|_F + \epsilon}$$

| Sound family | Reported mean non-additivity ratio |
| :--- | :---: |
| **Kick** | 30.3% |
| **Snare** | 29.0% |
| **Percussion** | 24.9% |
| **Hi-hat** | 22.9% |
| **Bass** | 21.5% |
| **Synthesizer lead** | 17.5% |

#### 6.2 Defensible Finding
Under the chosen ratio, the test dataset showed larger relative non-additivity for kick and snare examples than for sustained bass and synthesizer-lead examples. This is consistent with the hypothesis that transient-rich sounds produce stronger nonlinear encoder interactions. It does not prove that transientness is the causal factor, because sound family may also covary with duration, level, spectral bandwidth, crest factor, temporal placement, and background overlap.

#### 6.3 Temporal Spread
The reported interaction energy extended approximately two latent frames around transient onsets. The defensible interpretation is: *Under the selected alignment and aggregation procedure, interaction-related latent changes were not confined to the event’s nominal onset frame.* Calling this “encoder non-causality” requires verification of the actual receptive field, padding, and analysis alignment.

---

### 7. Experiment B: Direct versus Structured Deterministic Separation

#### 7.1 Research Question
Does explicitly estimating the target latent and interaction residual improve practical background reconstruction over direct prediction?
- **Direct model:** $\hat{z}_b = f_{\mathrm{direct}}(z_m, q)$
- **Structured model:** $(\hat{z}_s, \hat{r}_{\mathrm{int}}) = f_{\mathrm{structured}}(z_m, q) \implies \hat{z}_b = z_m - \hat{z}_s + z_0 - \hat{r}_{\mathrm{int}}$

The models were reported as capacity matched at approximately 1.25 million parameters.

#### 7.2 Reported Results

| Model | Validation latent $L_1$ | Decoded SDR | Improvement over reported mixture |
| :--- | :---: | :---: | :---: |
| **Direct background predictor** | 0.1415 | −23.69 dB | +8.52 dB |
| **Structured predictor** | 0.1006 | −26.65 dB | +5.56 dB |

#### 7.3 Defensible Finding
For the tested models, direct background prediction produced better decoded waveform fidelity than factorized target-plus-interaction prediction. The structured model achieved a lower reported latent $L_1$ loss but worse decoded SDR. This is strong evidence that the chosen latent $L_1$ metric was not sufficiently aligned with decoded waveform quality.

#### 7.4 Possible Explanations
Plausible explanations include:
- error compounding between $\hat{z}_s$ and $\hat{r}_{\mathrm{int}}$;
- different scaling of the two training targets;
- poor identifiability of the internal factorization;
- decoder sensitivity to specific combined error directions;
- optimization differences;
- or a direct model finding a better task-specific representation.

The result does not prove that structured modelling is intrinsically inferior. It demonstrates that this particular structured parameterization did not earn its additional assumptions.

---

### 8. Experiment C: Conditional Continuous Flow Matching

#### 8.1 Research Question
Does replacing deterministic point prediction with stochastic conditional generation improve background-latent recovery? The tested model used approximately 1.4 million parameters and twenty-step Euler integration.

#### 8.2 Reported Results

| Method | Reported SDR | Improvement over reported mixture |
| :--- | :---: | :---: |
| **Untouched mixture** | −32.21 dB | 0.00 dB |
| **Direct deterministic predictor** | −23.69 dB | +8.52 dB |
| **Conditional flow model** | −32.09 dB | +0.12 dB |

The flow generated different outputs across three samples, with a reported variance statistic of 0.22508.

#### 8.3 Defensible Finding
The tested flow baseline did not achieve useful reference-background recovery according to the implemented SDR metric. Its output variability establishes that the numerical sampling procedure produced different outputs. It does not establish that:
- the true conditional distribution is multimodal;
- the generated samples correspond to valid source modes;
- the samples are perceptually plausible;
- or the model learned the intended conditional distribution.

#### 8.4 Scope of the Negative Result
The negative result applies to the tested 1.4-million-parameter architecture, its conditioning mechanism, its training objective, twenty-step Euler integration, the available training data, and the reported evaluation pipeline. It does not resolve the prospects of larger conditional flows, joint multi-source generation, permutation-equivariant source axes, source-consistent conditioning, mixture-consistency guidance, pretrained generative backbones, or LoRA adaptation of such backbones. The report should not infer that deterministic prediction is generally necessary for source recovery.

---

### 9. Experiment D: Waveform Residual Adaptation

#### 9.1 Research Question
Can a waveform-domain network improve a decoded latent-separator output? The reported adapter used GroupNorm, a bounded output transformation, and multi-resolution STFT loss.

#### 9.2 Reported Score
The adapter achieved **−22.65 dB SDR**. If the direct latent separator’s decoded output at −23.69 dB was the adapter’s input, the incremental improvement was:
$$-22.65 - (-23.69) = \mathbf{1.04\text{ dB}}$$
This is the comparison that should be reported prominently.

#### 9.3 Required Clarification
The final report must state exactly whether the adapter received:
- The decoded direct-separator output;
- An oracle-subtracted latent decode;
- The untouched mixture decode;
- The original mixture waveform;
- Or a combination of these.

Without that information, the Stage D result cannot be interpreted reliably.

#### 9.4 Defensible Finding
Subject to clarification of its input baseline, waveform refinement appears to add approximately 1 dB over the practical direct latent separator. This supports further study of hybrid latent-context and waveform-refinement architectures. It does not establish that the frozen decoder is the ultimate bottleneck, that waveform refinement has recovered discarded phase information, that the adapter has inverted decoder information loss, or that joint decoder adaptation is required for complete demixing.

---

### 10. Decoder Gradient-Path Finding

The code path behaves as follows:
```python
if self.pretransform is not None:
    if self.pretransform.enable_grad:
        decoded = self.pretransform.decode(decoded)
    else:
        with torch.no_grad():
            decoded = self.pretransform.decode(decoded)
```
This means that end-to-end loss propagation through the pretransform decoder requires `pretransform.enable_grad = True` when the training objective depends on decoded audio.

Two separate experimental conditions must be distinguished:
1. **Frozen decoder with input gradients:** Decoder parameters remain frozen (`for p in decoder.parameters(): p.requires_grad_(False)`), but the forward pass is gradient tracked so that $\frac{\partial L}{\partial \hat{z}_b}$ can be calculated.
2. **Trainable decoder adaptation:** Some decoder parameters, adapters, or LoRA weights are updated.

The first condition tests decoder-aware latent training. The second tests whether adapting the decoder improves the interface. They must not be combined into one experiment and interpreted as one mechanism.

---

### 11. Revised Findings

- **Finding 1 (Interaction residual contains useful predictable structure):** A contextual oracle predictor improved decoded SDR by 0.79 dB over zero interaction correction. This establishes partial predictive structure under the tested compressed-input representation. It does not establish how much of the remaining error is irreducible.
- **Finding 2 (Latent metrics are not sufficient proxies for audio quality):** The structured separator achieved lower latent $L_1$ loss but worse decoded audio than the direct predictor. This is one of Phase 9’s strongest results and directly motivates decoder-aware objectives.
- **Finding 3 (Decoder error sensitivity is direction dependent):** Equal-norm errors in different tested latent directions produced different waveform degradation curves. This supports using decoded losses or decoder-local sensitivity weighting rather than uniform coordinate losses alone.
- **Finding 4 (Direct prediction remains the best tested deployable latent model):** The direct point separator outperformed the tested structured and flow models on the reported test split. This is an empirical architecture ranking, not a general proof that deterministic regression is optimal.
- **Finding 5 (The small flow baseline was inadequate):** The model generated variable outputs but did not recover the reference background effectively. The experiment does not establish genuine conditional multimodality or validate the outputs as plausible source samples.
- **Finding 6 (Waveform refinement is promising but modest):** The reported waveform-adapter result appears to improve the practical direct separator by approximately 1.04 dB, pending clarification of its exact input. This justifies further hybrid experiments but not a claim that waveform adaptation resolves the frozen-interface bottleneck.

---

### 12. Claims Not Supported by Phase 9

| Unsupported Claim | Replace With |
| :--- | :--- |
| *The remaining 60.9% of residual variance is irreducible phase ambiguity.* | The tested contextual model did not explain the remaining residual variation. Further experiments are required to distinguish missing information from model, data, optimization, and metric limitations. |
| *Conditional flow diversity confirms multimodality.* | The tested flow sampler generated variable outputs, but the experiment did not establish whether that variability corresponds to valid conditional source modes. |
| *The interaction residual remains on the natural latent manifold.* | At some matched perturbation magnitudes, interaction-residual directions caused less decoded waveform degradation than the tested isotropic-random directions. |
| *The frozen autoencoder interface is confirmed as the bottleneck.* | The results motivate testing whether decoder-aware training or decoder adaptation reduces the remaining performance gap. |
| *Full demixing requires joint decoder adaptation.* | Joint decoder adaptation is an untested next step made technically possible by enabling gradient propagation through the decoder path. |
| *Point estimation is strictly necessary for phase-aligned waveform separation.* | The tested deterministic model substantially outperformed the tested small conditional-flow baseline on reference-based SDR. |

---

### 13. Revised Hypothesis Scorecard

| Research Question | Evidence | Defensible Conclusion |
| :--- | :--- | :--- |
| **Does $r_{\mathrm{int}}$ contain predictable structure?** | Contextual oracle predictor reached reported $R^2 = 0.3911$ and improved SDR by 0.79 dB | Supported under the tested oracle condition |
| **Is residual unpredictability irreducible?** | No capacity or data scaling study; no latent-collision study | Unresolved |
| **Is the decoder directionally sensitive?** | Perturbation slopes differed by direction | Supported locally under the tested protocol |
| **Does structured factorization beat direct prediction?** | Structured model was 2.96 dB worse in decoded SDR | No, not for the tested architectures |
| **Does conditional flow improve recovery?** | Tested flow improved only 0.12 dB over reported mixture | No, not for the tested baseline |
| **Does flow variability demonstrate multimodality?** | Three generated outputs had nonzero variance | Not established |
| **Does waveform refinement help?** | Reported adapter score was 1.04 dB better than direct prediction | Promising, pending baseline clarification |
| **Is the frozen decoder the bottleneck?** | No controlled frozen-gradient versus adapter versus LoRA comparison | Unresolved |
| **Is decoder-gradient training technically possible?** | Code path requires `enable_grad = True` | Yes |

---

### 14. Next Experimental Gates

- **Gate 1 (Audit the metric scale):** Before further model training, verify the closed-form untouched-mixture SDR: $10 \log_{10} \frac{\|b\|_2^2}{\|s\|_2^2 + \epsilon}$. Compare this with the implemented metric. Report actual target-to-background ratios after rendering. Verify amplitude scaling, channel aggregation, and temporal alignment. Report original-waveform and autoencoder-output metrics separately. The −32.21 dB mixture score must be explained.
- **Gate 2 (Determine whether 39.1% is still capacity limited):** Run an oracle capacity and data ladder: current contextual model; 8-layer Transformer; 16-layer Transformer; TFLocoformer-style sequence model; progressively larger training datasets; progressively longer context. If performance continues increasing, language about a residual ceiling is premature.
- **Gate 3 (Test the missing-phase hypothesis directly):** Construct phase-altered waveform pairs whose isolated SAME latents are close but whose mixture interaction residuals differ. Measure $d_z = \|E(b_1) - E(b_2)\| + \|E(s_1) - E(s_2)\|$ against $d_r = \|r_{\mathrm{int}}(b_1, s_1) - r_{\mathrm{int}}(b_2, s_2)\|$. Then test whether adding complex-STFT phase features improves residual prediction.
- **Gate 4 (Separate gradient propagation from decoder adaptation):** Run: Latent-only training; Decoder-aware waveform loss with frozen decoder weights and `enable_grad = True`; Decoder adapter training; Decoder LoRA; Partial decoder unfreezing. Use identical datasets, separators, and training budgets.
- **Gate 5 (Test a credible generative baseline):** The next flow experiment should use: a substantially stronger architecture; explicit joint source representation; permutation-equivariant source processing; supervised source stems; source-consistent conditioning; better ODE integration; and mixture-consistency evaluation. The Phase 9 flow result should remain in the report as a negative baseline, not as evidence against generative separation generally.

---

### 15. Revised Conclusion

Phase 9 established that SAME’s nonlinear interaction residual is neither captured adequately by centered latent addition nor devoid of predictable structure. A contextual oracle predictor recovered useful residual structure and modestly improved decoded waveform fidelity. The decoder’s response to latent errors was direction dependent, and practical separation quality was not well predicted by the tested latent $L_1$ objective.

Among the tested deployable models, direct background-latent prediction performed best. Explicit factorization into target and interaction terms reduced the latent-coordinate loss but worsened decoded audio, indicating that internal decomposition errors can combine unfavourably under the nonlinear decoder. The tested small conditional flow model did not recover the reference background effectively, while waveform-domain refinement provided a modest additional improvement over the practical direct separator.

These findings support three immediate directions:
1. Audit and repair the waveform evaluation pipeline, particularly the unexpectedly low untouched-mixture SDR.
2. Train the direct separator using decoder-aware waveform and spectral losses with gradient flow through the frozen decoder.
3. Test decoder adapters and LoRA separately from the effect of merely enabling decoder gradients.

The experiments do not yet determine how much residual error is due to information absent from the isolated latents, insufficient model capacity, optimization, metric mismatch, or decoder sensitivity. They also do not establish that stochastic latent separation is inherently inferior to deterministic regression.

The central Phase 9 conclusion is therefore:
$$\boxed{ \text{The main demonstrated problem is not latent non-additivity alone, but the mismatch between latent-coordinate prediction errors and their decoded acoustic consequences.} }$$
That conclusion is both supported by the current results and directly actionable for the next phase.
