# V5 Proposal Review: Listening, Metrics and Single-GPU Throughput

**Date:** October 13, 2026
**Reviewer:** KUANG, for Kim, Antigravity and Wintermute (the validation step that V5 section 7.3 asks for)
**Reviews:** `V5_ARCHITECTURAL_ANALYSIS_AND_PROPOSAL.md` at `4426f77`, plus section 6 (training speed and precomputation) added in `d91ccb8`
**Checked against:**
- the code at `d91ccb8` (`fix/inharmonicity-aux`);
- SonicMaster `ab32b2d` (September 16, 2026) and stable-audio-metrics `fd55536` (March 21, 2024);
- the Synth-JEPA deep-read note in `SAO/papers` (arXiv 2609.31024).

**Evidence tags:**
- **[V]** verified in code here; file:line given.
- **[N]** from the team notes or the paper note.
- **[I]** inferred from code or arithmetic.
- **[U]** unverified or proposed.

Nothing here was run on this machine: it has no Surge, no ROCm GPU, no checkpoints and no stems.

---

## 0. Summary

The diagnosis is right in direction. No predicted patch is ever rendered in training or in the v4 per-item evaluation, so we have neither heard nor scored our generations as audio. Four findings change the plan.

1. **Much of V5 already exists.** The cheapest V5 step is to evaluate what we have, in audio space, with no training. Existing pieces:
   - the JADE + Adam JEPA search (`synth_jepa_search_v3.py`, checked against the paper);
   - renderer-in-the-loop refinement (`refine_v3.py`);
   - rendered best-of-k (`inference_v3.py`);
   - activity masking (`surge_activity.py`);
   - cross-attention pooling for the flow encoder (`models_v4.py`, `pool="queries"`);
   - the paper's own metrics (`reference_metrics.py`);
   - a human-alignment harness for metrics (`timbre_alignment_check.py`);
   - a blind listening tool (`listening_ab_v4.py`).
2. **The throughput numbers are budgets, not measurements.** "42 samples/s" is 12 000 × 64 / 5 h, the launcher's time budget.
   - The trainer renders through pedalboard (a VST3 host).
   - The surgepy renderer is built but has not been tried in the training loop.
   - Measure first: the fix may be the renderer rather than a dataset.
3. **Most of the SonicMaster and stable-audio-metrics list does not fit our audio: 0.8 s, mono, peak-normalized bass notes.**
   - Distribution metrics (FAD, OpenL3-FD) cannot score 24 stems.
   - The stereo, RT60, RMS and production-quality (PQ) metrics are degenerate here.
   - Five items are described wrongly.
   - Adopt the paper's own paired metrics first: log-mel L1, MSS, wMFCC, and CLAP audio-to-audio cosine. Put any further metric through the gate in section 3 before using it.
4. **Pre-encode audio, not mel and not JEPA tokens.** A render bank shared by every arm has four benefits:
   - it removes the CPU bottleneck, if there is one;
   - every arm sees the same data stream, which sharpens the paired comparisons;
   - it leaves the CPU free for rendering evaluation candidates;
   - we can listen to the exact training audio.

   Precomputed JEPA tokens are not a speedup. They don't work with the current recipe's condition noise, and they would make a different model.

## 1. What holds up

- Nothing in training or in the v4 per-item evaluation renders a prediction **[V]**. `param_mae` charges inaudible differences the same as audible ones **[V]**.
- The inharmonicity aux head is not established. This agrees with the earlier A-series review.
- The temporal ASP gain on amp-envelope (AEG) sustain tracking is large: +0.111 against a replicate gauge of 0.023, about five times the gauge **[U]**. These numbers come from the V5 table; `report_v4_2x2.md` is not committed, so I could not check them.
- Generating preference data offline rather than online, and caching reference embeddings, are the right instincts.
- We train far below the paper's scale: 12 000 versus 1 000 000 steps at batch 64. The team's own reference recipe was about 78 000 steps **[N]**. Effects measured at 12 000 steps may not hold at full length.

## 2. Corrections

| # | Claim in V5 | What the code and notes show | Consequence |
|---|---|---|---|
| 1 | Section 0: temporal ASP "established better" | In `eval_plan_v4_2x2.json` this contrast's primary metrics are `flow_mse` and `param_mae`. Neither is established, and `flow_mse` leans the other way (+0.0111, p = 0.125). Every "established" row is a secondary metric, which the plan calls descriptive. Compared with the replicate gauge: the overall ladder (+0.042 vs 0.026) and the filter EG (FEG) amount error (−0.0074 vs 0.0043) are under twice the gauge, and categorical accuracy (+0.51 vs 0.45 percentage points) only matches it. | Report it as "established on the AEG-sustain ladder; the pre-registered primary metrics were not established." Commit `report_v4_2x2.md` so the table can be checked. |
| 2 | Section 0, Disconnect 1 examples (drive without a waveshaper, detune without unison) | `canonicalize_vector` (surge_spec_v3.py:242) already zeroes both in every training sample and before every search score. It does the same for delay feedback without delay. FX are rendered off (`FX_RENDER_ENABLED = False`) **[V]**. The real gap is the 25 expanded parameters (oscillator 2 and 3 shape, width and octave; ring-mod and noise volumes; filter 2), which have no coupling rule. `surge_activity.py` has activity rules with 18/18 probe agreement (`activity_probe.json`), but only its own probe and test import it, not the trainer, the eval or inference **[V]**. It uses v4 mute flags, so v3 needs an adapter (volume 0 counts as muted) **[I]**. | Add `param_mae_active` to the eval now: it is cheap and needs no retraining. Extend canonicalization for the next training run. The "irreducible 0.12 floor" then becomes something we measure rather than assert. |
| 3 | Section 0: the loss is isotropic ℓ1 | It is MAE weighted by domain (`w_enc`): shape 3.5, cutoff 2.5, filter type 2.0, and so on **[V]** (train_realistic_bass_v3.py:238, surge_spec_v3.py DOMAIN_WEIGHTS). | Minor. |
| 4 | Sections 0 and 1: no audio feedback is possible in training | The JEPA path f<sub>p→a</sub>(E<sub>p</sub>(θ)) is a differentiable audio proxy we already have. It is exactly what the paper's search optimizes. | A renderer-free, audio-aware training signal is possible (candidate C, section 6). |
| 5 | Section 3.2: z<sub>a</sub> ∈ R<sup>512</sup>; "our optimizer is significantly more advanced [V]" | The A-series ran the `run_v3_recipe.sh` defaults: embedding 768, flow 768, feed-forward 2 048 **[V]**. "More advanced" is unverified. The A-series had no AdamW control, and earlier ablation notes found Muon/MONA worse than AdamW in one setting (fusion-optimiser manual, R-list). | Retag as [U]. Add an AdamW control arm once arms are cheap (section 5). |
| 6 | Section 3.1 understates the paper | In the paper, flow matching was the losing baseline: in-domain MSS 8.44 for JEPA search vs 12.75 for flow, and flow stops improving past about 16 ODE steps. Rendered best-of-k by log-mel L1 helps flow the most (Fig. 3) **[N]**. | This is the strongest argument for Proposal 4, and it can be tested on existing checkpoints. |
| 7 | Section 4, #19: MSS is "spectral convergence + log magnitude" | That is `audio_utils.MultiScaleSTFTLoss`, which the team found does **not** match the paper. The paper's MSS is multi-scale log-mel L1, ported exactly in `reference_metrics.mss`, alongside `wmfcc` and `rms_env_cos` **[N, V]**. The inventory leaves out wMFCC and the human-alignment results. | Use `reference_metrics`. |
| 8 | Section 4: descriptions of metrics #1, #2, #4, #14, #15 and #17 | **#2:** the KL is taken over dB mel values (≤ 0, ref = max) normalized to sum to 1, so the quietest bins weigh the most. It is not an energy distribution. **#4:** the script averages every aesthetics axis and never isolates PQ. **#14:** an amplitude ratio that returns 0 for mono input. **#15:** examples and cached stats use 44.1 kHz, normalization is peak −1 dBFS rather than LUFS, and it needs TensorFlow `openl3==0.4.2`. **#17:** text-to-audio only; there is no audio-to-audio mode. **#1:** SonicMaster uses the non-fusion 630k checkpoint, while stable-audio-metrics uses the fusion one. | See section 3. |
| 9 | Section 6.2C: both repos cache `clean_embeddings.npz` in `load/` | SonicMaster ships no cache; it writes one to a hard-coded path. stable-audio-metrics' `load/` holds OpenL3 μ/Σ and PaSST probabilities for public datasets, and nothing for CLAP. | Build our own cache (section 5.4). |
| 10 | Section 6.1: 24 stems in about 5 s on CPU | stable-audio-metrics' PaSST and CLAP code hard-codes CUDA. OpenL3 needs TensorFlow, and its README says GPU only. The 5 s figure was not measured. | On ROCm, TensorFlow is a liability: skip OpenL3. |
| 11 | Section 6.2: about 42 samples/s, 5.0 h, IPC-bound | 42/s is 12 000 × 64 / 18 000 s, taken from the launcher's `HOURS=5.0` **[I]**. A float32 clip is 141 KB, so 42/s means about 6 MB/s of inter-process traffic, which is no bottleneck **[I]**. The trainer renders through pedalboard (`init_synth`, surge_spec_v3.py:420). surgepy's "439 renders/s" test renders the same note over and over and never sets a parameter (test_surgepy_renderer.py:91), so the comparison isn't like for like. | Do the measurements in section 5.1. |
| 12 | Section 6.2E: 2 048 validation notes and 2 048 ladders, 15 min startup | At startup the trainer renders, in one process, 512 + 512 validation notes and a ladder set of 8 anchors per axis × 4 axes × 16 steps. It then estimates the mel normalizer over 250 batches, which is 16 000 worker renders **[V]** (train :344–345, :531–537, :597–602). The 2 048 figure is `heldout_eval_v4`'s extra set. | The startup cost is real; cache all three (section 5.4). |
| 13 | Section 6.2B: precompute frozen JEPA tokens for the flow | The flow has its own encoder (conv or temporal ASP), trained by the flow loss; the one established effect lives there. Condition noise (H6) noises each sample's mel at a fresh level every step (train :230–235). Frozen tokens computed from clean mels cannot reproduce that. "−40 % memory, −30 % time" was not measured. | Not a speedup. It is a different model (a flow conditioned on frozen JEPA tokens): a legitimate arm, but test it as one. |
| 14 | Proposal 1: cross-attention | `models_v4.py` already provides `pool="segments"` (attack and sustain tokens plus ASP) and `pool="queries"` (learned queries that cross-attend to the tokens) **[V]** (:11–14). | Run these as arms before building a DiT (diffusion transformer). |
| 15 | Proposal 2: RoPE | Training onsets are always at 0 s: `phrase_events` plays notes at 0–0.125 s and 0.25–0.5 s **[V]**. Shift equivariance only matters for real stems, which `onset_alignment_v4` already aligns. RoPE alone won't make a model trained on 0.8 s reliable on multi-second phrases **[I]**. | Low priority. If done, bundle it with the DualMel front end, since both need a fresh start. |
| 16 | Proposal 3: Flow-DPO "zero overhead" offline | DPO needs a frozen reference model's errors on the same noised pairs, which is one extra forward pass per sample **[I]**. The proposed reward (Mel-SSIM − 9-band − punch) is unvalidated. The team's check puts log-mel L1 first, with a Spearman correlation of 0.473 against human dissimilarity **[N]**. On synthetic data we already know θ\*; what reward training adds is learning from unlabeled real audio. | Later. Use a validated metric as the reward, and train on real bass audio that is not among the evaluation stems. |
| 17 | Proposal 4: hybrid inversion | The JADE + Adam search is implemented and checked against the paper (`synth_jepa_search_v3.py`), and renderer-in-the-loop refinement exists (`refine_v3.py`). The only new piece is seeding the JADE population with flow samples. | Evaluate it now on existing checkpoints. |
| 18 | Section 7.2: Bitwig projects (1 260 clips) | Not blind: the arm names are visible. | Fine for exploration. Make decisions with the protocol in section 4. |

Hygiene: the V5 document and the launchers contain absolute home-directory paths, and the repo is public at the moment.

## 3. Metrics for This Task

Our audio **[V]**:
- 0.8 s long;
- mono (the channel mean);
- peak-normalized to 1.0;
- 44.1 kHz;
- fundamentals from E1 to D3 (41–147 Hz);
- FX off;
- always the same two-note phrase.

| Metric | Paired per item? | Fit for 0.8 s mono bass | Cost | Verdict |
|---|---|---|---|---|
| log-mel L1 (our mel) | Yes | Best human alignment in our check (0.473); also the paper's best-of-k selector | Negligible | **Primary, now.** Also use it as the reranker: `inference_v3.py:117` still uses `MultiScaleSTFTLoss`, which scored 0.437 **[V, N]** |
| MSS (`reference_metrics.mss`) | Yes | 0.470; a paper metric | Milliseconds per clip | **Primary, now** |
| wMFCC (`reference_metrics.wmfcc`) | Yes | 0.392; a paper metric; tolerant of timing shifts (DTW) | Milliseconds | Secondary, now |
| RMS envelope cosine | Yes | 0.307 overall, but best on the envelope-driven datasets | Milliseconds | Secondary, for the envelope axes |
| CLAP audio-to-audio cosine | Yes | A paper metric. laion_clap repeats a 0.8 s clip 12 times and pads it to 10 s. stable-audio-metrics only offers text-to-audio, so we need our own short wrapper | GPU, about 10 s of audio per clip | Add after the gate |
| Mel-SSIM | Yes | Sensitive to time alignment: fine for synthetic items, needs onset alignment for stems | Milliseconds | Add after the gate |
| 9-band balance, onset strength ("punch"), spectral flatness, modulation spectrum | Yes | Coarse in the bass: only two bands lie below 250 Hz. At 0.8 s the modulation bins are about 1.2 Hz apart | Milliseconds | Per-axis diagnostics (attack, drive, detune), not for ranking |
| PaSST KL | Yes (by ID) | The 527 AudioSet classes are coarse for bass timbre. A 0.8 s clip goes in unpadded, despite the code comment saying such clips are dropped | GPU | Optional; low value |
| FAD (CLAP), OpenL3-FD | **No**: distribution-level | The covariance is rank-deficient below 512 clips, so these are meaningless on 24 stems. OpenL3 needs TensorFlow | — | Not for inversion; at most a distribution check on 2 048 or more synthetic renders |
| RT60, stereo ratio, global RMS, PQ/aesthetics, SonicMaster KL, SI-SDR | — | **RT60:** on a dry note it only proxies release time and is often NaN. **Stereo ratio:** always 0 for mono. **RMS:** fixed by peak normalization. **PQ:** has no reference and is out of domain. **KL:** computed on dB values. **SI-SDR:** breaks on free-running oscillator phase | — | Don't use |

**Gate before adopting any metric [U, proposed]:**
1. **Human alignment.** Add the metric as a candidate in `timbre_alignment_check.py`. Its 21 listening-test datasets are acoustic tones, so this is out of domain for us.
2. **In-domain monotonicity.** Render each step of the held-out ladders that `heldout_eval_v4` already builds. The metric's distance to the anchor should rise with step distance (Spearman per axis). This costs only renders.
3. **Agreement with the ear.** Compare the metric's choice with Kim's blind A/B answers (section 4).

**Cost:** every paired DSP metric takes milliseconds per clip. What costs time in audio-space evaluation is rendering the candidates. Render them once, store them, and score them on the CPU in a side process.

## 4. Listening Protocol

1. **A listening pack for every checkpoint.** It is built automatically by a CPU side job after each export.
   - 16 fixed held-out synthetic items whose ground truth is known. Each shows the target, flow draw 0, the flow's best of 8 by log-mel L1, and the JEPA search result.
   - 8 dev stems, each shown as stem and inversion.
   - One HTML page with metric values beside each clip.
   - The items are the same for every arm and checkpoint, so you hear progress and failure modes rather than hand-picked clips.
2. **Decisions between arms.** Use blind A/B with `listening_ab_v4.py`. It already conditions on the reference, randomizes sides and order, matches loudness, includes attention checks, and analyzes by stem bootstrap.
   - Use dev stems only. The test stems stay sealed until a final comparison.
   - One listener on 24 stems needs 18 wins to reach p < 0.05 (exact sign test). To detect anything smaller than a large difference, add synthetic items as trials.
3. **Answers double as metric validation.** Every A/B answer is a labeled triplet (reference, A, B), which feeds step 3 of the gate.
4. **Bitwig projects** remain useful for free exploration.

## 5. Single-GPU Throughput and What to Pre-Encode

### 5.1 Measure first (desktop, about an hour)

- **Single-core render rate.** `training.log` has the line "Rendered fixed validation sets (512 notes each) in X s". 1 024 / X gives renders per second per core with pedalboard; in aux arms this includes inharmonicity extraction.
- **Achieved throughput.** Samples per second, from the step timestamps in `training.log` or the `time` field in `val_kinds.jsonl`.
- **GPU ceiling.** Run the step loop for about 200 steps on one batch held in memory, with no DataLoader. Alternatively, watch GPU use in `rocm-smi` during a run.
- **Verdict.** If the ceiling is well above the achieved rate, training is data-bound.

**Rough expectation [I, order of magnitude only].** At d = 768 and batch 64, the model's forward and backward passes cost on the order of 1 TFLOP per step. The ModularOptimizer's Newton–Schulz iterations, over 100+ weight matrices in two models, are of the same order or larger. So the GPU ceiling may be a few hundred samples per second. That makes "300–500 samples/s" plausible but not assured: the optimizer may become the next bottleneck.

### 5.2 If data-bound: change the renderer before building a dataset

Move the workers to surgepy after a parity test:
- Render the same 200 prior patches through pedalboard and through surgepy.
- The mel L1 between the two must stay within the render repeat-noise floor: 0.06, measured October 6, 2026 **[N]** (surge_spec_v3.py:486).

| Trade-off | Assessment |
|---|---|
| Benefit | Keeps the endless supply of freshly rendered data. |
| Risk | Parameter names and state reset differ between the two hosts. |
| Reversibility | Easy: a flag. |

### 5.3 Shared render bank (recommended for any multi-arm experiment)

**Contents:**
- int16 audio: 35 280 samples, 70.6 KB per clip;
- the 48-value parameter vector and the MIDI note;
- the inharmonicity label and its valid flag;
- the 5 % random patches;
- a separate ladder bank, because every fourth step is a ladder batch for the ordinal loss.

A clip is fully determined by its vector and MIDI note, plus render noise: `render_patch` ignores `note_dur` **[V]**.

**Size:**
- 768 000 clips, one 12 000-step run with no repeats: about 54 GB as int16.
- 200 000 clips: about 14 GB.
- Mel in fp16 would be 20.7 KB per clip, about 16 GB for 768 000. But it would lock in the current front end while DualMel (`frontend_v4.py`) is built and untrained, and `ExactGpuMel` is cheap on the GPU **[I]**.
- So store audio, not mel.

**Benefits:**
- **Render once per experiment, not once per arm.** With 5 arms, that is 5 times fewer renders.
- **Identical data order across arms.** Seeds alone don't achieve this. A0 and A3′ both used seed 42, but they build different modules before the DataLoader draws its base seed, so their data streams differ **[I]** (train :379–551).
- **Listening.** The exact training audio is available for listening and for metrics.
- **CPU headroom.** It frees the CPU for rendering evaluation candidates.

**Risk: finite data.**
- 768 000 clips is one epoch of a 12 000-step run.
- For 60 000 to 78 000 steps, either run 4 to 5 epochs over about 1 million clips, or keep a background renderer appending shards.
- The prior is 134 preset anchors with covariance jitter **[N]**, so repeats are probably harmless **[I]**.
- To check for overfitting, watch the gap between training and held-out presets, which the trainer already logs.

### 5.4 Cache startup and evaluation artifacts

- **Startup sets.** Cache the validation sets, the held-out ladder set and the frozen mel normalizer, keyed by prior md5, renderer, seed and size. A shared normalizer also makes arms more comparable.
- **Evaluation.**
  - Store the target audio and target embeddings for the extra set and the stems.
  - Store each checkpoint's rendered candidates: render once, then score them with any metric later.

### 5.5 What not to pre-encode

- **Mel instead of audio:** see section 5.3.
- **Frozen JEPA tokens for the flow:** see correction 13.

## 6. Recommended Order

| | A. As proposed: DiT cross-attention, RoPE and Flow-DPO, then retrain | B. Recommended: measure, evaluate in audio space, then train what the evidence supports |
|---|---|---|
| Benefit | Unknown, because there is no audio-space baseline yet | Every later choice rests on audio-space numbers and listening |
| Cost | High: a new model, a DPO pipeline and fresh runs | Steps 1–3 below need no new model; step 2 needs CPU renders only |
| Risk | Gains would be judged with the same parameter-space metrics that V5 itself calls blind | Low; the main risk is that the surgepy parity check fails, in which case we keep pedalboard and the bank |
| Reversibility | Low: fresh start | High: flags and caches |

**Steps for option B:**
1. Measure throughput (section 5.1), run the surgepy parity check (section 5.2), and build the render bank and caches (sections 5.3 and 5.4).
2. Evaluate what exists in audio space, with no training:
   - **Checkpoints:** A0 to A3′.
   - **Items:** 512 held-out synthetic items plus the dev stems.
   - **Systems:** flow draw, flow best-of-k by log-mel L1, JEPA search, and JEPA search seeded with flow samples, each with and without `refine_v3`.
   - **Metrics:** log-mel L1, MSS, wMFCC and envelope cosine; CLAP once it passes the gate.
   - **Listening:** the listening pack plus one blind A/B test.
3. Fix the evaluation:
   - add `param_mae_active` using activity masks;
   - switch the reranker to log-mel L1.
4. Run new training arms, which are now cheap enough to replicate:
   - `pool="queries"` and `pool="segments"`;
   - canonicalization of the expanded parameters;
   - candidate C (below);
   - one AdamW control;
   - one long run of 60 000 to 78 000 steps.
5. Fine-tune with a reward or DPO on unlabeled real bass audio, not the evaluation stems, using a validated reward.

**Assumptions that would change option B:**
- If the GPU, not rendering, is the bottleneck, drop the speed argument for the bank but keep the bank for the shared data stream. Then look at the optimizer's cost and the batch size.
- If step 2 shows flow best-of-k performing about as well as JEPA search, Proposal 4 drops in priority.

**Candidate C: a renderer-free, audio-aware loss [U].**
- **Loss:** ‖f<sub>p→a</sub>(E<sub>p</sub>(θ̂<sub>1</sub>)) − sg(z<sub>a</sub>)‖². Here θ̂<sub>1</sub> = θ<sub>t</sub> + (1 − t)·v̂ is the one-step estimate of the endpoint.
- **Setup:** use the EMA JEPA, frozen. Weight the loss toward t → 1, and pass categoricals as soft one-hots.
- **Cost:** one forward and backward pass through E<sub>p</sub> and the predictor per step, and no renders.
- **Risk:** the flow may learn to exploit the proxy, so judge the result on rendered audio with the metrics in section 3.

## 7. Questions for the Desktop

1. Please commit `report_v4_2x2.md` and the per-item eval files so the V5 table can be checked.
2. What do the three measurements in section 5.1 show?
3. Has JEPA search, or search seeded with flow samples, been run on the v3 or v4 checkpoints and scored with `reference_metrics`? If so, where are the results?
4. Has anyone compared surgepy and pedalboard renders of the same patch?
