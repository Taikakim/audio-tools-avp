# V5 Proposal Review and Staged Plan: Listening, Metrics, Single-GPU Throughput

**Date:** October 13, 2026; **last updated October 15, 2026** (section "Status" below, sections 3, 6.1, 9, 10)
**Reviewer:** KUANG, for Kim, Antigravity and Wintermute (the validation step that V5 section 7.3 asks for)
**Reviews:** `V5_ARCHITECTURAL_ANALYSIS_AND_PROPOSAL.md` at `4426f77`, plus section 6 (training speed and precomputation) added in `d91ccb8`
**Checked against:**
- the code at `d91ccb8` (`fix/inharmonicity-aux`);
- SonicMaster `ab32b2d` (September 16, 2026) and stable-audio-metrics `fd55536` (March 21, 2024);
- the Synth-JEPA deep-read note in `SAO/papers`;
- the paper's listening-test section, read from the arXiv HTML version (2609.31024v1, section 4.3).

**Evidence tags:**
- **[V]** verified in code here; file:line given.
- **[N]** from the team notes or the paper.
- **[I]** inferred from code or arithmetic.
- **[U]** unverified or proposed.
- **[T]** measured in this branch's CPU tests on synthetic or stand-in audio; not on Surge.

Nothing here ran on Surge, a ROCm GPU, real checkpoints or stems. The laptop has none of them, and no C++ toolchain to build Surge. Every tool in section 9 is tested on CPU only, and the desktop smoke tests are listed there.

---

## Status, October 15, 2026 (read this first)

**Where things are** (branch `fix/inharmonicity-aux` holds the v5 tools since merge `5c53a3a`; `feature/v5-flow-seeded-search` adds the flow-seeded search and the bank code guard below):

| Stage | State | Evidence |
|---|---|---|
| 0a. 500-step throughput runs (live rendering) | **Not done** | `implementation_plan.md` step 0.3 |
| 0b. Render bank | Built on October 14 (`bank_v5_s0`, 786 432 items + 12 288 ladders, 65 GB, verified), then **made stale** the same night: `d795b1e` and `e6e07f4` changed the parameter layout from 48 to 53 values (amp/filter EG mode, decay shapes, `feg_release`), the ranges and the random-patch sampler. A rebuild was started and **stopped part-way** when the desktop agent ran out of tokens (Kim, October 15) | `walkthrough.md`, `surge_spec_v3.py` diff |
| 0c. Ladder metric check | **Done** on Surge (October 13, 48-value code, 149-preset manifold) | `ladder_gate.md`; summary in section 3 |
| 0d. Banked throughput (step 0.6) | Not done | — |
| 1–4 | Not started | — |

**Rules that follow (for any agent):**
1. **Bank code guard (new).** `render_bank_v5.py` now writes a `code_fingerprint` (md5 of `surge_spec_v3.py`, `realistic_bass_prior_v3.py`, `training_controls_v3.py` and the trainer's random-patch sampler with its helpers; line endings normalised) and `num_params` into `manifest.json`. The trainer refuses a bank whose fingerprint differs from the running code, `BankMainDataset` refuses vectors of the wrong length, and `--val_cache` keys include the fingerprint. A bank directory from before this guard (including a half-built one) is refused on resume: **delete it and rebuild**, or pass `build --trust_existing` only if every existing shard was rendered by exactly the current code. Any later edit to those files means rebuilding the bank (about 28 min at the measured 580 clips/s).
2. **A-series checkpoints (A0–A3′) are 48-value models.** The current 53-value code cannot load them (the flow head and the parameter encoder change shape) [I, from the code]. Stage 1 on A-series checkpoints must run in a worktree at a 48-value commit, for example `git worktree add ../sit48 aa734a8`, with the v5 evaluation files copied in (`git checkout feature/v5-flow-seeded-search -- scripts/synth_inversion/flow_seeded_search_v5.py scripts/synth_inversion/audio_eval_v5.py`), and with the manifold each run was trained on (md5 in its `run_meta.json`). New 53-value runs compare only with each other.
3. **Throughput, revised.** The bank build rendered 580 clips/s with 14 workers, about 41 clips/s per worker [N, `walkthrough.md`]. The trainer's 8 + 3 workers can therefore supply several hundred clips/s against the 42 samples/s the A-series achieved, so those runs were probably **not** waiting on Surge [I]; the GPU step, including ModularOptimizer's per-weight syncs (section 5.1), is the likelier limit. Steps 0a and 0d measure it. If confirmed, the bank's value is identical data across arms rather than speed, and the speed must come from model size and the optimizer.
4. **Laptop vs desktop.** Full-size model runs, timing and anything with Surge belong on the desktop; the laptop runs the CPU tests only.

## 0. Summary

The diagnosis is right in direction. No predicted patch is ever rendered in training or in the v4 per-item evaluation, so we have neither heard nor scored our generations as audio.

This branch (`feature/v5-staged`) turns the review into tools and a staged plan (section 7). It implements what Kim asked for, built and tested on CPU:
- **Mel-SSIM** (plain, and gated to non-silent pixels) and the other audio metrics;
- **cross-attention in the flow inverter**;
- **Flow-DPO**, ready to try now;
- a **paper-style listening test**;
- a **small-model preset** for finding the training paradigm.

It also adds the pieces those need: a shared render bank, cached validation sets, data-wait logging, an audio-space evaluator with a listening pack, and a ladder check for metrics.

Findings that shaped the plan:
1. **Much of V5 already existed.** The JEPA search, render-and-refine, best-of-k, activity masking, the paper's metrics, a human-alignment harness and blind A/B were all in the repo. The cheapest first step is still to evaluate what we have in audio space (stage 1).
2. **The throughput numbers are budgets, not measurements.** 42 samples/s is 12 000 × 64 / 5 h. A smaller model only trains faster if the GPU, not Surge, is the bottleneck. The trainer now logs both rates (stage 0), and the render bank removes Surge from the loop.
3. **Most of the SonicMaster and stable-audio-metrics list does not fit our audio: 0.8 s, mono, peak-normalized bass notes** (section 3).
   - Faithful ports of the SonicMaster metrics sit in the ladder check so the data can confirm or overturn that verdict.
   - A stand-in test finds that **log-mel L1, our primary metric, is blind to unison detune**: a re-render with new phases scores the same as twice the detune. The modulation spectrum separates them [T]. On Surge ladders no metric tracks detune reliably (best ρ ≈ 0.4; section 3), so detune is judged by ear for now [N].
4. **Pre-encode audio, not mel and not JEPA tokens** (section 5).

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
| 2 | Section 0, Disconnect 1 examples (drive without a waveshaper, detune without unison) | `canonicalize_vector` (surge_spec_v3.py:242) already zeroes both in every training sample and before every search score. It does the same for delay feedback without delay. FX are rendered off (`FX_RENDER_ENABLED = False`) **[V]**. The real gap is the 25 expanded parameters (oscillator 2 and 3 shape, width and octave; ring-mod and noise volumes; filter 2), which have no coupling rule. `surge_activity.py` has activity rules with 18/18 probe agreement (`activity_probe.json`), but only its own probe and test import it, not the trainer, the eval or inference **[V]**. It uses v4 mute flags, so v3 needs an adapter (volume 0 counts as muted) **[I]**. | Add `param_mae_active` to the eval: it is cheap and needs no retraining. Extend canonicalization for the next training run. The "irreducible 0.12 floor" then becomes something we measure rather than assert. |
| 3 | Section 0: the loss is isotropic ℓ1 | It is MAE weighted by domain (`w_enc`): shape 3.5, cutoff 2.5, filter type 2.0, and so on **[V]** (train_realistic_bass_v3.py:238, surge_spec_v3.py DOMAIN_WEIGHTS). | Minor. |
| 4 | Sections 0 and 1: no audio feedback is possible in training | The JEPA path f<sub>p→a</sub>(E<sub>p</sub>(θ)) is a differentiable audio proxy we already have. It is exactly what the paper's search optimizes. | A renderer-free, audio-aware training signal is possible (candidate C, section 6). |
| 5 | Section 3.2: z<sub>a</sub> ∈ R<sup>512</sup>; "our optimizer is significantly more advanced [V]" | The A-series ran the `run_v3_recipe.sh` defaults: embedding 768, flow 768, feed-forward 2 048 **[V]**. "More advanced" is unverified. The A-series had no AdamW control, and earlier ablation notes found Muon/MONA worse than AdamW in one setting (fusion-optimiser manual, R-list). | Retag as [U]. Add an AdamW control arm once arms are cheap (stage 2). |
| 6 | Section 3.1 understates the paper | In the paper, flow matching was the losing baseline: in-domain MSS 8.44 for JEPA search vs 12.75 for flow, and flow stops improving past about 16 ODE steps. Rendered best-of-k by log-mel L1 helps flow the most (Fig. 3) **[N]**. | This is the strongest argument for Proposal 4, and it can be tested on existing checkpoints (stage 1). |
| 7 | Section 4, #19: MSS is "spectral convergence + log magnitude" | That is `audio_utils.MultiScaleSTFTLoss`, which the team found does **not** match the paper. The paper's MSS is multi-scale log-mel L1, ported exactly in `reference_metrics.mss`, alongside `wmfcc` and `rms_env_cos` **[N, V]**. The inventory leaves out wMFCC and the human-alignment results. | Use `reference_metrics`. |
| 8 | Section 4: descriptions of metrics #1, #2, #4, #14, #15 and #17 | **#2:** the KL is taken over dB mel values (≤ 0, ref = max) normalized to sum to 1, so the quietest bins weigh the most. It is not an energy distribution. **#4:** the script averages every aesthetics axis and never isolates PQ. **#14:** an amplitude ratio that returns 0 for mono input. **#15:** examples and cached stats use 44.1 kHz, normalization is peak −1 dBFS rather than LUFS, and it needs TensorFlow `openl3==0.4.2`. **#17:** text-to-audio only; there is no audio-to-audio mode. **#1:** SonicMaster uses the non-fusion 630k checkpoint, while stable-audio-metrics uses the fusion one. | See section 3. |
| 9 | Section 6.2C: both repos cache `clean_embeddings.npz` in `load/` | SonicMaster ships no cache; it writes one to a hard-coded path. stable-audio-metrics' `load/` holds OpenL3 μ/Σ and PaSST probabilities for public datasets, and nothing for CLAP. | Build our own cache (section 5.4). |
| 10 | Section 6.1: 24 stems in about 5 s on CPU | stable-audio-metrics' PaSST and CLAP code hard-codes CUDA. OpenL3 needs TensorFlow, and its README says GPU only. The 5 s figure was not measured. | On ROCm, TensorFlow is a liability: skip OpenL3. |
| 11 | Section 6.2: about 42 samples/s, 5.0 h, IPC-bound | 42/s is 12 000 × 64 / 18 000 s, taken from the launcher's `HOURS=5.0` **[I]**. A float32 clip is 141 KB, so 42/s means about 6 MB/s of inter-process traffic, which is no bottleneck **[I]**. The trainer renders through pedalboard (`init_synth`, surge_spec_v3.py:420). surgepy's "439 renders/s" test renders the same note over and over and never sets a parameter (test_surgepy_renderer.py:91), so the comparison isn't like for like. | Stage 0 measures it. |
| 12 | Section 6.2E: 2 048 validation notes and 2 048 ladders, 15 min startup | At startup the trainer renders, in one process, 512 + 512 validation notes and a ladder set of 8 anchors per axis × 4 axes × 16 steps. It then estimates the mel normalizer over 250 batches, which is 16 000 worker renders **[V]** (train :344–345, :531–537, :597–602). The 2 048 figure is `heldout_eval_v4`'s extra set. | The startup cost is real. `--val_cache` and `--bank` now remove both parts. |
| 13 | Section 6.2B: precompute frozen JEPA tokens for the flow | The flow has its own encoder (conv or temporal ASP), trained by the flow loss; the one established effect lives there. Condition noise (H6) noises each sample's mel at a fresh level every step (train :230–235). Frozen tokens computed from clean mels cannot reproduce that. "−40 % memory, −30 % time" was not measured. | Not a speedup. It is a different model (a flow conditioned on frozen JEPA tokens): a legitimate arm, but test it as one. |
| 14 | Proposal 1: cross-attention | `models_v4.py` already provided `pool="segments"` (attack and sustain tokens plus ASP) and `pool="queries"` (learned queries that cross-attend to the tokens) **[V]**. | Per-block cross-attention is now built too (section 6). `pool="queries"` remains the cheaper alternative arm. |
| 15 | Proposal 2: RoPE | Training onsets are always at 0 s: `phrase_events` plays notes at 0–0.125 s and 0.25–0.5 s **[V]**. Shift equivariance only matters for real stems, which `onset_alignment_v4` already aligns. RoPE alone won't make a model trained on 0.8 s reliable on multi-second phrases **[I]**. | Low priority. If done, bundle it with the DualMel front end, since both need a fresh start. |
| 16 | Proposal 3: Flow-DPO "zero overhead" offline | DPO needs a frozen reference model's errors on the same noised pairs, which is one extra forward pass per sample **[I]**. The proposed reward (Mel-SSIM − 9-band − punch) is unvalidated. The team's check puts log-mel L1 first, with a Spearman correlation of 0.473 against human dissimilarity **[N]**. On synthetic data we already know θ\*; what reward training adds is learning from unlabeled real audio. | Implemented with log-mel L1 as the default ranking (section 6). Real-audio DPO comes after synthetic DPO shows a gain. |
| 17 | Proposal 4: hybrid inversion | The JADE + Adam search is implemented and checked against the paper (`synth_jepa_search_v3.py`), and renderer-in-the-loop refinement exists (`refine_v3.py`). The only new piece is seeding the JADE population with flow samples. | JEPA search is a system in `audio_eval_v5.py`. Flow-seeded search is not built yet. |
| 18 | Section 7.2: Bitwig projects (1 260 clips) | Not blind: the arm names are visible. | Fine for exploration. Make decisions with the protocol in section 4. |

Hygiene: the V5 document and the launchers contain absolute home-directory paths, and the repo is public at the moment.

### 2.1 Addendum: the audio benchmark added later (V5 section 8, `6e4c9b7`, and `eval_audio_perceptual_suite.py`)

1. **The arms it compares differ only by seed.** The renders come from the flow alone: `render_multitemp_eval_arms.py` loads `flow_*_best.pt` with `load_inverter_v4` and reranks with the STFT loss. The JEPA, and so the inharmonicity head, is never loaded **[V]**. The JEPA and the flow are separate modules with separate losses, and the flow is not conditioned on the JEPA. So for the audio:
   - A0 vs A1 and A2 vs A3 are flow replicates;
   - A3 vs A3′ is a full replicate.

   The one path from the aux head to the audio is checkpoint selection: `flow_*_best.pt` is the step with the best held-out JEPA loss (train :701–707) **[V]**. The "A1 failure mode" and "A3 synergy" explanations therefore have no mechanism.
2. **The replicate spread in its own table is as large as the effects it reports.**
   - **EMA Mel-SSIM:** A0 vs A1 differ by 0.052, A2 vs A3 by 0.026, A3 vs A3′ by 0.026. The between-arm differences it highlights are of the same size.
   - **Punch error:** online A3 vs A3′ differ by 0.10, more than A0 vs A3.
   - **No statistics:** there are no paired tests or confidence intervals; per-stem standard deviations are 0.12–0.34 on 24 stems.

   Nothing in the section is established. Under EVAL_PARADIGM_V4.md it would need paired, preset- or stem-clustered intervals that exceed the replicate gauge.
3. **Forking paths.** It reports 5 temperatures × 2 weight kinds × snapped/raw, highlighting the best setting per claim. Two further problems:
   - all 24 stems were used, so no dev/test split remains for a later confirmation;
   - the metrics are the SonicMaster ones that section 3 puts through the gate first.
4. **Naming.** "ASP" in this code base is attentive statistics pooling (`models_v4.py`), not "acoustic parameter sensitivity."
5. **The manifold change (`285a4c5`, 134 → 149 presets, 127 train / 22 held-out) affects every later comparison.**
   - **A-series checkpoints:** evaluate them with the manifold they were trained on. The md5 is in each run's `run_meta.json`; `audio_eval_v5.py --manifold` must point at that file. If the held-out split was re-drawn, some new held-out presets may have been training presets.
   - **New arms:** they train on a different prior, so compare them with each other, not with the A-series numbers.
   - **Render bank:** build it from the manifold the new arms use; the trainer refuses a mismatch.

## 3. Metrics for This Task

Our audio **[V]**:
- 0.8 s long;
- mono (the channel mean);
- peak-normalized to 1.0;
- 44.1 kHz;
- fundamentals from E1 to D3 (41–147 Hz);
- FX off;
- always the same two-note phrase.

All metrics below live in `audio_metrics_v5.py` as paired distances, where 0 means identical. The SonicMaster ones (`sm_*`) are faithful ports that keep their quirks, so the ladder check can test them alongside ours.

| Metric (`audio_metrics_v5` name) | Fit for 0.8 s mono bass | Cost | Verdict |
|---|---|---|---|
| log-mel L1 (`logmel_l1`, our model's mel) | Best human alignment in our check (0.473); also the paper's best-of-k selector. **Blind to unison detune** on a two-voice saw: 40 vs 80 cents scores the same as a re-render with new phases (ratio ≈ 1.0) **[T]** | Negligible | **Primary.** Also use it as the reranker: `inference_v3.py:117` still uses `MultiScaleSTFTLoss`, which scored 0.437 |
| MSS (`mss`) | 0.470; a paper metric | Milliseconds per clip | **Primary** |
| wMFCC (`wmfcc`) | 0.392; a paper metric; tolerant of timing shifts (DTW) | Milliseconds | Secondary |
| RMS envelope (`env_dist`, 1 − cosine) | 0.307 overall, but best on the envelope-driven datasets | Milliseconds | Secondary, for the envelope axes |
| **Mel-SSIM** (`mel_ssim_dist`) | Structure of envelopes and filter sweeps on our own mel; Surge ladder ρ 0.61 **[N]** | Milliseconds | **Secondary metric** (passed the ladder check) |
| **Gated Mel-SSIM** (`mel_ssim_gated_dist`) | The same, but averaged only over pixels within 40 dB of either clip's maximum **[T]**. On Surge not better overall than plain (0.57 vs 0.61) | Milliseconds | **Not adopted** (Surge ladder check); plain Mel-SSIM goes forward |
| Modulation spectrum (`modspec_dist`) | Separates 40 from 80 cents about 3.4× better than a re-render at 110 Hz on a synthetic saw **[T]**, but on Surge detune ladders only ρ = 0.37 **[N]** | Milliseconds | Weak detune diagnostic; detune judged by ear for now |
| Gated flatness (`flatness_gated_err`) | Fails the Surge ladder check (ρ 0.23 overall; drive 0.13, noise 0.07) **[N]** | Milliseconds | **Dropped** |
| Octave-band dB profile (`octave_db_l1`) | A bass-resolution replacement for the 9-band balance | Milliseconds | Diagnostic for tilt (cutoff, shape) |
| CLAP audio-to-audio (`clap_distances`) | A paper metric. laion_clap repeats a 0.8 s clip to fill 10 s. Optional; needs laion_clap and a checkpoint | GPU | Add after the gate |
| SonicMaster ports: `sm_band9_*`, `sm_mel_kl`, `sm_ssim_dist`, `sm_punch_err`, `sm_dr_err`, `sm_flatness_err`, `sm_modspec_dist` | Reasons in the previous answer and correction 8: linear 9-band shares, KL on dB values, one-number punch and dynamic range, flatness counting silence | Milliseconds | Expect "don't use"; kept in the ladder check so the data decides |
| FAD (CLAP), OpenL3-FD | Distribution-level; meaningless on 24 stems. OpenL3 needs TensorFlow | — | Not for inversion |
| RT60, stereo ratio, global RMS, PQ, SI-SDR | Degenerate for dry, mono, peak-normalized notes | — | Don't use |

**Ladder check on Surge (October 13, 2026; `ladder_gate.md`, 9 axes × 8 held-out anchors × 9 steps, 48-value code) [N]:** mean Spearman ρ over the nine axes: `mss` 0.65, `logmel_l1` 0.64, `octave_db_l1` 0.63, `mel_ssim_dist` 0.61, `wmfcc` 0.60, the SonicMaster SSIM and KL ports 0.60, `modspec_dist` 0.58, `mel_ssim_gated_dist` 0.57, `env_dist` 0.52; the remaining SonicMaster ports 0.30–0.42; `flatness_gated_err` 0.23. What this changes:
- **Primary metrics confirmed:** `mss` and `logmel_l1` (ρ ≈ 0.85–0.98 on cutoff, sustain, FEG amount, resonance).
- **Mel-SSIM:** the plain version (0.61) goes forward as the secondary structural metric. The gated version is not better overall (0.57); it is better only on AEG sustain (0.90 vs 0.76) and worse on drive (0.28 vs 0.67).
- **Detune:** no metric tracks it reliably on Surge. Best are `env_dist` 0.40, `modspec_dist` 0.37 [CI 0.10, 0.62] and the SonicMaster modulation port 0.34; `logmel_l1` is 0.15. The stand-in result (modulation spectrum sees detune, log-mel L1 does not) holds in direction only. Detune errors will be judged by ear for now.
- **Gated flatness fails** (drive 0.13, noise 0.07): dropped as a diagnostic. **Noise level and FM depth** are tracked weakly by every metric (best ρ about 0.16 and 0.51).
- **SonicMaster ports:** the band, punch, dynamic-range and flatness ports rank last, as expected. The SSIM and KL ports do as well as our Mel-SSIM, but they add nothing over `mss` and `logmel_l1`.
- **To re-run on the 53-value code:** the check reads the spec at run time, so rerun `ladder_metric_check_v5.py` after the bank rebuild if the new envelope modes need an axis.

**Gate before adopting any metric:**
1. **Human alignment:** `timbre_alignment_check.py`. Its datasets are acoustic tones, so this is out of domain.
2. **In-domain monotonicity:** `ladder_metric_check_v5.py` (built). Every metric is scored on held-out one-parameter sweeps over nine axes: the four training axes plus detune, drive, resonance, FM depth and noise. It reports the mean Spearman ρ with a CI over ladders, and the first step's distance relative to Surge's own re-render noise.
3. **Agreement with the ear:** answers from `listening_test_v5.py`. Every trial is a labeled triplet.

Once there are about 100 or more answers, a composite can be fitted to them (a logistic regression on metric differences). Do not hand-weight one, as V5's reward does.

## 4. Listening

### 4.1 The Synth-JEPA paper's test (section 4.3, read from the arXiv HTML) [N]

- **Listeners and trials:** 18 listeners judged 16 targets (8 in-domain, 4 NSynth, 4 FSD50K); 48 pairwise trials per participant (16 targets × 3 baselines).
- **Task:** reference-conditioned. Participants "compared a reference to two sound matching attempts and selected the closest."
- **Presentation:** trial order and A/B sides randomized; each trial's stimuli normalized to a shared −23 LUFS; headphones or good speakers.
- **Attention checks:** four, "containing the ground truth as an option"; all participants passed.
- **Statistics:** 95 % CIs by bootstrap over participants; no significance tests. Result: 85.1 % overall preference for Synth-JEPA (89.1 % in-domain, 81.0 % out-of-domain).

### 4.2 Our version (`listening_test_v5.py`, built)

The tool follows the same protocol:
- one page per listener, each with its own trial order and A/B sides;
- audio files with anonymous random names;
- every file normalized to −23 LUFS (BS.1770 integrated loudness: pyloudnorm when installed, otherwise a built-in meter that reads a 997 Hz sine at amplitude 0.1 as −23.0 LUFS [T]);
- ground-truth attention checks, with listeners who fail any check excluded;
- domains reported separately.

Two additions:
- **A second interval, bootstrapped over participants × targets.** With a handful of listeners, a participant-only bootstrap is unstable.
- **Per-target majorities.**

**Plan for one test:**
- **Targets:** 16, matching the paper's split. 8 held-out synthetic notes (in-domain) and 8 real dev stems (out-of-domain; keep the test stems sealed).
- **Comparisons:** 3, chosen at the end of stage 3. For example: DPO vs its reference flow (both single draws), cross-attention vs ASP (small), and the best system vs JEPA search.
- **Size and length:** 48 trials plus 4 checks; with 0.8 s clips that is about 15 minutes per listener.
- **Listeners:** as many as we can get. One listener on 24 stems needs 18 wins for p < 0.05 (exact sign test); with fewer than about 5 listeners, read the participants × targets interval.

**Listening pack** (`audio_eval_v5.py --listen 16`): the same 16 held-out items for every system and checkpoint, with scores beside each clip. It is not blind; it is for hearing progress and failure modes. Its folder is also the input to `listening_test_v5.py make`.

## 5. Single-GPU Throughput and What to Pre-Encode

### 5.1 Measurement (now built in)

- **The trainer's log line** reports samples/s and the share of wall time spent waiting on the data loader (`data wait N%`).
- **`ladder_metric_check_v5.py` and `render_bank_v5.py build`** report Surge renders per second.
- **Data-bound runs.** If the data wait is high, a smaller model will not train faster. Use the bank, or try surgepy after a parity check (render the same 200 patches both ways; the mel L1 must stay within the render-noise floor of 0.06 measured October 6, 2026 [N]).

**Rough expectation [I, order of magnitude only].** At d = 768 and batch 64, model compute is about 1 TFLOP per step. The ModularOptimizer's Newton–Schulz work is of the same order.

There is also a fixed per-step cost: the radial brake calls `.item()` once for every weight on every step (`modular_opt/optimizer.py:578` and `:612`). Each call forces a GPU→CPU sync, so a step costs about 100+ syncs per optimizer. At small model sizes this can dominate step time [I]. The fix is to compute the scale as a tensor, but that file is the shared optimizer, so it is not changed in this branch.

### 5.2 Shared render bank (`render_bank_v5.py`, built)

**Contents:**
- int16 audio;
- vector, MIDI note, inharmonicity label and random-patch flag;
- a ladder bank for the ordinal loss;
- `manifest.json` (prior md5, renderer, seeds, inharmonicity-config md5).

**Properties:**
- **Reproducible.** Each item has its own seed stream, so a build resumes shard by shard.
- **Same data for every arm.** `--bank DIR --bank_seed S` gives every arm the same items in the same order, whatever modules the arm builds [T]. Seeds alone don't: A0 and A3′ both used seed 42, yet got different data [I].
- **Checked.** The trainer refuses a bank whose prior, inharmonicity config, ladder length or random-patch share doesn't match the run.

**Sizing:** one 12 000-step run reads about 592 000 main items and 12 000 ladders. 786 432 items plus 12 288 ladders take about 69 GB, and about 6.5 h to build at the budgeted 42 clips/s. The build logs the real rate.

### 5.3 Caches and what not to pre-encode

- **`--val_cache DIR`** stores the startup validation renders under a key of everything that determines them [T].
- **Not mel:** a cached mel locks in the front end, while DualMel is built but untrained.
- **Not frozen JEPA tokens:** see correction 13.

## 6. The Three Proposals: Verdicts and What Was Built

**RoPE: not built; low value.** Training onsets are fixed, and real stems are onset-aligned. For the flow, parameters form a set, so they need per-parameter embeddings rather than rotary positions. Fold RoPE into a fresh-start run if one happens anyway (for example DualMel); it does not deserve its own arm.

**Cross-attention flow inverter: built** (`models_v4.py`, `--flow_xattn`, `FLOW_XATTN=1`).
- **How it works.** After every ResBlock, the flow state forms one query per head over the encoder's 41 time tokens. The state already carries x_t, t and τ, so different blocks and ODE times can read different parts of the note.
- **Safe start.** The output projection starts at zero, so at step 0 the model equals the flow without cross-attention [T].
- **Compatibility.** The encoder returns the tokens packed into the embedding tensor, so every existing caller that does `model.encoder(mel)` then `model(..., audio_emb=...)` still works, including best-of-k with a batch-1 condition [T].
- **Diagnostics.** `attention_maps()` shows where each block looks.
- **Checkpoints.** Exports store the configuration, including the head count, which can't be inferred from the weights.
- **Caveat.** This targets timing information, while the ladders that didn't move (cutoff, FEG amount) may be limited by the front end's 4 FFT bins below 131 Hz. Cross-attention can't recover resolution the mel never had.

**Flow-DPO: built** (`dpo_pairs_v5.py` and `train_flow_dpo_v5.py`).
- **Objective.** Diffusion-DPO (Wallace et al. 2023) adapted to flow matching. Winner and loser share one t and one noise draw; MSE inside DPO; a frozen reference copy.
- **Policy setup.** The policy's BatchNorm is frozen, so policy equals reference at step 0 and the loss is exactly log 2 [T].
- **Pairs.** Best vs worst of K = 8 by log-mel L1, dropping pairs whose gap is below each condition's measured render-noise floor.
- **Anchor.** Optional SFT term on the winner or the true vector.
- **What the tests show [T].** On a synthetic preference, the loss falls and held-out accuracy exceeds 0.6 while the reference stays frozen. The tool refuses pairs sampled from a different flow.
- **Before any DPO run: check the headroom.** `dpo_pairs_v5.py` reports mean log-mel L1 for candidate 0, best-of-K and worst-of-K, plus the share of conditions whose spread exceeds render noise. If best-of-K is barely better than candidate 0, DPO has nothing to learn.
- **Fix to the earlier pipeline.** At τ < 1 each candidate gets its own condition-noise draw (`flow_sampling_v5.py`). `predict_vectors_pinned` shares one noise draw across all candidates.

**Candidate C: not built** [U]. A renderer-free JEPA-proxy loss, ‖f<sub>p→a</sub>(E<sub>p</sub>(θ̂<sub>1</sub>)) − sg(z<sub>a</sub>)‖². It is cheap, but needs its own validation on renders.

### 6.1 Replacing the paper's JADE stage (added October 15, 2026)

**Built:** `flow_seeded_search_v5.py`. The paper's JADE stage (half of 2 048 evaluations: population 32, about 32 generations) only finds the right region, categoricals included. The flow already does that, so:
1. K flow samples per target (default 32, MIDI pinned) replace the JADE population. Optionally a share is drawn uniformly from the training box (`random_frac`), in case the flow misses a mode.
2. The paper's Adam stage (lr 0.1, cosine decay, worse half pruned 3 times) runs on **all targets in one batch**, with the training box, the couplings and the per-row note applied after every step.
3. The best candidate seen at any scoring point is returned, so the result is never worse than the best seed by the objective [T].

**Why not a newer DE variant:** SHADE's success memory needs hundreds of generations, and we run about 32. L-SHADE's standard setting starts at about 18 × D candidates and is tuned for CEC budgets of 10 000 × D evaluations, over 200× ours; its population-reduction idea is already the Adam stage's halving. NL-SHADE-RSP, EBJADE and mLSHADE-RL are tuned to CEC test functions. MetaDE/EvoX's useful part, scoring a whole population per GPU call, the search already does.

**Status:** CPU-tested on tiny random models (`test_flow_seeded_v5.py`, 6 tests): budget accounting, note pinning, box and couplings, never worse than the seeds, determinism, batched equals one-by-one, and the evaluator integration. Not run on real checkpoints; speed and audio quality are unmeasured.

**Desktop (not the laptop; full-size models need the 16 GB card):**
- Timing with random weights: `python flow_seeded_search_v5.py --bench --device cuda:0 --targets 16 --budgets 256 512 1024`. It prints the paper's search per target next to the batched flow-seeded search.
- Audio quality: `audio_eval_v5.py ... --flow A3p=RUN/flow_ema_latest.pt --jepa A3pJADE=RUN/jepa_ema_latest.pt --seeded A3pFS=RUN/flow_ema_latest.pt,RUN/jepa_ema_latest.pt --seeded_budgets 256 512 1024`. `summary.json` reports seconds per target for each system.
- **Decision:** adopt flow-seeded search if one budget matches JADE + Adam on log-mel L1 and MSS (paired, preset-clustered CI within the replicate gauge) at a fraction of the time. Keep JADE if it clearly wins on audio, which would mean the flow misses good regions.

## 7. Staged Plan

Each stage ends in a decision, and the evaluation rules of EVAL_PARADIGM_V4.md apply:
- paired comparisons with preset clusters;
- the replicate gauge;
- Holm correction on pre-registered primary metrics.

| Stage | What | Tools | Desktop cost (estimate) | Decision |
|---|---|---|---|---|
| **0. Infrastructure and measurement** | (a) 500-step runs at base and small size on live rendering: read `samples/s, data wait`. (b) Build bank seed 0 and verify it. (c) Ladder check of every metric. | trainer log line, `render_bank_v5.py`, `ladder_metric_check_v5.py` | (a) about 1 h; (b) about 6.5 h unattended at 42 clips/s; (c) about 700 renders, minutes | Is training data-bound? Which metrics, and which Mel-SSIM variant, go forward? |
| **1. Audio-space baseline, no training** | A0 and A3′ (EMA): flow draw 0, best-of-8 and JEPA search on 256 held-out notes; listening pack; one blind A/B by Kim | `audio_eval_v5.py`, `listening_test_v5.py` | About 256 × (2 × 8 + 1) ≈ 4 400 renders plus search, about 30 min | DPO headroom (draw 0 vs best-of-8); JEPA search vs flow on audio |
| **2. Small-model paradigm search** | `SIZE=small` on the bank, 12 000 steps: S0 = temporal ASP; S1 = S0 + cross-attention; two seeds each, same `BANK_SEED`; optional S2 = AdamW control | `run_v3_recipe.sh`, `heldout_eval_v4.py`, `audio_eval_v5.py` | Unknown until stage 0. GPU compute is about 1/8 of base; render cost is zero with the bank | Pre-register the primary metrics: `logmel_l1` and `mss` on draw 0, with plain Mel-SSIM as secondary (Surge ladder check, section 3). Cross-attention goes forward only if it beats the replicate gauge |
| **3. Flow-DPO** | On the stage-2 winner, and in parallel on A3′: 8 192 synthetic conditions × K = 8 (about 82 000 renders, about 35 min at 42/s); β ∈ {30, 100, 300}, lr 1e-5, 2 000 steps; with and without an SFT anchor of 0.2 | `dpo_pairs_v5.py`, `train_flow_dpo_v5.py`, `audio_eval_v5.py`, `heldout_eval_v4.py` | Pairs once (about 35 min); each DPO run is minutes on the GPU | Success means DPO's draw 0 closes at least half the gap to the reference's best-of-8, parameter-space metrics get no worse than the gauge, and listeners prefer it. A second round, sampled from the DPO checkpoint, only if round one helps. Real-audio DPO (3b) needs bass clips outside the eval stems, transcribed for MIDI notes |
| **4. Confirm at base size** | The winning recipe at `SIZE=base`, two seeds, against the A3′ recipe at two seeds; then the paper-style listening test | everything above | Two to four base runs | The decision that ships |

## 8. Small Model for Paradigm Search: What Transfers

The idea: choose the training paradigm on a model about 1/8 the compute, then confirm at base size. The `SIZE=small` preset is 384 / 384 / 1 024 with 4 JEPA layers, 4 flow blocks and a 768-wide predictor; base is unchanged at 768 / 768 / 2 048 / 8 / 8 / 1 024. This is sound for most decisions, with three conditions.

- **Likely to transfer:**
  - data and evaluation choices (the bank, metrics, listening protocol, best-of-k selector);
  - DPO's pairing and margin;
  - whether a conditioning change helps at all, when the effect is large.
- **May not transfer:**
  - **learning rate and weight decay.** ModularOptimizer's spectral updates are meant to be width-stable, but that is not verified here, and the preset halves the depth too;
  - **capacity-sensitive choices:** cross-attention adds capacity, the aux head, and the DPO β, since how far a policy can move depends on capacity. A small effect at small size can grow or flip.
- **Conditions:**
  1. Small is only faster if training is not data-bound (stage 0), and if the optimizer's per-weight syncs (section 5.1) don't set the floor.
  2. If the small model underfits badly, for example a much higher flow loss than base at the same step, conditioning decisions may not transfer. That is why stage 4 is mandatory, not optional.
  3. Keep the data identical across the two sizes (same bank and `BANK_SEED`), so only size changes.

## 9. Tools, Tests and Desktop Smoke Tests

| File | What | Status |
|---|---|---|
| `audio_metrics_v5.py` | Paired metrics, SonicMaster ports, optional CLAP | CPU-tested (an SSIM implementation matched to a slow reference, identity, monotonicity, beating, gating) |
| `ladder_metric_check_v5.py` | Metric check on one-parameter sweeps (9 axes) | CPU-tested on a fake renderer |
| `render_pool_v5.py` | Parallel Surge rendering, as the trainer renders | Pool path not run (needs Surge) |
| `render_bank_v5.py` | Build, verify, datasets, compatibility checks | CPU-tested with a fake renderer (reproducible, resumable, order fixed) |
| `train_realistic_bass_v3.py`, `run_v3_recipe.sh` | `--bank`, `--val_cache`, `--flow_xattn*`, size flags, data-wait logging, `SIZE` presets | End-to-end CPU run on tiny models with bank, cross-attention, condition noise, OT, EMA and ladders; defaults unchanged |
| `models_v4.py` | Cross-attention, packed tokens, `attention_maps`, config in exports | CPU-tested (identity at init, gradients, sampling, broadcast, save/load) |
| `flow_sampling_v5.py` | K candidates with MIDI pinned and per-candidate condition noise | CPU-tested |
| `dpo_pairs_v5.py`, `train_flow_dpo_v5.py` | Flow-DPO | CPU-tested (neutral at init, learns a synthetic preference, refuses off-policy pairs) |
| `audio_eval_v5.py` | Audio-space evaluation and listening pack | CPU-tested end to end on a fake renderer |
| `listening_test_v5.py` | Paper-style multi-listener test | CPU-tested (BS.1770 meter, make, analyze, exclusions, domains) |
| `flow_seeded_search_v5.py` | Flow-seeded, batched JEPA search replacing the paper's JADE stage (section 6.1); `--bench` timing | CPU-tested in `test_flow_seeded_v5.py` (6 tests) |
| `render_bank_v5.py` code guard | `code_fingerprint` and `num_params` in the bank manifest and the validation-cache key; `--trust_existing` | CPU-tested (`test_bank_refuses_other_code`) |

**Test results:** `test_v5.py` has 32 tests. Together with the existing `test_eval_v4.py`, `test_v4_components.py`, `test_activity_table.py` and the inharmonicity tests, 117 pass. One existing test fails because surgepy isn't installed here: `test_activity_table.py::test_v4_registry_coverage`, which doesn't touch anything this branch changes.

**Desktop smoke tests, before stage 0 proper:**
1. `render_bank_v5.py build --n_items 4096 --n_ladders 256 ...`, then `verify --n 32`: the vectors must be identical, and the median log-mel L1 must sit at the render-noise level.
2. `ladder_metric_check_v5.py --anchors 2 --steps 5 ...` runs and reports renders/s.
3. `SIZE=small BANK=... VAL_CACHE=... FLOW_ENCODER=temporal_asp FLOW_XATTN=1 MAX_STEPS=200 HOURS=1 run_v3_recipe.sh`: the log shows `samples/s, data wait`, and `heldout_eval_v4.py` loads the exports.
4. `dpo_pairs_v5.py --n 64 --k 4`, then `train_flow_dpo_v5.py --steps 50 --link_run ...`, then `audio_eval_v5.py --n 16`.
5. `listening_test_v5.py make` on that `listen/` folder.

## 10. Questions for the Desktop

1. ~~Commit `report_v4_2x2.md`~~ Done (`6e7ef56`); it matches the numbers in correction 1 (online kind: `flow_mse` +0.0111, p = 0.125, Holm 0.249; AEG-sustain ladder +0.1114).
2. What do steps 0a and 0d show: samples/s and data wait, live and from the bank? Renders per second are answered: about 41 clips/s per worker (580/s with 14).
3. Has JEPA search been scored with `reference_metrics` on the v3 or v4 checkpoints? If so, where are the results?
4. Is there a pool of real bass clips, outside the evaluation stems, for real-audio DPO (stage 3b)?
5. After the bank rebuild: does `render_bank_v5.py verify` pass on the 53-value bank, and does the manifest carry `code_fingerprint`?
