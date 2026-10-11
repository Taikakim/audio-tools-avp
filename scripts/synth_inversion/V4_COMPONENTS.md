# v4 components: what was built, what was measured, what is not done

Built October 11, 2026 by KUANG (Claude Sonnet 5.5) for Kim, on top of `fix/inharmonicity-aux` at `4dfc5ef`.
Everything here is **new files**. No existing module is edited; the one script that edits the trainer
(`apply_flow_encoder_patch_v4.py`) does so only when you run it. A0/A1 and the v3 trainer are unaffected unless that
script is run and `--flow_encoder` is used.

Evidence tags: **[R]** run here (CPU, laptop venv, stand-in synth); **[V]** verified by reading code or sources; **[U]** untested here.
**Every [R] result below is on a stand-in synth (`standin_synth_v4.py`), not Surge and not real stems.** It shows that a
component follows a control in a clean, controlled setting. It does not show that the component helps on Surge audio or
on the 24 stems. That needs the probe harness on the training machine, then blind listening.

## Why not the brief as written
The brief's losses (two-anchor STFT, psychoacoustic, prominence, harmonic-ratio) compare rendered audio, and Surge XT is not
differentiable, so they cannot be trained on in this pipeline (`V4_REVIEW/ERRORS.md` B1). The feasible form is: audio-domain
scoring at **inference** (rerank and refine), a better **encoder and front end** for the flow, and **measurements** that replace
untestable listening criteria. That is what is here.

## Files

| File | What | Status |
|---|---|---|
| `standin_synth_v4.py` | Additive stand-in bass synth with analytic controls (attack, decay, filter envelope, Q, FM, detune) | [R] used by every test |
| `onset_alignment_v4.py` | Bounded onset snapping (`snap_onset`) | [R] |
| `transient_descriptors_v4.py` | Log attack time, decay slopes, temporal centroid, attack/sustain centroid, per-band peak time and drop | [R] |
| `resonance_v4.py` | Harmonic-envelope resonance card (height, centre, width, **censored Q**), plus the brief's prominence for comparison | [R] |
| `buzz_v4.py` | Comb residual, its timeline (static vs enveloped buzz), harmonic wobble | [R] |
| `objectives_v4.py` | `TwoAnchorObjective`, log-mel and multi-scale baselines, harmonic-envelope distance | [R] |
| `inference_v4.py` | `candidate_scores`, `rank_candidates` (raw components; no tuned composite) | [R] |
| `frontend_v4.py` | `DualMel` (short and long window), band layouts with no redundant low bands. **Not wired into any trainer** | [R] shapes only |
| `models_v4.py` | `TemporalAudioEncoderV4` (asp, segments, queries), `FlowMatchingResMLPV4` (+ optional FiLM), `load_inverter_v4` | [R] unit tests; **not trained on Surge data** |
| `apply_flow_encoder_patch_v4.py` | Adds `--flow_encoder` and `--flow_film` to `train_realistic_bass_v3.py`; CRLF-safe; refuses on any anchor mismatch | [R] text edit only; trainer not run |
| `probe_metrics_v4.py` | Research-spec probe tables; pluggable renderer (`standin` here, `surge` adapter **[U]**) | [R] stand-in |
| `probe_encoders_toy_v4.py` | Toy training probe: pooled v3 encoder against the v4 pools | [R] stand-in |
| `probe_refine_v4.py` | Runs the real `refine_v3.Refiner` with the v4 objectives against a stand-in renderer | [R] stand-in |
| `enumerate_surge_v4.py` | Read-only discovery of Surge modulation sources, filter configurations and routing behavior | **[U]** needs surgepy |
| `test_v4_components.py` | 40 CPU tests | [R] all pass |
| `V4_COMPONENTS.md` | this file | |

## Re-review: what changed against the brief and the research report

- **CDPAM / psychoacoustic loss: dropped.** No training path, and the cited benchmark favors other representations (research report;
  `ERRORS.md` B14). Not implemented.
- **Resonance Q is censored, not labelled.** The card reports height and centre, and flags Q as censored when the peak is narrower than the harmonic
  spacing. Width is a half-height width, **not** calibrated to the filter's Q (it orders correctly; it is not a Q label).
- **The brief's prominence method is wrong for harmonic tones** (reproduced below, `ERRORS.md` B12).
- **Two-anchor windows are measured from a snapped onset**, never from the MIDI onset, and both clips get the same lead-in so the
  measurement bias cancels.
- **The brief's frame indices were wrong** (10 ms hop: 0 to 40 ms is frames 0 to 4, not 0 to 8). The segments pool computes its token indices from the real hop and token stride.
- **The research report's "one-module swap" for attentive pooling does not hold for the v3 flow encoder**: its time axis has only 6 positions after the strides
  (81, 41, 21, 11, 6). `TemporalAudioEncoderV4` keeps time stride 1 for the first three layers and ends with 41 tokens of about 20 ms (the Synth-JEPA token grid).
- **"Global pooling cannot perceive decay" (the brief, and the review's wording) is too strong.** In the toy probe the pooled v3 encoder reaches R-squared 0.96 on amp decay; the temporal encoders add about 0.02 on average, mostly on filter-envelope decay, and nothing on attack time.
- **The current mel front end has redundant low bands** [R, torchaudio Slaney filterbank, not `ExactGpuMel` itself]: 28 of 128 bands are fed by at most one FFT bin; the report's proposed short branch
  (`n_fft` 512, 64 mels from 20 Hz) would have 13 of 64. `DualMel` uses 32 bands from 100 Hz (short) and 96 from 30 Hz (long): 0 redundant bands in both.
- **Surge-dependent parts are not built**: enveloped sync/FM, `surge_spec_v4`, the Pedalboard against surgepy equivalence test, and any Surge probe run.
  `enumerate_surge_v4.py` is the discovery step that settles filter-configuration counts and routing availability from the pinned build.

## Measured [R, stand-in synth]

**Resonance card** (saw 1/k through a two-pole low-pass, cutoff 1 000 Hz; analytic amplitudes and rendered audio):
height orders by Q with Spearman 1.00 at f0 = 41.2, 82.4 and 146.8 Hz; the centre is within one harmonic spacing of the cutoff; Q = 0.7 reports no peak;
on a 146.8 Hz note Q of 10 or more is flagged censored (82.4 Hz: Q = 30). Width is resolved and ordered on a 41.2 Hz note for Q from 2 to 30.

**The brief's prominence** (rolling median over 500 Hz, above 800 Hz), rendered audio, Q from 0.7 to 30: at f0 = 41.2 Hz it is **22.4 dB for every Q (range 0.0 dB)**; at 82.4 Hz the range is 1.0 dB;
at 146.8 Hz the range is 4.0 dB and non-monotone (Spearman 0.72). The harmonic-envelope height over the same sweeps ranges 24.1, 24.2 and 19.5 dB. It measures harmonic-to-valley contrast, as `ERRORS.md` B12 predicted
(test: `test_brief_prominence_measures_the_comb_not_the_resonance`).

**Buzz** (comb residual): clean saw 0.0004 to 0.0014; static non-integer-ratio FM 0.31 to 0.48 at depth 1.0 and 0.03 to 0.07 at depth 0.3; integer-ratio FM 0.0006 to 0.0025 (on the comb, as it should be);
enveloped FM decays across the timeline (0.15 to 0.000 at 41 Hz) while static FM stays flat. A 25-cent detune on a single oscillator also raises it (0.03 to 0.10): the measure says "off-comb energy", not "wrong".

**Onset snapping:** with the expected onset moved by plus or minus 40 ms the snapped onset does not move (spread under 1 ms); error under 5 ms for attacks up to 10 ms at any pitch (0.8 to 4.8 ms); slow attacks on low notes are biased by up to about one waveform period
(14 to 18 ms for a 40 ms attack at 41 to 55 Hz) **in the same direction for every clip of that note**, which is why the objectives snap candidate and target the same way. A clip that is already sounding at sample 0 returns the expected onset with confidence 0.

**Descriptors:** late decay slope within 3% of the true time constant and early slope within 12%; log attack time orders attacks of 10 ms and longer on a 55 Hz note and 2 ms and longer on a 147 Hz note
(shorter attacks on low notes are not resolved: the RMS follows the waveform within a cycle); centroid ratio is 1.10 for a static filter and 1.51 for a closing one.

**Probe tables** (`probe_results_v4_standin.md`, 2 patches by 4 MIDI notes). Distance monotonicity is about 1.00 for all four distances on a clean stand-in, so it does not separate them. The nuisance block does
(distance caused by the nuisance divided by the mean distance of three real changes; lower is better):

| metric | shift 10 ms | shift 20 ms | room (RT60 0.3 s, 15% wet) |
|---|---|---|---|
| logmel_l1 | 2.06 | 2.36 | 0.65 |
| mss_l1 | 1.45 | 1.89 | 0.62 |
| two_anchor | 0.12 | 0.26 | 0.49 |
| harmonic_env | 0.10 | 0.21 | **2.43** |

Reading: onset snapping makes the two-anchor score about 17 times less sensitive to a 10 ms offset than plain log-mel. The harmonic-envelope distance ignores offsets but is **very sensitive to a room tail**: do not use it on stems with reverb unless they are dereverberated.
This is one synthetic room and one stand-in; it is a pointer, not a measurement of the stems.

**Toy encoder probe** (`probe_encoders_toy_v4.py`): each encoder plus the same small head regresses four stand-in controls from a 128 x 81 log-mel (1 200 train, 300 held-out clips, 600 steps, batch 32, 2 seeds; held-out R-squared, mean of 2 seeds):

| encoder | encoder params | attack time | amp decay | filter-env decay | Q | mean |
|---|---|---|---|---|---|---|
| conv + global average pool (v3) | 0.42 M | 0.956 | 0.963 | 0.892 | 0.971 | 0.946 |
| temporal, attentive stats pool | 1.55 M | 0.961 | 0.974 | 0.953 | 0.985 | 0.968 |
| temporal, attack/sustain segments | 1.62 M | 0.962 | 0.976 | 0.949 | 0.986 | 0.968 |
| temporal, 4 learned queries | 1.85 M | 0.482 (0.955 and 0.009) | 0.946 | 0.907 | 0.973 | 0.827 |

Reading, with the limits of a clean toy in mind (noise-free, one synth, 2 seeds, not the flow, not Surge):
- **The pooled v3 encoder is not blind to decay**: it reaches R-squared 0.96 on amp decay and 0.89 on filter-envelope decay. The review's and the brief's "cannot perceive decay" is too strong for this encoder.
- **The temporal encoders help modestly, on the control most tied to temporal structure**: filter-envelope decay 0.89 to 0.95 (seed spread about 0.005), amp decay +0.01, Q +0.015, **attack time +0.005 (no gain)**. The attack range tested is 2 to 40 ms against a 10 ms hop,
  which suggests the front end, not the pooling, limits attack (an inference from one toy; consistent with `ERRORS.md` B4).
- **The cost**: 3.7 times the encoder parameters and 7 times the encoder FLOPs for about +0.02 mean R-squared here.
- **The queries pool was unstable at this learning rate** (one seed failed on attack time, R-squared 0.009). Treat `temporal_queries` as needing tuning before any comparison; `temporal_asp` and `temporal_segments` behaved.
- What this does and does not say: Stage 2 on Surge data is plausible but not obviously large, and the gain, if any, is most likely in filter-envelope decay. It does not say it will transfer, and it argues for running the Stage 0 diagnostics first.


**Refiner probe** (`probe_refine_v4.py`, the real `refine_v3.Refiner` against a stand-in renderer, 3 perturbed starts, 2 passes, 141 renders each; recovered error is the mean absolute error of 7 controls in the Refiner's [0, 1] units, the held-out distance is `mss_l1`, which none of the objectives optimizes):

| objective | recovered error | held-out mss_l1 |
|---|---|---|
| v3 (`refine_v3.objective`) | 0.036 | 0.305 |
| log-mel L1 | 0.045 | 0.328 |
| two-anchor, the brief's weights (4, 2, 1) | 0.056 | 0.660 |
| two-anchor, no snapping | 0.052 | 0.657 |
| two-anchor, equal weights | 0.048 | 0.408 |
| two-anchor, no snapping, equal weights | 0.040 | 0.352 |

Reading: on this clean stand-in, **no two-anchor variant beat the current v3 objective**, and the brief's attack-heavy weights were the worst. Turning snapping off barely matters here, so snapping jitter is not the cause;
the likelier cause is the attack weight. Three starts with overlapping per-start ranges (v3 0.029 to 0.042; two-anchor 0.042 to 0.078) make this a pointer, not a result, and the stand-in has no onset misalignment or FX, which are the conditions where
two-anchor should help (see the nuisance table). Recommendation: keep `refine_v3.objective`; add two-anchor as an extra term, with equal weights as the starting point, and decide by blind A/B on Surge renders.


## Not done, and why

| Item | Why not | What it needs |
|---|---|---|
| Enveloped sync/FM, `surge_spec_v4.py` | Modulation routings are not Pedalboard parameters (`ERRORS.md` B7); needs the surgepy backend, a prior for the new depths, a new parameter vector and a retrain | Run `enumerate_surge_v4.py`; decide the v4 parameter space (`surge_v4_registry.py` has 72, the brief extends v3's 48); equivalence test; Kim's go |
| Pedalboard against surgepy equivalence test | No Surge here | Training machine |
| Probes on Surge audio | No Surge here | `python probe_metrics_v4.py --render surge --base-patch <json>` (adapter **[U]**) |
| Dual-mel in a trainer | Trainer and normalizer-statistics change | Stage 2 follow-up; decide after the single-mel temporal encoder result |
| Reward fine-tuning, CDPAM, psychoacoustic loss | No gradient path / unsupported (see above) | Only if Stage 1 shows the sampler is the limit |
| Real-stem validation, listening tests | Needs the stems and listeners | Stage 0 and Stage 1 gates in `V4_REVIEW/V4_REVIEW.md` |

## How to run

From `scripts/synth_inversion` (Python 3.10 or newer; numpy, scipy, torch, torchaudio; librosa only for the Refiner test and probe):

```
python -m pytest test_v4_components.py -q                      # 40 tests, about 40 s
python probe_metrics_v4.py                                      # stand-in tables, about 3 min
python probe_metrics_v4.py --render surge --base-patch P.json   # [U] on the training machine
python probe_refine_v4.py --seeds 3                             # Refiner plumbing with the v4 objectives
python probe_encoders_toy_v4.py --seeds 2                       # toy encoder comparison
python enumerate_surge_v4.py --out surge_v4_enumeration.json    # [U] needs surgepy
python apply_flow_encoder_patch_v4.py --check                   # then without --check to edit the trainer
```

Use two-anchor inside the existing Refiner (the Refiner looks up `objective` at call time; see `probe_refine_v4.use_objective`):
```
import refine_v3
from objectives_v4 import TwoAnchorObjective
ta = TwoAnchorObjective()
refine_v3.objective = lambda cand, target, env_weight=10.0: ta(cand, target)
```

Stage 2 A/B on the training machine, after the A0/A1 launch is fixed (`V4_REVIEW/ERRORS.md` Part A):
`run_v3_recipe.sh` with the same flags plus `--flow_encoder temporal_asp` (and later `temporal_segments`, `--flow_film`), a **new** `OUT_DIR`, the same `SEED`.
Measured at flow width 512, 8 blocks, cond_noise on: the v4 flow is +1.23 M (asp), +1.49 M (segments) or +1.72 M (queries) over the 5.87 M v3 flow (+21% to +29%); FiLM adds about 1.6 M more. The JEPA (53 M) is unchanged.

## Integration risks (not testable here)

- **Optimizer roles.** The modular optimizer assigns roles by parameter shape. New shapes: 4-D conv (as v3), 2-D linear and 2-D positional/query parameters, 2-D attention projections, 1-D norms. The positional embedding is 2-D on purpose. Confirm in the smoke test that no parameter falls into an unintended role.
- **Checkpoints.** A v4 flow and a v3 flow do not load into each other (different encoder keys); resume needs the same `--flow_encoder`. `load_inverter_v4` rebuilds either from `flow_*.pt` (kind inferred from the state dict; pass `v4_config` for non-default encoder kwargs).
- **EMA, cond_noise, OT coupling, MAE loss** are inherited unchanged from `FlowMatchingResMLP`.
- **Compute** [R, torch FlopCounterMode, forward, one clip]: v3 flow encoder 81 MFLOPs; temporal encoders 572 to 582 MFLOPs (about 7 times); the Synth-JEPA audio encoder 1 514 MFLOPs. At batch 64 the temporal encoder adds about 94 GFLOPs per step (forward and backward),
  roughly one third of the JEPA audio encoder's own cost. Measured step time on the training GPU: [U].
- **Two-anchor as the only refine objective was worse than the current one on the stand-in** (Refiner probe above). Add it as a term, start from equal weights, and validate by blind A/B before replacing anything.
- Weights (4, 2, 1) in `TwoAnchorObjective` are the brief's and are unablated. `rank_candidates` does not tune any weights.

## Decisions for Kim

1. Run Stage 0 on the training machine (`probe_metrics_v4.py --render surge`, per-parameter held-out error on A0, FX / no-FX split of the 24 stems) **before** any Stage 2 training. These components make Stage 0 and Stage 1 cheap; they do not replace them.
2. Whether to run Stage 2 (the temporal encoder A/B) after A0/A1, as one extra arm with the same seed. The toy evidence (+0.02 mean R-squared, mostly filter-envelope decay; 7 times the encoder FLOPs) argues for doing Stage 0 first and expecting a modest gain.
3. Whether the Surge-dependent work (enveloped sync/FM) is worth a retrain, after `enumerate_surge_v4.py` and Stage 0.
