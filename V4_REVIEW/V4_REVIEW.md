# Review of DESIGN_BRIEF_V4.md: feasibility, expected benefit, gaps, improved plan

Date: October 11, 2026. Reviewer: KUANG (Claude Sonnet 5.5) for Kim. Companion files: `ERRORS.md` (the register, IDs `A#` and `B#`),
`V4_RESEARCH_SPEC.md` (what to look up). Evidence tags as in `ERRORS.md`: [V] verified, [I] inferred, [M] from memory.

## Verdict
The brief identifies real problems and some of its diagnoses check out. As written it cannot be implemented as a training-loss
overhaul, because its three headline losses compare rendered audio and Surge XT is not differentiable (B1). A staged version is feasible
and worth doing, and its first stage costs no GPU time.

## What checks out [V]
- The flow's own audio encoder pools all time frames into one vector (`models_v3.py:13-38`), so *when* things happen is lost (B6 qualifies "cannot perceive").
- v3 sync and FM are static parameters; a surgepy renderer with modulation routings already exists (`surge_synth.py`), so enveloped sync and FM are reachable.
- Inference-time use of audio-domain objectives already exists (`refine_v3.py`, `inference_v3.py`), so two-anchor and contour terms can be tried without retraining.

## Expected benefit per item (estimates; nothing here was measured)
| Item | Expected benefit | Cost | Main risk |
|---|---|---|---|
| Attack and sustain aware conditioning with a better spectrogram (B4, B5, B6) | Modest on the toy test (+0.02 mean R-squared, mostly filter-envelope decay, none on attack; stand-in synth, not Surge); unknown on Surge. Larger only **if** in-distribution decay and filter-envelope errors are large | One training run per variant | The cause may be the FX gap (B2), not the encoder |
| Two-anchor and contour terms in rerank and refine | Medium | Low, no retraining | Metrics agree with humans only about 0.44 to 0.47 (B14) |
| Enveloped sync and FM (new parameters, surgepy backend) | High artistic value for goa and acid | High: backend, prior, retrain (B7, B10) | Identifiability; no like-for-like comparison with v3 |
| Prominence loss as written | Low | Medium | Measures harmonic-to-valley contrast (B12) |
| CDPAM | Low, unsupported (B14) | Medium | Licence and domain |
| Harmonic-ratio snapping | Uncertain | Low as a listening test | Removes intentional detune (B9) |

## Missing from the brief
1. A diagnosis of where the errors are: per-parameter held-out error on synthetic data, and the 24 stems split by FX present or absent (B2).
2. A baseline of v3 plus rerank plus refine at a named checkpoint size (B16).
3. A decision on the input representation (B4).
4. Any mention of the JEPA, or of the inharmonicity aux head, which already measures what the brief wants for "buzz" (B6, B9, B15).
5. An ablation ladder: the brief bundles about six changes (encoder, losses, parameters, backend, prior, reranking).
6. Measurable acceptance criteria with a synthetic-oracle noise floor and blind listening (B15).

## Improved plan (staged, each stage gated)
- **Stage 0, CPU only, no new training.** (a) Per-parameter held-out error from the A0 checkpoint (feg_decay, aeg_decay, resonance, sync, fm_depth).
  (b) Tag the 24 stems FX or no FX and compare errors. (c) Run candidate metrics and features through `timbre_alignment_check.py` and the three sweep probes in `V4_RESEARCH_SPEC.md` section 4.
  (d) Blind A/B on the harmonic-snapped renders that `render_muscriptor_top3_harmonic.py` already produces.
- **Stage 1, inference only.** Two-anchor and contour terms in rerank and refine. Gate: blind A/B on the 24 stems, several flow seeds per stem, paired comparison.
- **Stage 2, one training change.** Two-window spectrogram front end plus attack and sustain pooling in the flow's encoder; parameters and prior unchanged; compare with A0 at equal steps. Implemented and unit-tested (not trained on Surge data) on branch `feature/v4-components`: `models_v4.py`, `frontend_v4.py`, `apply_flow_encoder_patch_v4.py`; see its `V4_COMPONENTS.md` for the stand-in results and limits.
- **Stage 3.** surgepy backend with an equivalence test against Pedalboard, then enveloped sync and FM with a defined prior. Only if Stage 0 shows static sync and FM causing the artifact
  (use the inharmonicity or sideband measure).
- **Stage 4, optional.** A short rendered-reward fine-tuning stage, if Stage 1 shows the sampler, not the search, is the limit.

## Options
| Option | Benefit | Cost | Risk | Reversibility |
|---|---|---|---|---|
| A. Build the brief as written | Unknown | Highest | High: non-trainable losses, backend swap, no attribution | Low |
| **B. Staged plan above (recommended)** | Targets what the diagnostics show | Moderate | Low | High |
| C. Inference-only | Moderate | Lowest | Lowest | Full |

Assumption that would change the recommendation: if Stage 0 (a) shows the in-distribution decay and attack parameter errors are already small, drop Stage 2 and take C plus a fix
for the FX gap. If they are large, B stands.

## Sequencing with A0/A1
Stage 0 reads the A0 checkpoint and needs no GPU. Do not start Stage 2 or 3 until A0/A1 are done: the aux head effect and a v4 encoder change would otherwise be confounded.
The A0/A1 launch has blocking problems of its own (`ERRORS.md` A1 to A3); fix those first.
