# V6 review: the state of synth inversion, the V6 proposal, and how to implement it

**Owner:** KUANG (Claude Opus 5.5), for Kim. **Date:** October 19, 2026.
**Branch:** audio-tools-avp `feature/v6-ablation`, based on `origin/fix/inharmonicity-aux` at `a079a5f` (Antigravity's V4/V5
pipeline commit and `V6_DISCRIMINATORS_AND_ACOUSTIC_PROPOSAL.md`).
**Evidence tags:** **[V]** verified by reading code or running it here (CPU, laptop); **[T]** covered by a CPU test in
`test_v6.py`; **[I]** inferred, not measured; **[D]** needs the desktop (Surge, GPU, real stems).
Nothing here has run on Surge or a GPU. No result in this document is a listening or Surge measurement.

---

## 1. Summary

1. **The V4 components were a real step forward, but part of the V5 pipeline quietly turned them off.** The A-series
   (A1 inharmonicity aux head, A2 temporal attentive-statistics encoder, A3 both) improved held-out retrieval and ladder
   monotonicity beyond the evaluation's own noise (section 2.1). The V5 trainer reaches those components only when the launch command asks
   for them (`--flow_encoder conv`, `--inharm_weight 0`, `--ladder_every 0` are the defaults), and its bank holds no
   inharmonicity labels and no ladders [V]. Whether the Stage 4/5 runs used them is **unknown from the repository**:
   check each run's `run_meta.json` / training log first (section 6, step 0).
2. **A long-standing bug hid half the parameters from the JEPA in every V3-trainer run** (A-series included) [V][T]:
   the JEPA's parameter encoder took its sizes from the original `surge_spec` (20 continuous, 3 categorical) and the
   V3 trainer never passed its own (43 and 10). It silently ignored the oscillator octaves, oscillator 2 / 3 volumes,
   ring modulation, noise, filter 2, filter routing, FM routing and envelope modes. Antigravity's new shape check in
   `a079a5f` exposed it (6 tests failed on that commit). V4 / V5 trainers pass the right sizes. Fixed for new V3 runs;
   old checkpoints load as trained.
3. **The real-stem evaluation told the model the wrong note** [V][T]: `eval_real_stems_consensus_v5.py` normalized
   the pinned note with 12..96, every layout trains on 28..50. The model heard a note up to an octave above what it was
   told and could only reconcile that through the oscillator octave. This plausibly explains much of the "octave
   transposition drift" in section 1 of the V6 proposal [I]. Fixed.
4. **"Spurious" oscillators, filters and distortion are largely what the training prior teaches** [V]: 60 % of the V4 / V5
   prior draws are unconstrained random patches, where oscillator 2 is on 60 % of the time, oscillator 3 40 %,
   unison 35 %, feedback 35 %, filter routing uniform over 7 configurations, filter 2 always randomized. For an
   ambiguous sound the flow's posterior follows that prior. Discriminator vetoes treat the symptom.
5. **Distortion can be detected where it is audible and only there** (section 4). A label from the waveshaper setting
   is the wrong target. `ablation_v6.py` labels the measured effect of switching a feature off, measures whether
   audio can detect it, and simplifies inverted patches when a feature changes nothing (Occam test).
6. **Flow-DPO is built but has never run**; until this branch it only worked on V3 (92-d) flows. It now follows the
   flow's layout and can rank by a weighted metric sum.

---

## 2. Current state of the inversion code

### 2.1 V4 components (A-series), what the evidence supports

Reported in `V4_EXPERIMENT_REPORT.md` (one seed per arm, 12 000 steps, 512 held-out notes):

| | Baseline | A1 inharm aux | A2 temporal ASP | A3 both |
|---|---|---|---|---|
| held-out audio-to-preset retrieval (online) | 14.5 % | 20.1 % | 19.5 % | 21.9 % |
| flow ladder monotonicity (Spearman rho) | ~0.40 | 0.46 | 0.53 | 0.56 |

- The gain of each arm over the baseline (5 to 7 percentage points) is larger than the evaluation's own spread
  (about ±3.6 points at 95 %, as the report computes) [V: arithmetic]. The differences **between** A1, A2 and A3 are
  inside that spread, so "A3 is best" is not established; "each component helps" is plausible.
- One seed per arm, and the baseline is described as "pre-A1 v3" without a matched rerun: part of the gain may come
  from other changes made at the same time [I].
- All A-series JEPAs saw only 23 of 53 parameters (finding 2). Within the A-series the comparison is fair (same bug
  everywhere); absolute retrieval numbers would likely change with the fix [I].
- The temporal encoder and the inharmonicity head are the parts with evidence. DualMel, two-anchor objective and the
  resonance card are built and unit-tested on a stand-in, not trained or validated on Surge (`V4_COMPONENTS.md`).

### 2.2 V5 Stage 4 / 5 (`train_realistic_bass_v5.py`, 126-d codec)

- **Layout:** 60 stored parameters; osc 1 / 2 / 3 octaves became 7-class categoricals (126-d encoding). The codec and
  the spec digitize octaves with Surge's thresholds; rendering used plain rounding, which disagreed with the labels near
  a boundary (1.4 % of random-patch draws, computed) [V][T]. Fixed: the spec now uses the same thresholds.
- **Data:** live mode could not start: it used an unimported `RealisticBassPrior` (a V3-layout prior) [V]. Fixed
  (soundset prior). Bank mode works, but `render_bank_v4.py` stores zero inharmonicity labels (marked invalid), has
  an empty ladder dataset, and the trainer checked only the bank's format number [V]. Fixed: the bank can store real
  labels (`--inharm_config`) and records manifold md5s; the trainer refuses a bank that cannot serve the run.
- **Glitch redraws in the bank** reused one seed per MIDI note, so every glitching item on a note got the same
  replacement patch, and the stored MIDI note was the original, not the redrawn one [V]. Fixed (seed from the item's own
  vector; stored note = rendered note; source = "redraw").
- **Live and bank data differ** [V]: the trainer's own `sample_unconstrained_random_patch` (used for
  `--random_patch_frac` in live mode) turns the waveshaper on in 7 of 8 draws; the bank's prior in 1 of 10. Not changed;
  decide which one you want before comparing live and bank runs.
- **Validation** uses the soundset prior's val split, which is also 60 % fresh random patches [V]: "held-out presets"
  describes only 40 % of it.
- **Acoustic loss (Stage 5)** [V]: `1 - cos(JEPA predictor(params(x_hat_1)), z_a)`, applied at every flow time t with
  soft categoricals `softmax(x / 0.5)`. Two problems: at small t, `x_hat_1` is a blurred average, not a patch, so the
  loss pulls the velocity field toward acoustically "safe" averages; and soft categoricals are never the one-hots the
  JEPA was trained on. Added, off by default (Stage 5 reproduces exactly): `--acoustic_t_min`, `--acoustic_t_weight t`,
  `--acoustic_cat st` (straight-through one-hot) [T]. **The claim that Stage 5 "solved the muffled filter cutoff" has no
  measurement in the repository**; it needs the paired audio evaluation of section 6.

### 2.3 Real-stem evaluation (`eval_real_stems_consensus_v5.py`)

Fixed [V][T]: note range from the model's layout, out-of-range notes folded by octaves with the oscillators shifted
back; the second phrase note is the stem's own note an octave up when transcribed (pitch shifting only as fallback, it
adds phase-vocoder smearing the model never heard); per-candidate and winner oscillator-1 octaves recorded; Antigravity's
private scratch directory removed from `sys.path`; optional `--occam_tol`. Still true: the evaluation reranks by a
hand-weighted phrase score (MSS plus stereo and sub penalties) that has not been checked against listening.

### 2.4 Flow-DPO (`dpo_pairs_v5.py`, `train_flow_dpo_v5.py`, `flow_sampling_v5.py`, `audio_eval_v5.py`)

- Written October 13, 2026, CPU-tested, **never run on real pairs** [V].
- Was hard-wired to the V3 layout (codec, note range, render spec, prior) [V]. Now follows the flow's layout
  (`layout_v6.py`), so it runs on Stage 4 / 5 flows; `--metric "mss=1,modspec_dist=0.5"` ranks by a weighted sum and
  uses the same weighted render-noise floor as the margin [T].
- V6 step 1's composite `MSS + 0.5 D_contour + 1.0 D_harmonic`: `D_contour` exists only inside `inference_v3.py`
  (`compute_bin_energy_contour_loss`), `D_harmonic` does not exist in the code. Use metrics that passed the ladder gate
  (`mss`, `logmel_l1`) plus `modspec_dist` (the only one that sees detune) until the others are defined and gated.

---

## 3. The V6 proposal, section by section

| V6 item | Assessment | Recommendation |
|---|---|---|
| D1 distortion probe, label `ws > 0 or drive > 0.15` | Label from the setting; includes inaudible cases (soft shaper at low drive). Physics: on a steady periodic tone a waveshaped wave and another oscillator shape can give the same harmonics, so no detector can separate them there [I: standard nonlinear-systems result; verify per Surge config with the probe] | Label by audible effect; measure detectability per stratum; Occam test at inference (section 4) |
| "Immune to ring-modulation sidebands" | A hope, not a property: a learned probe uses whatever correlates | Measure with `ring_off` labels |
| D2 multi-oscillator, gate clamps osc 2 / 3 when P < 0.15 | Same label issue; threshold uncalibrated | Occam test does this exactly, no threshold to tune |
| D3 sub-oscillator, `osc_2_octave < 0` | Octaves are absolute per oscillator: osc 2 at -1 with osc 1 at -1 is not a sub | Label relative to osc 1, or by audible effect |
| D4 odd-only (square vs saw) | Ignores pulse width (only 50 % is odd-only) and asymmetric shapers (add even harmonics) | Drop as a gate; EOHR is fine as a descriptor |
| Lock osc 1 to ±1 octave | Right direction, half a fix: the flow predicts both the note (28..50) and osc 1's octave, so several pairs give the same pitch. That ambiguity, plus finding 3, is the octave drift [I] | Fix osc 1 at octave 0 and give the note to the flow as an input (section 5) |
| Detune ±0.2 semitones in 20 % of draws | Sensible. Log-mel distances are blind to it (October 13 finding); only `modspec_dist` sees it. `osc_2_pitch` already spans ±1 semitone in the spec; the ±0.2 mapping must be checked by render | Keep, score with `modspec_dist`, verify the mapping on Surge |
| Pitch head (128 classes + cents) | 22 of 128 classes ever occur; cent target is 0 except detune, so the fine head learns a constant; "frees z_a for timbre" is unsupported, it may as well make z_a pitch-dominated | Use the known note as a flow input instead |
| Flow-DPO first | Fine, after the evaluation fixes; check headroom before training | Section 6, step 4 |

---

## 4. Can a distortion discriminator be built?

Yes, for distortion that is audible; no, for the part that is not, and that part does not need detecting.

- **Ground truth by counterfactual render** (`ablation_v6.py labels`): render each patch twice with everything equal
  except the feature (waveshaper off, osc 2 off, ...), and once more unchanged for Surge's render-to-render noise.
  `delta = d(on, off)`, `audible = delta > k * noise` (default k = 2, plus an optional absolute floor). The summary
  says, per feature, how often the prior turns it on and how often it is then audible.
- **Detectability** (`ablation_v6.py probe`): an L2 logistic probe, 5-fold cross-validated, on clip-level log-mel
  statistics and, with `--jepa`, on the JEPA audio embedding V6 would put its heads on. AUROC (0.5 = chance) of the
  audible label, overall and per stratum (single oscillator vs multi / unison, static vs swept filter), next to the
  AUROC of the parameter label. Expected [I]: high AUROC where the waveshaper meets detuned or multiple oscillators
  (intermodulation off the harmonic comb) or a swept filter; near chance on a single static oscillator.
- **Use at inference: Occam test, not a learned veto** (`ablation_v6.py occam`, `eval_real_stems_consensus_v5.py
  --occam_tol`): for the chosen patch, try switching off each active feature (waveshaper, osc 2, osc 3, sub, unison,
  ring, noise, feedback, detune, drift); keep the removal that hurts the score least if the score rises by at most
  `max(tol_abs, tol_rel * score)`; repeat. Exact, needs no training, and directly addresses "distortion on clean plucks"
  and "extra oscillators on single-oscillator sounds". Cost: about 10 renders per round per item.
- **Only if the probe shows high AUROC in a stratum** is a learned head worth adding, and then as a training-time
  auxiliary loss on audible labels, not as an inference gate.

---

## 5. How to implement V6 correctly (design)

1. **Note as input, osc 1 octave fixed at 0.** Add the MIDI note (normalized) as a conditioning input to the flow
   (like `tau`), and remove `midi_note` and `osc_1_octave` from the predicted vector (osc 2 / 3 octaves become relative
   to osc 1, which they already are acoustically). At inference the note comes from the transcription (Muscriptor),
   as now. Out-of-range notes: fold by octaves (`layout_v6.fold_note`). This removes the note/octave ambiguity instead
   of narrowing it. Needs a new layout (v6 spec / codec), a new bank, a retrain.
2. **Simplicity-biased prior.** Replace the 60 % unconstrained random share by a sparse random patch: each optional
   feature on with probability about 0.2 to 0.3, filter routing Serial 1 in most draws, filter 2 off unless routed.
   Use `ablation_v6.py labels` on the old and new prior to check the share of features that are on but inaudible
   drops. Keep a small share (5 to 10 %) of fully random patches for coverage.
3. **Bring back the A-series components explicitly**: `--flow_encoder temporal_asp`, `--inharm_weight 100` with a
   labelled bank (`render_bank_v4.py build --inharm_config ...`), ladders in live mode or a ladder bank (none exists for
   V4 / V5 yet).
4. **Acoustic loss as an ablation arm, corrected:** `--lambda_acoustic L --acoustic_cat st --acoustic_t_min 0.5`
   against the same run without it, same seed and bank. Keep it only if the audio evaluation says so.
5. **Detune** as proposed, scored by `modspec_dist`.
6. **No discriminator gates.** Occam test at inference; probe-driven auxiliary heads only where detectable.
7. **Flow-DPO** on the chosen base, pairs ranked by a gated metric sum, beta sweep {30, 100, 300}.

---

## 6. Plan for the desktop (Antigravity runs, KUANG writes code)

Each step has a gate. Paths: `S=~/Projects/SAO/stable-audio-tools/scripts/synth_inversion`,
`B=/home/kim/sao_runs/real_manifold_v4_bass.npz`, `A=/home/kim/sao_runs/real_manifold_v4_all.npz`.

**Step 0: what did Stage 4 / 5 actually train with?** (5 minutes, no compute)
For each Stage 4 / 5 run directory: `run_meta.json` (`args`: `flow_encoder`, `inharm_weight`, `ladder_every`, `bank`,
`lambda_acoustic`, `acoustic_temp`) and the training log's `Aux ... (v=...)` field (v = 0.00 means the inharmonicity
head trained on nothing). Report the table to Kim. Gate: we know which A-series components each run had.

**Step 1: tests on the desktop.** `cd $S && python -m pytest test_v6.py test_v5.py test_eval_v4.py test_v4_components.py -q`
(laptop, October 19, 2026: 115 pass = 16 + 32 + 26 + 41, with the essentia / mido fakes test_v6 installs; run test_v6 first).

**Step 2: re-run the real-stem evaluation with the fixes.**
```
python eval_real_stems_consensus_v5.py --model A3 <A3 flow> --model S4 <stage 4 flow> --model S5 <stage 5 flow> \
    --out_dir ~/sao_runs/stems_v6_fixed --device cuda:0
python eval_real_stems_consensus_v5.py ... --occam_tol 0.02 --out_dir ~/sao_runs/stems_v6_occam
```
(A3 is a 48-parameter checkpoint: it needs the worktree at `aa734a8`, as noted on October 15; run it there or drop it.)
Read: `osc1_octave_candidates` / `osc1_octave_winner` per stem (does the drift persist with the right note?),
`notes[].second_note` (real vs pitch shift), `occam` (what was removed, score change). Kim listens to both packs.
Gate: if octave drift is gone, V6's octave lock is not urgent; if Occam removals are common and inaudible, the
discriminator gates are not needed.

**Step 3: ablation labels and detectability.**
```
python ablation_v6.py labels --layout v5 --bass_manifold $B --all_manifold $A --n 4096 --enrich waveshaper_off=0.5 \
    --metric mss --workers 14 --out ~/sao_runs/abl_v5
python ablation_v6.py probe --labels ~/sao_runs/abl_v5 --target waveshaper_off --jepa <stage 5 jepa_ema_latest.pt> --device cuda:0
python ablation_v6.py probe --labels ~/sao_runs/abl_v5 --target osc2_off
```
Read: `summary.json` (on share vs audible-when-on per feature; render-noise median) and `probe_*.json` (AUROC per
stratum). Gate: decide whether any learned head is worth it (AUROC well above 0.5 in a stratum that matters).

**Step 4: Flow-DPO headroom on the Stage 5 flow** (now layout-aware).
```
python dpo_pairs_v5.py --flow <S5 flow_ema_latest.pt> --bass_manifold $B --all_manifold $A --source synth \
    --n 4096 --k 8 --metrics mss logmel_l1 modspec_dist --workers 14 --out ~/sao_runs/dpo_pairs_s5 --device cuda:0
```
Gate: `best of 8` clearly below `candidate 0` and the spread above render noise in most conditions (the script prints
both). Then `train_flow_dpo_v5.py --metric "mss=1,modspec_dist=0.5" --beta 100` (and 30 / 300), evaluated with
`audio_eval_v5.py --bass_manifold $B --all_manifold $A --flow S5=... --flow S5DPO=...`.

**Step 5: V6 training (after KUANG writes the v6 layout of section 5).** Arms at small size, same bank and seed:
(a) V5 settings with A-series components on; (b) + note-as-input / osc 1 fixed; (c) + simplicity prior;
(d) + corrected acoustic loss. Two seeds for the arm that wins. Gate: paired audio metrics (`audio_eval_v5.py`) and
the real-stem pack, then a blind listening test (`listening_test_v5.py`).

---

## 7. What changed in this branch

| File | Change | Test |
|---|---|---|
| `layout_v6.py` (new) | flow width -> spec / codec (widths read from the codecs: 92 / 108 / 126); note range; octave folding and shift | [T] |
| `ablation_v6.py` (new) | counterfactual labels, Occam simplification, AUROC, cross-validated probe, strata; CLI labels / probe / occam | [T] fake renderer |
| `eval_real_stems_consensus_v5.py` | note range, octave folding, real second note, octave logging, `--occam_tol`, no scratch path | [T] helpers |
| `train_realistic_bass_v5.py`, `_v4.py` | live mode uses the soundset prior; real bank checks; V5: `--acoustic_cat`, `--acoustic_t_min`, `--acoustic_t_weight` (defaults = Stage 5) | [T] |
| `render_bank_v4.py`, `render_pool_v4.py` | `--inharm_config` labels, manifold md5s, `check_compatible` -> (hard, soft); redraw seed per item; stored note = rendered note | [T] checks |
| `surge_spec_v5.py` | octave class by Surge's thresholds (matches the codec labels) | [T] |
| `synth_jepa_model.py`, `synth_jepa_search_v3.py`, `train_realistic_bass_v3.py` | legacy truncation explicit; old checkpoints load as trained (warning); new V3 runs see all parameters; resume refuses to mix (`--legacy_jepa_params`) | [T] |
| `flow_sampling_v5.py`, `render_pool_v5.py`, `dpo_pairs_v5.py`, `train_flow_dpo_v5.py`, `audio_eval_v5.py` | layout-aware; DPO weighted metric; audio_eval soundset items with honest clusters | [T] + existing test_v5 |
| `test_eval_v4.py` | its tiny JEPA now uses the V3 sizes | |

Not done here: the v6 layout (note as input, osc 1 fixed), the simplicity prior, a ladder bank for V4 / V5. They change
the training data and need Kim's go after steps 0 to 3.
