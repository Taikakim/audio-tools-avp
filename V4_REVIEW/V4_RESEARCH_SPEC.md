# Research spec: existing solutions, Python components and papers for the v4 ideas

Date: October 11, 2026. Author: KUANG (Claude Sonnet 5.5) for Kim. For: an AI research agent (web access, a local Python environment).
Read first: `V4_REVIEW.md`, `ERRORS.md` (Part B), and `scripts/synth_inversion/DESIGN_BRIEF_V4.md` on branch `fix/inharmonicity-aux`.

## 0. Purpose and deliverables

The v4 brief wants better attack and decay, better resonance, dynamic sync and FM, and harmonic coherence in a Surge XT sound-inversion system. Our review
(`ERRORS.md` Part B) found that several of its methods do not fit the pipeline. **Find what already exists** (papers, Python libraries, datasets, code) that
could replace or fix those methods, and test the cheap candidates on our data. This research is not on the critical path for A0/A1.

Deliverables, written to `V4_REVIEW/research_output/` (do not commit or push):
1. `V4_RESEARCH_REPORT.md`: one section per question in section 5, in the format of section 3.
2. `candidates.csv`: one row per candidate evaluated (columns in section 3).
3. `search_log.md`: every query, source and date; also what returned nothing (negative results are results).
4. `probe_results/`: the raw tables from section 4 and the scripts that made them.

Order: P1 questions (Q1, Q2, Q3, Q6) first. **Stop after P1 and send the report so Kim can steer**; then P2 (Q4, Q5, Q7, Q8), then P3 (Q9, Q10).
Stop rule per question: three viable candidates evaluated, or a documented negative search across the source types in section 3.

## 1. Rules

1. **No invented sources.** Never give a title, author, year, arXiv ID, DOI, version or licence you did not open. Tag every statement:
   **[V]** you opened the source (paper PDF, README, code, release page) and name it; **[A]** abstract or README only; **[R]** you ran it here;
   **[M]** from memory, unconfirmed. If you cannot open a source, say so.
2. **Quote sparingly and exactly.** A quotation needs its page or section. Prefer a paraphrase plus the table or figure reference.
3. **Licences.** Record the licence of the code and, separately, of any model weights or data. Classify: permissive (MIT, BSD, Apache-2.0), weak copyleft
   (LGPL, MPL), strong copyleft (GPL, AGPL), non-commercial or research-only, unknown or none. Anything other than permissive: write
   "**needs Legal review before vendoring or redistribution**". Do not copy code into the repo; point to it.
4. **No audio leaves the machine.** Never upload stems, renders or clips to a web service, hosted model or API. Evaluate locally only.
5. **Read-only on the repo; CPU only; do not touch any running job.** Install into a throwaway virtual environment, never into `sat-venv` or the
   agent's Essentia environment. Do not push anything.
6. **Our review is a list of hypotheses, not ground truth.** If a source or an experiment contradicts an `ERRORS.md` B-entry, say so and show why.
7. **Separate three things in every recommendation:** what the source shows, what you tested here, and what you infer.
8. **Options** are compared on benefit, cost, risk and reversibility, with one recommendation and the assumption that would change it.
9. **Do not research** the inharmonicity aux head, the JEPA training recipe, the optimizer or A0/A1. You may mention how a finding interacts with them.

## 2. Context pack (all [V] from the code at commit `4dfc5ef` unless marked)

- **Task.** Given a 0.8 s mono clip of a bass note or phrase, predict Surge XT patch parameters. Audio is rendered by the real Surge XT through Pedalboard (VST3);
  44.1 kHz, peak-normalized. Notes are MIDI 28 to 50 (41.2 to 146.8 Hz). FX are OFF in all training renders (`surge_spec_v3.py:48`).
- **Parameter vector.** 48 values (`surge_spec_v3.py`, `PARAM_NAMES`): filter and filter 2, shape, width, sub, oscillator octaves and volumes, **static** `sync` and `fm_depth`,
  unison, cutoff, resonance, keytrack, FEG amount, decay and sustain, AEG decay, sustain and release, waveshaper, drive, chorus and delay mix. Six categorical, the rest continuous in [0, 1].
- **Models.** (1) A conditional flow-matching inverter: a residual MLP whose condition is one vector from its own conv encoder (four stride-2 conv layers, batch norm, then global average
  pooling; `models_v3.py:13-38`). (2) Synth-JEPA (arXiv 2609.31024): a transformer audio encoder (41 tokens plus a CLS token) and a Perceiver parameter encoder, trained jointly, used for search.
  The flow does not use the JEPA embedding.
- **Input features.** Log-mel, 128 bands, `n_fft` 1024, hop 441 (10 ms), 20 Hz to 16 kHz, 81 frames (`audio_utils.py:15`). Bin width 43 Hz.
- **Training.** Losses are in parameter and latent space only. Renders happen in DataLoader workers to make (audio, parameters) pairs. No audio-domain loss, and Surge cannot be differentiated.
- **Inference stack that exists.** Flow draws, then rerank (`inference_v3.predict_and_rerank_candidates`: STFT, pitch, sub, envelope, richness, stereo, modulation and a per-bin contour loss),
  then coordinate-search refinement with real renders (`refine_v3.py`: objective `mss_ref + 10 * (1 - rms_env_cos)`, about 150 renders per note, 3 to 5 s on CPU).
- **Existing evidence base.** `timbre_alignment_check.py` scores audio distances against 21 published listening-test datasets (Hayes' `timbremetrics`, expected at
  `/home/kim/Projects/timbre-dissimilarity-metrics` on the training machine). Spearman with human dissimilarity: log-mel L1 0.473, paper MSS 0.470, our MSS 0.437 (from the `refine_v3.py` docstring).
  Caveat in that script: most datasets are acoustic tones, not synth bass.
- **Real material.** 24 stems (`STEM_CATALOG`, `invert_stem_collection_v3_cpu.py:46`), MuScriptor-transcribed MIDI. 9 of 24 filenames mention reverb, delay, chorus, phaser, flanger, saturation or
  distortion, so the targets contain FX the renderer lacks.
- **Surge access paths.** Pedalboard sets `plugin.parameters[...]` (v3 renders). A surgepy wrapper with `set_mod_routing(source, dest, depth)` exists (`surge_synth.py`); `surge_v4_registry.py` lists 72
  parameters. Whether the VST3 exposes modulation routings to a host is **unverified**.
- **Machines [I from earlier reports].** Training: Linux, ROCm (RX 9070 XT locally, MI250X on LUMI); the agent machine has Python 3.10, Essentia 2.1-beta6-dev, numpy 1.26.4. Laptop: Windows, CPU-only
  Python 3.13 venvs, no Surge, no Essentia. **Every candidate must work on CPU and must not force CUDA-only code**; say if it needs torch GPU kernels and whether ROCm is plausible.
- **Known facts to build on.** Essentia `Inharmonicity` and `HarmonicPeaks` are already used by the aux head. Pedalboard has built-in effects (reverb, delay, chorus, phaser, distortion, compressor).
  `scipy.signal` has `find_peaks`, `peak_prominences` and `peak_widths` [M until you confirm in the docs].

## 3. Candidate card and report format

For each candidate (paper, library, model, dataset or code) give one card; the same fields go in `candidates.csv`.

| Field | Content |
|---|---|
| name, URL, version or commit, date accessed | exact; the commit or release you opened |
| type | paper, library, pretrained model, dataset, repo |
| what it does (one sentence) and the claim it supports for us | |
| evidence tag | [V], [A], [R] or [M] per section 1 |
| licence (code) and licence (weights or data) | with the classification of section 1 |
| install | pip or conda name, Python and numpy constraints, native dependencies |
| CPU, ROCm, CUDA-only | measured or stated by the source |
| differentiable | yes or no, and through what |
| sample rate and input length it expects | |
| cost | seconds per pair or per clip, on which CPU, measured if [R] |
| maturity | last release or commit date, open issues that matter, maintainers |
| fit | what it would replace in our pipeline, and which `ERRORS.md` B-entry it addresses |
| result of our probes (section 4), if run | the table row |
| verdict | adopt as a dependency, reimplement the idea, use for evaluation only, reject (with the reason) |

Per question, end with a **recommendation block**: options compared on benefit, cost, risk and reversibility; one recommendation; the assumption that would change it; the smallest
experiment that would confirm it, with its cost.
Source types to search: arXiv and ISMIR/ICASSP/DAFx/Interspeech/NeurIPS proceedings, Google Scholar forward citations of the papers you find, GitHub and PyPI (search the paper title and key terms),
Papers With Code, the Surge XT and Pedalboard repositories and issue trackers, and the DAFx and ISMIR proceedings for synth-specific work.

## 4. Probe protocol (use for every candidate metric, feature or front end in Q1 to Q4 and Q9)

Run on the training machine or a machine with Surge. If you have no Surge, say so, use Kim's render set if supplied, and otherwise label any result "synthetic stand-in" and say which stand-in.
Use `surge_spec_v3.render_patch` with FX off. Use 3 canonical patches (saw pluck, filtered saw, square) and notes {28, 36, 44, 50}.

| Probe | Sweep (others fixed) | Question it answers | Report |
|---|---|---|---|
| P-attack | `feg_decay`, `aeg_decay` (9 steps each) | Does the metric grow monotonically with the decay difference? | Spearman of distance against \|delta parameter\|, per note, then the mean |
| P-reso | `resonance` and `cutoff` (9 steps each) | Same, for resonance and cutoff, at the low and high end of the note range | same |
| P-sync/FM | `sync`, `fm_depth` (static) | Does it separate buzz from clean? | same |
| P-pitch | detune of 5, 10, 25 cents and octave shifts | Is it overly sensitive to pitch, as published spectral distances can be? [M] | distance as a fraction of the P-attack effect size |
| P-nuisance | level plus or minus 6 dB (before peak-normalization), onset shift of plus or minus 10 ms, and Pedalboard FX (small reverb, delay, chorus, phaser, saturation) at 2 strengths | Does FX or misalignment change the distance as much as a real parameter change? | signal-to-nuisance ratio per effect |
| P-human | `timbre_alignment_check.py` | Does it agree with human timbre ratings? | Spearman per dataset and mean, next to the four existing baselines |

Always report the four existing baselines (`logmel_l1`, `mss_ref`, `stft_ours`, `rms_env`) in the same table. **Do not set pass or fail thresholds.** Kim decides after seeing the table.
Report the cost per pair. Keep scripts re-runnable and record seeds.

## 5. Research questions

Each question gives: why it matters, what to find, seed leads, what to deliver. **Seed leads are from memory ([M]); confirm each exists before use and drop any you cannot find. Do not limit yourself to them.**

### P1: answers needed before any v4 build decision

**Q1. Audio distances and descriptors for sound matching of synth timbre** (B1, B3, B14, B15).
We need an objective that agrees with human timbre judgments better than log-mel L1 (0.473) and is robust to the pitch problem, usable at inference (rerank and refine) and, if differentiable, as a monitoring metric.
Find: papers comparing audio distances for synthesizer sound matching or timbre; libraries implementing them; pretrained audio embeddings usable as distances.
Seeds [M]: `auraloss` (Steinmetz and Reiss); a paper on spectral distances and pitch (Turian and Henry, "I'm Sorry for Your Loss"); `timbremetrics` (already in use); CDPAM and DPAM (Manocha et al.);
CLAP, MERT and DAC/EnCodec embeddings as distances; Frechet Audio Distance variants; the multi-scale spectral losses of DDSP and Masuda and Saito's synthesizer sound matching paper (ISMIR 2021).
Deliver: ranked candidates with the probe table (section 4). Say explicitly whether CDPAM has any evidence for patch-to-patch bass timbre, and record its licence, sample rate and training data (B14).

**Q2. Measuring resonance and spectral envelope on harmonic tones from 41 to 147 Hz** (B12).
`ERRORS.md` B12 argues that a rolling-median prominence on the raw spectrum measures harmonic-to-valley contrast. Find a method that estimates filter-like resonance (peak height above a smooth roll-off, centre
frequency, and bandwidth where it is resolvable) from a harmonic source.
Seeds [M]: `scipy.signal.peak_prominences` and `peak_widths` applied to the sequence of harmonic amplitudes; Essentia harmonic peaks and spectral envelope algorithms; true-envelope and cepstral envelope
estimation; LPC and formant tracking (Parselmouth/Praat, `pysptk`); WORLD CheapTrick (`pyworld`); papers on estimating filter cutoff and resonance of subtractive synths from audio.
Deliver: a recommended definition of resonance height, centre and Q with its failure modes (note range, bandwidth below the harmonic spacing); the P-reso table for each candidate; a statement of the minimum
window length and the effect of `n_fft` 1024 versus 4096 on the estimate.

**Q3. Attack and transient description, and onset alignment** (B13, B15).
Find: attack descriptors (log attack time, temporal centroid, spectral flux at onset, attack slope) that follow `feg_decay`, `aeg_decay` and attack parameters; onset-exaggerating auditory front ends
(adaptation loops, gammatone or ERB filterbanks) in Python; and a robust way to align a MIDI note onset (MuScriptor, error likely tens of milliseconds [I]) with the true onset in a stem.
Seeds [M]: Essentia `LogAttackTime`, `TCToTotal`, `Envelope`, onset detectors; librosa onset functions; `madmom`; Timbre Toolbox ports; Python gammatone or auditory-model packages;
`MoSQITo` (psychoacoustics: sharpness, roughness); DTW or per-note onset snapping.
Deliver: P-attack and P-nuisance tables; for alignment, the error of each method against a known onset on rendered clips with a deliberate shift of 0 to 50 ms.

**Q6. Surge XT: enveloped sync and FM, and the render backend** (B7, B8, B10, B11).
Find: how to create, set and vary a modulation routing (Filter EG to oscillator sync, to FM depth) from Python at training throughput; whether the Surge VST3 exposes modulation routings or depths to a host
(read the Surge source and docs, and test with Pedalboard if Surge is available); other Python hosts (DawDreamer and similar); the patch file format and whether routings can be injected per sample;
the real number of filter configurations in the installed Surge version; throughput of surgepy against Pedalboard.
Seeds [M]: `surgepy` (from the Surge repo, build from source), Pedalboard (`raw_state`, `load_preset`, `plugin.parameters`), DawDreamer, Surge XT patch `.fxp` XML, the Surge manual's modulation chapter.
Deliver: the recommended route with a minimal working snippet, renders per second per core for each route, and an equivalence test plan between Pedalboard and surgepy renders of the same patch (what difference is expected and why).
Say what a v4 parameter vector would need (sources, destinations, ranges) and which destinations Surge allows.

### P2: after the P1 checkpoint

**Q4. Time-frequency front ends for bass** (B4).
Find: front ends with enough low-frequency resolution and attack time resolution for 41 to 147 Hz fundamentals and 30 ms attacks: multi-resolution inputs, CQT or variable-Q, long plus short window pairs.
Seeds [M]: `nnAudio` (GPU spectrograms), `librosa` CQT and VQT, torchaudio transforms, multi-resolution spectrogram inputs in audio classification and sound-matching papers.
Deliver: two or three concrete front ends with shapes, cost, ROCm status and a precedent each; a probe result showing which one separates the P-reso and P-attack sweeps better than 128-band `n_fft` 1024.

**Q5. Temporal-aware conditioning of the flow** (B5, B6).
Find: ways to condition a flow-matching or diffusion model on an audio sequence rather than a pooled vector: attentive pooling, learned queries (Perceiver), cross-attention, FiLM; and precedents in
synth inversion. Include the option of conditioning on the JEPA transformer tokens.
Seeds [M]: Set Transformer pooling, Perceiver and Perceiver IO, DiT cross-attention conditioning, the Synth-JEPA flow conditioning (arXiv 2609.31024, already in the repo), AST and BEATs as encoders.
Deliver: two or three architectures with parameter counts at our sizes (flow width 512, 8 residual blocks: `FLOW_KWARGS`, `train_realistic_bass_v3.py:70`), precedent, and what a minimal swap in `FlowMatchingResMLP` would be. No training required.

**Q7. Training or fine-tuning with a non-differentiable renderer** (B1).
Find: methods to use rendered-audio rewards with a flow model: reward-weighted flow matching, best-of-N distillation, preference optimization for diffusion and flow models, reward fine-tuning, and any
neural-surrogate approach for synthesizers. Include known failure modes (reward hacking, loss of diversity) and typical sample cost.
Seeds [M]: Diffusion-DPO, DRaFT and ReFL, Flow-GRPO, rejection-sampling fine-tuning, neural proxy synthesizers in the sound-matching literature.
Deliver: a recommended method for our case (about 20 to 33 ms per render per core [I from `refine_v3.py`], batch 64), an estimate of the extra step time, and the minimal experiment that would show a gain.

**Q8. The sim-to-real FX gap** (B2).
Find: cheap ways to make the inverter robust to FX it cannot render: FX augmentation of training audio (Pedalboard effects on the rendered target), dereverberation or denoising preprocessing of stems (local tools only),
invariance or contrastive training, differentiable FX chains, and published treatments of the domain gap in sound matching.
Seeds [M]: Pedalboard built-in effects, `dasp-pytorch`, local dereverberation models and their licences, papers on real-recording sound matching.
Deliver: approaches ranked by cost, the P-nuisance numbers for what you can run, and a design for the cheapest first experiment, including a per-stem measure of how much FX alone changes the "decay" descriptors.

### P3: lower priority

**Q9. Objective measures of "buzz", roughness and harmonic coherence** (B9, B15).
Find: measures that tell static sync or FM ringing from a clean sustain: sideband energy relative to a harmonic comb, roughness and fluctuation strength, dissonance and inharmonicity measures.
Seeds [M]: Essentia `Inharmonicity`, `Dissonance`, `HarmonicPeaks`; `MoSQITo` roughness; Sethares-style dissonance; Parselmouth harmonics-to-noise ratio.
Deliver: P-sync/FM table; a recommended single number with its failure modes, and how it relates to the pooled inharmonicity label (so it can serve as the acceptance proxy for "no unmodulated buzz").

**Q10. Evaluation methodology** (B15).
Find: a blind listening protocol and software for comparing two systems on 24 stems with several flow seeds each; the statistics for paired preference; best-of-k evaluation conventions in sound matching.
Seeds [M]: webMUSHRA, BeaqleJS, ABX tooling in Python; the evaluation sections of the Synth-JEPA paper and the sound-matching papers above.
Deliver: a minimal protocol (clips per listener, order randomization, number of seeds, test) and its cost in listener minutes.

## 6. What a good report looks like

- Every P1 question has at least three cards or a documented negative search, the probe table, and a recommendation block.
- Each claim carries its tag. A reader can tell what you opened, what you ran, and what you guess.
- Findings that contradict `ERRORS.md` are stated first.
- Open items for Kim: licences needing Legal, anything that needs Surge or GPU time, and any source you could not open.
