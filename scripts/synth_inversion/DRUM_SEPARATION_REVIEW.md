# Drum separation: review of the test plan, the papers, and a 44.1 kHz path

**Author:** KUANG, with Kim's input. **Date:** October 20, 2026.
**Reviews:** `DRUM_SEPARATION_TEST_PLAN.md` (Antigravity, commit `0bf7946`).
**Status:** review and plan only. No model has been downloaded or run, and nothing here is measured.

Tags: [P] = stated in a paper, [C] = read in the code or configuration, [E] = KUANG's estimate, not measured.

## 1. Sources read

| System | Paper | Code (cloned to `C:\dev\private`) | Weights |
|---|---|---|---|
| BS-RoFormer (stage 1: drums out of the full mix) | none (ByteDance architecture, community checkpoints) | `openmirlab/bs-roformer-infer` | default SW 6-stem: license listed as **unknown** by the repo [C] |
| Inverse Drum Machine (IDM) | Torres, Peeters, Richard, IEEE TASLP 2026, arXiv 2505.03337 | `bernardo-torres/inverse-drum-machine` | `idm-44-train-kits` inside the repo [C] |
| Separate-and-Detect (MSG-LD) | Hsu et al., ISMIR 2026, arXiv 2608.01093 | `ddman1101/Separate-and-detect` | Hugging Face `ddman1101/Separate-and-Detect` (OB+TB) |
| MSG-LD base | Karchkhadze et al., ICASSP 2025, arXiv 2409.12346 | – | skimmed only |

The arXiv HTML versions lost the numbers inside formulas, so some IDM details (parameter count, STFT sizes) are not
checked here.

## 2. What the plan gets wrong or leaves open

1. **IDM's synthesized stems are training-kit sounds.** Its timbre input is a drum-kit classifier over the StemGMD kits
   it was trained on, which "restricts the model to synthesize only from drum kits observed during training" [P].
   On a 909 or psytrance kick, the synthesized `KD.wav` is an acoustic StemGMD kick. For Phase 4 (kick inversion),
   use the Wiener-masked output (the real audio, `masking='wiener'`) or only the transcription, never the synthesized
   stem.
2. **IDM expects a drums-only, mono input,** with one-shots one second long. The authors blame that length for weaker
   cymbal and hi-hat results [P]. Phase 3's "IDM directly on the full mix" is outside its training conditions:
   treat it as a sanity check, not a baseline.
3. **MSG-LD generates its stems; it does not separate them.** It runs at 16 kHz mono, with 64 mel bands covering
   30 Hz–8 kHz, and a HiFi-GAN vocoder renders the audio [P][C]. The paper deliberately reports mel-MSE and FAD
   (Fréchet Audio Distance), not SDR (signal-to-distortion ratio) [P]. The plan's BSS-Eval SDR is the wrong measure
   for it. Its stems cannot be inversion targets above 8 kHz. Its strengths are kick and snare onsets (MDB kick F1
   0.931 vs ADTOF 0.851 [P]) and clean silences. It is slow: about 25 s per 10 s of audio on an RTX 6000 Ada [P].
4. **The test sets are not all held out.** IDM was trained on StemGMD, and MSG-LD on StemGMD + IDMT-SMT-Drums [P].
   Only MDB-Drums and ENST-Drums are unseen by both.
5. **"24 in-house electronic loops" have no ground-truth stems** unless they come from Kim's projects (section 4).
6. **Missing scripts:** `test_idm_separation.py` and `run_drum_eval.sh` do not exist yet. All paths in the plan are
   desktop paths.
7. Phase 2's check of bass bleeding into the kick stem is the most useful step for psytrance, where kick and bass
   interleave. Keep it.

## 3. Can a 44.1 kHz Separate-and-Detect be trained?

Yes. The 16 kHz limit is not in the diffusion model. It is in the two frozen pretrained parts it sits on: the MusicLDM
VAE (latent 256 × 16, 8 channels) and the 16 kHz HiFi-GAN vocoder [C, `config/MSG-LD/integrated_musicldm.yaml`].
Changing the sample rate means replacing both.

- **Latent space:** retraining a MusicLDM-style VAE and vocoder at 44.1 kHz means two models to train, each a GAN.
  The cheaper route is the Stable Audio VAE already used in Kim's projects: 44.1 kHz stereo, decoding directly to
  a waveform (no vocoder), about 21.5 latent frames per second. Keep the MSG-LD recipe (the mixture latent as the
  condition, five stem latents denoised together, the onset branch) and replace the 2-D U-Net with a 1-D DiT
  (diffusion transformer). A 10 s clip is about 215 latent frames, which is short [E].
- **Data:** StemGMD (about 1 200 hours, 9 classes) and IDMT-SMT-Drums are both available at 44.1 kHz. LarsNet was
  trained on StemGMD in stereo. The StemGMD download is probably on the order of a terabyte of WAV; check before
  downloading [E]. Precomputed latents take about 2.8 kB/s per track at 16-bit, so 6 tracks × 1 200 h ≈ 70 GB [E].
- **Compute:** the paper does not report training time. It gives a 305M-parameter U-Net at batch size 3 on one GPU,
  and a commented-out checkpoint path in the config says epoch 14 [P][C]. Estimate: a 300M DiT on precomputed
  latents needs 100k–300k steps at batch size 16–32. That is several days to two weeks on one 24–48 GB GPU, plus a
  day or two to encode the latents [E]. Inference stays iterative: seconds per clip.

**Options compared**

| | Diffusion at 44.1 kHz (MSG-LD-style) | Mask-based separator (BS-RoFormer or similar) trained on StemGMD + electronic data |
|---|---|---|
| Benefit | Clean silences; best kick and snare onsets in the paper | Real audio, consistent with the mix; full band, usable as an inversion target |
| Cost | VAE decision, DiT training, slow inference | Existing training code; days on one GPU |
| Risk | Stems are generated, so transients and timbre can drift; hard to evaluate | Bleed between stems, musical noise |
| Reversibility | High sunk cost | Easy to swap or fine-tune further |

**Recommendation:** start with the mask-based separator, at 44.1 kHz. Mezza et al. (IEEE IS2 2024) benchmarked
general separators on StemGMD, which gives a reference to compare against. Treat diffusion as a second step, for
transcription rather than as the source of inversion targets. **The deciding assumption** is what the stems are
for. If the goal is MIDI and kick and snare onsets, MSG-LD's strengths count for more. If the goal is sounds to invert
into Surge patches, the real waveform is needed, and the separator gives it.

## 4. Training data: synthetic renders and Kim's 40 stem sets

StemGMD is acoustic kits (played on an electronic kit, rendered through kit samples), with no 909s, no psytrance
kicks and no sidechain. A model trained on it alone will do poorly on electronic tracks. Kim's concern is that
synthetic renders lack the sound of professionally produced tracks; that is right for naive renders. The answer is
to use both sources, each for what the other lacks.

**What naive synthetic renders miss, and what can be done about it**
- **Processing on the hits** (saturation, EQ, transient shaping): mostly solvable. Electronic producers mostly use
  pre-processed sample packs, so renders from commercial-grade one-shots and Surge-made kicks already carry much of
  the sonic print.
- **Bus and master processing** (glue compression, sidechain pumping, limiting): solvable, but the training target
  needs care. These processors break "stems sum to the mix". Because they are gain changes, compute the gain curve
  from the processed mix and apply the same curve to each stem; the targets then sum exactly to the processed mix.
  Bus saturation is the one part this cannot make exact: keep it mild, or put it on the stems instead.
- **Interplay with the rest of the track** (bass under the kick, masking, shared reverb, arrangement): the hard part
  to fake. It matters most for stage 1 (drums out of the full mix), less for splitting a drum stem into kit pieces.
- **Groove:** easy for quantized electronic music; Groove MIDI covers human timing.

**Kim's 40 project stem sets (about 4 hours)**
1. **Held-out test set:** keep about 10 projects aside, never trained on. Without it, nobody can tell whether the
   synthetic data transfers.
2. **Fine-tuning after synthetic pre-training,** with stem-remix augmentation: random gains, tempo and pitch shifts,
   stems combined across songs. The timbres stay Kim's palette and genre, which only matters if the model should
   work beyond that style. Combining across songs breaks the kick and bass sidechain timing, so keep some original
   combinations.
3. **Calibrating the renders:** measure Kim's real drum stems (loudness, crest factor, spectral tilt, stereo width,
   compression depth) and draw the synthetic processing settings from those distributions.

**Other real multitracks:** the Cambridge-MT "Mixing Secrets" library (hundreds of free multitracks with separate
kick, snare and overhead tracks; mostly band recordings), and part of MedleyDB. MUSDB has only one drum stem, so it
serves stage 1 only. Training-use terms for these and for commercial sample packs have not been checked; check
them before building a dataset.

**Open question for Kim:** are the 40 stem sets split per drum piece (kick, snare, hats), or one drum bus? Per-piece
stems can train the kit-piece split. A drum bus helps stage 1 only.

## 5. Next steps

1. **Cheapest decisive experiment:** train one 44.1 kHz separator twice, on synthetic data only and on synthetic data
   plus 30 of Kim's projects, and test both on the 10 held-out projects. A small gap means synthetic data carries the
   load and can be scaled. A large gap means more real multitracks are needed. Cost: a few days of GPU time [E].
2. **Code, CPU-testable here:** a synthetic drum-multitrack renderer. It takes MIDI patterns and plays them through
   one-shot packs or Surge patches, applies per-stem and bus processing (with the gain-curve-consistent targets from
   section 4), and writes per-stem outputs and a manifest. Either model needs it.
3. **The plan's Phase 1** (the IDM checkpoint on 5 loops) and **Phase 2** (BS-RoFormer drums, the 53-stem model) can
   run on the desktop as written, with the changes in section 2: for IDM use the masked output, and for MSG-LD use
   mel-MSE, FAD and onset F1 instead of SDR.
