# Drum separation on SA3 Medium Base: plan

**Author:** KUANG, for Kim. **Date:** October 23, 2026.
**Status:** plan, plus the gate G0 script (`stable-audio-3` branch `feature/sa3-drum-sep`,
`scripts/drum_sep/codec_ceiling.py`, CPU-tested with fake codecs). Nothing has run on SA3 or a GPU.
Follows `DRUM_SEPARATION_REVIEW.md` (October 20) and Kim's verdict of October 23. [E] = KUANG's estimate.

## 1. Why change course

Kim's listening: BS-RoFormer is good for the whole drum stem. The Inverse Drum Machine's stems sound like crops
around the target sounds, and Separate-and-Detect (MSG-LD) is far from the target.

Both match how the systems work. IDM's masked output is the mix filtered by masks built from its own synthesized
one-shots, which come from the StemGMD training kits. On other kits the masks are only coarse time-frequency
windows around each detected hit, which is a crop of the mix. MSG-LD generates its stems at 16 kHz mono, from a
64-band mel VAE and HiFi-GAN trained on acoustic kits, so its prior has never produced these sounds.

## 2. What SA3 Medium Base brings

- **A 44.1 kHz stereo codec (SAME-L):** a 256-channel latent at 44 100 / 4 096 = 10.77 frames per second.
- **A strong music prior:** a rectified-flow DiT with 24 blocks (about 1.4B parameters, per `EXPERIMENTS.md`),
  pre-APT, with a trained unconditional branch (cfg dropout 0.1) and T5-Gemma text conditioning.
- **Plumbing already in this fork:**
  - the DiT's native modular local-conditioning inlet: a zero-init per-block projection, frame-aligned and added
    per frame. `scripts/mir_control.py` installs it (`install_mir_control`) and meters its use with
    `ControlAblationCallback` (true, shuffled and zero control);
  - the LoRA/DoRA adapter path of `scripts/train_lora.py`;
  - the 257-channel inpaint slot, `local_add_cond` = inpaint mask (1 channel) + masked latent (256 channels),
    projected per layer.

## 3. Design: mixture-conditioned stem generation (MSG-LD's recipe on SA3)

- **Condition:** the drum-bus latent (256 × T) goes through the modular inlet as cond id `mix_lat`, on all 24
  blocks. Per-item dropout is 0.1; zeros through a zero-init projection are exactly the unconditioned model, which
  gives classifier-free guidance on the mix for free.
  - **Arm B, for comparison:** reuse the inpaint slot with mask = 1 and the bus as the masked input. This adds no
    parameters and starts by copying the bus, which is the trivial answer for a dominant stem. The LoRA then has to
    turn "copy" into "extract".
  - Compare A and B at 5 000 steps on held-out velocity loss and on the ablation meter (true vs shuffled vs zero
    mix).
- **Which stem:** a text prompt through the frozen T5-Gemma: "kick drum", "snare drum", "hi-hats", "cymbals",
  "toms", "percussion", optionally followed by the track's style caption. If the text path turns out weak for drum
  pieces, the fallback is a learned stem embedding in the global conditioning.
- **Trainable:** LoRA or DoRA at rank 32–64 on attention and MLP, plus the inlet projection; the base stays frozen.
  Optimizer: AdamW with warmup and cosine decay. SA3's base is AdamW-trained, and LoRA is the case where switching
  to Muon hurts least (arXiv 2605.10468, `papers/knowledge.md`), but the matched optimizer is the safe default.
- **Onset branch (optional, v2):** MSG-LD's best ablation (+OB) adds a training-only onset head. Here it would be a
  LatCH-style head on a mid DiT block predicting the stem's onset envelope. One latent frame is 93 ms, so the head
  needs an upsampler to about 10 ms resolution. The LatCH infrastructure is the template.
- **Inference:** medium-base, 30–50 Euler steps, text CFG 1–2. One pass per stem per 12–24 s window, with overlap
  and a crossfade.
- **Reconciling with the real mix (the fix for "far from target"):** compute Wiener masks from the decoded stems'
  spectrograms and apply them to the real drum bus (α = 1–2). The output is then the mix's own audio and sums to the
  bus; the generation only decides how to split it. Deliver both the generated and the masked stems and let
  listening decide.

## 4. Data

- **Contract:** per window, the bus latent and one latent per stem. Latents are not additive, so remix
  augmentation (random stem gains) has to be re-encoded; store K remixed buses per window (K ≈ 3).
- **Sources:**
  - StemGMD: about 1 200 h, but it renders the same performances through 10 kits, so a subset with 2–3 kits per
    performance (200–300 h) [E] keeps the variety.
  - Synthetic electronic multitracks: the renderer from `DRUM_SEPARATION_REVIEW.md` section 4.
  - Kim's 40 projects: 30 for fine-tuning, 10 held out and never trained on.
  - The Cambridge-MT multitracks.
- **Realistic stage-1 input:** where a full mix exists (Kim's projects, Cambridge-MT), condition on BS-RoFormer's
  drum output instead of the clean bus, with the true stems as targets. The separator then learns BS-RoFormer's
  artifacts instead of meeting them first at inference.
- **Storage:** 256 channels × 10.77 frames/s × 2 bytes ≈ 5.5 kB/s per track. 300 h × (1 bus + 6 stems + 3 remixes)
  ≈ 3 000 h of audio ≈ 60 GB [E].
- **Open question for Kim:** are the 40 stem sets split per drum piece, or one drum bus?

## 5. Compute [E]

- **Encoding** about 3 000 h of audio: SAME-L's encode speed on the desktop is unknown; G0 measures it.
- **Training:** LoRA, 12 s windows of 129 latent frames plus the text tokens, batch 8–16, 20 000–50 000 steps.
  On the order of 1–3 days on the 16 GB desktop GPU, shared with the LatCH queue (`gpu_guard`).
- **Inference:** seconds per stem per window (diffusion), so a 6-minute track with 6 stems is minutes, not seconds.

## 6. Gates, cheapest first

| Gate | What | Cost | Stop or redirect if |
|---|---|---|---|
| G0 | SAME-L round trip on isolated stems: LSD, onset F1 at 20 / 50 ms, attack energy, additivity. Script written: `python -m scripts.drum_sep.codec_ceiling --stems_dir ... --out_dir ...` | about 1 GPU hour | Onset F1 at 20 ms below 0.9, or attack loss above 3 dB, on kick or hats: the latent route caps quality, so only the masked output is usable (or drop the idea for hats) |
| G1 | Zero training: (a) text-only "solo kick drum, 140 BPM" and the like, does the prior make isolated pieces; (b) SDEdit on the bus with stem prompts, `init_noise_level` 0.3–0.7; (c) ZeroSep-style: invert the bus along the reverse rectified-flow ODE under "drums", re-denoise under the stem prompt at CFG 1 (`C:\dev\private\ZeroSep` has the AudioLDM version) | hours | The prior cannot make convincing isolated pieces: a learned stem embedding and more data become necessary |
| G2 | Small fine-tune: kick only, about 20 h (StemGMD plus synthetic), 5 000 steps, arms A and B | about half a day | No arm beats BS-RoFormer's 53-stem kick on the held-out projects |
| G3 | Full 5–6 stem training | 1–3 days | – |

**Baseline at every gate:** BS-RoFormer's MVSep 53-stem model (kick, snare, toms, hi-hat, ride and crash outputs;
in `bs-roformer-infer`). Plus IDM's masked output.

## 7. Evaluation

On the 10 held-out projects, per stem:
- mel_ssim_dist and LSD, for both the generated and the masked output;
- SI-SDR, masked output only;
- onset F1 at 20 and 50 ms;
- energy in regions where the true stem is silent (bleed; IDM's PES).

Then a blind A/B listening test by Kim, which is the deciding measure.

## 8. Options compared

| | SA3 LoRA separator (this plan) | Fine-tune BS-RoFormer on drum pieces | ZeroSep-style on SA3 (no training) |
|---|---|---|---|
| Benefit | Strong 44.1 kHz music prior; open vocabulary through text; reuses the fork's inlet and trainer | True masking separation, consistent with the mix | No data, no training; any prompt |
| Cost | Pair encoding plus 1–3 days of LoRA training; diffusion inference | Training code from the separator repos; days | Inference only, but inversion per stem |
| Risk | Generated stems can drift (mitigated by masking); the 93 ms latent frame (G0) | Bleed and musical noise; needs per-piece data | Weak separation of overlapping hits |
| Reversibility | Adapter only, base untouched | Separate model | Nothing to undo |

**Recommendation:** run G0 and G1 now, about a day of desktop time with no training, and measure the BS-RoFormer
53-stem baseline in parallel. Commit to the SA3 LoRA (G2) only if G0 passes. **The deciding assumption** is that
SAME-L keeps drum transients. If G0 shows it smears them, the latent route can only guide masks, and fine-tuning
a masking separator becomes the better spend.

**Code still to write after G0 and G1** (KUANG): `scripts/drum_sep/build_pairs.py`, which encodes bus, stems and
remixes into the npz contract, and a `--sep-data` mode for `train_lora.py`, which loads the dataset and installs the
`mix_lat` inlet through `install_mir_control`.
