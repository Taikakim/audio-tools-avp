# SAME / Stable Audio 3 chroma — handoff for training a control head

*Self-contained note. Everything here is needed to (a) compute the **exact** chroma targets the SAME autoencoder was trained against, and (b) train a control head that reads/steers that chroma. Confidence is marked per claim: **[verified-from-source]** = decoded from `stable-audio-tools` @ commit `3241adb`; **[inferred]** = our reconstruction or from offline information, not paper-stated.*

---

## 0. What this is and why it matters for a control head

The SAME autoencoder (Stability AI, arXiv 2605.18613 §3.3.2) — the VAE under Stable Audio 3 — was trained with auxiliary **semantic-regression heads** that predict a chromagram from the latent. Those heads were a *regularizer* to shape the latent, **not** a control interface **[inferred]**. The payoff for us: **chroma is, by training objective, linearly decodable from the SAME latent**. So a small head (linear, or a light transformer) can read pitch-class content out of the latent, and — run backward as guidance — push generation toward chosen pitches.

To train such a head you must compute the target chroma **bit-for-bit the way SAME did**, or the head sees out-of-distribution targets and learns garbage. Section 1–3 is that recipe. Section 4–6 is the head itself.

---

## 1. The exact target recipe [verified-from-source]

Source: `stable-audio-tools/training/autoencoders.py:567-597` + `models/transforms.py`, commit `3241adba4fc2a85cf5b29d9eb68d42f40a28e820`. Pipeline:

```
TightSpectrogram(n_fft=8192, normalized=True, power=1.0)   # hop FORCED to 4096 (= n_fft//2)
  -> per-channel |STFT|, then MEAN over channels            # NOT mono-mixed first
  -> log1p
  -> ChromaScale(sample_rate=44100, n_chroma=128, n_freqs=4097,
                 ctroct=center, octwidth=width, norm=1, base_c=True)
       for (center, width) in [(1.0, 1.0), (5.0, 1.5), (9.0, 1.0)]
  -> F.interpolate(target, size=n_latent_frames, mode='linear')
```

Output: **`(3, 128, T)`** float32 — three register-resolved chromagrams, at the SAME latent rate.

Constants (copy verbatim):
```python
SR            = 44100
N_FFT         = 8192
HOP           = 4096          # n_fft // 2, forced by TightSpectrogram
N_FREQS       = 4097          # n_fft//2 + 1, one-sided
N_CHROMA      = 128           # bins PER BAND (not 12)
CHROMA_CENTERS = (1.0, 5.0, 9.0)   # octave centers (ctroct)
CHROMA_WIDTHS  = (1.0, 1.5, 1.0)   # octwidth per band; mid band is widened
LATENT_HOP    = 4096          # = STFT hop, so chroma frames are 1:1 with latent frames
```

The three bands are **register-resolved** chroma. Octave reference is A0 = 27.5 Hz, so centers 1/5/9 sit at **55 Hz / 880 Hz / 14080 Hz** = **bass / mid / brilliance** (≈ A1 / A5 / A9). A plain folded chromagram is octave-invariant (bass C == treble C); the three octave-windowed bands restore a coarse register axis, and the mid band is widened to 1.5 octaves over the densest musical region **[inferred rationale; the paper states the three regressors but gives no reason for the band count or centers]**.

---

## 2. The seven subtleties — each silently breaks compatibility if wrong [verified-from-source]

1. **C is at bin 2.0, NOT bin 0.** With `base_c=True` and `n_chroma=128`, the code rolls by `-3*(128//12) = -30` bins, but the exact A→C offset is 32 bins, so it lands 2 bins off. Each semitone spans `128/12 ≈ 10.667` bins (non-integer). Pitch class `p` (C=0) sits at bin `(128*(p+3)/12 - 30) mod 128` → **C=2.0, A=98.0**. Never assume bin 0 = C, and never assume integer bins-per-semitone.
2. **No per-frame normalization.** `norm=1` only L1-normalizes the *filterbank rows*. The targets are unbounded `log1p`-compressed energy values that scale with signal level. ⟹ **train and steer with relative objectives (cosine / dot product), not absolute target matching.**
3. **`torch.stft(normalized=True)` divides by `sqrt(n_fft)`** (ATen "by_root_n" / frame_length mode). This is *not* the `sqrt(sum(win^2))` window normalization used by `torchaudio.transforms.Spectrogram` — they differ by exactly √2 here, and SAME uses the former. Verified to ATen source.
4. **Hermitian isometry:** one-sided STFT bins are scaled by √2 except DC and Nyquist.
5. **Frame alignment:** STFT hop 4096 = SAME latent hop (patch 256 × TRB stride 16) ⟹ chroma frames are natively **1:1** with latent frames at **~10.766 Hz** (44.1 kHz / 4096). Training `F.interpolate(..., mode='linear')` only mops up a possible off-by-one (`center=False` trims one frame). `n_latent_frames = ceil(n_samples / 4096)`.
6. **The window is a sqrt-Hann.** The "tight power-sine" window at hop = N/2 reduces exactly to `g[n] = sin(pi*n/N)` (amplitude 1). `TightSpectrogram.demodulate` only flips complex signs `(-1)^{k*m}` — irrelevant after `.abs()`.
7. **`torchaudio.prototype.transforms.ChromaScale` is deprecated upstream** — vendor the filterbank (NumPy/librosa port) so you don't depend on a moving target. Keep interpolation coords in **float32** to match ATen's float32 accumulation (float64 coords drift on long sequences).

---

## 3. Original heads & rate compatibility

- **Original readout heads [verified-from-source]:** `Conv1d(256 -> 128, kernel_size=1, bias=True)` per band — i.e. **affine, not pure linear**. **L1 loss**, per-band weights **`[0.035, 0.05, 0.2]`** (treble weighted ~6× bass). 10k-step warmup on `latents.detach()` before gradients reach the encoder. (There is also an **ILD head**: 32-band mel L/R log-magnitude difference, weight ×0.1, stereo only — relevant if you ever want a stereo-width control, since it means the latent is partly ILD-decodable.)
- **The trained heads are probably not released [inferred]**, but they don't need to be: the correlations live in the public-encoder latents and a linear probe re-extracts them. Refitting your own head on public latents is the intended route.
- **Rate gotcha:** the SAO-era LatCH dataset format is the **SAO** VAE = 2048× = **21.533 Hz**, 256 frames per 11.89 s crop. **SAME/SA3 is 4096× = ~10.766 Hz, 128 frames per 524288-sample crop.** Don't mix them silently. To feed a 256-frame SAO-format pipeline, `interpolate_linear(chroma, 256)`. To stay native SAME, align to `ceil(T/4096)`.

---

## 4. Refitting the head (the cheap, no-training-loop route)

If a plain linear/affine readout suffices (it should, given the training objective):
1. Encode varied audio with the **public SAME encoder** → latents `(256, L)`. **Resample audio to 44100 yourself first** so the encoder and the chroma target see literally identical samples.
2. Compute targets with the Section-1 recipe → `(3, 128, L)`, natively frame-aligned.
3. Per band, solve `latent → 128 bins` **with bias** by ridge least squares (streaming normal equations: accumulate `XᵀX` and `XᵀY`, solve `(XᵀX + λI)W = XᵀY`). Minutes on CPU, no GPU training loop.
4. **Gate on held-out R² / frame-cosine per band.** Bass band is usually the strongest linear readout; mid is the one that matters for melody. **If R² is near zero, suspect frame alignment (off-by-one) before doubting the method.**

---

## 5. If you train a *learned* head — the hard-won lessons

These come from actually training LatCH-style control heads (and apply to any head meant for **training-free guidance**):

- **The forward-noising schedule MUST match the diffusion model's objective.** SA3 `-base` is **rectified flow** (linear `z_t = (1-t)·z0 + t·noise`), **not** VP/DDPM cosine. A schedule mismatch ⟹ the head sees out-of-distribution noisy latents at sampling time ⟹ useless gradients. Record the schedule in the checkpoint and warn if it ≠ the model's objective.
- **Match the loss to the feature, identically at train and inference.** For chroma (a direction, not a magnitude): **cosine**. (`bce_logits` for beat/onset; `smooth_l1`/Huber for scalar regressions, with a clamp — dB features have −800 dB garbage on near-silent frames.)
- **Standardize regression targets (zero-mean/unit-std), un-standardize at inference via metadata.** A small-scale feature's head outputs at the feature's tiny scale → negligible guidance gradient → looks "dead." Standardizing fixes the gradient scale **and** makes the guidance gain uniform across features (same ρ/μ/weight work everywhere). This single change recovered features that looked uncontrollable.
- **Screen before you invest.** Two cheap probes: (1) **encodability** — a linear probe latent→feature; if R² is decent, the latent *encodes* it. (2) **head sensitivity** — `‖∂pred/∂z‖ / ‖z‖`; tiny ⟹ dead-for-guidance regardless of gain. A collapsed head ≠ an unencodable feature — usually it's the scale/standardization problem above, not missing signal.
- **Guidance window placement (rectified flow):** guidance strength scales with α, and the **first ~20% of steps is a dead zone** (σ≈1, α≈0). Put the window in the **back half** where it has power; ending early (e.g. at 0.8) discards the strongest region. (This is RF-specific; the index differs for VP models.)
- **The paper-canonical guidance gain (ρ=μ≈0.03) is inaudible in practice** — expect to push 100× higher, and use a separate **weight** knob (multiplies the gradient) as the real loudness control. Per-band perceptibility differs: bass/body are punchy, air (>2.5 kHz) is subtle — push harder there.
- **Mean guidance on the predicted-clean `z0|t` is the meaningful one** (vs variance guidance on the noisy `z_t`). `z0|t = z_t - t·v` for rectified flow.

---

## 6. Verification — the load-bearing discipline

**Never trust the head's own climbing self-report.** A head's loss going down during sampling does not mean the audio changed. **Decode the generated audio and re-measure with an independent extractor** (recompute the Section-1 chroma from the output waveform, compare to the target). And the sharpest test for "is this controllable at all": **A/B at opposite targets with the same seed** (e.g. all-C vs all-F#, or bass-locked-to-E vs bass-locked-to-A) — comparing two extremes is far more legible than one steered clip vs none.

Caveat worth stating to whoever runs this: relative *tracking* tends to be reliable (more target → more feature, in order), but the **absolute level is offset** — calibrate by the closed-loop decode-and-measure check, not by ear alone.

---

## 7. The make-or-break open question [inferred / untested]

It is **unverified by anyone, including the authors**, whether *steering* the latent along the chroma readout direction **audibly moves generated pitch** or merely **moves the meter**. This is the experiment that decides the whole architecture:
- Refit the heads (§4), guide sampling toward an "only C" profile, **listen**.
- If pitch moves cleanly → build on guidance alone, no base-model fine-tune.
- If the meter moves but the audio doesn't → fall back to a trained conditioner / LoRA (inject a frame-aligned chroma signal additively at the latent, the way SA3's own `local_add_cond` works).

Two latent reasons it might work for us where generic chroma guidance has underwhelmed elsewhere: (a) SAME's latent **was** chroma-regularized (linear decodability was a training objective); (b) a soft, near-static palette target is the "smooth/low-frequency control" regime that training-free guidance handles best, vs sparse fast note-contours.

---

## 8. Practical notes for building the UI / target

- Bin helpers: pitch class `p` → bin `(128*(p+3)/12 - 30) mod 128`; a semitone bump of width ≈ **0.25 semitones** matches real through-filterbank peak widths and gives ~18:1 on/off contrast after folding.
- **Three bands, two UI groups [inferred design]:** fold mid+air into one "melody" palette, keep **bass separate** (genre case: static goa/techno bass + free melody). Backend always emits 3×128; the fold is presentation-only. Asymmetric bleed favors this: a melody note at octave 4 reaches the bass band at weight ~0.011 (bass lock stays clean); bass harmonics do reach the mid band but land on root/5th/♭7, which a scale palette usually allows.
- **Lock-vs-prefer = per-band guidance gain:** bass high (a constraint), melody moderate (a palette preference, motion stays free), air ~0.3 or 0 (>8 kHz chroma is hats/noise).
- Chroma is pitch-**class**: the bass band pins *register* via the octave window, not an exact octave.

---

## 9. References

- **SAME** autoencoder: arXiv **2605.18613** (§3.3.2 = the three regressors).
- **Stable Audio 3** report: arXiv **2605.17991** (small 459M / medium 1.4B / large 2.7B; D_t=4096, d=256; flow-matching → distillation → adversarial post-training).
- Code: `github.com/Stability-AI/stable-audio-tools` @ `3241adb` — `training/autoencoders.py:567-597`, `models/transforms.py` (TightSpectrogram, ILDTransform, MeanChannelLog1pTransform).
- `torchaudio` v2.5.0 `prototype/functional/functional.py::chroma_filterbank`, `prototype/transforms/_transforms.py::ChromaScale.forward`.
- A working, adversarially-verified extractor (NumPy/SciPy core + optional torch GPU path, self-test 9/9, filterbank bit-exact in float64) exists as `src/harmonic/same_chroma.py` in the `mir-feature-extraction` repo: `compute_same_chroma(audio, sr) -> (3,128,T)`, plus `semitone_bin_centers()`, `expand_semitone_weights()`, `fold_to_12()`, `make_steering_target()`, `n_latent_frames()`.
