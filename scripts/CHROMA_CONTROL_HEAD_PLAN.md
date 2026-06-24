# Chroma control head — execution plan (2026-06-25)

*Builds on `scripts/CHROMA_HANDOFF.md` (the exact SAME-chroma target recipe + the hard-won guidance lessons) and the rhythm-control findings (`docs/book/findings/2026-06-23-…`, §6–§8). This is the **execution** layer: which target, which head, the make-or-break test, the fallback.*

## Why chroma is the most promising control we've tried

1. **It's the dense/smooth-palette regime.** §8 found training-free latent guidance steers *dense amount* features (onset density steered at 0.986) but is a **no-op on sparse structural** ones (beat/downbeat placement). Chroma is a soft, near-static *palette* — exactly the smooth/low-frequency regime guidance handles best.
2. **SAME's latent was chroma-regularized by training objective** — the SAME VAE trained auxiliary chroma-regression heads (a latent-shaping regularizer). So chroma is **linearly decodable from the latent by construction** — the strongest steerability prior of any feature we've probed (stronger than the §3 R² map suggests, because that used the wrong, essentia-hpcp target).

## Targets — the critical detail

Use **`compute_same_chroma` → (3,128,T)** (the SAME-compatible register-resolved chroma; `mir-same-chroma/src/harmonic/same_chroma.py`, pure numpy/scipy). **Not** the essentia `hpcp_ts` (12-bin, wrong recipe → "head learns garbage", per handoff §1). Three sources, now available:

| target | what | role |
|---|---|---|
| **full_mix** same_chroma | what the SAME latent was regularized *against* | the SAME-objective readout target (cleanest linear decodability) |
| **other** stem same_chroma | melody/harmony, drums excluded *(extracting now)* | the **melody palette** head — cleaner than full-mix |
| **bass** stem same_chroma | clean bassline pitch *(extracting now)* | the **bass-lock** head |

The two stems map onto the handoff's bass-band / melody-band split (§8 there) — but cleaner, because they're *source-separated* rather than *register-windowed*. **Decision to test empirically:** full-mix vs stem target — which gives the higher per-band readout R²? (Full-mix is SAME-objective; stems are cleaner but a subset of what the latent encodes.)

## The head

- **Readout (already refit — `mir-same-chroma/sc_run/chroma_heads*.npz`):** affine `Conv1d(256→128)` per band, ridge-fit on latents→same_chroma. **Step 1: re-fit/verify on the new stem targets**, gate on **per-band held-out R²** (bass usually strongest linear readout; **mid band = melody = the one that matters**). Minutes on CPU, no GPU. If R² is near zero → suspect frame alignment (off-by-one) before doubting the method (handoff §4).
- **Guidance head:** a temporal **LatCH head** (`train_latch.py` — now wired for the full wandb telemetry, the standing requirement), output **3×128** (or per-band), trained with the **RF noising schedule** (SA3-base is rectified flow), **cosine loss** (chroma is a *direction*, not a magnitude), **standardized** targets (the dead-gradient fix — §8's beat null was partly a scale problem; the handoff §5 makes the same point). adaln_zero / depth-4, as the rhythm heads.

## Guidance config (handoff §5, hard-won)

- **Gain ~100× the paper** (paper ρ=μ≈0.03 is inaudible) — the SA3-medium 48–96 range, with the separate **weight** knob as the loudness control.
- **Window in the BACK HALF** — RF's first ~20% of steps is a dead zone (σ≈1, α≈0); ending early discards the strongest region.
- **Mean guidance on `z0|t = z_t − t·v`** (the meaningful one), not variance on noisy `z_t`.
- **Per-band gain = the UX:** bass **high** (a lock/constraint), melody **moderate** (a palette preference, motion stays free), air (>8 kHz) **~0.3 or 0** (it's hats/noise).

## The make-or-break test (handoff §7 — the experiment that decides the architecture)

**A/B opposite palettes, same seed:** all-C vs all-F# (and bass-locked-E vs -A). **Decode the output and re-measure chroma with `same_chroma`** (never trust the head's climbing self-report — the exact §8 discipline), and **listen**.
- **Pitch moves cleanly →** guidance alone, no finetune. (§8 predicts chroma is on the steerable side.)
- **Meter moves but audio doesn't →** fall back to a **trained conditioner**: inject the frame-aligned chroma *additively at the latent* (`local_add_cond`), the Music-ControlNet / MuseControlLite path (Sourcebook Ch8/Ch11). This is the same boundary-fallback that beat/downbeat hit — but chroma is far likelier to clear it.

## Sequence

1. **[running]** Generate stem `same_chroma` (other, bass) for the dataset → `Lehto/latents_sa3_stem_chroma/<crop>.npz` `{other,bass}: (3,128,4096) fp16`.
2. **Readout screen:** refit ridge heads on full_mix vs other vs bass targets; compare per-band R². Pick the target(s) with the cleanest melody-band readout.
3. **Train the guidance head** (cosine / standardized / RF / telemetry) on the chosen target.
4. **Make-or-break A/B** (decode-and-measure + listen). 
5. If it steers → build the **two-group control**: bass-lock + melody-palette heads at independent gains. If not → the trained-conditioner fallback.

## LUMI

Readout + a single guidance head are small (CPU / single-GPU). LUMI-scale: the **full same_chroma extraction across all stems**, a **target sweep** (full-mix vs stems vs bands), and — if needed — the **trained chroma-conditioner** (a ControlNet-style branch, real training). Chroma is the first **dense** feature to add to the `DiT-block × feature` controllability map — it should land firmly on the *steerable* side, validating the dense-vs-sparse axis the rhythm work uncovered.
