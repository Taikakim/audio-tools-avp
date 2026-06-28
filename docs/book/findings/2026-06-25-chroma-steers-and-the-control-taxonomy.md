# Chroma steers — the first content control, and the three-way taxonomy

*Finding — 2026-06-25. Source: our own runs — trained a chroma LatCH head on SAME-compatible `(3,128,T)` stem chroma and ran the make-or-break A/B with a gain sweep. Builds on `scripts/CHROMA_HANDOFF.md` (the target recipe + the make-or-break test it set up) and the rhythm findings (`2026-06-23-…` §6–§8). House rule kept: the gain-64 "MOVED: True" was a degenerate noise verdict, corrected by the sweep.*

## The result: chroma steers (conclusively), with a clean gain–authority curve

Trained an **`other`-stem chroma LatCH head** (temporal, adaln_zero/depth-4, **cosine loss** on SAME `(3,128,T)` → 384-ch, 12 ep) and steered all-C vs all-F# palettes (same seed, no-pitch prompt), decoding and **re-measuring the output chroma with `same_chroma`** (the handoff's discipline — never trust the head's self-report). Sweeping gain:

| gain | C-steer → dominant | F#-steer → dominant | C↑F# separation |
|---|---|---|---|
| 64 | A | A | +0.004 (noise) |
| 256 | A | A | +0.012 |
| 1024 | A | **F#** | +0.036 |
| 1536 | **C** | **F#** | +0.047 |
| **2048** | **C** (0.107) | **F#** (0.110) | **+0.053** |

At gain ~1536–2048 **each palette makes its requested pitch class the *dominant* one** in the generated audio — request C → C-dominant, request F# → F#-dominant — monotonically with gain. The pitch genuinely moves in the requested direction. This is the **first content/pitch feature that steers** (onset was *amount*; beat/downbeat *don't* steer at all).

## Why the readout looked weak but the head reads great (the §6 pattern, again)

A **linear per-frame** ridge readout of chroma scored only cos ~0.14 (mid/melody band) — which would say "barely decodable." But the **temporal LatCH head** reads `other`-chroma at **cos ≈ 0.89** (training). Same lesson as rhythm (§6): per-frame-linear under-measures; the temporal transformer crushes it. (The screen's one real signal: the **bass stem reads best linearly** — matching the handoff's "bass band is the strongest linear readout.")

## The three-way control taxonomy (the capstone)

Combining all the control experiments, training-free latent guidance sorts features by *what kind of thing they are*:

| feature kind | examples | steers? | gain |
|---|---|---|---|
| **amount / density** | onset density, RMS | ✅ yes | moderate (48–96) |
| **content / pitch** | chroma (palette) | ✅ yes | **high (~1536–2048)** |
| **structure / timing** | beat, downbeat placement | ❌ no | (gradient can't reorganise) |

This validates both priors: **§8's dense-vs-sparse axis** (dense features move the latent, sparse structural ones can't) *and* the **handoff's chroma-regularised-latent hypothesis** (the SAME VAE trained auxiliary chroma heads → chroma is decodable → steerable; §7's open question answered **yes**).

## Honest caveats

- **It needs very high gain** (~1536–2048 — ~20–40× onset's). Consistent with the handoff's "push ~100× the paper."
- **Clear-dominance but modest magnitude.** The requested class lands at ~0.107–0.110 vs 0.083 chance — *dominant*, a real lean toward the target key, but **not a hard lock**. Per-band steering (bass-lock at high gain + melody-palette) and the back-half window should sharpen it.
- **Coherence at 2048 is by-ear.** The meter crosses the threshold; whether gain 2048 degrades audio quality is the listener's call (the sweet spot is likely the lowest gain that still flips the dominant class — ~1536).
- Trained on the **`other` stem** (cleaner melody); full_mix (the SAME-objective) is the obvious comparison.

## Infra built (committed)

- `train_latch.py`: **`--target-source chroma`** (reads per-crop stem `(3,128,T)` npz → 384-ch) + **`--loss cosine`** (`fork/latch-sa3-phase1` 5a32d1b). Full wandb telemetry applies.
- `latch_guided.py`: **cosine added to the single-guide sampler** (the multi-guide already had it; chroma needs it) (644c23c).
- Stem `same_chroma` extracted for 4907 crops (`other`, `bass`) → `Lehto/latents_sa3_stem_chroma/`. The trained `other` head + the A/B wavs persisted in `cu_reward_renders/analysis/`.

## Next

Per-band **bass-lock + melody-palette** at independent gains (the handoff's two-group UX, now with real stems); the **full_mix vs stem** target comparison; and on LUMI, chroma joins the `DiT-block × feature` controllability map as the first **content** feature — firmly on the steerable side, validating the taxonomy at scale.
