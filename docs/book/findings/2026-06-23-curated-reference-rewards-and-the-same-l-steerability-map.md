# Curated-reference rewards, and what the SAME-L latent actually exposes

*Finding — 2026-06-23. Source: our own runs — a 240-sample Audiobox aesthetic sweep on `medium-base`, a MERT-separation test on 48 aavepyora tracks vs goa/AI, and a 200-crop linear probe of the SAME-L latent against the recorded `TIMESERIES.npz` features. Advances the [Sourcebook](../CONTROL_METHODS_SOURCEBOOK.md) §Ch13 entry (Readout Guidance's relational/correspondence head) and §Ch8 (the conditioner). House rule kept: an over-optimistic small-sample claim was caught and corrected (§2).*

---

## 1. Audiobox aesthetics are a collapse-floor, not a reward

The chain started with "use a high Audiobox **Content Usefulness** score as a steering reward." A 240-sample sweep (10 genres × 24, medium-base) killed that as a *maximization* target and reframed it:

- **The axes carry the annotators' taste.** Per-genre means tilt hard toward "refined" music — on Content Enjoyment: jazz 7.2 / deep-house 7.1 / classical 6.9 / folk 6.7 at the top, ambient 4.0 / techno 5.8 / metal 6.0 at the bottom. **Maximizing any of CU/CE/PQ is a gradient toward jazz-house-folk** — the lowest common denominator of refined music, not what you want from a genre-spanning instrument.
- **CU ≈ PQ** (within-prompt +0.87 across every genre): "usefulness" is essentially a production-quality proxy, not a distinct axis. **PC (complexity)** is the only descriptive, non-taste axis.
- **Correction (the house rule):** a quality/complexity *tradeoff* (PQ↔PC −0.26) seen on an n=8 small-base pilot **did not replicate** on the 240-sample medium run (+0.25, genre-dependent). It was small-sample noise.

**Use:** Audiobox belongs in **eval as a collapse-floor** (`PQ<5`, `CU<5` = a broken-output red flag in a checkpoint/parameter sweep), never as a steering reward. Caveat: a `CE` floor wrongly condemns ambient/drone (CE 4.0 by design). Full numbers in `WORKLOG` 2026-06-22 and the `fk-steering-cu` recipe's `findings` block.

## 2. A curated reference set *is* a genre-neutral reward

The way out of the taste bias: let a curated set of tracks define the target instead of a global scalar. Tested on **48 aavepyora tracks** (the user's own catalogue), MERT-embedded (layers 3–6, 23), vs two contrasts:

- **vs default SA3 output (240 AI renders): centroid-AUC 0.999**, 1-NN purity 0.97 — near-perfect. This is the case that matters: a "be like this set" reward has a strong, clean gradient pulling generation *away from* generic output *toward* the target.
- **vs same-genre goa (decoded from `latents_sa3`): AUC ~0.87 at MERT layers 5–6** (timbre/production), but only **0.77 at layer 23** (semantic). Reading: MERT correctly says the set *is* goa (the semantic layer can't tell two goa apart), but its **production/timbre signature is distinguishable** — the distinctiveness is in the *sound*, not the genre.

**Practical:** use **MERT layers 5–6, mean-centred** (the raw cosines all sit ~0.95 — the space is anisotropic; centring lifted 0.84→0.87 for free), k-NN or centroid. **48 tracks suffices** — the centroid is stable; the limiter is the *metric*, not the data (for sharper within-genre precision, the discriminative MERIT `S_tim` head or a learned aave-vs-world direction beats raw cosine). **Two honesty caveats:** the goa contrast went through SAME-L (codec artifacts) while the references are pristine — so 0.87 is an *upper bound* on pure-style separation; and because SA3's own generations are SAME-L-decoded, the references should be **embedded after a SAME-L round-trip** to make the reward codec-fair. This is exactly the **relational/correspondence-head** idea flagged in Sourcebook §Ch13 (Readout Guidance) — now with our-data evidence that it has signal.

## 3. The SAME-L steerability map — measured, not borrowed

Which recorded features are *linearly decodable from the 256-d SAME-L latent*? That decodability **predicts steerability** (a LatCH head can only read what's linearly there). Per-frame ridge `latent → feature`, 200 crops, split by track. The negative control (`relative_position`, non-acoustic) lands at **0.044** — the probe is honest, no leakage.

| band | feature(s) | test R² |
|---|---|---|
| **easy** | spectral flux · spectral flatness | **0.88 · 0.80** |
| medium | spectral skew · bass energy · **chroma (hpcp)** · air energy | 0.55 · 0.50 · **0.42** · 0.35 |
| weak | onset envelope · beat activation · spectral kurtosis | 0.31 · 0.30 · 0.26 |
| **hard** | **downbeat** · body energy · mid energy | **0.16** · 0.11 · 0.11 |
| (control) | relative_position (non-acoustic) | 0.044 |

Three things worth keeping:
- **Spectral texture is densely encoded** (flux/flatness 0.80–0.88) — easy LatCH targets, as expected from a reconstruction autoencoder.
- **Energy is band-uneven:** bass 0.50 and air 0.35 are readable, but **mid/body energy ~0.11** — the latent under-exposes mid-band energy linearly (a non-obvious result worth remembering before promising a "make the mids louder" head).
- **Rhythm is the weakest family** (onset 0.31, beat 0.30, **downbeat 0.16**). This **explains the riffer's documented rhythm-transfer failure (MERIT ≈ 0)**: the bare latent doesn't linearly carry downbeat structure, so neither a linear head nor an opaque reference embedding can steer it. Chroma at 0.42 likewise matches the `chroma-steer` recipe needing ~15× the default gain.

Caveat: the **per-stem** onset/RMS features were dropped (absent in some crops — not every track was stem-separated); a stemmed-only re-run is pending.

## 4. The multi-band MERT conditioner — the design, and what gates it

§3 names the prize: rhythm (and mid energy) are *tangled-but-maybe-present* in the latent — the niche where the LatCH/heads approach is weakest. The idea (the user's): inject MERT, which exposes different abstraction levels at different layers, as **new conditioning streams with per-band on/off** — a disentangled, reference-driven control. Articulated for building:

- **Bands, not 25 layers** (adjacent layers are near-identical — we saw 3–6 behave as one). Group into ~3–5 bands.
- **One GLIGEN-gated decoupled cross-attn adapter per band** (`tanh(γ)`, init 0 = identity). Gate→0 = off; a strength slider = the on/off UX. This is a **multi-band extension of the existing riffer**, reusing `set_lora_strength`.
- **Pooled-global** (drop-a-clip → "be like this", timing-free). Per-frame time-varying is a *separate, harder* mode (needs a shared timeline).
- **Self-supervised training** on `latents_sa3` (each track conditions on its *own* MERT bands; swap a different clip in at inference) — data is free.

Three caveats that decide whether it's worth it:
1. **SAME-L is the gatekeeper.** Conditioning can only surface features SAME-L can *render but doesn't linearly expose* (§3's tangled-but-present, e.g. rhythm) — never features it *discards*.
2. **The riffer prior: explicit beats opaque.** A scalar onset-density head steered at corr +0.90 where the opaque riffer embedding scored MERIT ≈ 0. MERT-bands are *between* — build them to **complement** the explicit heads (the ineffable "make it like *this*"), not replace them.
3. **Disentanglement isn't free** — gating gives clean on/off, but bands may leak; a disentanglement objective may be needed.

## 5. Answered (2026-06-23): MERT exposes rhythm where SAME-L hides it

We ran the gating probe — 80 goa windows (30 s) decoded → MERT, each layer probed per-frame against the same targets as §3, split by track. The result is decisive, and *specifically about rhythm*:

| feature | SAME-L latent | best MERT layer | Δ |
|---|---|---|---|
| **beat activation** | 0.36 | **0.83 (L6)** | **+0.47** |
| **onset envelope** | 0.33 | **0.71 (L1)** | **+0.39** |
| **downbeat activation** | 0.17 | **0.39 (L6)** | **+0.22** |
| spectral flux | 0.86 | 0.85 (L3) | −0.01 (tie) |
| chroma (hpcp) | 0.42 | 0.37 (L3) | −0.05 (SAME-L wins) |
| relative_position (control) | <0 | <0.1 | — (correctly unpredictable) |

The precondition holds **for rhythm**: beat/onset/downbeat are far more linearly accessible from MERT (concentrated in **layers ~1–6**) than from the bare SAME-L latent. The signal *is* in the SAME-L-decoded audio — SAME-L renders rhythm fine, it just doesn't expose it *linearly* in the 256-d latent, and MERT recovers it. **Spectral flux and chroma show no MERT advantage** (SAME-L is as good or better), so the conditioner should not bother with them.

This **sharpens §4's design**: the multi-band conditioner's value is concentrated in a **single rhythm band (MERT layers ~1–6)**, not "all bands" — and it's the concrete fix for the riffer's rhythm-transfer failure (§3), giving the model the rhythmic handle the latent lacks.

The new fork (the next decision, not a closed door): because rhythm is **present-but-nonlinear** in the latent (MERT reads it back out of the decoded audio), a **nonlinear MLP readout head on the SAME-L latent** might recover it too — cheaper than a MERT-conditioned finetune, and target-driven ("hit this onset curve") rather than reference-driven ("groove like this clip"). The §2 "explicit beats opaque" prior says: probably **both** — a nonlinear latent head for *measurable* rhythm, a MERT-rhythm band for the ineffable feel. The cheapest next test is the MLP head (CPU, on data we have); the adapter is the bigger commit. *(→ §6 ran this and revises the conclusion.)*

## 6. Corrected (2026-06-23, same day): you don't need MERT for rhythm — the latent just needs a *temporal* readout

We built §5's cheap head — but as a **1D-CNN** (temporal, ~4 s receptive field), not only a per-frame MLP. It overturns §5's lean toward MERT:

| feature | linear/frame | MLP/frame | **CNN (temporal)** | MERT (linear probe) |
|---|---|---|---|---|
| beat | 0.29 | 0.47 | **0.86** | 0.83 |
| onset | 0.29 | 0.58 | **0.81** | 0.71 |
| downbeat | 0.13 | 0.03 | **0.58** | 0.39 |

A temporal head on the SAME-L latent **matches or beats MERT on all three** rhythm features. The ceiling §3/§5 hit was **missing temporal context — not nonlinearity, not absence**: the per-frame MLP barely moved beat/onset and *failed* on downbeat (0.03 — a single 256-d frame can't locate a downbeat); only ~4 s of context recovers it. So the earlier reading — "MERT exposes rhythm the latent hides" — was an **artifact of comparing a temporal model (MERT) against a per-frame *linear* latent probe**. The latent never hid rhythm; the readout was too local.

Consequences:
- **Rhythm control = a temporal (1D-CNN) LatCH head on the latent.** No MERT, no finetune; target-driven ("hit this onset/beat curve"), composes with the existing guidance. This is the recommended build — and the real fix for the riffer's rhythm failure (its readout was per-frame/opaque; not the latent's fault).
- **MERT's niche narrows** to the *reference-style* reward (§2 — genuinely reference-driven and ineffable) and any feature truly absent from the latent (none found yet). The §4 multi-band conditioner **loses its rhythm justification**.
- **The §3 steerability map is a per-frame *linear lower bound*** — it under-measures any temporally-structured feature. Re-read with a temporal head, rhythm's real steerability is high (downbeat 0.16→0.58, beat 0.36→0.86).

Honesty note: the comparison gives MERT a *linear* probe but the latent a *CNN* — not perfectly matched (a CNN on MERT might score higher still). Moot for the decision: the latent **alone**, with a temporal head, already reaches MERT's level → MERT is **not necessary** for rhythm. What remains is engineering: fold the temporal-head architecture into the LatCH head family and train beat/onset/downbeat heads for inference-time guidance.

## 7. Built and verified (2026-06-23): the onset head *steers* generation (corr 0.986)

§6 said a temporal latent head should work; we trained one and checked it end-to-end. **It turned out the existing LatCH head architecture is already temporal** (RoPE self-attention over the latent sequence) — so no new architecture was needed, just *training a head on a rhythm feature*. An **onset LatCH head** (`scripts/latch/train_latch.py`, production recipe: adaln_zero, depth 4, 4.9M params, standardized, smooth_l1) trained on `latents_sa3` + `onset_envelope_ts` (12 epochs, loss 0.29→0.17), then driven through the **production guidance path** (`model.generate(latch_configs=...)`, medium-base, gain 48, 5-level sweep):

| requested onset | measured onset-strength of the output |
|---|---|
| 0.40 | 0.741 |
| 0.90 | 0.752 |
| 1.30 | 0.772 |
| 1.80 | 0.807 |
| 2.30 | 0.830 |

**correlation(requested, measured) = 0.986, monotonic = True.** The head doesn't just *read* onset — it **steers** it: requesting denser onsets makes the generated audio measurably denser, reliably and in order. This realises §6 concretely — a temporal LatCH head on the SAME-L latent is a **working rhythm control**, no MERT, no MERT-conditioner, no base-model finetune, riding the existing guidance machinery.

Honest read (MASTER §5 "judge by spread, not just corr"): the *direction* is rock-solid (0.986, monotonic), but *authority* at gain 48 is **moderate** — a 0.74→0.83 onset-strength swing across the full range. Gain 96 (top of the SA3-medium range) or a wider/later guidance window should widen it — knob-tuning, not redesign. **Next:** sweep gain, train beat + downbeat heads the same way, and compose them (rhythm + the existing chroma/RMS heads) — the multi-head endgame. The trained head + verifier are persisted alongside the render set.
