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

## 5. The open question (what gates the build)

Does MERT actually **expose what SAME-L hides**? §3 says rhythm (downbeat 0.16) is the prime candidate, but a MERT-band only helps if the signal lives *somewhere* — so the next move is the **MERT-vs-SAME-L probe**: predict the same features from MERT layers and compare to the SAME-L R² above. If a MERT band predicts downbeat well past 0.16, a rhythm conditioner is justified; if MERT can't read it either, the adapter has nothing to grab and the honest answer is "rhythm isn't recoverable from a reference at all." That probe (decode + MERT-embed time-aligned audio — GPU) **scopes the whole adapter before a single training step**, and it's the cheapest next experiment that can say *no*.
