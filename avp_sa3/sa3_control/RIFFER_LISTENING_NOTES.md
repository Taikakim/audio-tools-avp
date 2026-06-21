# Riffer listening notes (by ear, Kim) — bracket6 old models

Verbatim qualitative audit of the riffer (audio-reference adapter) outputs. Confirms the
MERIT finding by ear: **Hallucinogen ≈ Morphem at the same settings (small variations only)** —
the riffer barely distinguishes references; it's a variation/character effect, not transfer.

## Gain axis (consistent across steps)
- **gain 8** — spectral noise / "robot speech" / "scifi ventilator", still has structure+repetition.
- **gain 4** — crusty spectral high-end (quantization-noise style) distortion, but *musically cool*
  ("clockwork steampunk beat", "psy d&b"). Often the most interesting-but-broken zone.
- **gain 2** — moderate distortion; straight 4/4 broken into rapidly-firing kicks, "trying to stay together".
- **gain 1.5** — almost clean; bass distorted; "4/4 → jazz drummer on speed with 90s eurodance saw leads".

## Step axis (6e-4_adamw_wu) — lower steps = cleaner / more ambient
- **step 3000, gain 2–4** = first genuinely *clean* zone: "Hearts of Space 90s deep ambient", pure pad
  droning, no rhythm. gain 1.5 = goth-ambient/dungeon-synth with structure. ← the listenable sweet spot.
- **step 1000–2000** = "veiled reverby spectral", vocodery transients, close-to-useful intros.
- **step 4000–6000** = more rhythmic but more vocodery/distorted; gain 1.5–2 most usable.
- **4e-4_adamw** ≈ 6e-4 but slightly *less* artifacty at matched settings.

## Takeaways
- Usable musical zones cluster at **lower steps (≤3000) + gain 1.5–4** (clean ambient / interesting-broken).
- Reference identity barely matters (Hallucinogen ≈ Morphem) — by-ear confirmation of MERIT's ~0 transfer.
- The "interesting-but-broken" gain-4 clockwork/steampunk zone is a real creative niche worth keeping.

## Onset-density head (lr8e-5, 10 epochs / step 54000) — first listen
- **gain 0.5**: track ramps up *non-linearly* toward density 20; **d20 still listenable**.
- **gain 1**: density 15 very clean; 20 disintegrates.
- **gain 3**: density 8 = kicks starting to fold in; beyond here most settings go bad.
- Direction: **low gain + high density** is the expressive sweet spot at 10 epochs. → probe gains 0.1–0.4
  with densities 0.1–50, and watch Audiobox PQ for the disintegration boundary.

### Corrected eval (in-range densities 2–9) + careful listening — the head is STYLE-ENTANGLED
- **Metric correction:** earlier low-gain "control" (corr +0.9 at gain 0.1–0.4) was an **OOD artifact** —
  it only looked good because requested 20–50 vs 0.1 made artificial contrast while the head saturated.
  Training labels (librosa onset_density) are **5–9 onsets/sec, max ~9.8**; requesting 20–50 is nonsense.
- **In-range truth:** gains <1 barely move the output (compress to ~6.4–7.4, no authority). **Gain ~1.0 is
  the real operating point** — output spans **5.7→9.2** for req 2→9 (corr +0.89) AND highest CE (6.1–6.8).
- **PQ is blind** (flat ~8.0 everywhere); **CE is the disintegration detector** — but use a **relative floor
  (~baseline−0.8 ≈ 5.0)**, not absolute (baseline CE ~5.8). Auto-stop brackets on CE, not PQ.
- **The head carries STYLE, not just density (LoRA-like).** By ear (Kim):
  - First distinct elbow at **gain 0.7, between d8 and d9: techno-ish → goa-ish.**
  - **g1_d2** = first genuinely sparse output (an even sparser techno beat than g0.75); gains <1 don't pull
    the sparse end off at these densities.
  - **Within gain 1, a style elbow across density:** d2 sparse techno → ~d5 *less* sparse → then shifts
    toward goa, busier through d9. So density and style co-vary (the head learned the joint goa-data manifold:
    sparse↔techno, busy↔goa). "Sounds a bit like a LoRA — carries stylistic markers."
  - **g1_d9 very listenable, no breakdown.** → fine-sweep gains 1.1–2.0 × d2–9 to map the sparse-end/style frontier.

### Fine gain sweep 1.1–2.0 (in-range d2–9) + the ear-vs-metric gap
- **Sparse end opens at gain ~1.3** (floor 5.7→2.2); **widest span gain 1.5–1.7** (2.3→10.7+, overshoots dense).
  Non-monotonic / U-shaped there (corr ~0.78, mid-densities sparsest) = the style elbow, not a clean knob.
- **CE/PQ held across gains 1.1–2.0** (CE 6.5–7, PQ ~8), only d9/gain≥1.9 broke (1/80). BUT —
- **THE EAR IS STRICTER: clean (no audible distortion) only at gain 1.0, at most 1.1** (Kim). Gains 1.3–1.7
  add range + variety but carry audible distortion that **Audiobox CE/PQ did NOT flag.** → metrics necessary
  but not sufficient; trust the ear for the clean ceiling. Capture **PC (Production Complexity)** too — may
  catch the high-gain style/distortion shift CE/PQ miss.
- **Operating recommendation:** **gain 1.0–1.1 = clean usable knob** (output ~4.5–9.2, monotonic-ish); 1.3–1.7
  = "wide but dirty" creative zone. Density↔style coupling is real (LoRA-like) — embrace it.
- *(in progress: Kim's 40-epoch run, ~32400+/216000 steps — evaluate when done.)*
