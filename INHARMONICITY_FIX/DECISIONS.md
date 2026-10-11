# Decisions Record

**D1: Scope of the metric**
- **Decision:** One window per played note, measuring exactly what the model renders.
- **Date:** October 10, 2026
- **Source:** Kim

**D2: Fundamental extraction**
- **Decision:** Use the nominal MIDI pitch as `minFrequency` and `maxFrequency` to isolate the fundamental, plus `require_fundamental_peak=True`.
- **Date:** October 10, 2026
- **Source:** Kim

**D3: Acceptance thresholds**
- **Decision:** Relaxed `max_excluded_energy_fraction` from 0.10 to 0.15 based on prior distribution.
- **Date:** October 10, 2026
- **Source:** Kim

**D4: The weight of the aux loss**
- **Decision:** `inharm_weight = 100.0`
- **Date:** October 10, 2026
- **Source:** Kim

**D5: Network targets vs raw scale**
- **Decision:** Softplus head instead of Sigmoid.
- **Date:** October 10, 2026
- **Source:** Kim

**D6: The render timeline**
- **Decision:** Use absolute timeline `(0.0, 0.125, 0.250, 0.500)` rather than relative phrase scale.
- **Date:** October 10, 2026
- **Source:** Kim

**D7: Restart or twin?**
- **Decision:** Restart both arms on the reference recipe (A0 and A1). The live run remains a pilot.
- **Date:** October 11, 2026
- **Source:** ROUND2_HANDOVER.md (Kim)

**D8: Resume across head change**
- **Decision:** Refuse loudly.
- **Date:** October 11, 2026
- **Source:** ROUND2_HANDOVER.md (Kim)

**D9: The launcher**
- **Decision:** Write `run_v3_recipe.sh` and mark `run_v3.sh` superseded.
- **Date:** October 11, 2026
- **Source:** ROUND2_HANDOVER.md (Kim)

**D10: Calibration acceptance gates**
- **Decision:** Confirmed gates in section 0b of ROUND2_HANDOVER.md.
- **Date:** October 11, 2026
- **Source:** ROUND2_HANDOVER.md (Kim)

**D11: Weight W and run length**
- **Decision:** W comes from measured gradient ratio + head start. Run twin until copied live checkpoint step.
- **Date:** October 11, 2026
- **Source:** ROUND2_HANDOVER.md (Kim)

**D12: Hyperball and escape velocity**
- **Decision:** Leave them out for now.
- **Date:** October 11, 2026
- **Source:** ROUND2_HANDOVER.md (Kim)
