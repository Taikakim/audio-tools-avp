# Refiner probe (stand-in synth, NOT Surge)

3 starts, each control perturbed by up to +-0.2; 2 passes; mean over starts.

| objective | start error | recovered error (mean abs, [0,1] units) | held-out mss_l1 | renders |
|---|---|---|---|---|
| v3 | 0.111 | 0.036 (per start: 0.029, 0.042, 0.038) | 0.305 | 141 |
| logmel | 0.111 | 0.045 (per start: 0.050, 0.059, 0.027) | 0.328 | 141 |
| two_anchor | 0.111 | 0.056 (per start: 0.078, 0.048, 0.042) | 0.660 | 141 |
| two_anchor_nosnap | 0.111 | 0.052 (per start: 0.080, 0.044, 0.032) | 0.657 | 141 |
| two_anchor_111 | 0.111 | 0.048 (per start: 0.055, 0.046, 0.044) | 0.408 | 141 |
| two_anchor_nosnap_111 | 0.111 | 0.040 (per start: 0.042, 0.046, 0.033) | 0.352 | 141 |

Stand-in, clean, single note, one target: plumbing and behavior in a clean setting only.
