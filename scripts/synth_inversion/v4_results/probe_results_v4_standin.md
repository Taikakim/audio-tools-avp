# Probe results (stand-in synth, NOT Surge)

Distance rows: Spearman(|sweep change|, distance to the first value), mean / worst over patch x note cells.

| sweep | logmel_l1 | mss_l1 | two_anchor | harmonic_env | descriptor (Spearman mean / worst) |
|---|---|---|---|---|---|
| attack (attack_s) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | log_attack_time: 0.82 / 0.36 |
| decay (amp_decay_s) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.57 / 0.39 | early_decay_db_per_s: 1.00 / 0.96 |
| filt_env (fc_decay_s) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | - |
| reso (q) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.96 / 0.86 | resonance_height: 1.00 / 1.00 |
| fm (fm_depth) | 1.00 / 1.00 | 1.00 / 1.00 | 0.99 / 0.94 | 1.00 / 1.00 | comb_residual: 1.00 / 1.00 |

Nuisance: distance caused by the nuisance divided by the mean distance of three real changes (amp decay x2, Q x3, filter envelope +1 octave); lower is better.

| metric | shift 10 ms | shift 20 ms | room (RT60 0.3 s, 15% wet) |
|---|---|---|---|
| logmel_l1 | 2.06 | 2.36 | 0.65 |
| mss_l1 | 1.45 | 1.89 | 0.62 |
| two_anchor | 0.12 | 0.26 | 0.49 |
| harmonic_env | 0.10 | 0.21 | 2.43 |

(patches pluck,sustain; MIDI notes 28,36,44,50; 179 s)
