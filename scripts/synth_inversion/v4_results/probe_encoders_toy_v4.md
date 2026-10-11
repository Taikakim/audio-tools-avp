# Toy encoder probe (stand-in synth, NOT Surge; held-out R-squared)

1200 train / 300 held-out clips, 600 steps, batch 32, AdamW 0.002, 2 seed(s) per encoder.

| encoder | encoder params | log10 attack_s | log10 amp_decay_s | log10 fc_decay_s | log10 q | mean |
|---|---|---|---|---|---|---|
| conv | 421,696 | 0.956 ±0.001 | 0.963 ±0.003 | 0.892 ±0.009 | 0.971 ±0.001 | 0.946 |
| temporal_asp | 1,550,145 | 0.961 ±0.001 | 0.974 ±0.006 | 0.953 ±0.005 | 0.985 ±0.000 | 0.968 |
| temporal_segments | 1,615,681 | 0.962 ±0.002 | 0.976 ±0.002 | 0.949 ±0.005 | 0.986 ±0.000 | 0.968 |
| temporal_queries | 1,846,848 | 0.482 ±0.473 | 0.946 ±0.003 | 0.907 ±0.021 | 0.973 ±0.000 | 0.827 |

Toy result on a clean stand-in task: it says whether pooling matters here, not on Surge or real stems.
