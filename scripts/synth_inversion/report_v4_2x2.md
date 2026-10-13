# v4 A-series 2x2 factorial: inharmonicity aux head (JEPA) and temporal ASP flow encoder (A0, A1, A2, A3, A3')

Paired differences a - b on identical held-out items; 95% CI from resampling held-out presets; gauge = largest |difference| between replicate runs of the same configuration (EVAL_PARADIGM_V4.md).

## kind: ema

| contrast | metric | a | b | a - b [95% CI] | p (Holm) | design eff. | gauge | verdict |
|---|---|---|---|---|---|---|---|---|
| aux_head | mrr_a2p | 0.3542 | 0.3576 | -0.0034 [-0.0208, 0.0156] | 0.702 (0.702) | 5.75 | 0.0129 | not established |
| aux_head | mrr_p2a | 0.3198 | 0.3318 | -0.0120 [-0.0282, 0.0047] | 0.157 (0.313) | 5.93 | 0.0160 | not established |
| aux_head | hit_a2p (sec.) | 0.2262 | 0.2283 | -0.0020 [-0.0157, 0.0132] | 0.756 | 2.01 | 0.0073 | not established |
| aux_head | hit_p2a (sec.) | 0.1997 | 0.2104 | -0.0107 [-0.0275, 0.0054] | 0.203 | 2.95 | 0.0137 | not established |
| aux_head | top5_a2p (sec.) | 0.4920 | 0.4963 | -0.0043 [-0.0347, 0.0297] | 0.768 | 7.83 | 0.0098 | not established |
| aux_head | top5_p2a (sec.) | 0.4421 | 0.4519 | -0.0098 [-0.0379, 0.0162] | 0.486 | 7.10 | 0.0229 | not established |
| aux_head | ladder_jepa_rho (sec.) | 0.8855 | 0.8975 | -0.0120 [-0.0313, 0.0025] | 0.123 | 1.13 | 0.0244 | not established |
| temporal_asp | flow_mse | 0.4063 | 0.3965 | 0.0098 [-0.0031, 0.0261] | 0.166 (0.332) | 18.19 | 0.0023 | not established |
| temporal_asp | param_mae | 0.1217 | 0.1221 | -0.0004 [-0.0018, 0.0008] | 0.569 (0.569) | 48.51 | 0.0009 | not established |
| temporal_asp | flow_mae (sec.) | 0.3038 | 0.3011 | 0.0027 [-0.0012, 0.0076] | 0.218 | 27.63 | 0.0015 | not established |
| temporal_asp | cat_acc (sec.) | 0.6859 | 0.6809 | 0.0050 [0.0012, 0.0096] | 0.006 | 5.97 | 0.0037 | established better |
| temporal_asp | ladder_flow_rho (sec.) | 0.5502 | 0.5088 | 0.0414 [0.0044, 0.0838] | 0.029 | 2.32 | 0.0259 | established better |
| temporal_asp | ladder_flow_rho:cutoff_low (sec.) | 0.4616 | 0.4475 | 0.0141 [-0.0400, 0.0588] | 0.597 | 1.25 | 0.0429 | not established |
| temporal_asp | ladder_flow_rho:aeg_sustain (sec.) | 0.4338 | 0.3281 | 0.1057 [0.0659, 0.1502] | 0.000 | 1.01 | 0.0319 | established better |
| temporal_asp | ladder_flow_rho:feg_amount (sec.) | 0.8440 | 0.8211 | 0.0229 [-0.0136, 0.0616] | 0.226 | 1.08 | 0.0363 | not established |
| temporal_asp | ladder_flow_rho:aeg_decay (sec.) | 0.4614 | 0.4384 | 0.0230 [-0.0734, 0.1539] | 0.696 | 2.36 | 0.0649 | not established |
| temporal_asp | param_err:cutoff (sec.) | 0.1201 | 0.1152 | 0.0049 [-0.0019, 0.0137] | 0.188 | 36.98 | 0.0026 | not established |
| temporal_asp | param_err:resonance (sec.) | 0.2041 | 0.2073 | -0.0032 [-0.0134, 0.0062] | 0.533 | 31.51 | 0.0051 | not established |
| temporal_asp | param_err:feg_amount (sec.) | 0.1254 | 0.1334 | -0.0080 [-0.0136, -0.0027] | 0.002 | 27.79 | 0.0032 | established better |
| temporal_asp | param_err:feg_decay (sec.) | 0.1089 | 0.1019 | 0.0069 [0.0008, 0.0123] | 0.030 | 29.31 | 0.0020 | established worse |
| temporal_asp | param_err:aeg_decay (sec.) | 0.1030 | 0.1037 | -0.0007 [-0.0055, 0.0042] | 0.751 | 28.73 | 0.0030 | not established |
| temporal_asp | param_err:aeg_sustain (sec.) | 0.3357 | 0.3398 | -0.0040 [-0.0183, 0.0085] | 0.569 | 45.85 | 0.0107 | not established |

## kind: online

| contrast | metric | a | b | a - b [95% CI] | p (Holm) | design eff. | gauge | verdict |
|---|---|---|---|---|---|---|---|---|
| aux_head | mrr_a2p | 0.3532 | 0.3610 | -0.0078 [-0.0266, 0.0132] | 0.428 (0.553) | 6.62 | 0.0208 | not established |
| aux_head | mrr_p2a | 0.3210 | 0.3308 | -0.0098 [-0.0271, 0.0076] | 0.277 (0.553) | 6.08 | 0.0193 | not established |
| aux_head | hit_a2p (sec.) | 0.2241 | 0.2314 | -0.0073 [-0.0228, 0.0089] | 0.376 | 2.42 | 0.0200 | not established |
| aux_head | hit_p2a (sec.) | 0.2017 | 0.2080 | -0.0063 [-0.0252, 0.0111] | 0.512 | 3.35 | 0.0146 | not established |
| aux_head | top5_a2p (sec.) | 0.4941 | 0.5061 | -0.0120 [-0.0422, 0.0223] | 0.448 | 7.10 | 0.0068 | not established |
| aux_head | top5_p2a (sec.) | 0.4440 | 0.4524 | -0.0084 [-0.0351, 0.0171] | 0.531 | 6.47 | 0.0269 | not established |
| aux_head | ladder_jepa_rho (sec.) | 0.8861 | 0.9024 | -0.0163 [-0.0343, -0.0026] | 0.014 | 1.24 | 0.0251 | below gauge |
| temporal_asp | flow_mse | 0.4049 | 0.3937 | 0.0111 [-0.0021, 0.0281] | 0.125 (0.249) | 18.25 | 0.0040 | not established |
| temporal_asp | param_mae | 0.1214 | 0.1218 | -0.0004 [-0.0018, 0.0008] | 0.536 (0.536) | 46.34 | 0.0013 | not established |
| temporal_asp | flow_mae (sec.) | 0.3031 | 0.3001 | 0.0030 [-0.0009, 0.0078] | 0.168 | 27.20 | 0.0023 | not established |
| temporal_asp | cat_acc (sec.) | 0.6892 | 0.6842 | 0.0051 [0.0019, 0.0090] | 0.000 | 3.86 | 0.0045 | established better |
| temporal_asp | ladder_flow_rho (sec.) | 0.5537 | 0.5119 | 0.0418 [0.0070, 0.0806] | 0.019 | 2.01 | 0.0256 | established better |
| temporal_asp | ladder_flow_rho:cutoff_low (sec.) | 0.4550 | 0.4377 | 0.0173 [-0.0402, 0.0645] | 0.554 | 1.37 | 0.0196 | not established |
| temporal_asp | ladder_flow_rho:aeg_sustain (sec.) | 0.4513 | 0.3399 | 0.1114 [0.0691, 0.1567] | 0.000 | 1.00 | 0.0234 | established better |
| temporal_asp | ladder_flow_rho:feg_amount (sec.) | 0.8436 | 0.8295 | 0.0142 [-0.0221, 0.0491] | 0.422 | 1.04 | 0.0376 | not established |
| temporal_asp | ladder_flow_rho:aeg_decay (sec.) | 0.4648 | 0.4405 | 0.0243 [-0.0666, 0.1477] | 0.658 | 2.33 | 0.0679 | not established |
| temporal_asp | param_err:cutoff (sec.) | 0.1205 | 0.1154 | 0.0050 [-0.0018, 0.0134] | 0.169 | 35.51 | 0.0016 | not established |
| temporal_asp | param_err:resonance (sec.) | 0.2033 | 0.2067 | -0.0034 [-0.0139, 0.0066] | 0.534 | 32.34 | 0.0061 | not established |
| temporal_asp | param_err:feg_amount (sec.) | 0.1248 | 0.1322 | -0.0074 [-0.0134, -0.0016] | 0.010 | 29.79 | 0.0043 | established better |
| temporal_asp | param_err:feg_decay (sec.) | 0.1089 | 0.1036 | 0.0053 [-0.0004, 0.0112] | 0.067 | 26.48 | 0.0029 | not established |
| temporal_asp | param_err:aeg_decay (sec.) | 0.1042 | 0.1076 | -0.0033 [-0.0084, 0.0019] | 0.201 | 29.86 | 0.0080 | not established |
| temporal_asp | param_err:aeg_sustain (sec.) | 0.3313 | 0.3371 | -0.0059 [-0.0205, 0.0074] | 0.410 | 45.73 | 0.0106 | not established |

## Overall (same call in every kind)

- aux_head / mrr_a2p: **not established**
- aux_head / mrr_p2a: **not established**
- aux_head / hit_a2p: **not established**
- aux_head / hit_p2a: **not established**
- aux_head / top5_a2p: **not established**
- aux_head / top5_p2a: **not established**
- aux_head / ladder_jepa_rho: **not established**
- temporal_asp / flow_mse: **not established**
- temporal_asp / param_mae: **not established**
- temporal_asp / flow_mae: **not established**
- temporal_asp / cat_acc: **established better**
- temporal_asp / ladder_flow_rho: **established better**
- temporal_asp / ladder_flow_rho:cutoff_low: **not established**
- temporal_asp / ladder_flow_rho:aeg_sustain: **established better**
- temporal_asp / ladder_flow_rho:feg_amount: **not established**
- temporal_asp / ladder_flow_rho:aeg_decay: **not established**
- temporal_asp / param_err:cutoff: **not established**
- temporal_asp / param_err:resonance: **not established**
- temporal_asp / param_err:feg_amount: **established better**
- temporal_asp / param_err:feg_decay: **mixed across kinds**
- temporal_asp / param_err:aeg_decay: **not established**
- temporal_asp / param_err:aeg_sustain: **not established**

## Runs

- A0/ema: step 12000, seed 42, flow FlowMatchingResMLP, aux False; parity: r_a2p logged 0.2246 vs 0.2109; r_p2a logged 0.2285 vs 0.2207; flow logged 0.3664 vs 0.3658; ladder_flow_rho_mean logged 0.5000 vs 0.5044; ladder_jepa_rho_mean logged 0.8640 vs 0.8619
- A0/online: step 12000, seed 42, flow FlowMatchingResMLP, aux False; parity: r_a2p logged 0.2344 vs 0.2305; r_p2a logged 0.2227 vs 0.2168; flow logged 0.3652 vs 0.3644; ladder_flow_rho_mean logged 0.5038 vs 0.5101; ladder_jepa_rho_mean logged 0.8657 vs 0.8609
- A0/sf: step 12000, seed 42, flow FlowMatchingResMLP, aux False; parity: r_a2p logged 0.2227 vs 0.2285; r_p2a logged 0.2246 vs 0.2246; flow logged 0.3673 vs 0.3666
- A1/ema: step 11570, seed None, flow FlowMatchingResMLP, aux True; parity: none
- A1/online: step 11570, seed None, flow FlowMatchingResMLP, aux True; parity: none
- A1/sf: step 11570, seed None, flow FlowMatchingResMLP, aux True; parity: none
- A2/ema: step 12000, seed None, flow FlowMatchingResMLPV4, aux False; parity: r_a2p logged 0.1992 vs 0.2070; r_p2a logged 0.2129 vs 0.2324; flow logged 0.3763 vs 0.3756; ladder_flow_rho_mean logged 0.5244 vs 0.5239; ladder_jepa_rho_mean logged 0.8266 vs 0.8316
- A2/online: step 12000, seed None, flow FlowMatchingResMLPV4, aux False; parity: r_a2p logged 0.1953 vs 0.2148; r_p2a logged 0.2227 vs 0.2246; flow logged 0.3756 vs 0.3749; ladder_flow_rho_mean logged 0.5252 vs 0.5238; ladder_jepa_rho_mean logged 0.8360 vs 0.8243
- A2/sf: step 12000, seed None, flow FlowMatchingResMLPV4, aux False; parity: r_a2p logged 0.1992 vs 0.2109; r_p2a logged 0.2109 vs 0.2227; flow logged 0.3764 vs 0.3757
- A3/ema: step 12000, seed None, flow FlowMatchingResMLPV4, aux True; parity: r_a2p logged 0.2129 vs 0.2129; r_p2a logged 0.2090 vs 0.1953; flow logged 0.3735 vs 0.3723; ladder_flow_rho_mean logged 0.5480 vs 0.5486; ladder_jepa_rho_mean logged 0.8542 vs 0.8558
- A3/online: step 12000, seed None, flow FlowMatchingResMLPV4, aux True; parity: r_a2p logged 0.2188 vs 0.1992; r_p2a logged 0.1953 vs 0.2031; flow logged 0.3739 vs 0.3727; ladder_flow_rho_mean logged 0.5597 vs 0.5584; ladder_jepa_rho_mean logged 0.8580 vs 0.8589
- A3/sf: step 12000, seed None, flow FlowMatchingResMLPV4, aux True; parity: r_a2p logged 0.2246 vs 0.2051; r_p2a logged 0.2051 vs 0.2051; flow logged 0.3740 vs 0.3728
- A3_prime/ema: step 12000, seed 42, flow FlowMatchingResMLPV4, aux True; parity: r_a2p logged 0.2168 vs 0.2090; r_p2a logged 0.1973 vs 0.2148; flow logged 0.3726 vs 0.3724; ladder_flow_rho_mean logged 0.5469 vs 0.5474; ladder_jepa_rho_mean logged 0.8805 vs 0.8739
- A3_prime/online: step 12000, seed 42, flow FlowMatchingResMLPV4, aux True; parity: r_a2p logged 0.2266 vs 0.2246; r_p2a logged 0.2051 vs 0.2402; flow logged 0.3716 vs 0.3713; ladder_flow_rho_mean logged 0.5591 vs 0.5653; ladder_jepa_rho_mean logged 0.8881 vs 0.8777
- A3_prime/sf: step 12000, seed 42, flow FlowMatchingResMLPV4, aux True; parity: r_a2p logged 0.2266 vs 0.2285; r_p2a logged 0.1992 vs 0.2344; flow logged 0.3726 vs 0.3724
