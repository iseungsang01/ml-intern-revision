# B.70 phase 2 -- the other readouts the paper cites, on D1

Registered: ces_prediction/experiments/PREREGISTRATION_B70.md (Addendum 2). r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed). Each readout is the batch's own code pointed at the D1 rows (Addendum 2, C1); its outputs are in `data/.b70p2/paper/`. Old values: `docs/paper/b*.json`, with the treatment named per batch. Checks: `docs/paper/b70_checks.md` (phase 2 section).

## Verdicts that change

- b66 R2: did not hold -> holds
- b65 I1: holds -> does not hold
- b65 I3: holds -> does not hold

## B.54 row readouts

Old: r2-r9, B.39 rows (random 40,000-sample subset per fold). D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| A1 | +0.946 [+0.928, +0.965] | holds | +0.941 [+0.925, +0.957] | holds |
| B1 | +0.684 [+0.640, +0.730] | holds | +0.700 [+0.658, +0.741] | holds |
| B2 | +0.661 [+0.591, +0.723] | holds | +0.656 [+0.574, +0.717] | holds |
| C1 | +0.092 [+0.062, +0.125] | holds | +0.098 [+0.066, +0.126] | holds |
| C2 | +0.096 [+0.066, +0.123] | holds | +0.101 [+0.071, +0.127] | holds |
| C3 | +0.463 [+0.355, +0.610] | does not hold | +0.462 [+0.354, +0.592] | does not hold |
| C4 | +0.874 [+0.681, +1.385] | holds | +0.845 [+0.678, +1.135] | holds |
| D1 | low +0.066 [+0.032, +0.100]; middle +0.091 [+0.044, +0.131]; high +0.110 [+0.061, +0.153] | holds | low +0.074 [+0.049, +0.098]; middle +0.090 [+0.046, +0.127]; high +0.116 [+0.066, +0.157] | holds |
| D2 | low +0.007 [-0.023, +0.031]; middle +0.042 [+0.009, +0.074]; high +0.092 [+0.040, +0.137] | does not hold | low +0.015 [-0.003, +0.035]; middle +0.040 [+0.008, +0.072]; high +0.097 [+0.041, +0.140] | does not hold |
| E1 | +0.069 [+0.010, +0.127] | holds | +0.074 [+0.030, +0.118] | holds |
| E2 | +0.054 [-0.095, +0.185] | does not hold | +0.039 [-0.086, +0.156] | does not hold |
| F1 | +0.087 [+0.053, +0.117] | holds | +0.090 [+0.060, +0.117] | holds |
| F2 | +0.037 [+0.007, +0.065] | holds | +0.039 [+0.011, +0.066] | holds |
| F3 | +0.074 [+0.033, +0.115] | holds | +0.074 [+0.040, +0.108] | holds |
| F4 | +0.020 [-0.014, +0.049] | does not hold | +0.020 [-0.005, +0.046] | does not hold |
| G1 | +0.151 [+0.093, +0.215] | holds | +0.151 [+0.098, +0.205] | holds |
| H1 | +0.054 [+0.026, +0.082] | holds | +0.060 [+0.031, +0.086] | holds |
| H2 | +0.874 [+0.734, +1.098] | holds | +0.889 [+0.746, +1.074] | holds |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| thirds low: network over causal GP | +0.066 [+0.032, +0.100] | +0.074 [+0.049, +0.098] | +0.008 |
| thirds middle: network over causal GP | +0.091 [+0.044, +0.131] | +0.090 [+0.046, +0.127] | -0.001 |
| thirds high: network over causal GP | +0.110 [+0.061, +0.153] | +0.116 [+0.066, +0.157] | +0.006 |
| thirds low: network over twin | +0.007 [-0.023, +0.031] | +0.015 [-0.003, +0.035] | +0.009 |
| thirds middle: network over twin | +0.042 [+0.009, +0.074] | +0.040 [+0.008, +0.072] | -0.002 |
| thirds high: network over twin | +0.092 [+0.040, +0.137] | +0.097 [+0.041, +0.140] | +0.005 |
| rows not flagged: increment | +0.092 [+0.062, +0.125] | +0.098 [+0.066, +0.126] | +0.006 |
| rows flagged: increment | +0.012 [-0.016, +0.036] | +0.015 [-0.010, +0.038] | +0.003 |
| p90 |e| ratio to causal GP | +0.946 [+0.928, +0.965] | +0.941 [+0.925, +0.957] | -0.005 |
| median |e| ratio to causal GP | +1.042 [+1.007, +1.082] | +1.037 [+1.001, +1.088] | -0.005 |
| overall: network over causal GP | +0.089 [+0.064, +0.112] | +0.094 [+0.072, +0.114] | +0.005 |
| overall: network over twin | +0.048 [+0.022, +0.073] | +0.052 [+0.029, +0.074] | +0.004 |

## B.55 sample timing

Old: r2-r9 networks, B.39 rows (40,000 subset); rows after an absent frame by B.55's own rule. D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| R1 | - | does not hold | - | does not hold |
| R2 | - | does not hold | - | does not hold |
| R3 | - | does not hold | - | does not hold |
| G1 | - | does not hold | - | does not hold |
| G2 | - | does not hold | - | does not hold |
| part D D1 | - | does not hold | - | does not hold |
| part D D2 | - | does not hold | - | does not hold |
| part D D3 | - | holds | - | holds |
| part D D4 | - | does not hold | - | does not hold |
| part D D5 | - | holds | - | holds |
| part D D6 | - | holds | - | holds |
| part D D7 | - | does not hold | - | does not hold |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| inputs as trained: over twin | +0.048 [+0.022, +0.073] | +0.052 [+0.029, +0.075] | +0.004 |
| inputs as trained: over causal GP | +0.089 [+0.063, +0.112] | +0.094 [+0.071, +0.115] | +0.005 |
| MCmean: over twin | +0.053 [+0.028, +0.078] | +0.058 [+0.038, +0.081] | +0.005 |
| MCmean: over gp | +0.094 [+0.069, +0.116] | +0.099 [+0.078, +0.120] | +0.006 |
| Gmc: over twin | +0.047 [+0.021, +0.072] | +0.052 [+0.029, +0.074] | +0.005 |
| Gmc: over gp | +0.088 [+0.063, +0.111] | +0.093 [+0.071, +0.114] | +0.005 |
| Gecei: over twin | +0.029 [+0.006, +0.050] | +0.031 [+0.010, +0.049] | +0.002 |
| Gecei: over gp | +0.071 [+0.048, +0.092] | +0.073 [+0.051, +0.092] | +0.002 |
| Gbes: over twin | +0.000 [-0.024, +0.022] | +0.002 [-0.018, +0.023] | +0.002 |
| Gbes: over gp | +0.043 [+0.020, +0.064] | +0.046 [+0.024, +0.066] | +0.003 |
| all fast channels one frame old (G): over twin | -0.027 [-0.050, -0.008] | -0.027 [-0.047, -0.010] | -0.001 |
| G: over causal GP | +0.017 [-0.004, +0.038] | +0.018 [-0.004, +0.037] | +0.000 |
| G: change skill | -0.079 [-0.096, -0.061] | -0.084 [-0.100, -0.069] | -0.005 |
| G, rows 1-3 after an absent frame: change skill | -0.247 [-0.335, -0.179] | -0.272 [-0.355, -0.204] | -0.024 |

## B.56 filters (refitted per D1 item 2.9)

Old: filters fitted on all non-test discharges, held values kept; pooled set r3-r9 (confirmation), r2 development; B.39 rows (40,000 subset); KX6 rows = not within ten rows after an absent frame. D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| KX1 | +0.049 [+0.040, +0.059] | holds | +0.049 [+0.040, +0.059] | holds |
| KX2 | +0.032 [+0.025, +0.039] | holds | +0.032 [+0.025, +0.039] | holds |
| KX3 | +0.036 [+0.012, +0.058] | holds | +0.042 [+0.022, +0.060] | holds |
| KX4 | +0.005 [-0.007, +0.017] | holds | +0.005 [-0.006, +0.015] | holds |
| KX5 | +0.921 [+0.906, +0.940] | holds | +0.921 [+0.906, +0.938] | holds |
| KX6 | +0.045 [+0.036, +0.055] | holds | +0.047 [+0.038, +0.056] | holds |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| confirm: kfx_over_kf0 | +0.049 [+0.040, +0.059] | +0.049 [+0.040, +0.059] | -0.000 |
| confirm: kg_over_kf0 | +0.032 [+0.025, +0.039] | +0.032 [+0.025, +0.039] | -0.000 |
| confirm: A_over_kfx | +0.036 [+0.012, +0.058] | +0.042 [+0.022, +0.060] | +0.006 |
| confirm: kc_over_kf0 | +0.042 [+0.033, +0.051] | +0.042 [+0.033, +0.051] | +0.000 |
| confirm: Tw_over_kf0 | +0.037 [+0.021, +0.053] | +0.039 [+0.024, +0.052] | +0.002 |
| confirm: kfx_over_Tw | +0.013 [-0.007, +0.032] | +0.011 [-0.006, +0.029] | -0.002 |
| confirm: kfx_over_G | +0.055 [+0.039, +0.070] | +0.054 [+0.040, +0.068] | -0.001 |
| confirm: kfx_over_O | -0.076 [-0.097, -0.056] | -0.076 [-0.095, -0.059] | -0.000 |
| confirm: kf0_over_G | +0.005 [-0.009, +0.020] | +0.005 [-0.008, +0.017] | -0.001 |
| confirm: A_over_G | +0.089 [+0.063, +0.112] | +0.094 [+0.071, +0.114] | +0.005 |
| confirm: A_over_Tw | +0.049 [+0.023, +0.073] | +0.052 [+0.029, +0.075] | +0.004 |
| confirm: kf0_over_G_90 | +0.005 [-0.007, +0.017] | +0.005 [-0.006, +0.015] | -0.001 |
| confirm: median_ratio_kfx_over_hold | +0.921 [+0.906, +0.940] | +0.921 [+0.906, +0.938] | -0.000 |
| rows not after an absent frame: kfx_over_kf0 | +0.045 [+0.036, +0.055] | +0.047 [+0.038, +0.056] | +0.002 |
| rows not flagged: kfx_over_kf0 | +0.104 [+0.092, +0.116] | +0.104 [+0.092, +0.116] | +0.000 |
| third coldest: kfx_over_kf0 | +0.001 [-0.007, +0.009] | +0.001 [-0.007, +0.009] | +0.000 |
| third coldest: A_over_kfx | +0.046 [+0.015, +0.072] | +0.055 [+0.038, +0.072] | +0.009 |
| third middle: kfx_over_kf0 | +0.042 [+0.031, +0.054] | +0.042 [+0.031, +0.053] | -0.000 |
| third middle: A_over_kfx | +0.005 [-0.036, +0.039] | +0.005 [-0.037, +0.037] | -0.001 |
| third hottest: kfx_over_kf0 | +0.101 [+0.085, +0.117] | +0.101 [+0.085, +0.117] | -0.000 |
| third hottest: A_over_kfx | +0.049 [+0.001, +0.086] | +0.057 [+0.005, +0.094] | +0.008 |
| median |e| kfx (eV) | 41.8 | 41.8 | -0.019 |
| median |e| kf0 (eV) | 43.3 | 43.3 | -0.047 |
| median |e| H (eV) | 45.4 | 45.4 | -0.004 |
| median |e| A (eV) | 45.8 | 45.7 | -0.119 |
| median |e| G (eV) | 44.1 | 44.1 | -0.054 |

## B.57 yardsticks

Old: r2-r9, B.39 rows (40,000 subset); filters of B.56; quiet = not within ten rows after an absent frame. D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| N1: every interval of MSE over the floor above 1 | A +1.331 [+1.246, +1.427]; Tw +1.415 [+1.325, +1.519]; G +1.472 [+1.389, +1.568]; H +2.123 [+2.031, +2.227]; O +1.279 [+1.222, +1.342]; P +1.630 [+1.587, +1.675]; K0 +1.469 [+1.387, +1.562]; KX +1.362 [+1.291, +1.444] | holds | A +1.328 [+1.255, +1.412]; Tw +1.419 [+1.343, +1.501]; G +1.477 [+1.399, +1.559]; H +2.128 [+2.050, +2.210]; O +1.281 [+1.226, +1.338]; P +1.630 [+1.592, +1.669]; K0 +1.472 [+1.400, +1.550]; KX +1.364 [+1.300, +1.435] | holds |
| N2: interval above 0.5 | +0.734 [+0.529, +0.936] | holds | +0.776 [+0.591, +0.963] | holds |
| N3: interval above 0.5 | +0.810 [+0.753, +0.866] | holds | +0.812 [+0.763, +0.863] | holds |
| N4: interval above 0.5 | +0.741 [+0.670, +0.812] | holds | +0.737 [+0.672, +0.803] | holds |
| N5: interval inside [-0.15, +0.15] | +0.016 [-0.052, +0.088] | holds | +0.015 [-0.038, +0.072] | holds |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| all rows: rms | 135.7 [126.4, 145.7] | 134.7 [125.6, 143.7] | -1.030 |
| all rows: mse_over_s2_A | +1.331 [+1.246, +1.427] | +1.328 [+1.255, +1.412] | -0.003 |
| all rows: mse_over_s2_Tw | +1.415 [+1.325, +1.519] | +1.419 [+1.343, +1.501] | +0.004 |
| all rows: mse_over_s2_G | +1.472 [+1.389, +1.568] | +1.477 [+1.399, +1.559] | +0.005 |
| all rows: mse_over_s2_H | +2.123 [+2.031, +2.227] | +2.128 [+2.050, +2.210] | +0.006 |
| all rows: mse_over_s2_O | +1.279 [+1.222, +1.342] | +1.281 [+1.226, +1.338] | +0.001 |
| all rows: mse_over_s2_KX | +1.362 [+1.291, +1.444] | +1.364 [+1.300, +1.435] | +0.002 |
| all rows: mse_over_s2_K0 | +1.469 [+1.387, +1.562] | +1.472 [+1.400, +1.550] | +0.003 |
| all rows: gap_closure_A | +0.734 [+0.529, +0.936] | +0.776 [+0.591, +0.963] | +0.042 |
| all rows: gap_closure_Tw | +0.354 [+0.200, +0.509] | +0.361 [+0.203, +0.503] | +0.008 |
| all rows: gap_closure_KX | +0.447 [+0.328, +0.567] | +0.444 [+0.336, +0.553] | -0.003 |
| all rows: net_twin_rho1 | +0.810 [+0.753, +0.866] | +0.812 [+0.763, +0.863] | +0.002 |
| all rows: kfx_kf0_rho1 | +0.741 [+0.670, +0.812] | +0.737 [+0.672, +0.803] | -0.003 |
| all rows: twin_kf0_rho1 | +1.122 [+1.038, +1.220] | +1.123 [+1.049, +1.215] | +0.001 |
| all rows: net_twin_beta0 | +0.693 [+0.589, +0.790] | +0.694 [+0.602, +0.788] | +0.001 |
| all rows: kfx_kf0_beta0 | +1.081 [+0.985, +1.172] | +1.071 [+0.958, +1.181] | -0.010 |
| all rows: removable_A_over_G | +0.298 [+0.191, +0.408] | +0.312 [+0.224, +0.399] | +0.015 |
| all rows: removable_A_over_Tw | +0.202 [+0.105, +0.303] | +0.218 [+0.130, +0.309] | +0.016 |
| all rows: removable_KX_over_G | +0.233 [+0.175, +0.288] | +0.236 [+0.196, +0.277] | +0.003 |
| all rows: net_twin_rho3_minus_rho1 | +0.016 [-0.052, +0.088] | +0.015 [-0.038, +0.072] | -0.001 |

## B.60 relative errors

Old: r2-r9 (filters' addendum r3-r9), B.39 rows (40,000 subset). D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| T20: share within 20 %, estimator minus causal GP; interval above 0 | -0.002 [-0.012, +0.006] | does not hold | -0.001 [-0.012, +0.006] | does not hold |
| T20h: share within 20 %, estimator minus hold; interval above 0 | +0.021 [+0.011, +0.031] | holds | +0.022 [+0.010, +0.031] | holds |
| T10: share within 10 %, estimator minus causal GP; interval above -0.02 | -0.007 [-0.020, +0.003] | does not hold | -0.006 [-0.022, +0.004] | does not hold |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| low third: within 20 %, network - GP | -0.038 [-0.070, -0.013] | -0.039 [-0.074, -0.014] | -0.001 |
| rows 1-3 after an absent frame: within 20 %, network - GP | +0.067 [+0.042, +0.092] | +0.065 [+0.045, +0.087] | -0.002 |
| all rows: share c|0.20 | +0.770 [+0.749, +0.791] | +0.771 [+0.750, +0.792] | +0.001 |
| all rows: share gp|0.20 | +0.772 [+0.753, +0.792] | +0.773 [+0.754, +0.792] | +0.001 |
| all rows: share c|0.10 | +0.562 [+0.539, +0.585] | +0.564 [+0.537, +0.588] | +0.001 |
| all rows: share gp|0.10 | +0.570 [+0.549, +0.591] | +0.570 [+0.549, +0.592] | +0.001 |
| filters, all rows: share kfx|0.20 | +0.783 [+0.763, +0.803] | +0.784 [+0.763, +0.803] | +0.000 |
| filters, all rows: share kfx|0.10 | +0.583 [+0.561, +0.606] | +0.584 [+0.562, +0.605] | +0.001 |
| filters, low third by median T_i: within 20 %, KFX - GP | +0.012 [+0.002, +0.023] | +0.011 [+0.001, +0.022] | -0.000 |
| filters, middle third by median T_i: within 20 %, KFX - GP | +0.012 [+0.008, +0.017] | +0.012 [+0.008, +0.016] | -0.000 |
| filters, high third by median T_i: within 20 %, KFX - GP | +0.009 [+0.006, +0.013] | +0.009 [+0.006, +0.013] | -0.000 |

## B.61 absolute and robust scores

Old: part A r2-r9 (filters r3-r9), B.39 rows (40,000 subset). D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| A1: Huber skill (200 eV) of the estimator over the causal GP | +0.074 [+0.052, +0.094] | holds | +0.081 [+0.064, +0.098] | holds |
| A2: Huber skill of the estimator over the twin | +0.052 [+0.031, +0.075] | holds | +0.056 [+0.032, +0.078] | holds |
| A3: trimmed skill (1 % of rows by hold error removed) over the causal GP | +0.067 [+0.029, +0.101] | holds | +0.068 [+0.027, +0.104] | holds |
| A4: trimmed skill over the twin | +0.070 [+0.037, +0.100] | holds | +0.073 [+0.034, +0.102] | holds |
| A5: Huber skill of KFX over KF0 (r3 to r9) | +0.067 [+0.058, +0.076] | holds | +0.067 [+0.058, +0.076] | holds |
| B1: the 64 discharges with the largest reduction removed: over the causal GP | +0.030 [+0.002, +0.055] | holds | +0.035 [+0.007, +0.057] | holds |
| B2: the 64 removed: over the twin | +0.027 [-0.004, +0.051] | does not hold | +0.026 [-0.006, +0.051] | does not hold |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| part A: rmse|c | 184.6 [174.4, 194.6] | 183.5 [173.7, 193.4] | -1.074 |
| part A: rmse|gp | 193.4 [182.9, 203.9] | 192.8 [182.4, 203.6] | -0.609 |
| part A: rmse_minus_gp_eV|c | -8.8 [-11.3, -6.3] | -9.3 [-11.5, -6.9] | -0.465 |
| part A: rmse_minus_gp_pct|c | -4.550 [-5.747, -3.277] | -4.805 [-5.911, -3.593] | -0.255 |
| part A: rmse_minus_gp_eV|t | -4.192 [-6.086, -2.238] | -4.256 [-6.064, -2.391] | -0.064 |
| part A: mae_minus_gp_eV|c | -3.026 [-4.668, -1.493] | -3.474 [-4.902, -1.958] | -0.448 |
| part A: skill_huber|c|hold | +0.265 [+0.244, +0.286] | +0.271 [+0.254, +0.288] | +0.006 |
| part A: skill_trim|c|hold | +0.231 [+0.188, +0.271] | +0.231 [+0.188, +0.271] | -0.000 |
| part A: skill_huber|c|gp | +0.074 [+0.052, +0.094] | +0.081 [+0.064, +0.098] | +0.007 |
| part A: skill_trim|c|gp | +0.067 [+0.029, +0.101] | +0.068 [+0.027, +0.104] | +0.001 |
| part A: skill_huber|c|t | +0.052 [+0.031, +0.075] | +0.056 [+0.032, +0.078] | +0.003 |
| part A: skill_trim|c|t | +0.070 [+0.037, +0.100] | +0.073 [+0.034, +0.102] | +0.003 |
| part A: skill_huber|c|off | -0.057 [-0.083, -0.030] | -0.048 [-0.072, -0.026] | +0.008 |
| part A: skill_trim|c|off | -0.070 [-0.119, -0.027] | -0.069 [-0.119, -0.025] | +0.002 |
| filters: skill_huber|kfx|kf0 | +0.067 [+0.058, +0.076] | +0.067 [+0.058, +0.076] | -0.000 |
| filters: skill_trim|kfx|kf0 | +0.084 [+0.074, +0.096] | +0.084 [+0.073, +0.096] | -0.000 |
| filters: skill_huber|kfx|gp | +0.085 [+0.073, +0.096] | +0.084 [+0.073, +0.095] | -0.000 |
| filters: skill_trim|kfx|gp | +0.073 [+0.054, +0.090] | +0.071 [+0.054, +0.088] | -0.001 |
| filters: rmse|kfx | 188.1 [177.5, 198.2] | 187.5 [177.1, 198.0] | -0.525 |
| filters: rmse|kf0 | 192.9 [182.5, 203.1] | 192.3 [182.1, 202.8] | -0.553 |
| part B: share of the reduction held by the 64 discharges | +0.7692 | +0.7356 | -0.034 |

## B.66 regimes

Old: r2-r9 (filters r3-r9), B.39 rows (40,000 subset). D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| R1: class A (no bursts): increment over the twin above zero | +0.056 [-0.008, +0.108] | does not hold | +0.051 [-0.018, +0.106] | does not hold |
| R2: class C (frequent bursts): increment over the twin above zero | +0.025 [-0.009, +0.052] | does not hold | +0.033 [+0.008, +0.061] | holds |
| R3: inter-burst phase of classes B and C: increment over the twin above zero | +0.050 [+0.027, +0.075] | holds | +0.058 [+0.037, +0.079] | holds |
| R4: increment in class A minus class C: interval excludes zero | +0.030 [-0.028, +0.082] | does not hold | +0.018 [-0.048, +0.071] | does not hold |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| class A: increment | +0.056 [-0.008, +0.108] | +0.051 [-0.018, +0.106] | -0.005 |
| class A: over causal GP | +0.060 [-0.006, +0.113] | +0.056 [-0.012, +0.108] | -0.004 |
| class A: discharges | 251.0 | 251.0 | +0.000 |
| class B: increment | +0.072 [+0.038, +0.105] | +0.079 [+0.047, +0.109] | +0.007 |
| class B: over causal GP | +0.109 [+0.070, +0.146] | +0.117 [+0.077, +0.153] | +0.008 |
| class B: discharges | 201.0 | 201.0 | +0.000 |
| class C: increment | +0.025 [-0.009, +0.052] | +0.033 [+0.008, +0.061] | +0.007 |
| class C: over causal GP | +0.091 [+0.059, +0.119] | +0.098 [+0.073, +0.123] | +0.007 |
| class C: discharges | 159.0 | 159.0 | +0.000 |
| between bursts (B, C): increment | +0.050 [+0.027, +0.075] | +0.058 [+0.037, +0.079] | +0.008 |
| between bursts (B, C): over causal GP | +0.095 [+0.070, +0.120] | +0.104 [+0.082, +0.124] | +0.008 |

## B.65: inclusive networks scored with the cut harness on the D1 rows

Old: inclusive population (values above 3 keV kept in training, scoring and references), r2-r4, 40,000 subset. D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| I1: network over causal GP, MSE skill (inclusive networks, cut population, D1 rows) | +0.109 [+0.070, +0.146] | holds | -0.092 [-0.168, -0.044] | does not hold |
| I2: network over twin, MSE skill (inclusive networks, cut population, D1 rows) | -0.002 [-0.017, +0.010] | does not hold | -0.040 [-0.104, -0.006] | does not hold |
| I3: network over causal GP, Huber skill (inclusive networks, cut population, D1 rows) | +0.037 [+0.003, +0.070] | holds | -0.151 [-0.200, -0.106] | does not hold |
| I4: network over twin, Huber skill (inclusive networks, cut population, D1 rows) | -0.012 [-0.038, +0.006] | does not hold | -0.040 [-0.071, -0.012] | does not hold |

| main-text quantity | old | D1 | difference |
|---|---|---|---|
| c over c39, MSE skill | - | -0.205 [-0.280, -0.148] | - |
| t over t39, MSE skill | - | -0.098 [-0.121, -0.076] | - |
| c39 over gp, MSE skill | - | +0.094 [+0.071, +0.114] | - |
| c39 over t39, MSE skill | - | +0.052 [+0.030, +0.075] | - |

## Delay replay (L = 1, 2) and 50 ms thinning replay

Old: four file-level splits, B.39 four-split network (cap 80, initialisation = split seed), B.30/B.42 row sets; a statement held on at least three of four splits. D1: r2-r4 pooled, every eligible row (no cap), refitted filters (training discharges, held removed).

| statement | old | old verdict | D1 | D1 verdict |
|---|---|---|---|---|
| T1 (L = 1) | s42 +0.135 [+0.095, +0.174]; s1 +0.097 [+0.066, +0.128]; s7 +0.137 [+0.094, +0.181]; s123 +0.099 [+0.063, +0.141] (4 of 4 above 0) | holds | +0.111 [+0.090, +0.131] | holds |
| T1 (L = 2) | s42 +0.165 [+0.123, +0.206]; s1 +0.075 [+0.043, +0.110]; s7 +0.130 [+0.082, +0.175]; s123 +0.088 [+0.052, +0.129] (4 of 4 above 0) | holds | +0.114 [+0.094, +0.134] | holds |
| T2 (L = 2, over the twin) | s42 +0.105 [+0.067, +0.146]; s1 +0.043 [+0.016, +0.073]; s7 +0.138 [+0.110, +0.166]; s123 +0.076 [+0.030, +0.127] (4 of 4 above 0) | holds | +0.087 [+0.066, +0.108] | holds |
| H1 (k = 5, over the causal GP) | s42 +0.127 [+0.091, +0.160]; s1 +0.070 [+0.035, +0.106]; s7 +0.110 [+0.080, +0.143]; s123 +0.100 [+0.066, +0.131] (4 of 4 above 0) | holds | +0.112 [+0.096, +0.129] | holds |
| H2 (k = 5, over the twin) | s42 +0.091 [+0.057, +0.128]; s1 +0.054 [+0.030, +0.081]; s7 +0.158 [+0.132, +0.183]; s123 +0.060 [+0.018, +0.109] (4 of 4 above 0) | holds | +0.089 [+0.069, +0.109] | holds |
