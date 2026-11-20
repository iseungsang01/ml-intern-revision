# Why the increment under the schedule rule is not established on the last block of shots

Unregistered diagnostic by the session lead, 2026-11-20, after B.75 and B.79 (`ces_prediction/experiments/b75_availability/lead_checks/last_block.py`). It uses the stored estimates of B.75 under the random design (V1) and rule S, target L5, and input-only quantities. Nothing here is a registered statement; no interval is given. Strata are on inputs or on the block of shots.

Question. Under the beam-modulation schedule the fast-channel increment holds in the blocks of shots up to 31991 and 32002 to 32310 and is not established from 32312 on (B.75: +0.066 [-0.163, +0.248] under the random design; B.79: with beam inputs +0.086 [-0.136, +0.231]). What is different there?

`py ces_prediction/experiments/b75_availability/lead_checks/last_block.py .`

```
(1) inputs on the S set (target L5), by block of the discharge
   block         rows  disch clus | age median (10-90 %) s | L5 median eV | RMSE of the frame's own reading / hold fresh / hold S | P1, P2 median MW | rows with NBI-2 off
   up to 31991  119655   349   44 | 0.66 (0.14-1.35) |    638 | 159.5 / 170.7 / 259.2 | 2.44, 1.63 | 0.063
   32002-32310   21002   104   14 | 0.60 (0.12-1.26) |    518 | 240.1 / 250.8 / 366.5 | 1.57, 1.54 | 0.168
   from 32312     7180    50   14 | 0.66 (0.14-1.31) |    404 | 247.4 / 257.3 / 203.0 | 2.11, 1.62 | 0.079
(2) how much the level moves since the newest visible value, and how it goes with the fast levels (no estimator)
   block        | sd of ln(L5 / M5 of the visible values) | Spearman of that with the ECEI change, with the BES change | sd of the ECEI change, of the BES change | relative scatter of the own reading |O/L5 - 1| median
   up to 31991  | 0.414 | +0.269, +0.287 | 0.342, 0.138 | 0.068
   32002-32310  | 0.562 | +0.329, +0.301 | 0.384, 0.194 | 0.109
   from 32312   | 0.569 | -0.013, +0.166 | 0.542, 0.121 | 0.125
(3) TX over T0 under S by the age of the newest visible value (V1, stored estimates)
   block        | 0.0-0.1 s (rows; T0, TX RMSE; skill) | 0.1-0.3 s (rows; T0, TX RMSE; skill) | 0.3-1.0 s (rows; T0, TX RMSE; skill) | 1.0-99.0 s (rows; T0, TX RMSE; skill)
   up to 31991  | 8347; 194, 171; +0.220 | 19430; 224, 180; +0.357 | 59590; 226, 177; +0.389 | 32288; 195, 171; +0.234
   32002-32310  | 1687; 182, 164; +0.187 | 3762; 234, 202; +0.253 | 10998; 240, 192; +0.360 | 4555; 270, 225; +0.303
   from 32312   | 500; 117, 102; +0.244 | 1208; 174, 157; +0.188 | 3506; 189, 177; +0.122 | 1966; 154, 169; -0.205
(4) discharges of the last block, S set: contribution to SSE(T0) - SSE(TX)
   discharges 50; total reduction 1.43e+07 eV^2 (skill +0.066); sum of the negative contributions -2.18e+07, of the positive 3.6e+07
   without the three discharges with the most negative contribution: skill +0.133; without the three most positive: +0.010
   shot  cluster rows | share of the block's SSE(T0) | RMSE T0, TX, own reading | L5 median | P1, P2 MW | age s | scatter
   32312   69   233 | 0.012 |   106   218    70 |   187 | 1.38 0.00 | 0.78 | 0.183   contribution -8.45e+06
   32333   70   146 | 0.009 |   115   182   164 |  1018 | 2.09 2.31 | 0.75 | 0.107   contribution -2.91e+06
   32405   73    80 | 0.017 |   217   267   300 |   160 | 2.00 1.43 | 0.81 | 0.213   contribution -1.93e+06
   32463   77    25 | 0.006 |   221   320   151 |   206 | 1.50 0.88 | 0.14 | 0.407   contribution -1.34e+06
   32406   73   215 | 0.086 |   293   302   302 |   369 | 2.17 1.43 | 0.66 | 0.127   contribution -1.18e+06
   32334   70   146 | 0.018 |   165   185   187 |   868 | 2.12 2.31 | 0.75 | 0.062   contribution -1.05e+06
   32662   85   162 | 0.038 |   224   193   125 |   309 | 1.51 0.96 | 0.42 | 0.300   contribution +2.09e+06
   32384   72    62 | 0.026 |   298   206   155 |   467 | 1.48 0.88 | 0.40 | 0.158   contribution +2.88e+06
   32410   73   120 | 0.038 |   262   204   315 |   178 | 2.18 1.53 | 0.62 | 0.331   contribution +3.23e+06
   32663   85   193 | 0.054 |   246   202   246 |   500 | 1.51 1.62 | 0.51 | 0.299   contribution +3.8e+06
   32338   70   136 | 0.035 |   235   164   240 |   678 | 2.45 2.33 | 0.75 | 0.137   contribution +3.82e+06
   32500   79    45 | 0.040 |   439   292   125 |   243 | 2.09 2.05 | 0.25 | 0.131   contribution +4.82e+06
   block up to 31991: discharges 349, share with a negative contribution 0.232 (last block: 0.420)
   block 32002-32310: discharges 104, share with a negative contribution 0.250 (last block: 0.420)
(5) by the level of the visible history (median of the last five visible values; a stratum on an input), S set
   level        | all blocks: rows; Spearman with the ECEI change, with the BES change; scatter of the own reading; TX over T0 | the same for the last block | for the two earlier blocks
   below 250 eV | 17884; +0.27, +0.23; 0.166; +0.401 | 1009; -0.06, +0.16; 0.192; +0.243 | 16875; +0.30, +0.23; 0.165; +0.409
   250-400      | 19008; +0.11, +0.21; 0.107; +0.222 | 2806; -0.08, +0.24; 0.119; -0.083 | 16202; +0.14, +0.21; 0.105; +0.255
   400-600      | 28651; +0.18, +0.29; 0.056; +0.128 | 1378; -0.17, +0.33; 0.143; +0.272 | 27273; +0.17, +0.29; 0.054; +0.122
   600-900      | 48729; +0.28, +0.40; 0.057; +0.269 | 1039; +0.19, -0.01; 0.101; +0.194 | 47690; +0.28, +0.41; 0.057; +0.271
   above 900    | 33565; +0.37, +0.20; 0.083; +0.404 | 948; +0.08, -0.12; 0.101; -0.020 | 32617; +0.38, +0.21; 0.082; +0.408
   share of S-set rows by level, per block (below 250 / 250-400 / 400-600 / 600-900 / above 900):
   up to 31991  0.094 / 0.109 / 0.204 / 0.361 / 0.232
   32002-32310  0.266 / 0.148 / 0.135 / 0.216 / 0.234
   from 32312   0.141 / 0.391 / 0.192 / 0.145 / 0.132
(6) by the state of NBI-2 at row t (a stratum on an input), S set: rows; Spearman with the ECEI change; TX over T0
   NBI-2 on   | all: 136162; +0.25; +0.322 | last block: 6612; -0.03; +0.105 | earlier blocks: 129550; +0.26; +0.330
   NBI-2 off  | all: 11675; +0.45; +0.414 | last block: 568; +0.22; -1.148 | earlier blocks: 11107; +0.46; +0.442
(7) the notch as a common event (no estimator): value in the first row after an absent step over the median of the five rows before it, by block of the discharge
   up to 31991  notches  1100 | T_i 0.819 (n 976) | ECEI 0.902 (10-90 %: 0.563-1.111) | BES 0.656
   32002-32310  notches   235 | T_i 0.837 (n 193) | ECEI 0.967 (10-90 %: 0.719-1.121) | BES 0.643
   from 32312   notches    73 | T_i 0.757 (n 54) | ECEI 1.001 (10-90 %: 0.352-1.283) | BES 0.669
(8) the ECEI channels themselves, per discharge (median over discharges of the block), grid rows of blocks of at least 100 rows
   up to 31991  discharges 374 | raw array mean, median 0.4078 | share of rows at or below zero 0.0045 | sd of the 100 ms log change of the smoothed level 0.128 | mean correlation between the four channels 0.911
   32002-32310  discharges 111 | raw array mean, median 0.9713 | share of rows at or below zero 0.0083 | sd of the 100 ms log change of the smoothed level 0.075 | mean correlation between the four channels 0.841
   from 32312   discharges  76 | raw array mean, median 0.2395 | share of rows at or below zero 0.0183 | sd of the 100 ms log change of the smoothed level 0.074 | mean correlation between the four channels 0.654
```

## Reading

1. **Not the age of the history.** The age of the newest visible value has the same distribution in the three blocks (median 0.66, 0.60, 0.66 s).
2. **Not the temperature level and not the beam mix.** At equal level of the visible history the earlier blocks keep the relation between the change of the ECEI level and the change of the ion-temperature level (Spearman +0.14 to +0.38 in every class) and the last block does not (-0.17 to +0.19). The state of NBI-2 does not separate the blocks either.
3. **The ECEI of the last block is another signal.** Without any estimator: from the newest visible value to the present, the change of the ECEI level goes with the change of the ion-temperature level in the two earlier blocks (Spearman +0.269 and +0.329) and not in the last (-0.013); the BES change goes with it in all three (+0.287, +0.301, +0.166). At a beam notch, an event common to all discharges, T_i falls in every block (0.819, 0.837, 0.757 of the level before) and so does BES (0.656, 0.643, 0.669); the ECEI level falls in the first block (0.902), less in the second (0.967) and not in the last (1.001, with a spread from 0.352 to 1.283). The four ECEI channels are correlated with each other at 0.911 in the first block, 0.841 in the second and 0.654 in the last (median over discharges), and the raw level is lowest there (0.24 against 0.41 and 0.97).
4. **Consequence for the estimate.** In the last block the reading is more scattered relative to its level (median relative departure 0.125 against 0.068 in the first block) and the value before the notch is a closer estimate of the following level than the previous frame is (RMSE 203.0 against 257.3 eV), so little is lost by the delay and the ECEI inputs add noise: at ages above 1 s the fast inputs raise the error (154 to 169 eV). Three discharges carry most of the loss (without them the skill is +0.133).

**What this is and is not.** It says that in the last block the electron-temperature diagnostic that carries the increment elsewhere does not follow the common events and is less coherent across its channels. It does not say why: the files hold no toroidal field, no ECEI settings and no channel radii. A change of the field or of the ECEI local oscillator moves the radius the channels see; a lower optical depth has the same effect on the signal. [needs data that do not exist here: toroidal field per shot, ECEI configuration.]

**What can be tested with the files.** Whether an input-only measure of the ECEI signal (the coherence of its four channels in a discharge, its response to a notch) orders the increment in every block, not only between blocks. That would turn the limit into a condition that can be checked before the estimator is used. Registered as B.80.
