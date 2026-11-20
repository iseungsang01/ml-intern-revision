# B.84 — the session lead's checks

Written 2026-11-21 after the batch was read, with code written independently of the implementing agent's readout
(`ces_prediction/experiments/b84_onefit/lead_checks/verify_b84.py`): the eleven statement values, the RMSE values and the increment inside the one fit pooled again from the stored estimates. Target L5; V1 is re-partition r2.

`py ces_prediction/experiments/b84_onefit/lead_checks/verify_b84.py .`

```
(A) statement values from the stored estimates, own pooling (target L5; V1 = re-partition r2)
   U1  V1  TXall over TX    under D650: +0.000  (160343 rows)
   U2  V1  TXall over TX    under D10 : -0.062  (160343 rows)
   U3  V1  TXall over TX    under S   : +0.047  (147837 rows)
   U4  V1  TXall over TXa   under D650: +0.502  (160343 rows)
   U5  V1  TXall over T0all under D650: +0.359  (160343 rows)
   U6  V1  TXall over T0    under K10 : -0.036  (60523 rows)
   U7  V3  TXall over TX    under D650: +0.080  (112199 rows)
   U8  V1  TXraw over TXall under D650: -0.327  (160343 rows)
   U9  V3  TXrel over TX    under D650: +0.089  (112199 rows)
   U10 V4a TXrel over T0    under S   : -0.004  (7180 rows)
   U11 V1  TXrel over TX    under D650: -0.032  (160343 rows)
(B) RMSE (eV), per-rule TX / applied TXa / TXall / TXraw / TXrel
   V1  D10   124.0 / - / 127.8 / 134.5 / -
   V1  D100  144.3 / 175.8 / 148.1 / 160.0 / -
   V1  D650  184.1 / 260.8 / 184.1 / 212.1 / 187.1
   V1  D1000 192.0 / 287.9 / 193.9 / 222.1 / -
   V1  S     177.9 / 212.1 / 173.7 / 189.1 / 178.6
   V1  K10   144.0 / - / 163.5 / 172.9 / -
   V3  D10   148.5 / - / 151.2 / 160.6 / -
   V3  D100  183.3 / 208.0 / 185.0 / 202.0 / -
   V3  D650  241.7 / 298.0 / 231.8 / 268.4 / 230.7
   V3  D1000 247.8 / 321.8 / 238.9 / 274.7 / -
   V3  S     218.5 / 239.4 / 217.7 / 241.5 / 211.3
   V3  K10   192.0 / - / 189.5 / 206.4 / -
   V4a D10   138.4 / - / 145.0 / 152.5 / -
   V4a D100  149.5 / 159.9 / 153.0 / 165.0 / -
   V4a D650  176.7 / 181.9 / 172.7 / 187.1 / 176.9
   V4a D1000 189.6 / 188.5 / 181.6 / 205.0 / -
   V4a S     182.3 / 181.0 / 174.0 / 202.8 / 185.2
   V4a K10   157.5 / - / 157.2 / 163.8 / -
(C) the increment inside the one fit, TXall over T0all, V1
   D10 +0.081, D50 +0.165, D100 +0.225, D300 +0.340, D650 +0.359, D1000 +0.372, S +0.281, K10 +0.162
```

All eleven values, the RMSE values of the per-rule, applied, single-fit, raw-window and level-free estimators and the increment of the single fit along the rules equal those of `b84_onefit.md`.
