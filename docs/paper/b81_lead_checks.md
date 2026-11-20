# B.81 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's readout
(`ces_prediction/experiments/b81_controls/lead_checks/verify_b81.py`): the seventeen statement values pooled again from the stored
estimates, and the correction of the permuted fit set against that of TX. Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; target L5.

`py ces_prediction/experiments/b81_controls/lead_checks/verify_b81.py .`

```
(A) statement values from the stored estimates, own pooling (target L5)
   C1 V1 TXp over T0, D650: -0.004 | C2 V1 TXp over T0, S: -0.000 | C3 V1 TX over TXo, D650: +0.273 | C4 V1 TX over TXo, S: +0.276 | C5 V2 TX over TXo, D650: +0.298
   C6 V3 TX over TXo, D650: +0.426 | C7 V4a TX over TXo, D650: +0.357 | C8 V2 TX over TXo, S: +0.296 | C9 V3 TX over TXo, S: +0.274
   E1 V2 TE over T0, D650: +0.163 | E2 V3 TE over T0, D650: +0.058 | E3 V2 TB over T0, D650: +0.225 | E4 V3 TB over T0, D650: +0.122
   E5 V2 TE over T0, S: +0.181 | E6 V3 TE over T0, S: +0.160 | E7 V2 TB over T0, S: +0.221 | E8 V3 TB over T0, S: +0.171
   TXo over T0: D650 V1 +0.087, V2 -0.023, V3 -0.411, V4a -0.187, V4b +0.030; S V1 +0.070, V2 -0.005, V3 -0.059, V4a +0.038, V4b -0.120
   TXp over T0: D650 V1 -0.004, V2 -0.005, V3 -0.001, V4a +0.012, V4b -0.012; S V1 -0.000, V2 +0.000, V3 -0.000, V4a +0.022, V4b +0.029
(B) correlation over the test rows between the correction of the permuted fit and that of TX (both minus T0)
   V1_r2_k0  D650 rows  26669: permuted +0.047; anchor row +0.472; sd of the correction (eV): TX 154.7, permuted 19.9, anchor row 108.7
   V1_r2_k0  S    rows  24422: permuted +0.048; anchor row +0.391; sd of the correction (eV): TX 127.7, permuted 22.1, anchor row 87.4
   V3_q2     D650 rows  43406: permuted +0.127; anchor row +0.376; sd of the correction (eV): TX 152.5, permuted 21.6, anchor row 228.5
   V3_q2     S    rows  40944: permuted +0.004; anchor row +0.432; sd of the correction (eV): TX 153.0, permuted 21.8, anchor row 136.1
   V4a       D650 rows  10146: permuted +0.174; anchor row +0.470; sd of the correction (eV): TX 119.3, permuted 23.7, anchor row 113.6
   V4a       S    rows   7180: permuted +0.189; anchor row +0.565; sd of the correction (eV): TX 94.4, permuted 28.2, anchor row 103.5
```

- A: the values of C1 to C9 and E1 to E8, and the reported cells of TXo over T0 and TXp over T0 for every design and both rules, equal those of `b81_controls.md`.
- B: over the test rows the correction of the permuted fit (TXp minus T0) has a standard deviation of 20 to 28 eV against 94 to 155 eV for TX, and its correlation with the correction of TX is +0.004 to +0.189: the permuted fit does not follow the row's fast inputs. The correction of the fit with the fast inputs of the anchor row is as large as that of TX and correlates with it at +0.376 to +0.565.
- An own fit of the two controls exists for one unit from the check of B.75 (`b75_lead_checks.md` part C: chronological fit to 31991, D650, own fast scalars): permuted +0.013 (here +0.012), at the anchor row -0.128 (here -0.187), present +0.205 over T0 (here +0.237). The differences are of the size by which fits on this unit move with the rounding of the fast scalars.
