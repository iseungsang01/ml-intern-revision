# B.80 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's
(`ces_prediction/experiments/b80_ecei_gate/lead_checks/verify_b80.py`): the coherence of the four ECEI channels per discharge
recomputed from the grid, and the class skills and the gated estimator under V1 from the stored estimates of B.75. Target L5.

`py ces_prediction/experiments/b80_ecei_gate/lead_checks/verify_b80.py .`

```
D1 discharges with the measure: 590 of 635; low / middle / high: 170 / 135 / 285; median 0.895
   up to 31991  low / middle / high: 87 / 95 / 211; median 0.911; share low 0.221
   32002-32310  low / middle / high: 40 / 32 / 43; median 0.864; share low 0.348
   from 32312   low / middle / high: 43 / 8 / 31; median 0.684; share low 0.524
V1, rule S: TX over T0 by class low / middle / high: +0.218 / +0.292 / +0.370; high minus low +0.151; TG over T0 +0.282; TG over TX -0.067
V1, rule D650: TX over T0 by class low / middle / high: +0.143 / +0.319 / +0.396; high minus low +0.253; TG over T0 +0.310; TG over TX -0.040
```

The class counts, the medians per block, the class skills, the differences of G1 and G4 and the gated estimator's skills equal those of `b80_ecei_gate.md`.
