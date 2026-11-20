# B.77 — the session lead's checks

Written 2026-11-20 after the batch was read, with code written independently of the implementing agent's
(`ces_prediction/experiments/b77_correct/lead_checks/verify_b77.py`). Skill of A over B = 1 - SSE(A) / SSE(B) on identical rows; target L5.

`py ces_prediction/experiments/b77_correct/lead_checks/verify_b77.py .`

```
(A) own pooling of the stored estimates
  A1 C0 over O, V1 +0.572 | A2 C0 over T0, V1 +0.147 | A3 CX over C0, V1 +0.048 | A4 V2 +0.013 | A5 V3 -0.009 | A6 V4a +0.010 (V4b +0.043) | A10 CE over C0, V1 +0.014 | placebo CXp over C0, V1 -0.004
(B) own refit
  V4a       rows 18277: C0 largest |mine - stored| 0.00e+00 eV; RMSE O 234.7, C0 120.7, CX mine 120.1 (stored 120.1); CX over C0 mine +0.010, stored +0.010
  V1_r2_k0  rows 35740: C0 largest |mine - stored| 0.00e+00 eV; RMSE O 188.0, C0 123.7, CX mine 121.1 (stored 121.2); CX over C0 mine +0.042, stored +0.040
```

- A: the values of the statements A1 to A6, A10 and of the placebo equal those of `b77_correct.md`.
- B: C0 refitted from B.75's cached D10 history inputs and the reported value equals the stored estimates bit for bit in both units. CX with the lead's own fast and anchor scalars (equal to the agent's up to rounding, `b75_lead_checks.md`) gives +0.010 under V4 (fit to 31991; stored +0.010) and +0.042 in the first fold of re-partition r2 (stored +0.040).
