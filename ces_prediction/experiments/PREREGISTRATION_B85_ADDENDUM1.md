# B.85 addendum 1 — dispositions of external design review

2026-11-22, before any input audit, smoke or training. Claude Opus reviewed a redacted methodological summary through the author-authorized API. Exact prompt, returned model, usage and response: docs/paper/b85_claude_design_review.md. The registration is unchanged.

- Normalization is fitted once per unit on all finite Ti and complete fast grid rows in TRAINING files, regardless of availability rule, and frozen for every train/validation/test rule. Hidden-value mutations in validation/test never refit statistics. This clarifies the registered train-file-only normalization, not scenario-specific scaling.
- L5 remains the B.73 target: same-segment next five nominal timestamps with at least three values. Report counts of available three/four/five values where feasible; these counts never select rows beyond the registered masks or enter inputs.
- State resets at every independently passed original segment, between files, between validation rules and between evaluated scenarios. Every rule is applied consistently to the entire input prefix. This is why random training conditions change only at boundaries, not inside a stateful sequence.
- S visibility is monotone within a segment: a newly observed gap replaces the previous pre-gap cutoff with a later pre-gap cutoff. Before the first gap no CES is visible. Claude's proposal to alter S on grounds of backward movement is rejected; include a worked real-file or index example in checks.
- Fixed-delay eligibility combines the time tolerance with a strictly earlier row index. Sorted timestamps and strictly prior indexing prevent the target row from entering its own input. Do not alter B.75/B.78 rules or tolerance based on the review.
- Retain exact comparator metadata, training/validation file lists, support masks and missing-baseline labels. Neural early stopping reserves files that trees used in fitting; disclose this difference rather than conceal it.
- A lagged cross-correlation is not a proof of timing causality and is not added as a leakage pass criterion. Use input-prefix mutation tests and existing acquisition-timing evidence. Additional architecture ablations require a new registration; no loss/model/hyperparameter changes are introduced here.

The review is methodological advice, not an independent experimental result or an acceptance estimate. No raw shot arrays, secrets or author identifying details were sent.
