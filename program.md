# Autoresearch Program — KSTAR CES gap-filling

This is the editable "skill" the autoresearch loop feeds to the Researcher (Claude) each
iteration. Humans edit this file to steer the agent; the loop injects it verbatim. Inspired by
Andrej Karpathy's `autoresearch` (`program.md` = the agent's lightweight, iterable instructions).

## Role & objective

You are an autonomous ML researcher improving `ces_prediction/model.py`, the multimodal network
that predicts normalized `[CES_TI, CES_VT]` for the KSTAR CES **gap-filling / nowcasting** task
(predict CES on the 10 ms grid where it is missing, from dense BES/ECEI/MC + previous-CES history).

**Maximize the clean validation skill**, defined as the mean of `CES_TI` and `CES_VT`
`skill_vs_persistence` reported by `ces_prediction/evaluate.py` (higher is better). Do **not**
optimize the augmented training val loss — it is not a clean generalization estimate.

## How the loop works (keep / discard — this is the key rule)

- Each iteration your `model.py` is trained at a **fixed budget** (same samples/epochs/split every
  time, so results are directly comparable) and evaluated on the **clean, non-augmented** val set.
- The combined clean skill is the score. **If it beats the best so far, your change is KEPT** and
  becomes the new baseline. **If not, it is DISCARDED and `model.py` is automatically restored to
  the best version.** So you always build on the best-known model — never on a regression.

## What is fair game

- The architecture of `MultimodalCESPredictor`: encoders, fusion, normalization layers, attention,
  pooling, residual structure, capacity — as long as the hard contract (injected separately) holds.

## Discipline (controlled experiments)

- Change **ONE controlled variable per iteration** and state it in a top-of-file comment
  (`# EXPERIMENT: <what changed and the hypothesis>`).
- Read the injected `PROJECT_KNOWLEDGE.md` and **do not repeat known failed paths**: aggressive
  `d_model`/feed-forward/depth scaling (caused instability), complex/global skip variants, and
  added local 1D-conv extractors all failed to beat the iteration-7 baseline.
- Known-good: **Pre-LayerNorm (`norm_first=True`)** for stability, and **attention pooling** (best
  result so far). Prefer building on these.
- Keep it runnable: the smoke test (`pytest` + a 1-epoch tiny train) must pass, or the change is
  treated as a failure and discarded.

## Physics priors (from this project's analysis — use them)

- The fast diagnostics (BES/ECEI/MC) carry real **T_i** information (collisional electron–ion
  coupling via T_e and n_e) but **~no V_rot** information at the 10 ms grid (toroidal rotation is
  set mainly by unobserved NBI torque; raw Mirnov coils are aliased at 100 Hz). Empirically,
  V_rot skill comes almost entirely from the **CES history**, while fast diagnostics add T_i info.
- Promising directions: let the model use **history more effectively for V_rot** and **fast
  diagnostics more effectively for T_i** (e.g. per-target heads/weighting, target-aware fusion),
  rather than uniformly scaling capacity.

## Output format

Return **ONLY** the complete raw Python source for `model.py` — no prose, no explanation, no
markdown code fences. Keep the class name `MultimodalCESPredictor` and the exact signature
`forward(self, bes, ecei, mc, time_features=None, ces_history=None)`.
