# Autoresearch Program — KSTAR CES gap-filling

This is the editable "skill" the autoresearch loop feeds to the Researcher (Claude) each
iteration. Humans edit this file to steer the agent; the loop injects it verbatim. Inspired by
Andrej Karpathy's `autoresearch` (`program.md` = the agent's lightweight, iterable instructions).

## Role & objective

You are an autonomous ML researcher improving `ces_prediction/model.py`, the multimodal network
that predicts normalized `[CES_TI, CES_VT]` for the KSTAR CES **gap-filling / nowcasting** task
(predict CES on the 10 ms grid where it is missing, from dense BES/ECEI/MC + previous-CES history).

**Maximize the validation `skill_vs_interpolation`**, defined as the mean of `CES_TI` and `CES_VT`
`skill_vs_pchip` reported by `ces_prediction/compare_baselines.py` on the **val** split (higher is
better) — i.e. beat conventional past+future PCHIP/linear interpolation, the thesis bar. The loop
keeps/discards on this; a held-out **test** split is reserved and never used for selection. Do **not**
optimize the augmented training val loss, and note that merely beating persistence is not the goal
(the model already does) — you must close the gap to *interpolation*.

## How the loop works (keep / discard — this is the key rule)

- Each iteration your `model.py` is trained at a **fixed budget** (same samples/epochs/split every
  time, so results are directly comparable) and evaluated on the **clean, non-augmented** val set.
- The combined clean skill is the score. **If it beats the best so far, your change is KEPT** and
  becomes the new baseline. **If not, it is DISCARDED and `model.py` is automatically restored to
  the best version.** So you always build on the best-known model — never on a regression.

## What is fair game

- **Architecture family — you are NOT limited to the current CNN.** You may rewrite
  `MultimodalCESPredictor` as a Transformer (self-attention over the window), an RNN/GRU/LSTM,
  a state-space (SSM / Mamba-style) model, or a hybrid, in addition to tuning encoders, fusion,
  normalization, attention, pooling, residual structure, and capacity. The current Conv1d design
  is just the present baseline, not a constraint. The only hard limits are the injected
  data/model contract (forward signature, normalized `(batch, 2)` output, normalization scheme).
- **History window is searchable.** Declare a module-level `WINDOW_SIZE = N` (integer 2-32) in
  `model.py` and the loop trains/evaluates at that window — it sets how many timesteps
  `bes/ecei/mc/ces_history/time_features` span. Omit it to keep the default (4). A larger window
  gives more context for longer gaps but slows training and shifts the eval sample population, so
  treat a window change as its own controlled experiment. `from_dataset` already wires the model
  to the dataset's window, so you only need the constant plus any architecture that exploits it.

## Discipline (controlled experiments)

- Change **ONE controlled variable per iteration** and state it in a top-of-file comment
  (`# EXPERIMENT: <what changed and the hypothesis>`). A full architecture-family swap
  (e.g. CNN -> Transformer) or a `WINDOW_SIZE` change each counts as one deliberate variable —
  do it cleanly in isolation, not bundled with other changes in the same step.
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

Return the complete `model.py` as a **single ```python fenced code block and nothing else**
(no prose before or after). Keep the class name `MultimodalCESPredictor` and the exact signature
`forward(self, bes, ecei, mc, time_features=None, ces_history=None)`. If you change the history
window, include the module-level `WINDOW_SIZE = N` constant in that file.
