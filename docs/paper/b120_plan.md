# Focused TCN training plan

Author requested execution,2026-11-28. Registration: ces_prediction/experiments/PREREGISTRATION_B120.md.

One original B105 TCN35 unit/seed, same architecture/data/F+L5 objective, fresh seeded trajectory because old checkpoints lack RNG state. First original trajectory must reproduce numerically. Minimum epochs and a terminal low-learning-rate plateau replace the old six-epoch early stop; safety caps are not convergence. TRAIN/VAL pilot only; no TEST inference or GP+fast improvement claim. Successful one-percent VAL gate motivates a separate registered evaluation.

Resource/process checks preserve other jobs. In particular a paused B119 is left paused; an active B119 must finish/stop before this queue starts GPU work. Qualification is real-data forward/causal/stream/backward with zero optimizer steps. One production fit only. Full recovery state is saved at each complete validation boundary. CPU/NumPy checks precede published numerical reports and commit/push.

Live: data/.b120/workflow/status.json, data/.b120/progress.json; human status docs/paper/b120_status.md. Final audited pilot: docs/paper/b120_training.json/md. No automatic evaluation or additional fits.

Operational background adapter: PREREGISTRATION_B120_ADDENDUM1.md -> b120_background/run120.py installs hidden subprocess policy in queue and scientific children before imports; workflow/background1 preserves the original wait clock. Original zero-update queue receipts retained.

ADDENDUM2 corrects the hidden adapter to rewrite Python launches only, leaving Git source arguments untouched. Background1 failed before qualification/fit; its receipt remains. Active operation uses workflow/background2 and original wait deadline.
