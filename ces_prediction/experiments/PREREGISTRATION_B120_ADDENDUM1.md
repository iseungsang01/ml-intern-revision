# B120 operational background adapter

2026-11-28, before qualification, first fit, or any optimizer update. Author explicitly requested fewer/no console windows and background execution. The original queue was stopped during its resource wait; no scientific child or fit directory existed. Its original started/status receipts are preserved. This addendum does not edit the original registration or scientific sources.

New b120_background/run120.py installs a process-wide subclass of subprocess.Popen adding CREATE_NO_WINDOW to all children, including inherited Git/provenance/runtime/nvidia-smi commands. Frozen train120.py and audit120.py are invoked through the same adapter, which installs this policy before their imports. The adapter does not alter numerical functions, scientific arguments, environment pins, model/optimizer/RNG, stopping rules, selection or evaluation. One original fresh fit remains queued, zero prior optimizer steps. No scientific retry or new initialization is added.

Use a separate workflow/background1 operational receipt with original scientific source hashes and committed operational source/addendum hashes. Preserve the original twelve-hour wait clock; do not reset it. Qualification, fit and audit stage safety durations are unchanged and have not started. The original task stays disabled; launch a new hidden task with no timed repeat or automatic restart. Resource gate still waits for active experiment processes and GPU usage. Existing B119 is untouched.

After audited numerical completion, independently retain operational hash/original-clock/unchanged-source receipt, then original publication/commit/push scope. No TEST access. New implementation and addendum committed/pushed before dispatch.
