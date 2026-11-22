# B.85 CUDA runtime incident and recovery

2026-11-22. This is a runtime audit, not a scientific performance readout. The author requested GPU execution. No unrelated GPU process was stopped or modified.

## Failure isolated before the registered batch

The prior smoke child completed all numerical checks and its one real-data epoch, printed `Complete`, then exited with native code `3221226505` (`0xc0000409`). Windows Application events named Python 3.14 and `ucrtbase.dll`. A completion file alone therefore did not establish a clean process exit.

`ces_prediction/experiments/b85_unified/diagnose_runtime85.py` isolated the failure using existing real grid segments and model weights. No generated observations were used. Commands used `py -X faulthandler -u` from the repository root.

| Probe mode | Calculation | Native process exit |
|---|---|---|
| `--mode cuda` | initialize CUDA, synchronize | 0 |
| `--mode linear --threads-before` | CUDA linear operation on actual model weights and real inputs | 0 |
| `--mode forward --threads-before` | cuDNN LSTM real-segment forward | `0xc0000409`, after normal return |
| `--mode backward --threads-before` | cuDNN LSTM real-segment forward/backward | `0xc0000409`, after normal return |
| `--mode backward --cleanup --threads-before` | same, explicit object deletion, GC, CUDA sync and cache cleanup | `0xc0000409` |
| `--mode backward --cleanup --threads-before --worker` | same on a joined worker thread | `0xc0000409` |
| `--mode backward --threads-before --disable-cudnn` | same LSTM equations through native ATen CUDA | 0 |

Observed software: PyTorch `2.11.0+cu128`, cuDNN `91900`, Python 3.14, RTX 5060 Laptop GPU. The tests localize the failure to the cuDNN LSTM execution/teardown path on this installation; they do not identify the exact underlying vendor defect. Ordinary CUDA initialization and linear algebra did not reproduce it. Explicit Python cleanup and worker-thread teardown did not fix it. No `os._exit` bypass was introduced.

## Registered correction and completion gate

Preregistration addendum 3 was committed and pushed before new smoke/full fitting. The B.85 model/trainer now set `torch.backends.cudnn.enabled=False`, retain CUDA tensors and float32, and record `cudnn_enabled: false` in configuration metadata. This is native ATen **CUDA**, not CPU training. Architecture, targets, loss, input availability, batch size, seeds and optimizer settings stay registered.

The supervising runner records `process_exit.json` atomically inside each run directory after the child exits, including return code and the completed configuration hash. `verify_complete` requires that attestation to have return code zero and the matching current configuration hash. Prediction and readout use the same verifier. An artifact from an unclean or unattested child cannot silently become accepted by a later skip-only invocation.

The 21:49 unclean smoke was preserved under `data/.b85/quarantine/smoke_X_native_abort_2149`. Earlier failed attempts remain in their existing quarantine/log locations. Diagnostic observations are recorded above; repeatable probe source remains in the new diagnostic script.

## Clean smoke and launch

Fresh X and H smokes both passed real-data forward/backward/finite-gradient, recurrent step agreement, reset and save/load checks, completed their one epoch, and exited **0**. Their per-run `process_exit.json` hashes match the new configurations. The two-smoke supervising runner also exited zero.

Preserved clean logs: `data/.b85/logs/smoke_aten_X_clean.log` and `smoke_aten_H_clean.log`. The hidden queue was launched with Python PID **947056**, stdout `data/.b85/queue_aten.log`, stderr `data/.b85/queue_aten.stderr.log`. It verifies/reuses the clean smokes and proceeds to the registered batch. Runtime truth is `queue_status.json`, `docs/paper/b85_status.md` and individual run logs. Full scientific results remain unavailable until all registered fits and checks finish.

At 22:00 local time the first registered full run, `V1_r2_k0__1701__X` (training PID 956032), had completed epoch 0 and continued into the next epoch. GPU observation at that time was 346 MiB allocated/used and 13% utilization; no out-of-memory error occurred. Queue start through the first logged epoch took approximately two minutes including data construction, validation-set construction and its internal plumbing checks. This is not a steady-state epoch-time estimate and is not an estimate for the complete 90-run batch.
