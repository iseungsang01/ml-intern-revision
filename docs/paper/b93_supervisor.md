# B.93 durable supervision and observed interruption

The earlier CPU_r2 queue had a stale `pilot` phase record while its Python processes were no longer alive. Five completed pilot fits were independently rechecked against current execution-source/configuration hashes, normal exit0 attestations, manifests/statistics, prepared-cache identity and readable best/latest checkpoint hashes: V1 A/C/T and V2 A/C. They remain preserved. The reason all earlier processes ended is unresolved; absent error logs or an old phase timestamp do not establish a cause or continued progress.

The parent explicitly resumed the existing CPU queue at06:08:40KST, PID958344. A separate Windows Task Scheduler watcher now observes that exact PID AND Windows process creation identity. It never starts, stops or retries the observed queue. Task `B93_CPU_r2_Watch` runs as the current interactive user with limited privileges, Hidden=true, RestartCount=0, MultipleInstances=IgnoreNew and an execution limit equal to the original remaining24hours plus120seconds. The windowless `pythonw.exe` watcher is spawned by service process `svchost.exe` PID3004, independently of the Codex process tree; its current PID is recorded in the status file. Heartbeat updates every5seconds in `data/.b92/cpu_r2/supervisor_status.json`. Future owned training children use the original `python.exe` with CREATE_NO_WINDOW, rather than changing the qualified training interpreter.

This watcher makes interruption observable; it does NOT migrate the already-running CPU queue out of its original job. Future queue launches can be owned directly by the scheduled durable supervisor. No automatic scientific retries or silent continuation after a failed child are implemented. A dead or reused PID, stale heartbeat, terminal queue result and original deadline are distinguished. `running` is process-liveness evidence, not proof of checkpoint or scientific progress. User logout, OS sleep, power loss or termination of Task Scheduler are not resolved by this interactive-user task.

A timestamp-only plumbing task `B93_Durability_Plumbing` actually ran through Task Scheduler and returned LastTaskResult0. It wrote a real process/time record at `data/.b93/task_plumbing/plumbing_timestamp.json`; it used no observations, models or fabricated scientific data. Task registration worked without elevation, so breakaway fallback was not needed. No training was launched by this infrastructure test.

## Interfaces

Read status, including stale detection:

```powershell
py ces_prediction/experiments/b93_gpu/status93.py --cache data/.b92/cpu_r2
```

Register a future ONE-SHOT supervisor task without starting it (use the exact parent-approved queue/cache; GPU qualification and scientific gates remain mandatory):

```powershell
py ces_prediction/experiments/b93_gpu/durable93.py register --name B93_GPU_r3 --queue ces_prediction/experiments/b93_gpu/queue93.py --cache data/.b92/gpu_r3
```

The example queue path is a placeholder until the parent supplies the production queue. Registration alone launches no fit. After qualification, source/configuration approval and verification that no existing queue is alive:

```powershell
py ces_prediction/experiments/b93_gpu/durable93.py start --name B93_GPU_r3 --cache data/.b92/gpu_r3
```

The task starts one supervisor and one owned queue; it does not restart either on failure. The supervisor records PID/creation identity and heartbeat, waits for the actual queue return code and terminates only its OWN queue subtree after the original deadline+120seconds. Existing-PID watch mode never terminates the observed process. Duplicate live queues are rejected before a supervisor launch. The original clock is read from `data/.b92/suite_clock.json`, never reset. Explicit `--deadline` allows the parent to pass that same registered epoch time.

If Task Scheduler is denied, `breakaway` uses explicit Windows CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP. A permission failure is reported; it does not silently use an ordinary child with inherited job lifetime. This fallback has not been needed or claimed tested on this machine.
