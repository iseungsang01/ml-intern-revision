# B116 temporal architecture plan

Author criterion: compare mean MSE on identical rows with GP+fast and describe input conditions associated with larger errors. No all-interval or all-cell superiority requirement.

| Candidate | Change | Fits |
|---|---|---|
| LSTM | Matched full-sequence temporal control | four units x three seeds |
| Decay GRU | Actual-time-conditioned hidden-state decay | four units x three seeds |
| Innovation GRU | Explicit hidden correction on a new causal CES source | four units x three seeds |

All candidates read the same real thirty-five-channel sequences. Maximum sixty epochs or fifteen minutes per fit, no early stopping. Thirty-six fits can allocate up to nine GPU hours plus smoke and inference. Resource waiting and all GPU work share a fixed twelve-hour boundary from launch; independent CPU verification has two more hours. This is an upper allocation, not a promise of a particular finish time. Actual validated epochs and elapsed time will be shown. No fit retry, no other-process termination.

All fits finish before B116 TEST predictions. Each unit selects its own architecture by seven-rule common-support VAL ensemble MSE. Failure cuts come only from equally weighted TRAIN inputs: CES age, current-versus-anchor fast change, fast level and availability/gap status. Results include all architectures and four validation designs, equal-cell macro MSE and paired cluster/discharge intervals preserving overlapping groups. These are exploratory reused-TEST results.

Pipeline and readers: root README and PREREGISTRATION_B116.md. queue116.py runs smoke -> fits/VAL -> selection/cuts -> TEST prediction -> staged readout -> independent functional/numerical audit -> publication/checks/commit/push. Live status: data/.b116/workflow/status.json, progress.json and docs/paper/b116_status.md. Published artifacts after audit only: b116_comparison.json/md, b116_average.png, b116_failure.png and b116_audit.json. No raw data/checkpoints committed. Earlier B115 remains complete.

Prelaunch verification: existing pytest suite passed89tests; record source checks passed with only the already-known log token. Frozen B105/B112 input guards passed. Separate reviewer found and corrected receipt-list, cluster/discharge mapping, F float32-rounding, age eligibility, dt strata and smoke-gate mismatches before any fit. All four TRAIN normalizations and eighty-eight real first-two-segment role/rule input/target reconstructions passed. Complete U0 D10 TRAIN/VAL raw-input/target/common-support checks passed independently. Static review has no remaining launch blocker; full-epoch feasibility and GPU numerical qualification require the actual registered smoke and subsequent training.

**B116 RUNNING (2026-11-26):** Registration/source aab851a pushed before actual CUDA smoke. All three real-TRAIN forward/backward smokes and independent CPU functional replay passed (maximum difference below0.00005eV). First production fit V1_r2_k0/lstm/s1701 is observed started; fixed queue contains36fits. Mean identical-support error, no all-cell positivity requirement; TRAIN-only failure cuts and per-unit VAL architecture choice. Hidden task CES_B116_Temporal auto-follows allfits -> selection -> TEST -> staged readout -> independent audit -> verified publication/records/commit/push. Original GPU boundary 2026-11-27T00:00:33KST, CPU boundary 2026-11-27T02:00:33KST; maximum allocation, not a promised finish time. No retries/duplicate launch or frozen-source edits. Live data/.b116/workflow/status.json and progress.json; plan docs/paper/b116_plan.md.
