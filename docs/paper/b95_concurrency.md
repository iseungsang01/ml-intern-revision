# B.95 whole-workload concurrency qualification

Source: `data/.b95/decision95.json`, `audit95.json`, `production_policy95.json` and preserved worker/exit files. No TEST scores were read.

| Global workers | Cohort wall seconds | Jobs/minute | Peak GPU MiB | All normal exits |
|---|---|---|---|---|
| 2 | 110.92, 65.94 | 1.082, 1.820 | 1687 | True |
| 3 | 107.03 | 1.682 | 2530 | True |
| 4 | 127.02 | 1.889 | 3373 | True |
| 6 | 175.54, 183.30 | 2.051, 1.964 | 5059 | True |

Selected global capacity: **2**. Six workers were feasible, but their slower repeated throughput divided by the faster two-worker reference was 1.0792, below the registered 1.10 threshold.

The extra four-worker tie confirmation was cancelled before launch: its already-observed throughput bounds the minimum of any future pair below the required threshold. The mathematical certificate is preserved in `tie_cancelled95.json`; no fictitious extra cohort was reported.

The optimized continuation reuses two verified complete GPU pilot fits, all three clean smokes and prepared caches. Two author-interrupted partial attempts were archived; only those jobs restart with the original seed/budgets. The duplicate CPU confirmation queue stays stopped. See B.94 addenda2/3 and `b94_status.md`.

Observed cohort timings, not predictive confidence intervals or a whole-suite acceleration estimate.
Reference timing varied substantially; the registered decision compares the slower candidate cohort with the faster reference.
Memory feasibility alone does not establish useful throughput. Numeric fit code and fixed budgets remain unchanged.
