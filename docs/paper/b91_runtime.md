# B.91 measured runtime

Source: data/.b91/runtime_report.json and immutable runtime_decision.json. All timings below use dropout 0 for numerical equivalence; production dropout 0.1 remains untested by these probes and requires fresh clean-exit smokes.

| Runtime path | Median batch seconds | Repetition range | Clean exits |
|---|---:|---|---|
| cpu_dense_mkldnn2 | 0.241310 | 0.230743 to 0.242683 | 3/3 |
| cpu_dense_mkldnn8 | 0.116811 | 0.116440 to 0.128039 | 3/3 |
| cpu_packed_native2 | 1.285420 | 1.185812 to 1.290907 | 3/3 |
| cuda_dense_cudnn | 0.056705 | 0.056425 to 0.060039 | 3/3 |
| cuda_dense_native | 0.534945 | 0.506136 to 0.545333 | 3/3 |
| cuda_packed_cudnn | 0.086468 | 0.049850 to 0.089207 | 3/3 |
| cuda_packed_native | 0.625471 | 0.597343 to 0.630594 | 3/3 |

Selected: cuda_dense_cudnn; measured batch factor 11.0302; total matrix wall time 651.20 seconds.

The training measurement includes real collation, transfer, forward/full frozen masked loss/backward, clipping and AdamW. Source preparation and seven-rule validation costs are reported separately in each child artifact. This is not a measured full-epoch or full-fit speedup. There are three point repetitions and no confidence interval.

Every child passed real-prefix outputs, gradients/None patterns, one AdamW update/state, masked padding-gradient invariance, streamed valid-prefix/reset/future-invariance/save-load checks. Both validators retained actual labels/masks and equal-rule/equal-head weighting. Supervisor-observed ordinary exit 0 and result/source hashes are recorded separately for every child.
