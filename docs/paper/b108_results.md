# Colab T4 runtime qualification

30 normal children, 12 clean groups. GPU Tesla T4, torch 2.14.1+cu130, CPU count 2.

| Backend | Workers | Steps/s,median3 | Maximum child reserved MiB |
|---|---:|---:|---:|
| cudnn | 1 | 18.515 | 534.0 |
| native | 1 | 2.610 | 528.0 |
| cudnn | 4 | 21.379 | 568.0 |
| native | 4 | 1.929 | 532.0 |

Within-T4 cuDNN/native ratios: workers1=7.093, workers4=11.082.

Three fixed repetitions; medians are descriptive, no95% interval or inferential speed claim.
Single worker uses unitA; four workers use A-D with different padded lengths. This does not isolate pure concurrency scaling.
Repeated-batch qualification is not full-epoch throughput or model MSE. No exact local/T4 speedup was measured.
Independent audit checks normal-exit records,hash bindings and timing arithmetic; GPU-prefix/finite checks remain producer attestations.
