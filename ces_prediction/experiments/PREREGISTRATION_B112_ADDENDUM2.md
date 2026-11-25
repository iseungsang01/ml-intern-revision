# B112 addendum2: preserve original split and factor exact transport

2026-11-25, before any B112 scientific fit/inference/readout. Corrected
export completed U0 then stopped at an unnecessarily stronger V2 cluster
assertion. The inherited internal TRAIN/VAL partition is by files and may
share clusters; do not invent a stricter partition or change its membership.
Bind and preserve exactly the original B105/B106 manifest with disjoint files;
report inherited TRAIN/VAL cluster overlap, and only assert the original
outer cluster grouping where it actually holds. This is not a new scientific
split. Preserve failed U0 output as export_attempt2; no model has used it.

The flat exact float64 U0 payload alone is192MB. Reduce delivery size by
factoring common history columns per rule, current fast columns per unit,
and anchor columns per unit/rule into separate arrays indexed by the fixed
B111 anonymous row code. The small per-unit/rule table retains targets,GP,
anonymous row/group IDs and split tags; _x_parts points to the three files.
common112.npz reconstructs the same float64 column concatenation for the
runner and independent auditor, with the original logical X hash retained.
Factor arrays carry no original row/shot/time identifiers; manifest hashes
bind every file. Full selected X values must remain bit-identical to the
original TX helper; no quantization,subsampling,new observations,model or
target change. Only storage/serialization and the incorrect extra assertion
change. Commit/push before re-export and fitting; keep allocated B111 T4.
