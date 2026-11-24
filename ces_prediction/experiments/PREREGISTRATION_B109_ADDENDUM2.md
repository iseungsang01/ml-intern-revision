# B109 pre-fit transport recovery after remote workspace loss

2026-11-24,before any B109 training launch. Transfer stopped at73,531,392
acknowledged bytes when the next run-cell response had no output. Read-only
inspection confirmed both B108 and B109 /content workspaces absent while the
runtime was still a Tesla T4. Cause unknown. No B109 fit had started; all
original checkpoints and the complete private archive remain local.

research/recover_colab_b109.py recovers TRANSPORT only. Recheck T4,assert
no launch.json,verify surviving partial SHA256 against the exact local prefix,
then append from that proven byte offset. A lost workspace permits one fresh
transfer on the new T4. Use1MiB binary chunks to reduce notebook RPC overhead;
never fetch notebook cells while an encoded large transfer cell is present.
An empty acknowledgement triggers a short owned-cell file-size/hash inspection.
Abort on inconsistent bytes or a second workspace loss. No scientific retry.
Training source/payload/hash/optimizer/data/stopping/allocation stay unchanged.
Commit/push before replacement delivery or fit. No local GPU work or security
setting changes. Do not bypass the denied browser-direct transfer permission.
