# B111 addendum2: reject stale notebook outputs during transport inspection

2026-11-24, no B111 scientific launch or fits. Addendum1's first read obtained
the prior chunk's bytes-only output while the cell source had already changed
to inspection code. Missing T4 raised KeyError; no prefix was accepted, no
new bytes were sent, and no scientific child was launched. Preserve that
failed attempt and the original transfer backup.

The transport helper must require inspection-specific output fields before
accepting a response; a bytes-only prior output is not an inspection result.
Add an explicit --resume-transport option to continue after the preserved
failed inspection, retaining all earlier state and exact-prefix checks.
Only a valid T4/no-launch/size/SHA inspection can authorize writing bytes.
This fixes operational acknowledgement handling only; the archive and all
scientific sources,allocation,fit parameters,results and judgements stay fixed.
Commit/push the helper amendment before use. No automatic scientific retry.
