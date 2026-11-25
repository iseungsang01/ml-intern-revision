# B112: matched direct and GP-residual scalar learning

**COMPLETE / INDEPENDENT AUDIT PASSED, 2026-11-25 11:44:13 KST.**
The existing B111 T4 completed all24 MLP and8 HGB fits; every unit returned
normally by11:39:28. All32 scenario/design readouts completed. Native result
download finished at11:42:26; local receipt verification and independent
CPU audit passed at11:44:13. Source-bound original inputs/splits/TRAIN
normalization, portable MLP/HGB algebra and all points/intervals passed.
No local fit/GPU/pytest/Claude calls. No additional experiments queued.

Final evidence: `b112_comparison.json` and `b112_comparison.md`.
R1 (residual over direct in all fresh-history designs), R2 (residual MLP
over matched HGB in all fresh-history designs), and R3 (same all scenarios)
were not established. Every D10 residual/direct interval includes zero;
matched residual-MLP/HGB D10 intervals are positive in V1,V2,V4 and negative
in V3. Report every seed/design/scenario, not only favorable comparisons.
The experiment does not support making GP-residual neural learning the new
estimator. It remains a fixed-budget exploratory scalar comparison on reused
TEST data; a matched unused-observation study could reverse this conclusion.
Historical run/transport entries below are superseded; do not rerun B112.

**SCIENTIFIC RUN LAUNCHED, 2026-11-25 11:34:57 KST.**
Native upload completed and exact archive/T4/launch-script guards passed.
Observed B112_LAUNCHED PID12848, torch2.14.1+cu130/CUDA13.0/TeslaT4,
attempt26c2ba0a64fa4da8ae18bb57faa2f0a7. At11:35:11 the controller had
admitted all four units, PIDs12900/12901/12902/12903, no failure. Initial
data loading preceded first epoch output; do not equate process admission
with completed model fits. Same existing B111 notebook/runtime, B112 isolated
under /content/ces_b112. Training sources are now frozen; do not edit them.
The prepared cell follows real progress and downloads only after normal
completion. Local watch112.py waits for that receipt-bound result and runs
independent CPU audit. Earlier upload/failed-export entries are historical.

**2026-11-25 11:23 KST: native upload / automatic launch waiting.**
The author specified the already allocated B111 T4, so B112 is a separate
cell in that same COLAB_B111.ipynb; B111 code/results remain intact. Original
export membership/extra-cluster assertions failed before any scientific
work; failed directories are preserved as export_attempt1/2. Addenda1/2
correct the membership conversion, preserve original V2 internal file split
(TRAIN/VAL share clusters; TEST remains cluster-disjoint), and factor exact
float64 TX columns. All selected X still matches the original helper.
Successful export and addendum3 native delivery sources are committed/pushed.

Exact native archive:485998873bytes, SHA-256
b04bfe9bf9d7a49bf90cd77d27c7278c55050def92de95fde260115366ef464c.
At11:23 the runtime had received about127MB. A small owned cell waits on
the actual file size, then verifies the exact launch-script hash before
executing the original archive/T4/no-duplicate guards and frozen runner.
Scientific launch has NOT yet been observed. The launch script was
normalized to its already registered LF source bytes before upload; its
SHA is f27aa61b92f96d6bc5d0c43fc91269f9437b119b013f9b7cd4152f8c17395bce.
Local download/audit observer PID25664 has a fresh waiting_download heartbeat.
No local pytest/fits/GPU or new Claude calls. Do not rerun the waiting cell
or allocate a second runtime; inspect its outputs for B112_LAUNCHED and epochs.

2026-11-25: author requests a focused next Colab experiment and explicitly
omits local test suites. Scope PREREGISTRATION_B112.md: one small MLP,
direct/residual target learning, four unchanged outer validation designs,
three initializations, seven-rule mixture and held-out K10 availability.
Exactly24 neural fits and8 matched HGB controls, no further candidate search.

Wiring: b112_residual/export112.py -> ignored data/colab_b112 input bundle;
prepare112.py -> standalone one-cell Colab T4 notebook; run112.py/model112.py
-> fit histories, model parameters and all scenario predictions; stats112.py
-> same-row MSE/RMSE and paired cluster/discharge intervals; audit112.py
-> independent CPU verification and b112_comparison.json/md after full pass.
watch112.py receives only the expected receipt-bound download and invokes
that audit; its live state is data/colab_b112/watch_status.json.

No local fitting/GPU, pytest or Claude API calls. Source/data integrity and
causal split assertions remain mandatory. B111 and older scientific sources
remain unchanged. This is exploratory work on reused TEST observations;
the manuscript purpose and estimator are unchanged. No launch is claimed
before an actual Colab B112_LAUNCHED marker and live epoch progress.
