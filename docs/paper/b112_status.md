# B112: matched direct and GP-residual scalar learning

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
