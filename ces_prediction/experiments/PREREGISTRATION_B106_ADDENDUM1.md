# B106 pre-fit operational runtime correction

2026-11-24, before any B106 smoke, GPU fit, correction fit, or new readout.
Source4f7bd65 passed89 existing tests and was pushed before cache preparation.
Prelaunch inspection identified that the new LSTM prediction entry set thread
count but omitted runtime92.configure. Unlike a TCN, the LSTM would then allow
the default CuDNN backend instead of the registered qualified native CUDA path.
Call the existing configure function during prediction, matching training and
memory qualification. No architecture, numerical objective, data, split, rule,
budget, comparison, or statement changes. No Claude/API calls.

The owned initial preparation process was stopped without any fit having run.
Its partial caches remain under ignored data/.b106_prelaunch_20261124_1015/;
do not reuse them. Commit and push the corrected sources and this addendum
before regenerating the entire B106 preparation from unchanged real inputs.
The corrected preparation/source bindings are required before launch.
