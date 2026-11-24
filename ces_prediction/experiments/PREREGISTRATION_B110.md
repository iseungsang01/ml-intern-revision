# B110: small D10/F validation residual diagnostic on Colab

2026-11-24, before export, inference or residual readout. Follows the author's
request to decompose the prediction problem. B109 completed its nine cloud
paths and independent retrieval audit; all twenty selected paths exist.
No Claude calls, local GPU work, new scientific fitting, TEST inference,
post-correction fitting or model selection in this batch.

## Fixed scope and sources

Use all four B105 units V1_r2_k0,V2_s1_k0,V3_q3,V4a, one per validation design,
seed1701, current reported target F, D10 only. Load each selected outer LSTM
from data/colab_b109/selected_paths.json, verifying its recorded SHA256.
Use the original B106 prepared VAL sequences, TRAIN-only statistics and
split manifests. Check disjoint TRAIN/VAL/TEST and statistics' TRAIN binding.
Retain original segment order, reset state at every original segment, export
first22 active input columns, infer float32/cuDNN on T4 with no AMP/TF32,
dropout disabled in eval. No parameter updates. Prefix agreement with the
original frozen model is required on CPU and T4 (atol/rtol1e-4 on T4).

Seven fixed rows: raw LSTM,TCN22,TCN35,GP,hold,B105 fair GP-boost specialized,
and B105 fair GP-boost mixture. The last two are the already fitted TRAIN-
matched B105 controls, not an unmodified original incumbent, not T0, and not
the LSTM's OOF post-correction. Read only their saved D10/F VAL predictions;
TCN uses F head0. Align all seven on the saved specialized tree's VAL idx,
requiring exact coverage, finite values, increasing unique idx and VAL-only
file membership. Do not silently drop unmatched rows. GP/hold/age come from
the frozen B75 D10 inputs. Reusing B105's input-building definitions preserves
the data treatment: stuck targets dropped,Ti>3000eV masked,TRAIN-only scaling,
per-shot/expanding normalization off,VT/MC inactive. Export anonymous unit,
row,discharge and shot-cluster codes, with no original IDs or raw timestamps.
No TEST values or predictions in the private payload. Never commit data.

## Problem decomposition, fixed before readout

Report five marginal groups in each unit, without choosing favorable groups:
all; source age<=10.000001ms; source age>10.000001ms; abs(GP-hold)<=q75;
abs(GP-hold)>q75. q75 is NumPy's linear 75th percentile of abs(GP-hold) on
that unit's TRAIN rows eligible under the frozen B75 common_F mask, never
VAL/TEST. This is an input-defined history-change proxy, not observed future
temperature change or a physically identified transient. Groups overlap
across the two axes; never sum their contributions. No age x change scan.

For each model/group report row-weighted MSE in eV^2, mean error in eV,
and centered MSE satisfying MSE=bias^2+centered_MSE. This is empirical error
decomposition, not physical noise or irreducible error. Paired contrasts:
each non-GP model versus GP, plus TCN35/fair_specialized,
LSTM/fair_specialized,LSTM/TCN35. Skill=1-MSE(candidate)/MSE(reference).
For each model's MSE,bias,centered_MSE and each skill report point estimates,
pointwise95% percentile intervals and zero-exclusion status, using20000
whole-cluster resamples and separately whole-discharge resamples, preserving
row weighting. Deterministic independent RNG per unit/group/resampling form:
SeedSequence([20261124,crc32(unit+'/'+group+'/'+form)]).
Resample all included groups, with no count/CI-dependent filtering. If fewer
than two rows or two resampling units, mark that interval unavailable with
the reason. Empty groups remain explicit. Any nonfinite draw or zero
reference MSE makes that contrast unavailable; do not redraw. No multiplicity
claim, inferential pass/fail, pooling across designs or early model choice.

## Interpretation and stopping

VAL selected checkpoints and has already informed earlier choices; all new
directions are exploratory/TEST-informed at project level. A small bias can
coexist with large MSE; a large bias is diagnostic, not permission to fit an
intercept on the same VAL. A poor stratum can motivate one separately
registered residual study, never a TEST-tuned hard rule. These comparisons
do not isolate fast-channel value (no paired H versus H+fast controls here),
NBI-conditional information, architecture effects, physical transport or
algorithm originality. No final LSTM+post-correction merit claim.

If direct-model residuals are concentrated in input-defined groups, prioritize
a minimal conditional residual hypothesis for later registration. If spread
across groups, examine shared bias/fit or support mismatch before proposing
gates. An interval spanning zero leaves the contrast unresolved; it does not
establish equivalence or lack of diagnostic information. A later matched
held-out comparison with lower MSE and interval excluding zero would change
the unfavorable conclusion for the corresponding fixed comparison.

export110.py performs CPU-only serialization/prefix/provenance checks;
run110.py performs T4 inference and cloud CPU readout, no fitting;
research/colab_b110_operator.py delivers and retrieves bounded hash-checked
chunks through the existing official MCP connection;
audit110.py independently recomputes row alignment, decomposition, paired
points and bootstrap intervals from returned per-row predictions using
separate arithmetic, then renders docs/paper/b110_residuals.json/md.
Private artifacts are under data/colab_b110. A failed check stops this batch;
no silent retry/retraining. Never alter B106-B109 sources or registrations.
Commit/push this registration and all executable code before export/run.
