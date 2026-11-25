# B112: matched direct versus GP-residual tabular learning

2026-11-25, after independently audited B111 and before new export, fitting,
prediction or readout. Author requests another focused Colab experiment and
asks to omit local test suites. No local fits/GPU or Claude API calls.

## Question and expectation

B111 did not establish a consistent LSTM-post increment over GP+fast.
Test whether preserving the causal GP as a fixed skip during learning helps
a small neural regressor, with identical scalar inputs/support and strong
boosted-tree controls. This is an exploratory TEST-informed follow-up, not
independent confirmation, algorithm originality or physical identification.
The purpose remains pedestal-top reported-F causal nowcasting, not forecasting.
Expectation is uncertain; no favorable outcome or expansion is assumed.

## Fixed allocation

Use B105/B106 original four outer units V1_r2_k0,V2_s1_k0,V3_q3,V4a,
unchanged internal TRAIN/VAL/TEST files and all original common_F or S_F rows.
All designs reported; none primary. Input exactly B75 TX history scalars,
TRAIN-normalized fast8 and source-anchor scalars from SF.Ctx, no rule ID.
Seven rules D10,D50,D100,D300,D650,D1000,S form TRAIN/VAL; K10 is TEST only.
No future CES input, new synthetic data, support trimming or split changes.
Export float64 physical inputs; neural normalization then casts to float32;
HGB uses the same float64 inputs. Train fast summaries use internal TRAIN only.

Exactly24 MLP fits: four units, seeds1701/1702/1703, direct/residual forms.
Both read the same inputs. Direct predicts F; residual predicts F-GP and adds
the fixed causal GP back at inference. Network is Linear(d,64),GELU,
Linear(64,64),GELU,Linear(64,1), no dropout. Each form has its own rule-balanced
TRAIN target mean/std (std>=1eV); input mean/std rule-balanced TRAIN only,
std floor1e-8. Target normalization differs by form and is disclosed as part
of the learning parameterization, not a pure architecture intervention.
AdamW lr1e-3,weight_decay1e-4,batch8192,max80epochs,patience8; fixed seeds,
one shuffle of all pooled TRAIN rows per epoch, loss mean(weight*error^2),
weight N_total/(7*N_rule). VAL selection uses equal-rule mean physical MSE,
earliest strict minimum. No TEST-based stopping or candidate selection.
For each form report all seeds and a pre-fixed arithmetic prediction ensemble.

Exactly8 HGB fits: direct/residual per unit, identical pooled rows/inputs/
weights; random_state20261116,early_stoppingFalse,max_iter100,other installed
sklearn defaults recorded. No VAL selection. Compare the two learning forms
within MLP and HGB; retain old B105 fair-mixture, original GP+fast, TCN35,
B111 raw and corrected LSTM, GP and hold as deployment references. Different
past training budgets remain disclosed; no architecture-only comparison.

All neural learning runs on allocated Colab T4, float32,no AMP/TF32,
one Torch/BLAS thread per child. At most four unit workers admitted with
1630MiB free GPU memory each; each worker fits its six MLPs/two HGBs serially.
No fit retry; a failed child prevents successful batch completion and is
preserved. Twelve-hour child guard is an emergency incomplete-run boundary.
No discretionary one-hour budget. Do not modify frozen code while running.
Explicit CES pins: stuck dropped,3keVcut,no per-shot/expanding scaling,
inactive MC/rotation,F target,seven-rule mixture; original data builder bound.

## Evaluation and interpretation

After a unit's fits finish normally, predict every TEST rule on exact common
rows. Final readout requires every unit normal exit and all32 cells complete.
Physical MSE/eV^2 and RMSE/eV first. Use20000 paired cluster and separately
discharge multinomial bootstrap draws (chunks500), seed
[20261125,CRC32(unit+'/'+rule+'/F/'+form)]; retain fixed-fit pointwise95%
intervals, skill=1-MSE(candidate)/MSE(reference), zero exclusion.
All seed outputs and arithmetic ensembles reported without favorable choice.
Initialization variability is reported separately, not hidden in row bootstrap.

Registered statements: R1 residual-ensemble/direct-ensemble lower cluster
skill endpoint>0 in every D10 design; R2 residual-ensemble/new HGB-residual
lower endpoint>0 in every D10 design; R3 same contrast lower endpoint>0 in
all32 cells. Also report ensemble versus original GP+fast/fair-mixture,
HGB residual/direct, each individual MLP seed versus HGB-residual, ensemble
versus GP/TCN35 and raw B111 LSTM. All statements regardless of outcome.
If absent, say consistent increment not established; no equivalence or neural
impossibility. A matched adequate-budget comparison on unused observations
could reverse the conclusion. No new pedestal physics inferred from F errors.

## Delivery and verification

b112_residual/export112.py serializes real TX inputs with exact original
helper equality, split/hash checks and B111 anonymous row/file/cluster mapping.
No raw filenames,timestamps,identity,credentials or .env in the payload.
common112.py pins sources; model112.py/run112.py fit on Colab; stats112.py
produces the fixed readout. prepare112.py makes a one-cell T4 notebook that
checks archive/source hashes, refuses duplicate launches, executes once and
downloads b112_results.zip after normal completion. Use native Colab upload,
no repeated MCP source transfer. Results retain weights, histories,configs,
portable trees,predictions,exit codes,source/input hashes and attempt receipt.
audit112.py independently reconstructs model algebra, tree outputs, split/
normalization metadata and paired arithmetic on CPU before public final
docs/paper/b112_comparison.json/md. Local pytest is omitted at author request;
required dataset/launch integrity assertions remain. Commit/push sources and
registration before export/run. Operational changes need separate addenda.

watch112.py observes only the expected newer Downloads/b112_results.zip,
validates the attempt/archive/manifest/T4/normal-exit receipt, preserves it
locally and invokes the independent CPU audit without fitting or restarting.
