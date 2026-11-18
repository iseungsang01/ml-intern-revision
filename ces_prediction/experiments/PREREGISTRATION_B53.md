# B.53 — networks that read one component of the beam-emission array, and training under a fixed budget

Registered 2026-11-18, before any run of this batch. GPU, 20 runs; designed by the session lead, runner written and checked without training by a delegated agent. Nothing in the shared training code is edited.

## 1. Why

Two points of the eleventh independent grading that only training can answer.

(C) B.51 split the beam-emission array into its common mode, in which a change of the beam appears in full, and the differences between channels, in which a beam interruption is attenuated to 0.121 of its size. A second stage on the twin found skill in the differences alone on four of four splits. That shows what the components carry beyond the twin, not what a network trained on them reaches. The measurement named there is a network that reads the differences alone.

(F) The history-only twin stopped early after 11 and 12 epochs in two trainings of B.39, and early stopping on a validation loss decided by few rows is unstable. An under-trained twin inflates the fast-channel increment. Training the estimator and the twin for the same fixed number of epochs, with no stopping rule, removes that route.

## 2. Seen before

B.39: the BES network over the twin +0.074, +0.026, +0.105, +0.063 (four of four, one initialisation); the estimator over the twin of the same initialisation +0.071, +0.022, +0.123, +0.060; epochs run by the estimator at initialisation = split seed 53, 22, 44, 26. B.51: second stage with the differences +0.044 to +0.055 over the same stage without BES, with the common mode +0.056 to +0.073, with both +0.071 to +0.075; the differences add to the common mode on one split. B.52: a linear filter gains +0.032 to +0.055 from the fast channels. Not seen: any network trained on a component of the array, any training of the estimator of B.39 or its twin without early stopping.

## 3. Design

T_i, cut population, the four file-level splits, the protocol of B.39 (causal standardization with the statistics of the training discharges, 80-epoch cap with early stopping unless stated, initialisation = split seed), scored by `eval_seq` on the test rows.

**Arm C — components (12 runs).** The history-only twin network with extra inputs to its T_i branch, through the feature-table route of the shared code (`CES_SEQ_EXTRA_TI`). The components are those of B.51 (`b51_bes_modes/run_b51.py::components`): the common mode c and the four differences g1, g2, h1, h2 of the logarithms of the nine BES channels. Each component enters as its value minus its mean over the training discharges of the split, divided by its standard deviation there; the tables are written per split with these constants and are passed through unscaled, so nothing of the estimated discharge enters the standardization. Rows with a non-positive BES channel carry zeros. Three networks per split: C (the common mode), D (the four differences), CD (all five).

**Arm F — fixed budget (8 runs).** The estimator (causal variant) and the twin, each trained for exactly 40 epochs with the final weights kept (`CES_SEQ_FIXED_EPOCHS=1`), on the four splits.

Checks before any GPU run (the batch does not start if one fails): a dry run of every arm; the extra channels reach the T_i branch and only it (the input width of the T_i encoder grows by the number of components, the rotation branch is unchanged); the feature tables match the grid rows of every file; the constants of a split come from its training discharges only; with no extra table the runner reproduces the environment of the twin of B.39 exactly.

Intervals: 95 %, 2,000 resamples of the test discharges; one initialisation, so the spread between initialisations is not in them.

## 4. Statements

Four splits; a statement holds if it holds on at least three.

- **C1.** The network D (differences) has skill over the twin with the interval above zero.
- **C2.** The network C (common mode) has skill over the twin with the interval above zero.
- **F1.** Under the fixed budget the estimator has skill over the twin with the interval above zero.
- **F2.** Under the fixed budget the estimator has skill over the causal GP with the interval above zero.
- Reported: CD over the twin, over C and over D; D, C and CD against the BES network of B.39 and against the causal GP; the fixed-budget twin against the early-stopped twin, the fixed-budget estimator against the early-stopped one, and the increment under the fixed budget minus the increment of B.39 at the same initialisation; epochs and validation losses; median absolute errors.

## 5. What goes into the paper

- C1 holds: the sentence on the beam reads that a network that sees only differences between BES channels, in which a beam interruption is attenuated about eightfold, keeps the increment, with the numbers; C2 and CD beside it.
- C1 does not hold and C2 holds: the paper states that a trained network needs the common mode, where the beam and the density are not separated, and the second-stage result of B.51 is quoted as what the differences carry but a network of this kind does not extract.
- Neither holds: stated; the route of the extra inputs is then a weaker reader than the BES network, and nothing is concluded about the components.
- F1 and F2 hold: one sentence that the increment does not rest on early stopping, with the fixed-budget numbers beside those of B.39. Either fails: stated as a limit, and the main table says which twin it uses.

## 6. Expectations

D over the twin: +0.01 to +0.07, C1 holds on two to four splits. C over the twin: +0.02 to +0.09, C2 holds on three or four. CD over the twin within 0.02 of the BES network of B.39. Fixed budget: the twin differs from the early-stopped twin by −0.03 to +0.03 in skill; the estimator has +0.02 to +0.12 over the twin and F1 holds on three or four splits; F2 holds.

## Addendum 1 (2026-11-18, before any run)

Found while the runner was written, with no network trained. The twin class of the shared code (`v2h`) ignores channels appended through the feature-table route: its T_i branch reads the seven CES-derived channels by position. Arm C therefore uses a subclass of the twin (`b53_components/model_seq_v2hx.py`, registered at run time; no shared file is edited) whose only change is that the first layer of the T_i branch reads the seven channels and the K extra ones. With K = 0 it is the twin: same shapes, same initial weights under the same seed, same output (checked). With K > 0 the initial weights of the T_i branch differ from those of the twin and the rest are the same. The environment of an arm-C run is that of the twin of B.39 with three changes: the model name, the feature table, the output directory.

Details the registration left open, fixed here: the standardization constants of a split are computed on the grid rows of its training discharges without the first and the last row of each file and without rows with a non-positive BES channel; in the readout the fixed-budget increment and that of B.39 are compared on the same resamples of the discharges.

The checks of section 3 passed on all four splits before any run (`docs/paper/b53_checks.md`): the extra channels widen the first layer of the T_i branch by 1, 4 and 5 inputs and leave the rotation branch unchanged; perturbing them changes the T_i output and not the rotation output; the tables match every grid row of the 641 files (759 rows filled with zeros); the constants are reproduced from the training discharges alone and change when test discharges are added; with no table the runner gives the environment of the twin of B.39 variable by variable; an arm-F run differs from its B.39 counterpart in the number of epochs, the fixed-budget switch and the output directory only.

## 7. Limits known in advance

One initialisation; B.39 found that the spread between initialisations decides intervals on single splits. Five summary signals of the array stand in for its nine channels, so CD is not the BES network. The fixed budget of 40 epochs is one choice; with no stopping rule a network may be past its best validation epoch. The test rows are those of every earlier readout.
