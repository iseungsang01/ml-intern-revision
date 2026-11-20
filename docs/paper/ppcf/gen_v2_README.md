# Data layer of the rewritten manuscript (tables, macros, figures)

Generated hand-off for the writer of `main_ppcf.tex` / `supp_ppcf.tex` (line of PAPER_DESIGN.md, 2026-11-20).
This file is written by `make_tables_v2.py`; do not edit it by hand.

## How to use

- `\input{gen_v2.tex}` in the preamble (after `booktabs`). Every table is a macro `\TabXxx` (a complete
  `tabular`, already sized at `\footnotesize` or `\scriptsize` to fit the 16 cm text width) with a caption stub
  `\CapXxx` that names target, availability rule, design and interval. Use it as
  `\begin{table}\centering\caption{\CapXxx}\label{tab:xxx}\TabXxx\end{table}`; edit the stub in the
  manuscript if the prose needs it (keep its numbers; they are registered tokens).
- `gen_v2.tex` defines `\ci{point}{[lo, hi]}` (point above its interval) and `\tcell`.
- Comparisons are printed as the reduction of the MSE in per cent (100 x skill, one decimal) with the 95 %
  interval from the shot-cluster bootstrap (MAST: session bootstrap); the point is bold when that interval
  excludes zero. RMSE in eV with one decimal; ratios with three decimals. Statement tables print the
  registered value as registered (skill as a signed fraction, three decimals).
- Design names: V1 random, V2 shot clusters held out, V3 rolling origin, V4 chronological fit to 31991 (V4a)
  and fit to 32310 (V4b); MAST M1 random, M2 sessions held out, M3 rolling origin, M4 chronological.
- Regenerate: `py docs/paper/ppcf/make_tables_v2.py` then `py docs/paper/ppcf/fig_v2.py` (needs the caches
  `data/.b73`, `data/.b75`, `data/.b79` for figures 1 and 7 only). Compile test: `latexmk -pdf test_v2.tex`
  in this folder. Number check: `py docs/paper/ppcf/check_numbers_v2.py` (manuscript) or `--self`.
- Provenance of every printed number: `gen_v2_tokens.json` (token -> JSON file, key path, transformation).


## Tables (`gen_v2.tex`)

| macro | shows | source (file and key) | paper map |
|---|---|---|---|
| `\TabSets` (caption `\CapSets`) | row sets, discharges and clusters (KSTAR), shots and sessions (MAST) | b75 sets/designs/main.*.counts; b73 increments.V1/L5; b78 counts.V1.L5; b77 row_set; b76 main.M1/L5/*.counts | sec. 2/3 (data and method) |
| `\TabScored` (caption `\CapScored`) | scored rows/discharges/clusters per design and set | b75 counts.*; b78 counts.*.L5; b76 main.M*/L5/common.all.counts | sec. 3 / supplement |
| `\TabInputs` (caption `\CapInputs`) | estimators and their inputs (definitions from PAPER_DESIGN.md sec. 6; beam scalar names from b79) | PAPER_DESIGN.md sec. 6 (text); b79 beam_scalars | sec. 3 (main) |
| `\TabRulesVone` (caption `\CapRulesVone`) | RMSE by availability rule (V1), cost of staleness, increment, recovered share | b75 main.V1/L5/common.all.{D10..D1000}, main.V1/L5/S.all.{S,D10S} | 4.1 (main) |
| `\TabRulesDesigns` (caption `\CapRulesDesigns`) | RMSE by rule for V2, V3, V4 (both fits), with cost, increment, recovered share | b75 main.{V2,V3,V4a,V4b}/L5/{common,S}.all.* | supplement (4.1) |
| `\TabIncrement` (caption `\CapIncrement`) | fast-channel increment by design and rule, KSTAR D10/D650/S and MAST D5/D100 | b75 main.*/L5/common.all.{D10,D650}.TX/T0, main.*/L5/S.all.S.TX/T0; b76 main.M*/L5/common.all.{D5,D100}.TX/T0 | 4.3 (main); 4.4 for the MAST rows |
| `\TabIncrementLadder` (caption `\CapIncrementLadder`) | increment along the whole delay ladder, every design, KSTAR and MAST | b75 main.*/L5/common.all.D*.TX/T0; b76 main.M*/L5/common.all.D*.TX/T0 | supplement (4.1, 4.3, 4.4) |
| `\TabFifths` (caption `\CapFifths`) | V3 by scored fifth and the size-matched control V3c (D10, D650) | b75 v3c.*; designs.v3_fifths | 4.3 (main or supplement) |
| `\TabWhich` (caption `\CapWhich`) | ECEI only, BES only, both; beam alone; both given beam; beam given both (D650, S, per design) | b75 main.{V1,V4a,V4b}.{D650,S}.{TE,TB}/T0 and main.*.{D650,S}.TX/T0; b81 main.{V2,V3}.{D650,S}.{TE,TB}/T0; b79 main.*.{D650,S}.{TN/T0,TXN/TN,TXN/TX} | 4.2 (main) |
| `\TabControls` (caption `\CapControls`) | permuted-input and anchor-row controls per design under D650 and S (KSTAR head rules) | b81 main.*.{D650,S}.{TXp/T0,TXo/T0,TX/TXo} | 4.2 (main) |
| `\TabControlsOther` (caption `\CapControlsOther`) | controls of the other conditions (sparse CES, beam scalars, correction, MAST) | b78 main.V1/L5.all.K10; b79 main.V1/L5/common.all.D650.{TXNp/TN,TNp/T0}; b77 designs.V1.all.CXp/C0; b76 main.M1/L5/common.all.controls | supplement |
| `\TabRefit` (caption `\CapRefit`) | B.79 main readout vs refit without the 13 files with invalid NBI-2 power columns | b79 refit_without_13_files.statements_beside_main.* | supplement (4.2) |
| `\TabMastRmse` (caption `\CapMastRmse`) | MAST RMSE by delay (M1, M4): hold, GP, T0, TX, O, closed ratio form RT, RTa and its alpha | b76 main.{M1,M4}/L5/common.all.D*.rmse_eV; rta_coefficients.*.alpha.mean | 4.4 (main) |
| `\TabMastCost` (caption `\CapMastCost`) | MAST cost of staleness, increment, recovered share, TX vs O by design and delay | b76 main.M*/L5/common.all.D{25,50,100,150}.{staleness T0,TX/T0,recovered share,TX/O} | 4.4 / supplement |
| `\TabMastDalpha` (caption `\CapMastDalpha`) | MAST D-alpha inputs (TD, TA) against T0 and TX, M1 and M4 | b76 main.{M1,M4}/L5/DA.all.{D5,D100} | 4.4 / supplement |
| `\TabSparse` (caption `\CapSparse`) | RMSE of T0 and TX by spacing of the CES values (V1), increment, cost, recovered share, equivalent spacing | b78 main.V1/L5.all.K* | 4.5 (main) |
| `\TabSparseDesigns` (caption `\CapSparseDesigns`) | increment by spacing K1..K50 and design | b78 main.*/L5.all.K*.TX/T0 | 4.5 (main or supplement) |
| `\TabBlocks` (caption `\CapBlocks`) | increment under S by block of shots (V1) and under V4 (both fits), with and without beam inputs | b75 main.V1/L5/S.<block>.S, main.V4*/L5/S.all.S; b79 main.*/L5/S.*.S.TXN/TN | 4.6 (main) |
| `\TabBlockDescr` (caption `\CapBlockDescr`) | last-block diagnostic: age, Spearman ECEI/BES, notch response T_i/BES/ECEI, channel coherence | docs/paper/b75_last_block.md code block sections (1), (2), (7), (8) (parsed) | 4.6 (main) |
| `\TabGate` (caption `\CapGate`) | B.80: increment by class of ECEI channel coherence (S and D650, V1), gated estimator TG | b80 cells.V1/{S,D650}.{class low,class middle,class high,all,low+high} | 4.6 (main or supplement) |
| `\TabGateBlocks` (caption `\CapGateBlocks`) | B.80 class counts by block of shots and share of the low class | b80 distribution.D1 discharges.* | 4.6 / supplement |
| `\TabCorrect` (caption `\CapCorrect`) | B.77: correcting the frame's reading, RMSE and fast increments by design | b77 designs.*.all | 4.6 (main) / supplement |
| `\TabDten` (caption `\CapDten`) | B.73: D10 increments by design and target (trees, filters, ECEI only, BES only, lagged) | b73 increments.*/{L5,F}.cells | 4.6 (main: the D10 column) / supplement |
| `\TabNetwork` (caption `\CapNetwork`) | B.73: every estimator incl. the recurrent network and two-stage estimator, targets F and L5 | b73 per_estimator.V1/{F,L5} | supplement (previous frame visible) |
| `\TabTiming` (caption `\CapTiming`) | computation time per row (GP + TX) | b75 timing.* | 4.1 / sec. 5 |
| `\TabAsFitted` (caption `\CapAsFitted`) | estimators fitted at D10 and applied to a stale history vs refitted under the rule | b75 main.*.{D50..D1000,S}.{rmse_eV.T0a/TXa, applied T0/TX: refit over applied, applied TXa/T0a} | sec. 3 (method sentence) / supplement |
| `\TabAsFittedChrono` (caption `\CapAsFittedChrono`) | the same for V4a, V4b | b75 main.V4*.* | supplement |
| `\TabAnchorScalars` (caption `\CapAnchorScalars`) | TX8 (no anchor scalars) and TX against T0, V1 and V4, D10/D650/S | b75 main.{V1,V4a,V4b}.{D10,D650,S}.{TX8/T0,TX/T0,rmse_eV} | 4.2 / supplement |
| `\TabBursts` (caption `\CapBursts`) | B.74 in readable form: rows with/without a burst inside the exposure, ratios, excursions | b74 A.W10, B.W10.{ti,l5,vt,ec}_ratio, E.W10 | supplement (sec. 2) |
| `\TabStuckEcei` (caption `\CapStuckEcei`) | constant-ECEI rows by block and the increment on moving/constant rows (unregistered) | docs/paper/b75_stuck_ecei.md code block (1), (3) (parsed) | sec. 2 / supplement |
| `\TabVsReading` (caption `\CapVsReading`) | TX against the frame's own reading O, D650 and S, per design | b75 main.*.{D650,S}.{TX/O, rmse_eV.TX/O} | 4.1 / 4.6 |
| `\TabNbiSources` (caption `\CapNbiSources`) | NBI sources: rows on, shots, voltage | docs/paper/nbi_files.md sec. 2 (parsed) | supplement (sec. 2 / 4.2) |
| `\TabNbiNotch` (caption `\CapNbiNotch`) | power of each NBI source before/after an absent frame | docs/paper/nbi_files.md sec. 3 (parsed) | sec. 2 (main or supplement) |
| `\TabNbiStates` (caption `\CapNbiStates`) | share of rows with a T_i value by beam state | docs/paper/nbi_files.md sec. 4 (parsed) | supplement (sec. 2) |
| `\TabStatBseventythree` (caption `\CapStatBseventythree`) | registered statements of B.73 (previous frame visible: targets, designs, network rows) | b73_targets.json statements.* | supplement (statement tables) |
| `\TabStatBseventyfour` (caption `\CapStatBseventyfour`) | registered statements of B.74 (bursts inside the exposure) | b74_exposure.json statements.* | supplement (statement tables) |
| `\TabStatBseventyfive` (caption `\CapStatBseventyfive`) | registered statements of B.75 (stale CES history) | b75_availability.json statements.* | supplement (statement tables) |
| `\TabStatBseventysix` (caption `\CapStatBseventysix`) | registered statements of B.76 (MAST, stale CXRS history) | b76_mast_availability.json statements.* | supplement (statement tables) |
| `\TabStatBseventyseven` (caption `\CapStatBseventyseven`) | registered statements of B.77 (correcting the frame's reading) | b77_correct.json statements.* | supplement (statement tables) |
| `\TabStatBseventyeight` (caption `\CapStatBseventyeight`) | registered statements of B.78 (a CES value every k-th frame) | b78_sparse.json statements.* | supplement (statement tables) |
| `\TabStatBseventynine` (caption `\CapStatBseventynine`) | registered statements of B.79 (beam waveforms as inputs) | b79_beam_inputs.json statements.* | supplement (statement tables) |
| `\TabStatBeighty` (caption `\CapStatBeighty`) | registered statements of B.80 (coherence of the ECEI channels) | b80_ecei_gate.json statements.* | supplement (statement tables) |
| `\TabStatBeightyone` (caption `\CapStatBeightyone`) | registered statements of B.81 (controls of the fast inputs; ECEI and BES halves under V2, V3) | b81_controls.json statements.* | supplement (statement tables) |
| `\TabStatBeightytwo` (caption `\CapStatBeightytwo`) | registered statements of B.82 (where the increment sits; robust error measures; specificity) | b82_localize.json statements.* | supplement (statement tables) |
| `\TabStatBeightythree` (caption `\CapStatBeightythree`) | registered statements of B.83 (previous frame visible: against interpolation and the offline GP) | b83_interp.json statements.* | supplement (statement tables) |
| `\TabStatBeightyfour` (caption `\CapStatBeightyfour`) | registered statements of B.84 (one estimator fitted for every availability rule) | b84_onefit.json statements.* | supplement (statement tables) |
| `\TabDataset` (caption `\CapDataset`) | the data set: discharges, rows, absent steps, T_i and L5 ranges, beam power and sources | b82 part_E | sec. 2 (main) |
| `\TabRobust` (caption `\CapRobust`) | robust error measures (MAE, median, p90, within 10/20 %) of T0, TX, O for V1 at D10, D650, S | b82 part_B.V1.{D10,D650,S} | 4.1 (main) |
| `\TabRobustFull` (caption `\CapRobustFull`) | robust error measures for H, G, T0, TX, O, V1 at D10, D650, S | b82 part_B.V1.{D10,D650,S} | supplement |
| `\TabTau` (caption `\CapTau`) | increment by time since the last notch (V1, V2, V3, V4a; D650, S; TXN vs TN for V1) | b82 part_A.*.{D650,S}.all.tau | 4.2 or 4.3 (main) |
| `\TabTauRows` (caption `\CapTauRows`) | rows per class of the time since the last notch, every design | b82 part_A.*.tau.*.counts | supplement |
| `\TabSpecificity` (caption `\CapSpecificity`) | Mirnov in place of/added to the fast inputs, another discharge's inputs, single BES columns | b82 part_C.{V1,V3,V4a}.{D650,S} | 4.2 (main) |
| `\TabSeeds` (caption `\CapSeeds`) | seed spread of the increment (V4a, V4b, V1 r2 pooled; D650, S) | b82 part_D | supplement |
| `\TabChange` (caption `\CapChange`) | increment by thirds of the change since the anchor row (V1, V3; D650, S) | b82 part_A.{V1,V3}.{D650,S}.all.c | supplement |
| `\TabRobustDesigns` (caption `\CapRobustDesigns`) | robust measures TX vs T0 per design and rule, per-discharge skill | b82 part_B.* | supplement |
| `\TabLadder` (caption `\CapLadder`) | the main nowcast table: references, T0, TX, network rows on F and L5 (V1, previous frame visible) | b83 estimators.V1.*; b73 per_estimator.V1/{F,L5}; b82 part_B.V1.D10 (within 10 % on L5) | 4 (main) |
| `\TabVsReferences` (caption `\CapVsReferences`) | TX (and T0) against hold, PCHIP, causal and offline GP by design; ratio block V1 | b83 pairs.*.*.MSE skill and ratios; b73 per_estimator.*/L5.{TX[L5],T0[L5]}.over_G | 4 (main) |
| `\TabLadderDesigns` (caption `\CapLadderDesigns`) | the F columns of the main table for V2, V3, V4a | b83 estimators.{V2,V3,V4a}.* | supplement |
| `\TabOneFit` (caption `\CapOneFit`) | one fit for every rule vs per-rule fits (V1 r2, V3, V4a; D10..S, K10) | b84 cells.* | supplement / sec. on use |
| `\TabMastRadius` (caption `\CapMastRadius`) | MAST by radius band: rho, T_i, rows, RMSE TX, within 10 %, cost of staleness, Thomson increment M1/M3 D5/D100 | b86 counts.*.L5_common; bands.*.{M1,M3}.{D5,D100} | supplement or 4.4 |
| `\TabMastRadiusAcross` (caption `\CapMastRadiusAcross`) | B.86 across-band statements R5-R7 | b86 statements (band = null) | supplement or 4.4 |
| `\TabStatSummary` (caption `\CapStatSummary`) | count of registered statements per batch B.73-B.86: hold, do not hold, expectations met | statements blocks of b73..b86 JSON (counted) | supplement (replaces the nine statement tables) |
| `\TabStatNotHeld` (caption `\CapStatNotHeld`) | registered statements that do not hold (B.73-B.77) | statements blocks of the batch JSON (holds = false) | supplement |
| `\TabStatNotHeldB` (caption `\CapStatNotHeldB`) | registered statements that do not hold (B.78-B.82) | statements blocks of the batch JSON (holds = false) | supplement |
| `\TabStatNotHeldC` (caption `\CapStatNotHeldC`) | registered statements that do not hold (B.83-B.86) | statements blocks of the batch JSON (holds = false) | supplement |

## Prose macros (`gen_v2.tex`, end of file)

Naming: `Inc<Rule><Design>` = fast-channel increment (TX vs T0), `IncBeam` = TXN vs TN, `BeamAlone` = TN vs T0, `EceiOnly`/`BesOnly` = TE/TB vs T0, `IncMast`, `IncKten`, `Cost`, `Rec` (recovered share), `N...` counts. Rule: Dten, Dsixfifty, S, Dfive, Dhundred, Kten. Design: Vone, Vtwo, Vthree, Vfoura (fit to 31991), Vfourb (fit to 32310), Mone..Mfour. A macro gives `point~[lo, hi]` (per cent for skills); the same name with `Pt` gives the point only. The writer adds `\%` and the words.

| macro | value is | source |
|---|---|---|
| `\RefitOverAppliedTXDsixfiftyVone` | refitted TX vs applied TX, D650, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.applied TX: refit over applied |
| `\RefitOverAppliedTXDsixfiftyVonePt` | refitted TX vs applied TX, D650, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.applied TX: refit over applied |
| `\RefitOverAppliedTZeroDsixfiftyVone` | refitted T0 vs applied T0, D650, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.applied T0: refit over applied |
| `\RefitOverAppliedTZeroDsixfiftyVonePt` | refitted T0 vs applied T0, D650, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.applied T0: refit over applied |
| `\RmseAppliedTZeroDsixfiftyVone` | RMSE (eV) of T0a, D650, V1 (applied = fitted at D10) | docs/paper/b75_availability.json main.V1/L5/common.all.D650 |
| `\RmseAppliedTXDsixfiftyVone` | RMSE (eV) of TXa, D650, V1 (applied = fitted at D10) | docs/paper/b75_availability.json main.V1/L5/common.all.D650 |
| `\RmseRefitTZeroDsixfiftyVone` | RMSE (eV) of T0, D650, V1 (applied = fitted at D10) | docs/paper/b75_availability.json main.V1/L5/common.all.D650 |
| `\RmseRefitTXDsixfiftyVone` | RMSE (eV) of TX, D650, V1 (applied = fitted at D10) | docs/paper/b75_availability.json main.V1/L5/common.all.D650 |
| `\StuckShare` | share of grid rows with a constant ECEI | docs/paper/b75_stuck_ecei.md |
| `\NStuckDischarges` | discharges with any constant-ECEI row | docs/paper/b75_stuck_ecei.md |
| `\NStuckDischargesHalf` | discharges with a constant ECEI in more than half of the rows | docs/paper/b75_stuck_ecei.md |
| `\TXvsOSVone` | TX against the frame's own reading under S, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TX/O |
| `\TXvsOSVonePt` | TX against the frame's own reading under S, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TX/O |
| `\TXvsOSVtwo` | TX against the frame's own reading under S, V2 (per cent, interval) | docs/paper/b75_availability.json main.V2/L5/S.all.S.TX/O |
| `\TXvsOSVtwoPt` | TX against the frame's own reading under S, V2 (per cent, point) | docs/paper/b75_availability.json main.V2/L5/S.all.S.TX/O |
| `\TXvsOSVthree` | TX against the frame's own reading under S, V3 (per cent, interval) | docs/paper/b75_availability.json main.V3/L5/S.all.S.TX/O |
| `\TXvsOSVthreePt` | TX against the frame's own reading under S, V3 (per cent, point) | docs/paper/b75_availability.json main.V3/L5/S.all.S.TX/O |
| `\TXvsOSVfoura` | TX against the frame's own reading under S, V4a (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TX/O |
| `\TXvsOSVfouraPt` | TX against the frame's own reading under S, V4a (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TX/O |
| `\TXvsOSVfourb` | TX against the frame's own reading under S, V4b (per cent, interval) | docs/paper/b75_availability.json main.V4b/L5/S.all.S.TX/O |
| `\TXvsOSVfourbPt` | TX against the frame's own reading under S, V4b (per cent, point) | docs/paper/b75_availability.json main.V4b/L5/S.all.S.TX/O |
| `\SeedMinDsixfiftyVfoura` | smallest increment over ten seeds, D650, V4a (per cent) | docs/paper/b82_localize.json |
| `\SeedMaxDsixfiftyVfoura` | largest increment over ten seeds, D650, V4a (per cent) | docs/paper/b82_localize.json |
| `\SeedMinSVfoura` | smallest increment over ten seeds, S, V4a (per cent) | docs/paper/b82_localize.json |
| `\SeedMaxSVfoura` | largest increment over ten seeds, S, V4a (per cent) | docs/paper/b82_localize.json |
| `\WithinTenTZeroDtenVone` | share of rows within 10 % of the level, T0, D10, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.D10.estimators.T0.share within 10 % |
| `\WithinTenTZeroDtenVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.D10.estimators.T0.share within 10 % |
| `\WithinTenTXDtenVone` | share of rows within 10 % of the level, TX, D10, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.D10.estimators.TX.share within 10 % |
| `\WithinTenTXDtenVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.D10.estimators.TX.share within 10 % |
| `\WithinTenODtenVone` | share of rows within 10 % of the level, O, D10, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.D10.estimators.O.share within 10 % |
| `\WithinTenODtenVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.D10.estimators.O.share within 10 % |
| `\WithinTenTZeroDsixfiftyVone` | share of rows within 10 % of the level, T0, D650, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.D650.estimators.T0.share within 10 % |
| `\WithinTenTZeroDsixfiftyVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.D650.estimators.T0.share within 10 % |
| `\WithinTenTXDsixfiftyVone` | share of rows within 10 % of the level, TX, D650, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.D650.estimators.TX.share within 10 % |
| `\WithinTenTXDsixfiftyVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.D650.estimators.TX.share within 10 % |
| `\WithinTenODsixfiftyVone` | share of rows within 10 % of the level, O, D650, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.D650.estimators.O.share within 10 % |
| `\WithinTenODsixfiftyVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.D650.estimators.O.share within 10 % |
| `\WithinTenTZeroSVone` | share of rows within 10 % of the level, T0, S, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.S.estimators.T0.share within 10 % |
| `\WithinTenTZeroSVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.S.estimators.T0.share within 10 % |
| `\WithinTenTXSVone` | share of rows within 10 % of the level, TX, S, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.S.estimators.TX.share within 10 % |
| `\WithinTenTXSVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.S.estimators.TX.share within 10 % |
| `\WithinTenOSVone` | share of rows within 10 % of the level, O, S, V1 (per cent) | docs/paper/b82_localize.json part_B.V1.S.estimators.O.share within 10 % |
| `\WithinTenOSVonePt` | the same, point | docs/paper/b82_localize.json part_B.V1.S.estimators.O.share within 10 % |
| `\MaeRatioDsixfiftyVone` | MAE(TX)/MAE(T0), D650, V1 (B1) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.MAE ratio |
| `\MaeRatioDsixfiftyVonePt` | MAE(TX)/MAE(T0), D650, V1 (B1) (point) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.MAE ratio |
| `\MedianRatioDsixfiftyVone` | median abs. error ratio, D650, V1 (B2) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.median absolute error ratio |
| `\MedianRatioDsixfiftyVonePt` | median abs. error ratio, D650, V1 (B2) (point) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.median absolute error ratio |
| `\MaeRatioDsixfiftyVthree` | MAE ratio, D650, V3 (B3) | docs/paper/b82_localize.json part_B.V3.D650.TX over T0.MAE ratio |
| `\MaeRatioDsixfiftyVthreePt` | MAE ratio, D650, V3 (B3) (point) | docs/paper/b82_localize.json part_B.V3.D650.TX over T0.MAE ratio |
| `\MaeRatioSVone` | MAE ratio, S, V1 (B4) | docs/paper/b82_localize.json part_B.V1.S.TX over T0.MAE ratio |
| `\MaeRatioSVonePt` | MAE ratio, S, V1 (B4) (point) | docs/paper/b82_localize.json part_B.V1.S.TX over T0.MAE ratio |
| `\MaeRatioDsixfiftyVfoura` | MAE ratio, D650, V4a (B7) | docs/paper/b82_localize.json part_B.V4a.D650.TX over T0.MAE ratio |
| `\MaeRatioDsixfiftyVfouraPt` | MAE ratio, D650, V4a (B7) (point) | docs/paper/b82_localize.json part_B.V4a.D650.TX over T0.MAE ratio |
| `\PninetyRatioDsixfiftyVone` | 90th-percentile abs. error ratio, D650, V1 (B8) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.p90 absolute error ratio |
| `\PninetyRatioDsixfiftyVonePt` | 90th-percentile abs. error ratio, D650, V1 (B8) (point) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.p90 absolute error ratio |
| `\WithinTenDiffDsixfiftyVone` | share within 10 %, TX minus T0, D650, V1 (B5; percentage points) (per cent, interval) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.share within 10 % difference |
| `\WithinTenDiffDsixfiftyVonePt` | share within 10 %, TX minus T0, D650, V1 (B5; percentage points) (per cent, point) | docs/paper/b82_localize.json part_B.V1.D650.TX over T0.share within 10 % difference |
| `\ShareDischargesPositiveDsixfiftyVone` | share of discharges with positive own skill, D650, V1 (B6) (per cent, interval) | docs/paper/b82_localize.json part_B.V1.D650.per discharge.share of discharges with positive skill |
| `\ShareDischargesPositiveDsixfiftyVonePt` | share of discharges with positive own skill, D650, V1 (B6) (per cent, point) | docs/paper/b82_localize.json part_B.V1.D650.per discharge.share of discharges with positive skill |
| `\IncTauFarDsixfiftyVone` | increment on rows more than 300 ms after the last notch, D650, V1 (per cent, interval) | docs/paper/b82_localize.json part_A.V1.D650.all.tau.above 300 ms.TX/T0 |
| `\IncTauFarDsixfiftyVonePt` | increment on rows more than 300 ms after the last notch, D650, V1 (per cent, point) | docs/paper/b82_localize.json part_A.V1.D650.all.tau.above 300 ms.TX/T0 |
| `\IncTauFarSVone` | increment on rows more than 300 ms after the last notch, S, V1 (per cent, interval) | docs/paper/b82_localize.json part_A.V1.S.all.tau.above 300 ms.TX/T0 |
| `\IncTauFarSVonePt` | increment on rows more than 300 ms after the last notch, S, V1 (per cent, point) | docs/paper/b82_localize.json part_A.V1.S.all.tau.above 300 ms.TX/T0 |
| `\IncTauFarDsixfiftyVthree` | increment on rows more than 300 ms after the last notch, D650, V3 (per cent, interval) | docs/paper/b82_localize.json part_A.V3.D650.all.tau.above 300 ms.TX/T0 |
| `\IncTauFarDsixfiftyVthreePt` | increment on rows more than 300 ms after the last notch, D650, V3 (per cent, point) | docs/paper/b82_localize.json part_A.V3.D650.all.tau.above 300 ms.TX/T0 |
| `\MirnovAloneDsixfiftyVone` | Mirnov in place of the fast inputs (TM vs T0), D650, V1 (per cent, interval) | docs/paper/b82_localize.json part_C.V1.D650.TM/T0 |
| `\MirnovAloneDsixfiftyVonePt` | Mirnov in place of the fast inputs (TM vs T0), D650, V1 (per cent, point) | docs/paper/b82_localize.json part_C.V1.D650.TM/T0 |
| `\TXoverMirnovDsixfiftyVone` | fast inputs against Mirnov (TX vs TM), D650, V1 (per cent, interval) | docs/paper/b82_localize.json part_C.V1.D650.TX/TM |
| `\TXoverMirnovDsixfiftyVonePt` | fast inputs against Mirnov (TX vs TM), D650, V1 (per cent, point) | docs/paper/b82_localize.json part_C.V1.D650.TX/TM |
| `\TXoverPchipVone` | TX vs PCHIP on F, V1 (per cent, interval) | docs/paper/b83_interp.json pairs.V1.TX over PCHIP.MSE skill |
| `\TXoverPchipVonePt` | TX vs PCHIP on F, V1 (per cent, point) | docs/paper/b83_interp.json pairs.V1.TX over PCHIP.MSE skill |
| `\TXoverOfflineVone` | TX vs offline GP on F, V1 (per cent, interval) | docs/paper/b83_interp.json pairs.V1.TX over offline GP.MSE skill |
| `\TXoverOfflineVonePt` | TX vs offline GP on F, V1 (per cent, point) | docs/paper/b83_interp.json pairs.V1.TX over offline GP.MSE skill |
| `\TXoverGpFVone` | TX vs causal GP on F, V1 (per cent, interval) | docs/paper/b83_interp.json pairs.V1.TX over causal GP.MSE skill |
| `\TXoverGpFVonePt` | TX vs causal GP on F, V1 (per cent, point) | docs/paper/b83_interp.json pairs.V1.TX over causal GP.MSE skill |
| `\TXoverGpLfiveVone` | TX vs causal GP on L5, V1 (per cent, interval) | docs/paper/b73_targets.json per_estimator.V1/L5.TX[L5].over_G |
| `\TXoverGpLfiveVonePt` | TX vs causal GP on L5, V1 (per cent, point) | docs/paper/b73_targets.json per_estimator.V1/L5.TX[L5].over_G |
| `\TXoverPchipVtwo` | TX vs PCHIP on F, V2 (per cent, interval) | docs/paper/b83_interp.json pairs.V2.TX over PCHIP.MSE skill |
| `\TXoverPchipVtwoPt` | TX vs PCHIP on F, V2 (per cent, point) | docs/paper/b83_interp.json pairs.V2.TX over PCHIP.MSE skill |
| `\TXoverOfflineVtwo` | TX vs offline GP on F, V2 (per cent, interval) | docs/paper/b83_interp.json pairs.V2.TX over offline GP.MSE skill |
| `\TXoverOfflineVtwoPt` | TX vs offline GP on F, V2 (per cent, point) | docs/paper/b83_interp.json pairs.V2.TX over offline GP.MSE skill |
| `\TXoverGpFVtwo` | TX vs causal GP on F, V2 (per cent, interval) | docs/paper/b83_interp.json pairs.V2.TX over causal GP.MSE skill |
| `\TXoverGpFVtwoPt` | TX vs causal GP on F, V2 (per cent, point) | docs/paper/b83_interp.json pairs.V2.TX over causal GP.MSE skill |
| `\TXoverGpLfiveVtwo` | TX vs causal GP on L5, V2 (per cent, interval) | docs/paper/b73_targets.json per_estimator.V2/L5.TX[L5].over_G |
| `\TXoverGpLfiveVtwoPt` | TX vs causal GP on L5, V2 (per cent, point) | docs/paper/b73_targets.json per_estimator.V2/L5.TX[L5].over_G |
| `\TXoverPchipVthree` | TX vs PCHIP on F, V3 (per cent, interval) | docs/paper/b83_interp.json pairs.V3.TX over PCHIP.MSE skill |
| `\TXoverPchipVthreePt` | TX vs PCHIP on F, V3 (per cent, point) | docs/paper/b83_interp.json pairs.V3.TX over PCHIP.MSE skill |
| `\TXoverOfflineVthree` | TX vs offline GP on F, V3 (per cent, interval) | docs/paper/b83_interp.json pairs.V3.TX over offline GP.MSE skill |
| `\TXoverOfflineVthreePt` | TX vs offline GP on F, V3 (per cent, point) | docs/paper/b83_interp.json pairs.V3.TX over offline GP.MSE skill |
| `\TXoverGpFVthree` | TX vs causal GP on F, V3 (per cent, interval) | docs/paper/b83_interp.json pairs.V3.TX over causal GP.MSE skill |
| `\TXoverGpFVthreePt` | TX vs causal GP on F, V3 (per cent, point) | docs/paper/b83_interp.json pairs.V3.TX over causal GP.MSE skill |
| `\TXoverGpLfiveVthree` | TX vs causal GP on L5, V3 (per cent, interval) | docs/paper/b73_targets.json per_estimator.V3/L5.TX[L5].over_G |
| `\TXoverGpLfiveVthreePt` | TX vs causal GP on L5, V3 (per cent, point) | docs/paper/b73_targets.json per_estimator.V3/L5.TX[L5].over_G |
| `\TXoverPchipVfoura` | TX vs PCHIP on F, V4a (per cent, interval) | docs/paper/b83_interp.json pairs.V4a.TX over PCHIP.MSE skill |
| `\TXoverPchipVfouraPt` | TX vs PCHIP on F, V4a (per cent, point) | docs/paper/b83_interp.json pairs.V4a.TX over PCHIP.MSE skill |
| `\TXoverOfflineVfoura` | TX vs offline GP on F, V4a (per cent, interval) | docs/paper/b83_interp.json pairs.V4a.TX over offline GP.MSE skill |
| `\TXoverOfflineVfouraPt` | TX vs offline GP on F, V4a (per cent, point) | docs/paper/b83_interp.json pairs.V4a.TX over offline GP.MSE skill |
| `\TXoverGpFVfoura` | TX vs causal GP on F, V4a (per cent, interval) | docs/paper/b83_interp.json pairs.V4a.TX over causal GP.MSE skill |
| `\TXoverGpFVfouraPt` | TX vs causal GP on F, V4a (per cent, point) | docs/paper/b83_interp.json pairs.V4a.TX over causal GP.MSE skill |
| `\TXoverGpLfiveVfoura` | TX vs causal GP on L5, V4a (per cent, interval) | docs/paper/b73_targets.json per_estimator.V4a/L5.TX[L5].over_G |
| `\TXoverGpLfiveVfouraPt` | TX vs causal GP on L5, V4a (per cent, point) | docs/paper/b73_targets.json per_estimator.V4a/L5.TX[L5].over_G |
| `\MedRatioTXOfflineVone` | median absolute error ratio TX / offline GP, V1, F | docs/paper/b83_interp.json |
| `\MedRatioTXOfflineVonePt` | the same, point | docs/paper/b83_interp.json |
| `\OneFitDsixfiftyVone` | one fit TXall vs per-rule TX, D650, V1 (r2) (per cent, interval) | docs/paper/b84_onefit.json cells.V1.D650.skills.TXall over TX |
| `\OneFitDsixfiftyVonePt` | one fit TXall vs per-rule TX, D650, V1 (r2) (per cent, point) | docs/paper/b84_onefit.json cells.V1.D650.skills.TXall over TX |
| `\OneFitDtenVone` | one fit TXall vs per-rule TX, D10, V1 (r2) (per cent, interval) | docs/paper/b84_onefit.json cells.V1.D10.skills.TXall over TX |
| `\OneFitDtenVonePt` | one fit TXall vs per-rule TX, D10, V1 (r2) (per cent, point) | docs/paper/b84_onefit.json cells.V1.D10.skills.TXall over TX |
| `\OneFitKtenVone` | one fit TXall vs per-rule TX, K10, V1 (r2) (per cent, interval) | docs/paper/b84_onefit.json cells.V1.K10.skills.TXall over TX |
| `\OneFitKtenVonePt` | one fit TXall vs per-rule TX, K10, V1 (r2) (per cent, point) | docs/paper/b84_onefit.json cells.V1.K10.skills.TXall over TX |
| `\NStatements` | registered statements B.73-B.86 | make_tables_v2.py (sum over the statements blocks of b73..b86) |
| `\NStatementsHold` | of them hold | make_tables_v2.py (sum over the statements blocks of b73..b86) |
| `\NStatementsNotHeld` | of them do not hold | make_tables_v2.py (sum over the statements blocks of b73..b86) |
| `\NStatementsDefinite` | with a definite registered expectation | make_tables_v2.py (sum over the statements blocks of b73..b86) |
| `\NStatementsMet` | definite expectations met | make_tables_v2.py (sum over the statements blocks of b73..b86) |
| `\IncDtenVone` | fast-channel increment, D10, V1, L5 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D10.TX/T0 |
| `\IncDtenVonePt` | fast-channel increment, D10, V1, L5 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/common.all.D10.TX/T0 |
| `\IncDsixfiftyVone` | fast-channel increment, D650, V1, L5 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.TX/T0 |
| `\IncDsixfiftyVonePt` | fast-channel increment, D650, V1, L5 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.TX/T0 |
| `\IncSVone` | fast-channel increment, S, V1, L5 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TX/T0 |
| `\IncSVonePt` | fast-channel increment, S, V1, L5 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TX/T0 |
| `\IncBeamDsixfiftyVone` | fast diagnostics after the beam scalars (TXN vs TN), D650, V1 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V1/L5/common.all.D650.TXN/TN |
| `\IncBeamDsixfiftyVonePt` | fast diagnostics after the beam scalars (TXN vs TN), D650, V1 (per cent, point) | docs/paper/b79_beam_inputs.json main.V1/L5/common.all.D650.TXN/TN |
| `\BeamAloneDsixfiftyVone` | beam scalars alone (TN vs T0), D650, V1 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V1/L5/common.all.D650.TN/T0 |
| `\BeamAloneDsixfiftyVonePt` | beam scalars alone (TN vs T0), D650, V1 (per cent, point) | docs/paper/b79_beam_inputs.json main.V1/L5/common.all.D650.TN/T0 |
| `\IncBeamSVone` | fast diagnostics after the beam scalars (TXN vs TN), S, V1 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V1/L5/S.all.S.TXN/TN |
| `\IncBeamSVonePt` | fast diagnostics after the beam scalars (TXN vs TN), S, V1 (per cent, point) | docs/paper/b79_beam_inputs.json main.V1/L5/S.all.S.TXN/TN |
| `\BeamAloneSVone` | beam scalars alone (TN vs T0), S, V1 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V1/L5/S.all.S.TN/T0 |
| `\BeamAloneSVonePt` | beam scalars alone (TN vs T0), S, V1 (per cent, point) | docs/paper/b79_beam_inputs.json main.V1/L5/S.all.S.TN/T0 |
| `\IncKtenVone` | fast-channel increment, K10, V1 (per cent, interval) | docs/paper/b78_sparse.json main.V1/L5.all.K10.TX/T0 |
| `\IncKtenVonePt` | fast-channel increment, K10, V1 (per cent, point) | docs/paper/b78_sparse.json main.V1/L5.all.K10.TX/T0 |
| `\IncDtenVtwo` | fast-channel increment, D10, V2, L5 (per cent, interval) | docs/paper/b75_availability.json main.V2/L5/common.all.D10.TX/T0 |
| `\IncDtenVtwoPt` | fast-channel increment, D10, V2, L5 (per cent, point) | docs/paper/b75_availability.json main.V2/L5/common.all.D10.TX/T0 |
| `\IncDsixfiftyVtwo` | fast-channel increment, D650, V2, L5 (per cent, interval) | docs/paper/b75_availability.json main.V2/L5/common.all.D650.TX/T0 |
| `\IncDsixfiftyVtwoPt` | fast-channel increment, D650, V2, L5 (per cent, point) | docs/paper/b75_availability.json main.V2/L5/common.all.D650.TX/T0 |
| `\IncSVtwo` | fast-channel increment, S, V2, L5 (per cent, interval) | docs/paper/b75_availability.json main.V2/L5/S.all.S.TX/T0 |
| `\IncSVtwoPt` | fast-channel increment, S, V2, L5 (per cent, point) | docs/paper/b75_availability.json main.V2/L5/S.all.S.TX/T0 |
| `\IncBeamDsixfiftyVtwo` | fast diagnostics after the beam scalars (TXN vs TN), D650, V2 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V2/L5/common.all.D650.TXN/TN |
| `\IncBeamDsixfiftyVtwoPt` | fast diagnostics after the beam scalars (TXN vs TN), D650, V2 (per cent, point) | docs/paper/b79_beam_inputs.json main.V2/L5/common.all.D650.TXN/TN |
| `\BeamAloneDsixfiftyVtwo` | beam scalars alone (TN vs T0), D650, V2 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V2/L5/common.all.D650.TN/T0 |
| `\BeamAloneDsixfiftyVtwoPt` | beam scalars alone (TN vs T0), D650, V2 (per cent, point) | docs/paper/b79_beam_inputs.json main.V2/L5/common.all.D650.TN/T0 |
| `\IncBeamSVtwo` | fast diagnostics after the beam scalars (TXN vs TN), S, V2 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V2/L5/S.all.S.TXN/TN |
| `\IncBeamSVtwoPt` | fast diagnostics after the beam scalars (TXN vs TN), S, V2 (per cent, point) | docs/paper/b79_beam_inputs.json main.V2/L5/S.all.S.TXN/TN |
| `\BeamAloneSVtwo` | beam scalars alone (TN vs T0), S, V2 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V2/L5/S.all.S.TN/T0 |
| `\BeamAloneSVtwoPt` | beam scalars alone (TN vs T0), S, V2 (per cent, point) | docs/paper/b79_beam_inputs.json main.V2/L5/S.all.S.TN/T0 |
| `\IncKtenVtwo` | fast-channel increment, K10, V2 (per cent, interval) | docs/paper/b78_sparse.json main.V2/L5.all.K10.TX/T0 |
| `\IncKtenVtwoPt` | fast-channel increment, K10, V2 (per cent, point) | docs/paper/b78_sparse.json main.V2/L5.all.K10.TX/T0 |
| `\IncDtenVthree` | fast-channel increment, D10, V3, L5 (per cent, interval) | docs/paper/b75_availability.json main.V3/L5/common.all.D10.TX/T0 |
| `\IncDtenVthreePt` | fast-channel increment, D10, V3, L5 (per cent, point) | docs/paper/b75_availability.json main.V3/L5/common.all.D10.TX/T0 |
| `\IncDsixfiftyVthree` | fast-channel increment, D650, V3, L5 (per cent, interval) | docs/paper/b75_availability.json main.V3/L5/common.all.D650.TX/T0 |
| `\IncDsixfiftyVthreePt` | fast-channel increment, D650, V3, L5 (per cent, point) | docs/paper/b75_availability.json main.V3/L5/common.all.D650.TX/T0 |
| `\IncSVthree` | fast-channel increment, S, V3, L5 (per cent, interval) | docs/paper/b75_availability.json main.V3/L5/S.all.S.TX/T0 |
| `\IncSVthreePt` | fast-channel increment, S, V3, L5 (per cent, point) | docs/paper/b75_availability.json main.V3/L5/S.all.S.TX/T0 |
| `\IncBeamDsixfiftyVthree` | fast diagnostics after the beam scalars (TXN vs TN), D650, V3 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V3/L5/common.all.D650.TXN/TN |
| `\IncBeamDsixfiftyVthreePt` | fast diagnostics after the beam scalars (TXN vs TN), D650, V3 (per cent, point) | docs/paper/b79_beam_inputs.json main.V3/L5/common.all.D650.TXN/TN |
| `\BeamAloneDsixfiftyVthree` | beam scalars alone (TN vs T0), D650, V3 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V3/L5/common.all.D650.TN/T0 |
| `\BeamAloneDsixfiftyVthreePt` | beam scalars alone (TN vs T0), D650, V3 (per cent, point) | docs/paper/b79_beam_inputs.json main.V3/L5/common.all.D650.TN/T0 |
| `\IncBeamSVthree` | fast diagnostics after the beam scalars (TXN vs TN), S, V3 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V3/L5/S.all.S.TXN/TN |
| `\IncBeamSVthreePt` | fast diagnostics after the beam scalars (TXN vs TN), S, V3 (per cent, point) | docs/paper/b79_beam_inputs.json main.V3/L5/S.all.S.TXN/TN |
| `\BeamAloneSVthree` | beam scalars alone (TN vs T0), S, V3 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V3/L5/S.all.S.TN/T0 |
| `\BeamAloneSVthreePt` | beam scalars alone (TN vs T0), S, V3 (per cent, point) | docs/paper/b79_beam_inputs.json main.V3/L5/S.all.S.TN/T0 |
| `\IncKtenVthree` | fast-channel increment, K10, V3 (per cent, interval) | docs/paper/b78_sparse.json main.V3/L5.all.K10.TX/T0 |
| `\IncKtenVthreePt` | fast-channel increment, K10, V3 (per cent, point) | docs/paper/b78_sparse.json main.V3/L5.all.K10.TX/T0 |
| `\IncDtenVfoura` | fast-channel increment, D10, V4a, L5 (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/common.all.D10.TX/T0 |
| `\IncDtenVfouraPt` | fast-channel increment, D10, V4a, L5 (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/common.all.D10.TX/T0 |
| `\IncDsixfiftyVfoura` | fast-channel increment, D650, V4a, L5 (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.TX/T0 |
| `\IncDsixfiftyVfouraPt` | fast-channel increment, D650, V4a, L5 (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.TX/T0 |
| `\IncSVfoura` | fast-channel increment, S, V4a, L5 (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TX/T0 |
| `\IncSVfouraPt` | fast-channel increment, S, V4a, L5 (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TX/T0 |
| `\IncBeamDsixfiftyVfoura` | fast diagnostics after the beam scalars (TXN vs TN), D650, V4a (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4a/L5/common.all.D650.TXN/TN |
| `\IncBeamDsixfiftyVfouraPt` | fast diagnostics after the beam scalars (TXN vs TN), D650, V4a (per cent, point) | docs/paper/b79_beam_inputs.json main.V4a/L5/common.all.D650.TXN/TN |
| `\BeamAloneDsixfiftyVfoura` | beam scalars alone (TN vs T0), D650, V4a (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4a/L5/common.all.D650.TN/T0 |
| `\BeamAloneDsixfiftyVfouraPt` | beam scalars alone (TN vs T0), D650, V4a (per cent, point) | docs/paper/b79_beam_inputs.json main.V4a/L5/common.all.D650.TN/T0 |
| `\IncBeamSVfoura` | fast diagnostics after the beam scalars (TXN vs TN), S, V4a (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4a/L5/S.all.S.TXN/TN |
| `\IncBeamSVfouraPt` | fast diagnostics after the beam scalars (TXN vs TN), S, V4a (per cent, point) | docs/paper/b79_beam_inputs.json main.V4a/L5/S.all.S.TXN/TN |
| `\BeamAloneSVfoura` | beam scalars alone (TN vs T0), S, V4a (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4a/L5/S.all.S.TN/T0 |
| `\BeamAloneSVfouraPt` | beam scalars alone (TN vs T0), S, V4a (per cent, point) | docs/paper/b79_beam_inputs.json main.V4a/L5/S.all.S.TN/T0 |
| `\IncKtenVfoura` | fast-channel increment, K10, V4a (per cent, interval) | docs/paper/b78_sparse.json main.V4a/L5.all.K10.TX/T0 |
| `\IncKtenVfouraPt` | fast-channel increment, K10, V4a (per cent, point) | docs/paper/b78_sparse.json main.V4a/L5.all.K10.TX/T0 |
| `\IncDtenVfourb` | fast-channel increment, D10, V4b, L5 (per cent, interval) | docs/paper/b75_availability.json main.V4b/L5/common.all.D10.TX/T0 |
| `\IncDtenVfourbPt` | fast-channel increment, D10, V4b, L5 (per cent, point) | docs/paper/b75_availability.json main.V4b/L5/common.all.D10.TX/T0 |
| `\IncDsixfiftyVfourb` | fast-channel increment, D650, V4b, L5 (per cent, interval) | docs/paper/b75_availability.json main.V4b/L5/common.all.D650.TX/T0 |
| `\IncDsixfiftyVfourbPt` | fast-channel increment, D650, V4b, L5 (per cent, point) | docs/paper/b75_availability.json main.V4b/L5/common.all.D650.TX/T0 |
| `\IncSVfourb` | fast-channel increment, S, V4b, L5 (per cent, interval) | docs/paper/b75_availability.json main.V4b/L5/S.all.S.TX/T0 |
| `\IncSVfourbPt` | fast-channel increment, S, V4b, L5 (per cent, point) | docs/paper/b75_availability.json main.V4b/L5/S.all.S.TX/T0 |
| `\IncBeamDsixfiftyVfourb` | fast diagnostics after the beam scalars (TXN vs TN), D650, V4b (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4b/L5/common.all.D650.TXN/TN |
| `\IncBeamDsixfiftyVfourbPt` | fast diagnostics after the beam scalars (TXN vs TN), D650, V4b (per cent, point) | docs/paper/b79_beam_inputs.json main.V4b/L5/common.all.D650.TXN/TN |
| `\BeamAloneDsixfiftyVfourb` | beam scalars alone (TN vs T0), D650, V4b (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4b/L5/common.all.D650.TN/T0 |
| `\BeamAloneDsixfiftyVfourbPt` | beam scalars alone (TN vs T0), D650, V4b (per cent, point) | docs/paper/b79_beam_inputs.json main.V4b/L5/common.all.D650.TN/T0 |
| `\IncBeamSVfourb` | fast diagnostics after the beam scalars (TXN vs TN), S, V4b (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4b/L5/S.all.S.TXN/TN |
| `\IncBeamSVfourbPt` | fast diagnostics after the beam scalars (TXN vs TN), S, V4b (per cent, point) | docs/paper/b79_beam_inputs.json main.V4b/L5/S.all.S.TXN/TN |
| `\BeamAloneSVfourb` | beam scalars alone (TN vs T0), S, V4b (per cent, interval) | docs/paper/b79_beam_inputs.json main.V4b/L5/S.all.S.TN/T0 |
| `\BeamAloneSVfourbPt` | beam scalars alone (TN vs T0), S, V4b (per cent, point) | docs/paper/b79_beam_inputs.json main.V4b/L5/S.all.S.TN/T0 |
| `\IncKtenVfourb` | fast-channel increment, K10, V4b (per cent, interval) | docs/paper/b78_sparse.json main.V4b/L5.all.K10.TX/T0 |
| `\IncKtenVfourbPt` | fast-channel increment, K10, V4b (per cent, point) | docs/paper/b78_sparse.json main.V4b/L5.all.K10.TX/T0 |
| `\EceiOnlyDsixfiftyVone` | ECEI only (TE vs T0), D650, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.TE/T0 |
| `\EceiOnlyDsixfiftyVonePt` | ECEI only (TE vs T0), D650, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.TE/T0 |
| `\BesOnlyDsixfiftyVone` | BES only (TB vs T0), D650, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.TB/T0 |
| `\BesOnlyDsixfiftyVonePt` | BES only (TB vs T0), D650, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.TB/T0 |
| `\EceiOnlySVone` | ECEI only (TE vs T0), S, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TE/T0 |
| `\EceiOnlySVonePt` | ECEI only (TE vs T0), S, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TE/T0 |
| `\BesOnlySVone` | BES only (TB vs T0), S, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TB/T0 |
| `\BesOnlySVonePt` | BES only (TB vs T0), S, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/S.all.S.TB/T0 |
| `\EceiOnlyDsixfiftyVfoura` | ECEI only (TE vs T0), D650, V4a (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.TE/T0 |
| `\EceiOnlyDsixfiftyVfouraPt` | ECEI only (TE vs T0), D650, V4a (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.TE/T0 |
| `\BesOnlyDsixfiftyVfoura` | BES only (TB vs T0), D650, V4a (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.TB/T0 |
| `\BesOnlyDsixfiftyVfouraPt` | BES only (TB vs T0), D650, V4a (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.TB/T0 |
| `\EceiOnlySVfoura` | ECEI only (TE vs T0), S, V4a (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TE/T0 |
| `\EceiOnlySVfouraPt` | ECEI only (TE vs T0), S, V4a (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TE/T0 |
| `\BesOnlySVfoura` | BES only (TB vs T0), S, V4a (per cent, interval) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TB/T0 |
| `\BesOnlySVfouraPt` | BES only (TB vs T0), S, V4a (per cent, point) | docs/paper/b75_availability.json main.V4a/L5/S.all.S.TB/T0 |
| `\CtrlPermDsixfiftyVone` | control: fast inputs permuted (TXp vs T0), D650, V1 (per cent, interval) | docs/paper/b81_controls.json main.V1.D650.TXp/T0 |
| `\CtrlPermDsixfiftyVonePt` | control: fast inputs permuted (TXp vs T0), D650, V1 (per cent, point) | docs/paper/b81_controls.json main.V1.D650.TXp/T0 |
| `\CtrlAnchorDsixfiftyVone` | control: fast inputs at the anchor row (TXo vs T0), D650, V1 (per cent, interval) | docs/paper/b81_controls.json main.V1.D650.TXo/T0 |
| `\CtrlAnchorDsixfiftyVonePt` | control: fast inputs at the anchor row (TXo vs T0), D650, V1 (per cent, point) | docs/paper/b81_controls.json main.V1.D650.TXo/T0 |
| `\PresentOverAnchorDsixfiftyVone` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V1 (per cent, interval) | docs/paper/b81_controls.json main.V1.D650.TX/TXo |
| `\PresentOverAnchorDsixfiftyVonePt` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V1 (per cent, point) | docs/paper/b81_controls.json main.V1.D650.TX/TXo |
| `\CtrlPermSVone` | control: fast inputs permuted (TXp vs T0), S, V1 (per cent, interval) | docs/paper/b81_controls.json main.V1.S.TXp/T0 |
| `\CtrlPermSVonePt` | control: fast inputs permuted (TXp vs T0), S, V1 (per cent, point) | docs/paper/b81_controls.json main.V1.S.TXp/T0 |
| `\CtrlAnchorSVone` | control: fast inputs at the anchor row (TXo vs T0), S, V1 (per cent, interval) | docs/paper/b81_controls.json main.V1.S.TXo/T0 |
| `\CtrlAnchorSVonePt` | control: fast inputs at the anchor row (TXo vs T0), S, V1 (per cent, point) | docs/paper/b81_controls.json main.V1.S.TXo/T0 |
| `\PresentOverAnchorSVone` | present fast inputs over anchor-row inputs (TX vs TXo), S, V1 (per cent, interval) | docs/paper/b81_controls.json main.V1.S.TX/TXo |
| `\PresentOverAnchorSVonePt` | present fast inputs over anchor-row inputs (TX vs TXo), S, V1 (per cent, point) | docs/paper/b81_controls.json main.V1.S.TX/TXo |
| `\CtrlPermDsixfiftyVtwo` | control: fast inputs permuted (TXp vs T0), D650, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.D650.TXp/T0 |
| `\CtrlPermDsixfiftyVtwoPt` | control: fast inputs permuted (TXp vs T0), D650, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.D650.TXp/T0 |
| `\CtrlAnchorDsixfiftyVtwo` | control: fast inputs at the anchor row (TXo vs T0), D650, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.D650.TXo/T0 |
| `\CtrlAnchorDsixfiftyVtwoPt` | control: fast inputs at the anchor row (TXo vs T0), D650, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.D650.TXo/T0 |
| `\PresentOverAnchorDsixfiftyVtwo` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.D650.TX/TXo |
| `\PresentOverAnchorDsixfiftyVtwoPt` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.D650.TX/TXo |
| `\CtrlPermSVtwo` | control: fast inputs permuted (TXp vs T0), S, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.S.TXp/T0 |
| `\CtrlPermSVtwoPt` | control: fast inputs permuted (TXp vs T0), S, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.S.TXp/T0 |
| `\CtrlAnchorSVtwo` | control: fast inputs at the anchor row (TXo vs T0), S, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.S.TXo/T0 |
| `\CtrlAnchorSVtwoPt` | control: fast inputs at the anchor row (TXo vs T0), S, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.S.TXo/T0 |
| `\PresentOverAnchorSVtwo` | present fast inputs over anchor-row inputs (TX vs TXo), S, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.S.TX/TXo |
| `\PresentOverAnchorSVtwoPt` | present fast inputs over anchor-row inputs (TX vs TXo), S, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.S.TX/TXo |
| `\CtrlPermDsixfiftyVthree` | control: fast inputs permuted (TXp vs T0), D650, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.D650.TXp/T0 |
| `\CtrlPermDsixfiftyVthreePt` | control: fast inputs permuted (TXp vs T0), D650, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.D650.TXp/T0 |
| `\CtrlAnchorDsixfiftyVthree` | control: fast inputs at the anchor row (TXo vs T0), D650, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.D650.TXo/T0 |
| `\CtrlAnchorDsixfiftyVthreePt` | control: fast inputs at the anchor row (TXo vs T0), D650, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.D650.TXo/T0 |
| `\PresentOverAnchorDsixfiftyVthree` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.D650.TX/TXo |
| `\PresentOverAnchorDsixfiftyVthreePt` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.D650.TX/TXo |
| `\CtrlPermSVthree` | control: fast inputs permuted (TXp vs T0), S, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.S.TXp/T0 |
| `\CtrlPermSVthreePt` | control: fast inputs permuted (TXp vs T0), S, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.S.TXp/T0 |
| `\CtrlAnchorSVthree` | control: fast inputs at the anchor row (TXo vs T0), S, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.S.TXo/T0 |
| `\CtrlAnchorSVthreePt` | control: fast inputs at the anchor row (TXo vs T0), S, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.S.TXo/T0 |
| `\PresentOverAnchorSVthree` | present fast inputs over anchor-row inputs (TX vs TXo), S, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.S.TX/TXo |
| `\PresentOverAnchorSVthreePt` | present fast inputs over anchor-row inputs (TX vs TXo), S, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.S.TX/TXo |
| `\CtrlPermDsixfiftyVfoura` | control: fast inputs permuted (TXp vs T0), D650, V4a (per cent, interval) | docs/paper/b81_controls.json main.V4a.D650.TXp/T0 |
| `\CtrlPermDsixfiftyVfouraPt` | control: fast inputs permuted (TXp vs T0), D650, V4a (per cent, point) | docs/paper/b81_controls.json main.V4a.D650.TXp/T0 |
| `\CtrlAnchorDsixfiftyVfoura` | control: fast inputs at the anchor row (TXo vs T0), D650, V4a (per cent, interval) | docs/paper/b81_controls.json main.V4a.D650.TXo/T0 |
| `\CtrlAnchorDsixfiftyVfouraPt` | control: fast inputs at the anchor row (TXo vs T0), D650, V4a (per cent, point) | docs/paper/b81_controls.json main.V4a.D650.TXo/T0 |
| `\PresentOverAnchorDsixfiftyVfoura` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V4a (per cent, interval) | docs/paper/b81_controls.json main.V4a.D650.TX/TXo |
| `\PresentOverAnchorDsixfiftyVfouraPt` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V4a (per cent, point) | docs/paper/b81_controls.json main.V4a.D650.TX/TXo |
| `\CtrlPermSVfoura` | control: fast inputs permuted (TXp vs T0), S, V4a (per cent, interval) | docs/paper/b81_controls.json main.V4a.S.TXp/T0 |
| `\CtrlPermSVfouraPt` | control: fast inputs permuted (TXp vs T0), S, V4a (per cent, point) | docs/paper/b81_controls.json main.V4a.S.TXp/T0 |
| `\CtrlAnchorSVfoura` | control: fast inputs at the anchor row (TXo vs T0), S, V4a (per cent, interval) | docs/paper/b81_controls.json main.V4a.S.TXo/T0 |
| `\CtrlAnchorSVfouraPt` | control: fast inputs at the anchor row (TXo vs T0), S, V4a (per cent, point) | docs/paper/b81_controls.json main.V4a.S.TXo/T0 |
| `\PresentOverAnchorSVfoura` | present fast inputs over anchor-row inputs (TX vs TXo), S, V4a (per cent, interval) | docs/paper/b81_controls.json main.V4a.S.TX/TXo |
| `\PresentOverAnchorSVfouraPt` | present fast inputs over anchor-row inputs (TX vs TXo), S, V4a (per cent, point) | docs/paper/b81_controls.json main.V4a.S.TX/TXo |
| `\CtrlPermDsixfiftyVfourb` | control: fast inputs permuted (TXp vs T0), D650, V4b (per cent, interval) | docs/paper/b81_controls.json main.V4b.D650.TXp/T0 |
| `\CtrlPermDsixfiftyVfourbPt` | control: fast inputs permuted (TXp vs T0), D650, V4b (per cent, point) | docs/paper/b81_controls.json main.V4b.D650.TXp/T0 |
| `\CtrlAnchorDsixfiftyVfourb` | control: fast inputs at the anchor row (TXo vs T0), D650, V4b (per cent, interval) | docs/paper/b81_controls.json main.V4b.D650.TXo/T0 |
| `\CtrlAnchorDsixfiftyVfourbPt` | control: fast inputs at the anchor row (TXo vs T0), D650, V4b (per cent, point) | docs/paper/b81_controls.json main.V4b.D650.TXo/T0 |
| `\PresentOverAnchorDsixfiftyVfourb` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V4b (per cent, interval) | docs/paper/b81_controls.json main.V4b.D650.TX/TXo |
| `\PresentOverAnchorDsixfiftyVfourbPt` | present fast inputs over anchor-row inputs (TX vs TXo), D650, V4b (per cent, point) | docs/paper/b81_controls.json main.V4b.D650.TX/TXo |
| `\CtrlPermSVfourb` | control: fast inputs permuted (TXp vs T0), S, V4b (per cent, interval) | docs/paper/b81_controls.json main.V4b.S.TXp/T0 |
| `\CtrlPermSVfourbPt` | control: fast inputs permuted (TXp vs T0), S, V4b (per cent, point) | docs/paper/b81_controls.json main.V4b.S.TXp/T0 |
| `\CtrlAnchorSVfourb` | control: fast inputs at the anchor row (TXo vs T0), S, V4b (per cent, interval) | docs/paper/b81_controls.json main.V4b.S.TXo/T0 |
| `\CtrlAnchorSVfourbPt` | control: fast inputs at the anchor row (TXo vs T0), S, V4b (per cent, point) | docs/paper/b81_controls.json main.V4b.S.TXo/T0 |
| `\PresentOverAnchorSVfourb` | present fast inputs over anchor-row inputs (TX vs TXo), S, V4b (per cent, interval) | docs/paper/b81_controls.json main.V4b.S.TX/TXo |
| `\PresentOverAnchorSVfourbPt` | present fast inputs over anchor-row inputs (TX vs TXo), S, V4b (per cent, point) | docs/paper/b81_controls.json main.V4b.S.TX/TXo |
| `\EceiOnlyDsixfiftyVtwo` | ECEI only (TE vs T0), D650, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.D650.TE/T0 |
| `\EceiOnlyDsixfiftyVtwoPt` | ECEI only (TE vs T0), D650, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.D650.TE/T0 |
| `\BesOnlyDsixfiftyVtwo` | BES only (TB vs T0), D650, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.D650.TB/T0 |
| `\BesOnlyDsixfiftyVtwoPt` | BES only (TB vs T0), D650, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.D650.TB/T0 |
| `\EceiOnlySVtwo` | ECEI only (TE vs T0), S, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.S.TE/T0 |
| `\EceiOnlySVtwoPt` | ECEI only (TE vs T0), S, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.S.TE/T0 |
| `\BesOnlySVtwo` | BES only (TB vs T0), S, V2 (per cent, interval) | docs/paper/b81_controls.json main.V2.S.TB/T0 |
| `\BesOnlySVtwoPt` | BES only (TB vs T0), S, V2 (per cent, point) | docs/paper/b81_controls.json main.V2.S.TB/T0 |
| `\EceiOnlyDsixfiftyVthree` | ECEI only (TE vs T0), D650, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.D650.TE/T0 |
| `\EceiOnlyDsixfiftyVthreePt` | ECEI only (TE vs T0), D650, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.D650.TE/T0 |
| `\BesOnlyDsixfiftyVthree` | BES only (TB vs T0), D650, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.D650.TB/T0 |
| `\BesOnlyDsixfiftyVthreePt` | BES only (TB vs T0), D650, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.D650.TB/T0 |
| `\EceiOnlySVthree` | ECEI only (TE vs T0), S, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.S.TE/T0 |
| `\EceiOnlySVthreePt` | ECEI only (TE vs T0), S, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.S.TE/T0 |
| `\BesOnlySVthree` | BES only (TB vs T0), S, V3 (per cent, interval) | docs/paper/b81_controls.json main.V3.S.TB/T0 |
| `\BesOnlySVthreePt` | BES only (TB vs T0), S, V3 (per cent, point) | docs/paper/b81_controls.json main.V3.S.TB/T0 |
| `\IncMastDfiveMone` | MAST Thomson increment, D5, M1 (per cent, interval) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D5.TX/T0 |
| `\IncMastDfiveMonePt` | MAST Thomson increment, D5, M1 (per cent, point) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D5.TX/T0 |
| `\IncMastDhundredMone` | MAST Thomson increment, D100, M1 (per cent, interval) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D100.TX/T0 |
| `\IncMastDhundredMonePt` | MAST Thomson increment, D100, M1 (per cent, point) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D100.TX/T0 |
| `\IncMastDfiveMtwo` | MAST Thomson increment, D5, M2 (per cent, interval) | docs/paper/b76_mast_availability.json main.M2/L5/common.all.D5.TX/T0 |
| `\IncMastDfiveMtwoPt` | MAST Thomson increment, D5, M2 (per cent, point) | docs/paper/b76_mast_availability.json main.M2/L5/common.all.D5.TX/T0 |
| `\IncMastDhundredMtwo` | MAST Thomson increment, D100, M2 (per cent, interval) | docs/paper/b76_mast_availability.json main.M2/L5/common.all.D100.TX/T0 |
| `\IncMastDhundredMtwoPt` | MAST Thomson increment, D100, M2 (per cent, point) | docs/paper/b76_mast_availability.json main.M2/L5/common.all.D100.TX/T0 |
| `\IncMastDfiveMthree` | MAST Thomson increment, D5, M3 (per cent, interval) | docs/paper/b76_mast_availability.json main.M3/L5/common.all.D5.TX/T0 |
| `\IncMastDfiveMthreePt` | MAST Thomson increment, D5, M3 (per cent, point) | docs/paper/b76_mast_availability.json main.M3/L5/common.all.D5.TX/T0 |
| `\IncMastDhundredMthree` | MAST Thomson increment, D100, M3 (per cent, interval) | docs/paper/b76_mast_availability.json main.M3/L5/common.all.D100.TX/T0 |
| `\IncMastDhundredMthreePt` | MAST Thomson increment, D100, M3 (per cent, point) | docs/paper/b76_mast_availability.json main.M3/L5/common.all.D100.TX/T0 |
| `\IncMastDfiveMfour` | MAST Thomson increment, D5, M4 (per cent, interval) | docs/paper/b76_mast_availability.json main.M4/L5/common.all.D5.TX/T0 |
| `\IncMastDfiveMfourPt` | MAST Thomson increment, D5, M4 (per cent, point) | docs/paper/b76_mast_availability.json main.M4/L5/common.all.D5.TX/T0 |
| `\IncMastDhundredMfour` | MAST Thomson increment, D100, M4 (per cent, interval) | docs/paper/b76_mast_availability.json main.M4/L5/common.all.D100.TX/T0 |
| `\IncMastDhundredMfourPt` | MAST Thomson increment, D100, M4 (per cent, point) | docs/paper/b76_mast_availability.json main.M4/L5/common.all.D100.TX/T0 |
| `\CostDsixfiftyVone` | cost of staleness of T0, D650, V1 (R1) (with interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.staleness T0 |
| `\CostDsixfiftyVonePt` | cost of staleness of T0, D650, V1 (R1) (point) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.staleness T0 |
| `\CostSVone` | cost of staleness of T0, S, V1 (with interval) | docs/paper/b75_availability.json main.V1/L5/S.all.S.staleness T0 |
| `\CostSVonePt` | cost of staleness of T0, S, V1 (point) | docs/paper/b75_availability.json main.V1/L5/S.all.S.staleness T0 |
| `\RecDsixfiftyVone` | recovered share, D650, V1 (with interval) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.recovered share |
| `\RecDsixfiftyVonePt` | recovered share, D650, V1 (point) | docs/paper/b75_availability.json main.V1/L5/common.all.D650.recovered share |
| `\RecSVone` | recovered share, S, V1 (with interval) | docs/paper/b75_availability.json main.V1/L5/S.all.S.recovered share |
| `\RecSVonePt` | recovered share, S, V1 (point) | docs/paper/b75_availability.json main.V1/L5/S.all.S.recovered share |
| `\RecDsixfiftyVthree` | recovered share, D650, V3 (with interval) | docs/paper/b75_availability.json main.V3/L5/common.all.D650.recovered share |
| `\RecDsixfiftyVthreePt` | recovered share, D650, V3 (point) | docs/paper/b75_availability.json main.V3/L5/common.all.D650.recovered share |
| `\RecDsixfiftyVfoura` | recovered share, D650, V4a (with interval) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.recovered share |
| `\RecDsixfiftyVfouraPt` | recovered share, D650, V4a (point) | docs/paper/b75_availability.json main.V4a/L5/common.all.D650.recovered share |
| `\CostMastDhundredMone` | MAST cost of staleness, D100, M1 (S1) (with interval) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D100.staleness T0 |
| `\CostMastDhundredMonePt` | MAST cost of staleness, D100, M1 (S1) (point) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D100.staleness T0 |
| `\RecMastDhundredMone` | MAST recovered share, D100, M1 (with interval) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D100.recovered share |
| `\RecMastDhundredMonePt` | MAST recovered share, D100, M1 (point) | docs/paper/b76_mast_availability.json main.M1/L5/common.all.D100.recovered share |
| `\EqRateKten` | equivalent spacing k' at K10, V1 (frames, interval) | docs/paper/b78_sparse.json main.V1/L5.all.K10.equivalent rate |
| `\EqRateKtenPt` | equivalent spacing k' at K10, V1 (frames) | docs/paper/b78_sparse.json |
| `\EqRateKtenMs` | equivalent spacing at K10 in ms (10 x k') | docs/paper/b78_sparse.json |
| `\EqRateKtenFactor` | spacing ratio k/k' at K10 | docs/paper/b78_sparse.json |
| `\AgeMedianS` | median age (s) of the newest value the analysis could deliver (S set, L5) | docs/paper/b75_availability.json |
| `\AgeQuartLoS` | 25th percentile of that age (s) | docs/paper/b75_availability.json |
| `\AgeQuartHiS` | 75th percentile of that age (s) | docs/paper/b75_availability.json |
| `\RmseTXSVone` | RMSE of TX under S, V1 (eV) | docs/paper/b75_availability.json |
| `\RmseOSVone` | RMSE of the frame's own reading on the S set, V1 (eV) | docs/paper/b75_availability.json |
| `\RmseTZeroSVone` | RMSE of T0 under S, V1 (eV) | docs/paper/b75_availability.json |
| `\TimeMedian` | computation time per row, median (ms) | docs/paper/b75_availability.json |
| `\TimePnn` | computation time per row, 99th percentile (ms) | docs/paper/b75_availability.json |
| `\NDoneRows` | D1 rows | docs/paper/b75_availability.json |
| `\NDoneDischarges` | D1 discharges | docs/paper/b75_availability.json |
| `\NDoneClusters` | D1 shot clusters | docs/paper/b75_availability.json |
| `\NGridRows` | 10 ms grid rows without the first and last rows of the files | docs/paper/nbi_files.md sec. 1 |
| `\NFiles` | shot files | docs/paper/b73_targets.json |
| `\NCommonRows` | common set rows (L5) | docs/paper/b75_availability.json |
| `\NCommonDischarges` | common set discharges | docs/paper/b75_availability.json |
| `\NCommonClusters` | common set clusters | docs/paper/b75_availability.json |
| `\NSRows` | S set rows (L5) | docs/paper/b75_availability.json |
| `\NSDischarges` | S set discharges | docs/paper/b75_availability.json |
| `\NSClusters` | S set clusters | docs/paper/b75_availability.json |
| `\NKRows` | K set rows (L5) | docs/paper/b78_sparse.json |
| `\NKDischarges` | K set discharges | docs/paper/b78_sparse.json |
| `\NMastRows` | MAST common rows (L5) | docs/paper/b76_mast_availability.json |
| `\NMastShots` | MAST shots | docs/paper/b76_mast_availability.json |
| `\NMastSessions` | MAST sessions | docs/paper/b76_mast_availability.json |
| `\NLastBlockDischarges` | discharges of the last block on the S set | docs/paper/b75_availability.json |
| `\NLastBlockRows` | rows of the last block on the S set | docs/paper/b75_availability.json |
| `\IncSLastBlockVone` | increment under S on the last block, V1 (per cent, interval) | docs/paper/b75_availability.json main.V1/L5/S.block from 32312.S.TX/T0 |
| `\IncSLastBlockVonePt` | increment under S on the last block, V1 (per cent, point) | docs/paper/b75_availability.json main.V1/L5/S.block from 32312.S.TX/T0 |
| `\IncBeamSLastBlockVone` | TXN vs TN under S on the last block, V1 (per cent, interval) | docs/paper/b79_beam_inputs.json main.V1/L5/S.block from 32312.S.TXN/TN |
| `\IncBeamSLastBlockVonePt` | TXN vs TN under S on the last block, V1 (per cent, point) | docs/paper/b79_beam_inputs.json main.V1/L5/S.block from 32312.S.TXN/TN |
| `\GateSpearman` | rank correlation of per-discharge skill (V1, S) with C_E | docs/paper/b80_ecei_gate.json spearman.all blocks |
| `\NGateSpearmanDischarges` | discharges of that rank correlation | docs/paper/b80_ecei_gate.json |

## Figures (`fig_v2.py`; `figv2_<name>.pdf/.png/.json`, the json holds every plotted number and `printed_tokens`)

Canvas widths are final: 16 cm for full-width figures, 10 cm for `figv2_sparse`; fonts 8 to 9 pt at that size. Include them at their natural size (`width=\textwidth` for 16 cm, `width=10cm` for the sparse figure) so nothing is scaled.

| figure | shows | source | paper map |
|---|---|---|---|
| `figv2_problem` | (a) the example discharge: reported T_i, absent frames (vertical lines), NBI-1 power (NB1B + NB1C), BES and ECEI array means over their discharge medians, grey band where the four ECEI channels are constant (rows kept); (b) distribution over the S-set rows (L5) of the age of the newest CES value the beam-modulation analysis could deliver (median 0.65 s; last bin holds ages above 2 s) | caches `data/.b73/grid.npz`, `data/.b79/beam_cols.npz`, `data/.b75/inputs/S.npz` (`age`, ms) on `S_L5` of `data/.b75/sets.npz` | sec. 2 |
| `figv2_rules` | schematic of the availability rules on a 10 ms frame axis: fast samples to t; D10; Dd (d = 60 ms drawn); S (before the latest notch); Kk (k = 5 drawn) | none (drawn) | sec. 3 |
| `figv2_ladder` | RMSE against the delay of the history: (a) KSTAR V1 (hold, causal GP, T0, TX; O as dashed line, named in the legend; rule S as open diamonds at the median age 0.65 s, S set), (b) MAST M1 | b75 `main.V1|L5|common.all.D*.rmse_eV`, `main.V1|L5|S.all.S.rmse_eV`; b76 `main.M1|L5|common.all.D*.rmse_eV` | 4.1, 4.4 |
| `figv2_designs` | forest plot of the fast-channel increment (%, cluster/session interval; filled = excludes zero): KSTAR D10, D650, S x V1..V4b; MAST D5, D100 x M1..M4 | b75, b76 `TX/T0` | 4.3, 4.4 |
| `figv2_which` | (a) D650, (b) S: per design TE/T0, TB/T0 (V1, V4 from B.75; V2, V3 from B.81), TX/T0 (B.75), TN/T0, TXN/TN; (c) the B.81 controls under D650 per design (fast inputs permuted, fast inputs at the anchor row, present over anchor row) and MAST M1 D100 (B.76) | b81, b75, b79, b76 | 4.2 |
| `figv2_sparse` | RMSE of T0 and TX against the spacing of the CES values (V1), O as dashed line, equivalent spacing of TX at K10 (k' = 4.67 frames = 46.7 ms, cluster interval as error bar) | b78 `main.V1|L5.all.K*` | 4.5 |
| `figv2_trace` | the example discharge in three 2 s windows (centred at the 25/50/75 % quantiles of its scored times): reported values, following level L5, T0 and TX under rule S (stored estimates), absent frames | caches `data/.b75/fits/V1_r2_k*__{T0,TX}__S__L5.npz`, `data/.b73/rows.npz`, `grid.npz` | 4.1 / 4.6 |
| `figv2_trace_q25` | the same for the 25th-percentile discharge (second example) | same | supplement |
| `figv2_nowcast` | the example discharge 31076 with the previous frame visible, target F: reported T_i, hold, PCHIP (waits), causal GP, TX[F] (B.73 stored), three 1 s windows, absent frames | caches `data/.b73/trees/V1_r2_k*__TX__F.npz`, `data/.b83/refs.npz`, `data/.b73/rows.npz` | 4 (main) |
| `figv2_tau` | increment TX vs T0 (%, cluster interval) by the time since the last notch, (a) D650, (b) S, one series per design V1, V2, V3, V4 (fit to 31991); filled = excludes zero | b82 `part_A.*.{D650,S}.all.tau` | 4.2/4.3 (main) |
| `figv2_limits` | (a) increment under S by block of shots, TX vs T0 and TXN vs TN (V1); (b) first row after a notch over the level before, T_i, BES, ECEI, by block (after the fact, no interval) | b75, b79 `main.V1|L5|S.<block>.S`; `b75_last_block.md` section (7) | 4.6 |

**Example discharge (selection rule, no hand picking).** Among the test discharges of unit group V1,
re-partition r2 (folds k0 to k4), with at least 500 rows in the S set (target L5) and at least three absent
steps (15 ms < time step < 45 ms inside a segment of the grid), the one whose skill of TX over T0 under S
(stored B.75 estimates, its own scored rows) is the median (lower median for an even count). Second example:
the 25th percentile. The chosen shots, the number of candidates and the skills are in `figv2_problem.json`
(`example`).


## Missing from the result files (not computed here; listed for the session lead)

1. (resolved by B.81) TE and TB under V2 and V3 and the controls per design under D650 and S. The earlier
   one-unit unregistered B.75 control (`b75_lead_checks.md` part C: permuted +0.013, anchor row -0.128) is
   superseded by B.81 and no longer printed.
2. No anchor-row control for B.79 (beam inputs); none for MAST under rules other than D100.
3. (none)
4. TX against TX8 (the anchor scalars given the other inputs) is not stored in b75; `\TabAnchorScalars`
   shows TX8 vs T0 and TX vs T0 side by side instead. TX8 exists for V1 and V4 only.
5. Computation time is measured for rule D650 only (one unit, shared machine); none for S or for MAST.
6. The age distribution of the newest deliverable value exists in the JSON as quantiles and bands only; the
   histogram of `figv2_problem` is computed from the stored ages in the cache (no estimator).
7. B.71 (replay when updates stop) and B.74 (bursts) have no table beyond the B.74 statement table; B.71's
   statements are not tabulated (the paper map puts it in the supplement; ask if wanted).

