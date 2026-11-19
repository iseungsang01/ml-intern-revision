# Checks behind `critique_20261120_cxrs.md`

Read-only checks on the 10 ms files, the raw streams and the stored D1 estimates of B.70.
Nothing is trained and nothing in the repository is written: every script writes its
intermediate tables beside the first argument, which should be a file in a scratch folder
outside the repository.

Run from the repository root, in this order. `S` is a scratch folder; the script names in
the table stand for `docs/paper/review_loop/critique_20261120_cxrs_checks/<name>`, and the
`.` argument is the repository root.

| id | command | reads | writes (in `S`) | what it prints |
|---|---|---|---|---|
| C0 | `py c00_load_all.py data S/grid.pkl` | `data/s*.csv` | `grid.pkl` | 641 files, 245,925 grid rows (first and last row of each file dropped) |
| C1 | `py c01_values.py S/grid.pkl` | grid | — | distribution of T_i and V_rot, repeated values |
| C2 | `py c02_held_rotation.py S/grid.pkl` | grid | `triples.npz` | one-frame scatter of T_i where V_rot is fresh or repeated |
| C3 | `py c03_excursions.py S/grid.pkl` | grid, triples | — | single-frame excursions, their link to BES and ECEI, spikes by band |
| C4 | `py c04_join_d1.py S/grid.pkl .` | `data/.b70/stage2_*`, `data/.b70p2/b54/rows_*` | `d1rows.pkl` | reproduces the D1 skills of `b70_d1.md` |
| C5 | `py c05_error_by_row_class.py S/d1rows.pkl` | d1rows | — | squared error and reductions by class of the target row |
| C6 | `py c06_following_frames_first.py S/grid.pkl .` | as C4 | `d1z.pkl` | first version of the following-frames reference; BES ratio of each scored row |
| C7 | `py c07_following_frames_build.py S/grid.pkl .` | as C4, `data/.b70p2/b56`, `data/.b70p2/replay` | `d1all.pkl` | all estimators against the frame's own value and against the following frames |
| C7b | `py c07b_following_frames_intervals.py S/d1all.pkl` | d1all | — | the same with two-level intervals |
| C8 | `py c08_rotation_witness.py S/grid.pkl` | grid | — | T_i excursions against V_rot excursions of the same fit |
| C9 | `py c09_shot32027_rows.py S/grid.pkl .` | grid | — | rows of 32027 around the two interruptions |
| C10 | `py c10_beam_off_rows.py S/grid.pkl` | grid | `beamoff.pkl` | frames stamped with the beam emission at its beam-off level |
| C11 | `py c11_reference_variants.py S/grid.pkl .` | d1all | `d1all.pkl` (columns added) | four future references; absent-frame counts |
| C12 | `py c12_value_age.py S/grid.pkl` | grid | — | position of the values relative to absent frames; age of the newest deliverable value |
| C13 | `py c13_low_high_ends.py S/d1all.pkl` | d1all | — | scored rows at the low and high ends |
| C14 | `py c14_correction_slopes.py S/d1all.pkl` | d1all | — | slope of the fast-channel correction on the own frame and on the following frames |
| C15 | `py c15_beam_off_raw.py S/beamoff.pkl .` | beamoff, `data/raw_mus/.cache` | — | raw BES during the flagged exposures |
| C16 | `py c16_beam_off_campaign.py S/grid.pkl` | grid | — | campaign-wide tally of values written with the beam emission low |
| C18 | `py c18_dose_table.py S/grid.pkl` | grid | `dose.pkl` | T_i, V_rot and ECEI of a frame against the BES sample at its stamp |
| C19 | `py c19_tail_rows.py S/grid.pkl` | grid | — | values after the beam emission ends (31747) |
| C20 | `py c20_raw_31747.py .` | raw BES of 31747 | — | raw BES level after 8.0 s |
| C21 | `py c21_bracket_status.py S/grid.pkl .` | grid, d1all | — | values inside and outside a pair of beam-off frames |
| C22 | `py c22_held_rotation_by_status.py S/grid.pkl` | grid | — | repeated V_rot values by that position |
| C23 | `py c23_scored_rows_by_bes.py S/grid.pkl` | d1all, d1z | — | scored rows by BES level; the 31747 tail in D1 |
| C24 | `py c24_following_frames_by_bes.py S/grid.pkl` | d1all, d1z | — | following-frames test by BES level of the frame |
| C25 | `py c25_following_frames_by_bes_intervals.py S/grid.pkl` | d1all, d1z | — | the same with two-level intervals |
| C26 | `py c26_reduction_share_by_bes.py S/grid.pkl` | d1all, d1z | — | share of the fast-channel reduction by BES level |

Conventions used in all of them: grid rows only; T_i above 3 keV and repeats of the previous
value are missing (as in D1) unless a script says "inclusive"; `S` (single-frame excursion) =
ln T_t minus the mean of ln T_(t-1) and ln T_(t+1) on rows whose two neighbours are 10 ms
away; "following-frames reference" = median of the reported values of rows t+1 to t+3 that
lie within 35 ms (at least two of them); intervals are two-level (a re-partition of r2-r4
drawn, then its discharges resampled).

The machine was loaded by other jobs when these ran; each takes one to five minutes with
two threads.
