# B.58 Beam blips as a calibration of the reported value, and the lag away from blips

**The definitions and thresholds were fixed after an exploratory look at the same data.** Registration: `ces_prediction/experiments/PREREGISTRATION_B58.md`. Runner: `ces_prediction/experiments/b58_blip_calibration/run_b58.py`. CPU only, no model, real files. Intervals: 95 %, 2,000 resamples of discharges. Halves: parity of the rank of the discharge file in the sorted list of 641 files.

Frozen definitions (copied unchanged): `uc_blip_shape.py` sha256 `f3fcf47582aef2ce0e18fa7dced573a0eee2dfd2e63ff3438fcdc1175a706630`, `uc_lag_quiet.py` sha256 `3dfd0fd4e626b9b7ce204a5031d2edc91d2bf618d684663eccf4834d163bcba8`. The per-discharge re-implementation of `uc_lag_quiet.py` in the runner reproduces its all-discharge medians exactly: True.

## Verdicts

| statement | all discharges | even half | odd half | verdict | expected |
|---|---|---|---|---|---|
| S1 shape | holds | holds | holds | **holds** | 0.01 and 0.34, holds, also in each half |
| S2 signal-level part < 1/3 of the first-row deficit | does not hold | does not hold | does not hold | **does not hold** | upper end 0.046 against 0.061, holds on all; a half may reach 1/3 |
| S3 control | holds | holds | holds | **holds** | holds |
| L1 lag ratio, BES common < 0.5; heating-route prediction > 0.7 | holds | holds | holds | **holds** | 0.15 against 0.90, holds |
| L2 lag ratio, ECEI < 0.5 | holds | holds | holds | **holds** | about 0.07, holds |
| L3 positive control (windows with a blip) | does not hold | does not hold | does not hold | **does not hold** | holds, ratio 0.6 to 1.0 |

L3 does not hold everywhere: by the registration, **L1 and L2 are withdrawn as uninformative**.

## Part S, shape (registered set: one absent frame)

| quantity | all | even half | odd half | threshold |
|---|---|---|---|---|
| events (discharges) | 1262 (489) | 619 (246) | 643 (243) | |
| unexplained share, T_i proportional to ECEI | 0.009 [0.005, 0.032] | 0.021 [0.011, 0.066] | 0.010 [0.006, 0.057] | upper end < 0.10 |
| unexplained share, T_i proportional to BES | 0.333 [0.286, 0.376] | 0.337 [0.276, 0.388] | 0.331 [0.260, 0.405] | lower end > 0.20 |
| b x first-row BES deficit | 0.030 [0.001, 0.065] | 0.039 [-0.002, 0.081] | 0.035 [-0.017, 0.087] | upper end < 1/3 of first-row T_i deficit |
| one third of the first-row T_i deficit | 0.062 | 0.062 | 0.062 | |
| first-row T_i deficit | 0.186 [0.165, 0.210] | 0.186 [0.160, 0.221] | 0.185 [0.160, 0.216] | |
| share of first-row deficit proportional to BES | 0.162 [0.006, 0.329] | 0.209 [-0.013, 0.414] | 0.188 [-0.096, 0.442] | reported |
| a (per unit ECEI deficit) | 2.114 [1.756, 2.544] | 1.943 [1.467, 2.506] | 2.241 [1.730, 2.814] | reported |
| b (per unit BES deficit) | 0.088 [0.003, 0.188] | 0.113 [-0.007, 0.235] | 0.103 [-0.049, 0.255] | reported (an upper bound) |
| shape index ti (rows 4-6 / row 1) | 0.487 [0.427, 0.550] | 0.516 [0.440, 0.588] | 0.455 [0.364, 0.561] | reported |
| shape index ecei (rows 4-6 / row 1) | 0.559 [0.467, 0.670] | 0.583 [0.442, 0.727] | 0.560 [0.389, 0.770] | reported |
| shape index bes (rows 4-6 / row 1) | 0.075 [0.063, 0.085] | 0.083 [0.063, 0.100] | 0.069 [0.054, 0.080] | reported |
| control: max deviation from 1, T_i / ECEI / BES | 0.0111 / 0.0025 / 0.0058 | 0.0112 / 0.0069 / 0.0057 | 0.0183 / 0.0052 / 0.0066 | 0.02 / 0.015 / 0.015 |

### Event-median profiles, all discharges (ratio to the median of the six rows before)

| row | T_i | ECEI level | BES level | control T_i | control ECEI | control BES |
|---|---|---|---|---|---|---|
| +1 | 0.814 [0.790, 0.835] | 0.929 [0.918, 0.942] | 0.658 [0.653, 0.663] | 1.005 | 1.002 | 1.006 |
| +2 | 0.841 [0.826, 0.857] | 0.923 [0.913, 0.938] | 0.872 [0.866, 0.878] | 1.005 | 1.000 | 1.002 |
| +3 | 0.858 [0.838, 0.878] | 0.938 [0.924, 0.949] | 0.931 [0.926, 0.937] | 1.003 | 1.002 | 1.004 |
| +4 | 0.877 [0.867, 0.891] | 0.944 [0.932, 0.955] | 0.951 [0.946, 0.957] | 1.011 | 1.002 | 1.003 |
| +5 | 0.915 [0.897, 0.925] | 0.961 [0.954, 0.968] | 0.977 [0.973, 0.982] | 1.007 | 1.000 | 1.003 |
| +6 | 0.936 [0.923, 0.947] | 0.976 [0.967, 0.984] | 0.995 [0.991, 1.000] | 1.009 | 1.000 | 1.002 |
| +7 | 0.962 [0.952, 0.969] | 0.986 [0.976, 0.995] | 1.013 [1.009, 1.018] | 1.007 | 1.000 | 1.000 |
| +8 | 0.977 [0.965, 0.986] | 0.991 [0.983, 1.000] | 1.016 [1.013, 1.021] | 1.005 | 1.000 | 1.004 |
| +9 | 0.986 [0.977, 0.997] | 1.000 [0.994, 1.009] | 1.020 [1.015, 1.025] | 1.006 | 1.000 | 1.002 |
| +10 | 0.996 [0.989, 1.006] | 1.000 [0.994, 1.007] | 1.029 [1.023, 1.033] | 1.005 | 1.000 | 1.004 |

### Event by event, rows 4 to 6 (reported)

| association | all | even half | odd half |
|---|---|---|---|
| rank_ti_bes | 0.468 [0.413, 0.523] | 0.463 [0.380, 0.542] | 0.473 [0.388, 0.551] |
| partial_ti_bes_given_ecei | 0.399 [0.336, 0.458] | 0.415 [0.328, 0.504] | 0.378 [0.278, 0.465] |
| rank_ti_ecei | 0.366 [0.309, 0.424] | 0.303 [0.216, 0.383] | 0.430 [0.358, 0.501] |
| partial_ti_ecei_given_bes | 0.261 [0.202, 0.319] | 0.209 [0.128, 0.292] | 0.316 [0.239, 0.394] |

### Two and three absent frames (reported apart, not statements)

| events | n (discharges) | T_i row 1 | ECEI row 1 | BES row 1 | shape T_i / ECEI / BES | unexplained ECEI / BES | b |
|---|---|---|---|---|---|---|---|
| 2_absent_frames | 67 (29) | 0.926 | 0.882 | 0.461 | -1.50 / -0.17 / -0.03 | 0.534 [0.321, 0.803] / 0.994 [0.906, 1.000] | -1.516 [-2.634, -0.215] |
| 3_absent_frames | 65 (33) | 0.830 | 0.926 | 0.601 | 0.42 / 0.15 / 0.03 | 0.236 [0.070, 0.875] / 0.460 [0.305, 0.839] | 0.144 [-0.332, 0.577] |

## Part L, lag on quiet stretches (30 ms blocks; + lag = X leads T_i)

Median over discharges of the per-discharge rank correlation of first differences of block means.

| X | lag -30 ms | lag 0 | lag +30 ms | lag +60 ms | ratio +30/0 (all) | ratio even half | ratio odd half |
|---|---|---|---|---|---|---|---|
| BES common mode | +0.001 [-0.019, +0.019] | +0.314 [+0.295, +0.338] | +0.047 [+0.030, +0.064] | -0.006 [-0.021, +0.009] | +0.149 [+0.094, +0.204] | +0.151 [+0.093, +0.219] | +0.145 [+0.047, +0.225] |
| BES radial difference | +0.010 [-0.010, +0.023] | +0.239 [+0.209, +0.264] | +0.017 [-0.004, +0.030] | +0.005 [-0.011, +0.017] | +0.073 [-0.016, +0.131] | +0.134 [+0.033, +0.245] | -0.012 [-0.108, +0.084] |
| ECEI level | -0.044 [-0.061, -0.031] | +0.257 [+0.234, +0.282] | +0.019 [+0.007, +0.028] | -0.025 [-0.046, -0.012] | +0.073 [+0.026, +0.114] | +0.095 [+0.018, +0.145] | +0.041 [-0.001, +0.149] |
| prediction, heating route (registered kernel, part S) | -0.201 [-0.220, -0.183] | +0.549 [+0.539, +0.555] | +0.502 [+0.483, +0.514] | +0.113 [+0.079, +0.129] | +0.914 [+0.880, +0.945] | +0.925 [+0.871, +0.956] | +0.898 [+0.869, +0.957] |
| prediction, kernel frozen in uc_lag_quiet | -0.197 [-0.213, -0.178] | +0.553 [+0.544, +0.561] | +0.500 [+0.483, +0.517] | +0.091 [+0.058, +0.108] | +0.905 [+0.871, +0.932] | +0.909 [+0.857, +0.945] | +0.894 [+0.860, +0.951] |

Nulls (circular shift), all discharges, lag 0: BES common mode -0.013, BES radial difference -0.004, ECEI level -0.003.

### Positive control: windows that contain a one-frame blip (BES common mode)

45 positions on the 10 ms grid, the absent frame at position 15 + shift, NaN there. Registered reading: shift 0 (the absent frame opens a block). Shifts 1 and 2 are reported.

| shift | half | discharges | lag 0 | lag +30 ms | ratio +30/0 |
|---|---|---|---|---|---|
| 0 | all | 331 | +0.444 [+0.397, +0.475] | +0.083 [+0.050, +0.113] | +0.188 [+0.112, +0.262] |
| 0 | half_even | 164 | +0.443 [+0.364, +0.479] | +0.069 [+0.004, +0.132] | +0.155 [+0.009, +0.316] |
| 0 | half_odd | 167 | +0.455 [+0.390, +0.487] | +0.099 [+0.058, +0.126] | +0.217 [+0.126, +0.290] |
| 1 | all | 331 | +0.350 [+0.325, +0.374] | +0.211 [+0.184, +0.242] | +0.603 [+0.516, +0.713] |
| 1 | half_even | 164 | +0.343 [+0.308, +0.368] | +0.232 [+0.177, +0.281] | +0.677 [+0.518, +0.825] |
| 1 | half_odd | 167 | +0.363 [+0.316, +0.403] | +0.199 [+0.165, +0.241] | +0.550 [+0.449, +0.687] |
| 2 | all | 332 | +0.455 [+0.414, +0.490] | +0.057 [+0.037, +0.079] | +0.124 [+0.079, +0.182] |
| 2 | half_even | 165 | +0.449 [+0.408, +0.503] | +0.065 [+0.028, +0.119] | +0.145 [+0.062, +0.259] |
| 2 | half_odd | 167 | +0.460 [+0.391, +0.493] | +0.051 [+0.021, +0.077] | +0.111 [+0.045, +0.183] |

### Partial association at lag 0 (reported)

| | all | even half | odd half |
|---|---|---|---|
| ti_bes_given_ecei | +0.242 [+0.219, +0.265] (n 464) | +0.242 [+0.211, +0.272] (n 233) | +0.242 [+0.209, +0.276] (n 231) |
| ti_ecei_given_bes | +0.188 [+0.164, +0.214] (n 464) | +0.194 [+0.153, +0.220] (n 233) | +0.178 [+0.158, +0.218] (n 231) |

## S2 bound in eV on quiet rows (reported)

Upper end of b (0.188) x per-discharge s.d. of the 30 ms changes of the BES common mode (median 0.0771) x per-discharge median T_i (median 609 eV), 464 discharges: median 8.10 eV, quartiles 4.55 to 13.43 eV, 90th percentile 22.72 eV (a bound on the r.m.s. size of a signal-level bias per 30 ms change; not a statement).

## Raw-stream add-on, twelve discharges (not a statement)

Blips matched to absent frames: 45 (41 with one absent frame). Duration of the interruption from the raw BES (0.1 ms level below half of its 200 ms running median, at least 3 ms).

| association (one absent frame) | rank correlation | n |
|---|---|---|
| rank_duration_vs_ti_ratio_row1 | -0.013 | 37 |
| rank_beam_return_vs_ti_ratio_row1 | +0.167 | 37 |
| rank_duration_vs_ecei_ratio_row1 | -0.296 | 41 |
| rank_beam_return_vs_ecei_ratio_row1 | -0.222 | 41 |
| rank_duration_vs_bes_ratio_row1 | -0.760 | 41 |
| rank_beam_return_vs_bes_ratio_row1 | -0.739 | 41 |

| shot | absent frames | duration (ms) | beam return after last stamp (ms) | T_i row 1 | ECEI row 1 | BES row 1 |
|---|---|---|---|---|---|---|
| 31097 | 1 | 11.0 | +9.3 | 0.797 | 0.427 | 0.591 |
| 31097 | 1 | 10.8 | +9.1 | 0.839 | 0.455 | 0.633 |
| 31097 | 1 | 10.7 | +8.9 | 0.792 | 0.489 | 0.631 |
| 31114 | 1 | 10.3 | +8.6 | 1.152 | 0.808 | 0.729 |
| 31114 | 1 | 10.3 | +8.6 | 0.239 | 0.837 | 0.678 |
| 31114 | 1 | 10.2 | +8.5 | 0.576 | 0.881 | 0.732 |
| 31357 | 1 | 10.5 | +8.8 | 0.911 | 0.958 | 0.683 |
| 31357 | 1 | 10.4 | +8.7 | 0.483 | 0.950 | 0.681 |
| 31357 | 1 | 10.2 | +8.5 | 1.525 | 0.955 | 0.723 |
| 31359 | 1 | 10.1 | +8.4 | 0.785 | 1.025 | 0.755 |
| 31359 | 1 | 9.6 | +8.1 | - | 1.456 | 0.805 |
| 31747 | 1 | 10.4 | +8.7 | 1.193 | 0.942 | 0.812 |
| 31747 | 1 | 10.6 | +8.9 | 0.915 | 2.537 | 0.587 |
| 31747 | 1 | 10.5 | +8.8 | 2.049 | -4.857 | 0.708 |
| 31747 | 1 | 10.5 | +8.7 | 0.729 | -3.166 | 0.614 |
| 31873 | 1 | 10.5 | +8.8 | 0.889 | 0.997 | 0.669 |
| 31873 | 1 | 10.5 | +8.8 | 0.869 | 0.906 | 0.700 |
| 31873 | 1 | 10.7 | +9.0 | 0.823 | 0.946 | 0.649 |
| 31873 | 1 | 10.7 | +8.9 | 0.820 | 0.950 | 0.659 |
| 31873 | 1 | 10.7 | +8.9 | 0.732 | 0.933 | 0.618 |
| 31873 | 1 | 10.6 | +8.8 | - | 0.965 | 0.621 |
| 31902 | 3 | 26.7 | +6.7 | 0.956 | 1.342 | 0.721 |
| 31902 | 1 | 10.8 | +9.1 | 1.306 | 0.868 | 0.625 |
| 31902 | 1 | 10.5 | +8.8 | 0.426 | 0.963 | 0.664 |
| 31902 | 3 | 26.7 | +6.7 | 1.059 | 1.033 | 0.631 |
| 31902 | 1 | 10.6 | +8.9 | 1.109 | 1.098 | 0.607 |
| 31902 | 1 | 10.5 | +8.8 | 0.999 | 0.849 | 0.635 |
| 31902 | 3 | 26.4 | +9.2 | 1.078 | 0.973 | 0.510 |
| 31921 | 1 | 10.4 | +8.7 | 0.180 | 0.958 | 0.661 |
| 31921 | 1 | 10.4 | +8.7 | 0.761 | 0.968 | 0.647 |
| 31921 | 1 | 10.0 | +8.3 | 0.965 | 0.958 | 0.982 |
| 31923 | 1 | 10.6 | +8.7 | 0.459 | 0.966 | 0.658 |
| 31923 | 1 | 9.9 | +8.2 | - | 1.027 | 0.881 |
| 32027 | 1 | 10.9 | +9.2 | 0.916 | 0.483 | 0.632 |
| 32027 | 1 | 10.8 | +9.1 | 0.600 | 0.709 | 0.638 |
| 32027 | 1 | 10.5 | +8.8 | 1.438 | 0.990 | 0.716 |
| 32092 | 1 | 10.6 | +8.9 | 0.961 | 1.018 | 0.658 |
| 32092 | 1 | 10.4 | +8.7 | 0.917 | 0.806 | 0.702 |
| 32092 | 1 | 10.5 | +8.8 | 0.931 | 0.916 | 0.607 |
| 32092 | 1 | 10.2 | +8.5 | - | 0.844 | 0.741 |
| 32097 | 2 | 27.2 | +12.4 | 0.728 | 0.598 | 0.290 |
| 32097 | 1 | 10.7 | +8.8 | 1.423 | 1.030 | 0.705 |
| 32097 | 1 | 10.7 | +9.0 | 0.984 | 1.218 | 0.623 |
| 32097 | 1 | 6.4 | +8.8 | 3.977 | 1.112 | 0.721 |
| 32097 | 1 | 10.3 | +8.6 | 0.837 | 1.571 | 0.742 |

## Interpretations made in running the registration

1. Kernel of the heating route: the registration names the event-average T_i deficit profile of part S; `uc_lag_quiet.py` hard-codes the exploratory profile of all events (one to three absent frames). The verdict of L1 uses the profile of the registered set (one absent frame, all discharges, event medians, rows 1-10, normalized to sum 1); the frozen kernel is reported beside it (`L1_holds_with_frozen_kernel` in the json).
2. Positive control windows: the registration does not fix the window; it was fixed in the runner before any result was read: one-frame blips, 15 rows before the absent frame and 29 after, all 10 ms contiguous, the absent position as NaN, absent frame opening a block (shift 0). Shifts 1 and 2 are reported.
3. L3 'above zero' and 'above 0.5' are read as the lower ends of the intervals (as for the other statements).
4. S2: the upper end of the interval of b x first-row BES deficit is compared with one third of the point first-row T_i deficit of the same set (as in the expectation '0.046 against 0.061'); the ratio with its interval is reported.
5. S3: point event-median control profiles, rows 1 to 10; controls are not split by number of absent frames.
6. Halves for part S use the file rank of each event's discharge; the S2 eV bound uses b from all discharges.

Run time 119 s.
