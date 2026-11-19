# A0 poster for the PPCF manuscript: what each part condenses

Poster: `docs/presentation/KSTAR_CES_poster_PPCF.pptx` (.png), built by `docs/presentation/build_poster_ppcf.py`;
figures by `docs/presentation/make_figures_poster_ppcf.py` into `docs/presentation/figures/poster_ppcf/`.
Line numbers refer to `docs/paper/ppcf/main_ppcf.tex` as of 2026-11-19 (830 lines) unless another file is
named; `b39_generated.tex` is `docs/paper/ppcf/b39_generated.tex` (the number macros and generated tables).
Every table value was checked as a "point [low, high]" pair against `b39_generated.tex`, `main_ppcf.tex` and
`docs/paper/b65_inclusive.md`; all 33 pairs are found verbatim.

## Title band

| Poster element | Source |
|---|---|
| Title | the manuscript title, lines 59–60 |
| Subtitle (recurrent network with a boosted second stage, five-parameter filter, references that read the CES history alone) | abstract 76–86; Table `tab:refs` 361–382 |
| Author line, data line | the brief (author: Seungsang Lee, Department of Nuclear Engineering, Seoul National University; data: KSTAR team) |

## 1 Motivation

| Poster element | Source |
|---|---|
| Problem: CES reports the C VI temperature per 10 ms frame [1]; chord selected as the pedestal top; weak signal; missing more often at low beam emission, not completely at random | abstract 72–74; introduction 102–110 |
| Need: estimate formed when the frame is taken, without the next value; hold is the simplest reference | 138–140 |
| Available: BES and ECEI view the same location without gaps; neither measures T_i | 110–112 |
| Question box | abstract 74–75 |

## 2 Diagnostics and data

| Poster element | Source |
|---|---|
| Data: 641 files, 2022 campaign, 10 ms grid, 639 discharges enter the splits; 9 BES, 4 ECEI, 2 Mirnov, one chord at R = 2.198 m | 186–190, 108; 245,925 rows: `docs/paper/ppcf/supp_ppcf.tex` 120 and `PROJECT_KNOWLEDGE.md` 71 |
| Fig. 1 (geometry) and caption | figure `fig_geometry` and caption 197–209; drawing re-laid out from `fig_paper.fig_geometry` |
| What a row holds: single raw sample at the stamp; 20–40 ms step = beam blip, frames without beam left out | 212–219 |
| Fig. 2 (a row) and caption (frame starts within about 2 ms of the stamp; causal estimate uses fast samples up to t, CES values before t) | figure `fig_rows` and caption 227–236; 221–224 |
| Target: reported C VI temperature, main-ion equality not tested; above 3 keV taken as failed fits [5] and treated as missing; held values removed | 239–251, 224–225 |
| Fig. 3 (example traces) and caption | figure `fig_trace_causal` and caption 257–268 (placed in the data section, as in the manuscript, 254–255) |

## 3 Estimators and scoring

| Poster element | Source |
|---|---|
| Fig. 4 (a) two-stage estimator, (b) five-parameter filter | figure `fig_schematic` and caption 327–338 |
| Fig. 4 caption: network reads last value, age, observed flag [4]; second stage is a boosted regression [3] fitted only on out-of-fold errors of non-test discharges; filter: common mode, radial difference, robust weight; KF0, KFX | 166–168, 283–287, 298–313, 341–349 |
| Table 1 (references; causal definition; history-only control defined once with the brief's wording; two-stage control) | 277–281, 287–290, 293–296, 320–322; Table `tab:refs` 361–382 |
| Skill note (small) | 390–392 |
| Scoring and pooled cross-validation; registration | 386–387, 392–397; 418–421 |

## 4 Main result

| Poster element | Source |
|---|---|
| Table 2, skill rows: network over hold, causal GP, offline GP, history-only control (r2–r9); two-stage estimator over hold, causal GP, offline GP, two-stage control (r2–r4) | Table `tab:main`(a) 459–486 = `\TabMain` in `b39_generated.tex` 216–246 (MSE column) |
| Table 2, error rows on r2–r4: RMSE, RMSE minus causal GP, median abs. error ÷ hold, within 20 % minus causal GP | `\TabMainC` (`b39_generated.tex` 261–276) and `\TabTwoPairsB` (277–304; RMSE differences) |
| Table 2 caption (3 keV cut; eight vs three re-partitions; one initialisation per network) | 461–480, 454–455 |
| Bullet "Network" | 432–437 |
| Bullet "Two-stage estimator" (six registered statements met; over the offline GP above 0 where the network alone is below 0) | 440–453 |

## 5 What carries the information

| Poster element | Source |
|---|---|
| Table 3 (BES one frame old at test time; retrained L, B, E, N; network over B; KFX over KF0) | Table `tab:carry` 538–553 = `\TabCarry` in `b39_generated.tex` 419–449 |
| Bullet "Present frame, BES" | 555–561 |
| Bullet "Beam or plasma" (BES is light of the heating beam; the filter's increment of the same size; no beam power, so a beam change is not excluded) | 530, 502–507, 562–569 |

## 6 Where the gain is; fit or plasma

| Poster element | Source |
|---|---|
| Fig. 5 (blip event means, panel (a) of `fig_blips`) and caption | figure `fig_blips` and caption 644–653; 634–637 (T_i deficit follows ECEI, not BES, which recovers within two rows) |
| Table 4, increment by row class and by temperature third, without the 64 discharges | Table `tab:where` 593–610 = `\TabWhere` in `b39_generated.tex` 467–504 |
| Table 4, ρ1 | 692–702; `\RhoNet` in `b39_generated.tex` 150 (`\TabPersist` 383–392) |
| Table 4, floor s and network MSE over the floor | 407–416, 509–512; `\FloorRms`, `\FloorNet` in `b39_generated.tex` 139–140 |
| Bullet "Where" | 612–617, 640–642 |
| Bullet "Fit or plasma" (next frame fitted independently; most of the correction lasts; the floor accounts for most of the network's error) | 692–702, 509–512 |

## 7 Conclusion

| Poster element | Source |
|---|---|
| Takeaway box | 798–800 |
| ✓ Two estimators agree; present frame | 800–802 |
| ✓ Two-stage estimator (intervals for the increment, the offline GP, the median ratio to hold) | 805–809 |
| ✓ The correction lasts into the next, independently fitted frame | 803–804 |
| Caveats for the network alone | abstract 88–91 (also 802–805) |

## 8 Limitations and outlook

| Poster element | Source |
|---|---|
| 1 Cut population, Table 5 (values above 3 keV kept: network over causal GP +0.109, over control −0.002) | `docs/paper/b65_inclusive.md`, statements I1 and I2 (newer than the manuscript, allowed by the brief); main 177–179, 772–774 |
| 1 → per-frame fit uncertainties | 811–813 |
| 2 One chord, one campaign, one device; C VI impurity temperature; no beam power → beam power, equilibrium mapping, further chords | 790, 768–771, 775–776; 811–813 |
| 3 No untouched hold-out → temporal confirmation on the last 97 discharges (running, no result) | 785–789; the brief (running) |
| 4 Real time is a prospect; use needs present-frame BES and a CES value at most two frames late → a CES analysis that needs no beam-off frame [2] | 318–320, 91–92, 809–811; 733–737 |

## Footer

| Poster element | Source |
|---|---|
| References [1]–[5] | `docs/paper/ppcf/refs.bib`: Ko2010ChargeExchange, Lee2026NeuralNetworkbased, Friedman2001GreedyFunction, Che2018RecurrentNeural, Kim2024KineticProfile |
| Acknowledgement | the brief; main 823–825 |
