# Paper final text pass — change log (2026-11-09)

Files touched: `docs/paper/main.tex`, `docs/paper/main_ko.tex`, `docs/paper/refs.bib` (plus the rebuilt
`main.pdf` / `main_ko.pdf` and their build logs). No figure includes added, nothing committed.

Sources used: design document §0 (claims / forks / prior-art matrix / confirmed main theme),
`docs/lit_2026-11-09/lit_problem_models.md`, `lit_statistics.md`, `lit_interpretation.md`,
THESIS_RESULTS.md §8bz, §8bz-2, §8bz-3, §8bz-4, `docs/paper/paper_numbers.json`,
`docs/paper/wording_pass_2026-11-09.md` (rules only).

## TASK A — main theme: title, abstract, contributions, conclusion

### Title
- en `\title`: "Causal Nowcasting of 10 ms Charge-Exchange Spectroscopy Gaps in KSTAR: an LSTM with Lower
  Error than Acausal Interpolation and the 20-Parameter Physical Filter It Computes"
  (`\hypersetup{pdftitle=...}` changed to match).
- ko `\title`: "10 ms 하전교환분광 결측의 인과 나우캐스팅: 미래 보간보다 낮은 오차의 LSTM과 그것이 계산하는
  20모수 물리 필터" (owner's wording verbatim; `pdftitle` and `pdfsubject` changed to match).

### Abstract (both)
Rewritten in the confirmed order: problem (missing rows, causal nowcast vs acausal interpolation) →
T_i headline (+0.17…+0.26 cut / +0.23…+0.32 incl, 4/4 both pops; causal GP +0.08…+0.17, 8/8; offline GP
−0.05…+0.11, interval ∋ 0 on 7/8) + replicate ([+0.19,+0.25] / [+0.19,+0.31]) + both stress tests
(persistence +0.28…+0.44 8/8; campaign +0.17…+0.20 4/4) → T_i↔V_rot asymmetry as the converse evidence
(bit-identical; two-level [+0.07,+0.70] cut; median 0.94–1.00, interval ∋ 1 on 3/4; scoped to standard CES
10 ms, core to pedestal top, NBI torque unobserved) → the equation (four registered stages; K0 0.27–0.30,
K∞ 0.48–0.53, s_K 39–47 ms, θ 2.9–3.0σ, Σa_BES +48…+53, Σa_ECEI +44…+50 eV/σ; 72–78 % of PCHIP skill,
87–90 % of persistence skill on 4/4 cut, registered bar 0.70; +0.034…+0.042 over the increment-form
equation, 4/4; remaining 22–28 % = nonlinear event handling) → mean vs median estimators and the
calibrated Student-t head as secondary (0.91×; CRPS +0.229 / +0.363; coverage 0.49–0.51 / 0.91–0.92) →
limits. The ko abstract keeps the `fielddesc` structure (배경·질문·설계·프로토콜·주 결과·회전·식·중앙값과
분포·한계·공개) with the same content and section pointers.

Removed from the abstract (still in the body): the 21k-rung / width-sweep / context-vs-structure / family /
conformal-Winkler sentences (architecture claims were dropped per design §0; they remain as contribution 6).

### Contributions (sec. 1 "Contributions." / "기여.") — now 7 items, same order en/ko
1. T_i headline + replicate + the full-grid framing gate (+0.081, [+0.067,+0.096], 16/16) + "we found no
   precedent in the fusion literature for causal gap-filling of CES/CXRS T_i or V_rot from other diagnostics,
   nor for interpolation-relative skill as the evaluation bar".
2. **The equation (B.28 stages 1–4)** — replaces the old "preliminary readout" item 7 and now ranks right
   after the headline. Stage 1 73–82 % / 77–85 % (persistence), stage 2 50–66 % (PCHIP, below 0.70),
   stage 3 form (100 ms high-pass, flat to 2σ, off beyond 3σ, no state dependence), stage 4 72–78 % /
   87–90 %, +0.034…+0.042 vs stage 2 (4/4), K(s) parameters, θ, channel gains, remaining 22–28 %; "we found
   no precedent for extracting a closed-form equation from a trained fusion predictor and retraining it
   under the same protocol against a pre-registered retention bar".
3. T_i↔V_rot asymmetry as the converse evidence (former item 4, with the scope qualification added).
4. Mean and median from separate estimators + calibrated distribution (merges former items 8 and 9; adds
   the dist-head numbers already in sec:dist).
5. Two stress tests (former item 5, unchanged numbers).
6. "The ceiling is the input set, not the estimator" (merges former items 1-tail, 6, 7, 10, 11: peak
   stratum, non-adjacent regime, 21,498-param rung, width sweep, families, context vs structure, conformal
   Winkler 32/32, CPU latency).
7. One line for the statistics methodology (former item 2), pointing to sec:eval, with "each piece is
   standard in another field; we found no precedent in fusion diagnostic ML …; the combination is reported
   as a combination, not as new".
"was first recorded" → "was initially recorded" (en item 3) to avoid the word "first".

### Conclusion (both)
Rewritten in five paragraphs: (1) headline + replicate + stress tests; (2) what the nowcaster computes
(stage-4 equation and parameters, 72–78 % / 87–90 %, +0.034…+0.042, remaining 22–28 %, where the skill
comes from); (3) the same equation explains the V_rot converse (bit-identical, [+0.07,+0.70], 1/4·2/4,
0.94–1.00 ∋ 1 on 3/4, NBI torque, √2 photon floor 8/8, scope); (4) two estimands (filter 0.91×, M1 +0.06…
+0.39 / +0.09…+0.27, 4/4; CRPS/coverage); (5) what is not claimed and what would overturn it (offline GP
bound, value cut, 54–68 % / one twentieth, marginal calibration, 45 ms interval −0.004 [−0.30,+0.25],
equation bands read in the cut population only, one device; size/family axes closed; levers = NBI torque,
raw Mirnov, per-exposure fit uncertainty; statistics as the condition of belief). All numbers are the ones
already in the previous conclusion / contributions / sec:model:backbone.

## TASK B — prior work and references

### Problem lineage (sec. 2 / 관련 연구), both languages
- "Profile fitting and gap-filling" paragraph: persistence-baseline precedent added — Abbate et al. 2025
  (NF 65:056014, 10.1088/1741-4326/adc283) + "we found no fusion precedent for Murphy's skill against
  persistence and against acausal interpolation as the evaluation bar, nor for the term *nowcasting* in its
  meteorological sense".
- "Cross-diagnostic reconstruction" paragraph: Jia et al. 2026 (NF 67:016012, 10.1088/1741-4326/aeb432)
  added next to Wang 2025 — "CNN–BiGRU–transformer … bidirectional, hence acausal, described by its authors
  as offline signal restoration"; Diag2Diag description made explicit in axis (ii): "memory-less multilayer
  perceptron mapping same-instant inputs to each Thomson time point, scored against the measured points
  with no interpolation or persistence baseline".
- "To our knowledge the conjunction … has not yet been addressed" → "A literature check (OpenAlex, arXiv,
  Crossref; 2026-11-09; full-text search incomplete) found no precedent … on DIII-D, JET, ASDEX Upgrade,
  EAST or KSTAR; … 'not found' is not 'does not exist'." The "extension, not absence" framing and the
  one-sentence limitation are kept.
- RTCAKENN (Shousha 2024) and NN-CES (Lee, FED 222:115518) were already cited; unchanged. NN-CES year:
  Crossref `issued` = 2026-01 (volume 222) → bib year 2026 kept; the text prints no year (numeric style),
  so en/ko are consistent.

### New paragraph "Methods lineage." / "방법의 계보." (end of sec. 2, both languages), one sentence each
pre-registration (Nosek & Lakens 2014; NeurIPS pre-registration workshop PMLR 148), Nadeau & Bengio 2003,
Bates–Hastie–Tibshirani 2024, hierarchical bootstrap (Saravanan–Berman–Sober 2020; Ren et al. 2010),
Murphy 1988, Gneiting & Raftery 2007, Sussillo & Barak 2013, Cranmer et al. 2020, LeCun–Denker–Solla 1990
(OBD), Allen & Bishop 1992, Roth–Özkan–Gustafsson 2013, Agamennoni–Nieto–Nebot 2012; closes with the
"no precedent found … not absence" statement and the search limits (full-text incomplete, publisher texts
unread).

### New `refs.bib` entries (15; DOIs re-verified via Crossref on 2026-11-09 unless noted)
| key | reference | DOI / id |
|---|---|---|
| Jia2026TimeSeries | Jia et al., Nucl. Fusion 67, 016012 (2026) | 10.1088/1741-4326/aeb432 |
| Abbate2025Combining | Abbate, Fable, Tardini, Fischer, Kolemen, Nucl. Fusion 65, 056014 (2025) | 10.1088/1741-4326/adc283 |
| Viezzer2020Dynamics | Viezzer et al., Plasma Phys. Control. Fusion 62, 024009 (Crossref issued 2020-01; lit report wrote "2019") | 10.1088/1361-6587/ab5b1d |
| Cavedon2017Pedestal | Cavedon et al., Plasma Phys. Control. Fusion 59, 105007 (2017) | 10.1088/1361-6587/aa7ad0 |
| Nosek2014RegisteredReports | Nosek & Lakens, Social Psychology 45(3), 137–141 (2014) | 10.1027/1864-9335/a000192 |
| Bertinetto2021Preregistration | Proc. NeurIPS 2020 Workshop on Pre-registration in ML, PMLR 148 (2021) | proceedings.mlr.press/v148 (no DOI) |
| Saravanan2020Hierarchical | Saravanan, Berman, Sober, NBDT 3(5) (2020) | bioRxiv 10.1101/819334 (journal page has no DOI; preprint DOI used, noted in entry) |
| Ren2010Nonparametric | Ren et al., J. Appl. Stat. 37(9), 1487–1498 (2010) | 10.1080/02664760903046102 |
| Gneiting2007StrictlyProper | Gneiting & Raftery, JASA 102(477), 359–378 (2007) | 10.1198/016214506000001437 |
| Sussillo2013OpeningBlackBox | Sussillo & Barak, Neural Comput. 25(3), 626–649 (2013) | 10.1162/neco_a_00409 |
| Cranmer2020DiscoveringSymbolic | Cranmer et al., NeurIPS 2020 | arXiv:2006.11287 (no DOI) |
| LeCun1990OptimalBrainDamage | LeCun, Denker, Solla, NIPS 2, 598–605 (1990) | none (proceedings) |
| Allen1992NeuralNetwork | Allen & Bishop, Plasma Phys. Control. Fusion 34(7), 1291–1302 (1992) | 10.1088/0741-3335/34/7/008 (lit report's OpenAlex DOI 10.1016/0741-3335(92)90121-L is an alias; Crossref resolves both to this paper; the IOP form is used) |
| Roth2013StudentT | Roth, Özkan, Gustafsson, ICASSP 2013, 5770–5774 | 10.1109/ICASSP.2013.6638770 |
| Agamennoni2012ApproximateInference | Agamennoni, Nieto, Nebot, IEEE TSP 60(10), 5024–5037 (2012) | 10.1109/TSP.2012.2208106 |
Nadeau2003Inference, Bates2024CrossValidation, murphy1988, Jalalvand2025MultimodalSuperresolution,
Shousha2023MachineLearningbased, Wang2025TimeSeries, Lee2026NeuralNetworkbased already existed and are
reused. All 15 new keys resolve in both `.bbl` files (0 undefined citations).

## TASK C — qualifications

1. **Asymmetry scope** (en/ko): added once at the end of sec:asym ("holds for standard CES at 10 ms, from
   the core to the pedestal top, with the NBI torque unobserved; edge E_r and edge rotation are known to
   move within an ELM cycle at sub-millisecond resolution [Viezzer 2020; Cavedon 2017], which a 10 ms
   single-sightline exposure does not resolve") and once in Limitations (en: new item "Scope of the
   asymmetry."; ko: new row 12 of the 한계 table). Also stated in abstract, contribution 3 and conclusion.
2. **OU anchor** (en/ko, sec:physfilter "How the measurement model was found"): "an Ornstein–Uhlenbeck
   prior … our modelling assumption for rotation, for which we found no literature precedent in tokamak
   rotation". "for the first time" → "which no earlier anchor had done" (same fact, no "first").
3. **Two-level bootstrap** (en/ko, sec:eval "Replication beyond the four splits"): "a hierarchical
   bootstrap in the sense of Saravanan et al. and Ren et al. with the training repeat as the outer level,
   assembled from standard pieces and not claimed as new"; "a cell *beats*" → "a cell passes".
4. Extra limit added (en item "The equation is three quarters of the backbone, read in one population";
   ko row 13): 22–28 % nonlinear remainder; inclusive ratio 0.43–0.81 and BES-gain sign flip under the
   fit-failure spikes (§8bz-4 inclusive secondary). ko 한계 intro/caption "열한 항목" → "열세 항목"; the table
   was converted from a floating `table`+`tblr` to a page-breaking `longtblr` because 13 rows no longer fit
   one page (row 11 overflowed into the footer).

## TASK D — Korean front matter

Checked every hand-transcribed number in the 초록 fielddesc, 심사자용 한 쪽 요약, 논증 사슬 and the
반증/한계 tables against `paper_numbers.json` (headline per-seed skills and CIs, PR4 pass counts,
replicate NB and two-level intervals, dual-readout medians, MNAR reweighted skills and in-domain fractions,
campaign per-init skills, peak stratum, scaling widths/params, b14 median 0.894 / M1 +0.128 / +0.285 /
deficit 1, data ledger fractions 8.2 / 23.9 / 41.1 / 35.0 / 0.53 %):
- **No drift found** — every number already matched (8.2 %, 23.9 %, 41.1 %, 65.0 %/35.0 %, 0.53 %,
  +0.17…+0.26 / +0.23…+0.32, +0.08…+0.17, −0.05…+0.11 (7/8), [+0.19,+0.25] / [+0.19,+0.31],
  [+0.07,+0.70], 1/4·2/4, 0.94–1.00, 0.91×, +0.28…+0.44, +0.14…+0.28 (2/4·4/4), +0.17…+0.20,
  +0.11…+0.16, 2/4·0/4, +0.45…+0.72, 32 cells, 21k / 34k→879k / ±0.01, −0.09…+0.03, 0.02 band,
  8 % Huber (B.14 original M1 +0.081), 96 / 60–66 shots, 70–83 %, 153.8 vs 173.6, 54–68 % / 4–6 %).
- **Added** the B.28 stage-4 summary where the equation is now summarised: 초록 item [식]; 심사자용 요약
  "답 셋" → "답 넷" with a new 셋 (K0 0.27–0.30, K∞ 0.48–0.53, s_K 39–47 ms, θ 2.9–3.0σ, Σa_BES +48…+53,
  Σa_ECEI +44…+50 eV/σ, 72–78 % / 87–90 %, remaining 22–28 %); 논증 사슬 figure: new bold node
  "백본이 계산하는 것" (§4.4) between 결과 and 이 사슬이 끊길 수 있는 자리, caption "두 마디" → "세 마디".
- The 반증 table (six pre-registered falsification points, two tripped) is left as is: the B.28 bar was a
  registered retention rule of its own batch, not one of the six, and the one-pager text and node 7 count
  those six.

## Build / collector

- `main.tex`: pdflatex → bibtex → pdflatex ×2, exit 0; `main.log` has 0 lines starting with "!" and 0
  "undefined"; 40 pages.
- `main_ko.tex`: lualatex → bibtex → lualatex ×2 (+ one more pass after the longtblr change), exit 0;
  `main_ko.log` 0 "!" / 0 "undefined"; 45 pages.
- bibtex warnings "empty journal" for five pre-existing arXiv entries (Noh2026, Shukla2021, Rubanova2019,
  Ilse2018, Shi2015) — pre-existing, not introduced here.
- `PYTHONIOENCODING=utf-8 python ces_prediction/collect_paper_numbers.py` from the repo root: the
  "transcription:" line reports a match for sec:physfilter, sec:dist, sec:bulk B.14, tab:replicate and the
  sec:model B.28 paragraph. `paper_numbers.json` regenerated identically (no diff).

## Not verified / notes for the owner

- Viezzer et al.: the lit report and the task say "2019"; Crossref gives the issue date 2020-01 (PPCF 62,
  024009). The bib entry carries 2020; numeric citations print no year.
- Saravanan et al. 2020: the journal (NBDT) page lists no DOI; the entry uses the bioRxiv DOI with a note.
- "we found no precedent" statements repeat the lit reports' limits (OpenAlex full-text search
  incomplete, Semantic Scholar 429, IOP/AIP full texts unread); they are stated as "not found", never as
  absence, in the abstract-free body text (related work, contributions, conclusion).
- Forbidden-word scan after the pass: no new "first"/"최초" as a novelty claim; remaining occurrences of
  "beats"/"이긴다"/"동률" are in definitions, hypotheses and registered wordings (allowed by the 2026-11-09
  wording pass) and in text this pass did not touch.
