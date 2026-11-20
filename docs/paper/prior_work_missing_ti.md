# Prior work on estimating ion temperature where the charge-exchange value is missing

Checked 2026-11-20 by the session lead at the author's request ("is there really no work anywhere that fills in missing values?"). Answer: there is. What was not found is the form used here.

Search: OpenAlex title and abstract search (26 queries, 254 works), OpenAlex full-text phrase search (14 queries), the lists of works citing the four closest papers, and web search. The arXiv API could not be reached from this machine; arXiv items came through OpenAlex and web search. Conference contributions, theses and non-English work are covered poorly. "Not found" below means not found in this search.

Read level: F = full text read, A = abstract, P = abstract and part of the body, T = the table of signals in the PDF.

## Work that estimates T_i without the charge-exchange measurement of that time

| Work | What it does | Without CX? | Time history | Compared with hold or interpolation | Read |
|---|---|---|---|---|---|
| RTCAKENN: Shousha et al., Nucl. Fusion 64 (2024) 026006, doi 10.1088/1741-4326/ad142f (DIII-D) | real-time Thomson, CER, MSE and magnetics-only equilibrium inputs to seven fitted profiles of the offline kinetic reconstruction (CAKE), ion temperature and rotation among them; runs in the plasma control system | yes: inputs of Thomson or CER are dropped at random in training, so the model runs when one is absent | none; the paper calls it a model with no memory and uses the most recent valid data of each input | no; compared with the offline reconstruction and with the existing real-time fits | F |
| Attention network on EAST, arXiv 2610.00243 | global discharge signals and x-ray crystal spectra to T_i and T_e profiles | yes: stated to work from the global signals when the spectra are missing | none (signals averaged on a slow common time base, offline) | no | P |

Details of RTCAKENN that matter here (full text):
- No accuracy of the ion temperature without CER is given. The case is shown once, in a hardware test with the CER input withheld, and described as qualitatively accurate; a sensitivity study of missing data is named as future work.
- The target of the ion-temperature output is the fitted profile of the offline reconstruction, not a channel measurement. That reconstruction leaves out the edge measurements of the carbon temperature and extrapolates the edge ion temperature from the electron temperature by an assumed stiffness of their ratio.
- The paper states that most of the variability of its outputs is in the rotation and the carbon ion temperature.

## Filling or sharpening other channels

| Work | Channel | Read |
|---|---|---|
| Diag2Diag: Jalalvand et al., Nat. Commun. 16:8506 (2025), doi 10.1038/s41467-025-63492-1 (DIII-D) | Thomson T_e, n_e at a higher rate from other diagnostics; charge exchange is an input | A |
| Temporal super-resolution at COMPASS, PPCF 2026, doi 10.1088/1361-6587/ae72c6 | T_e, n_e profiles from soft x-ray and AXUV | A |
| Missing T_e on EAST (time series extrinsic regression), Nucl. Fusion 2025, doi 10.1088/1741-4326/addb5f | T_e | A |
| Missing Z_eff on EAST, Nucl. Fusion 2026, doi 10.1088/1741-4326/aeb432 | Z_eff, offline, bidirectional | A |

## Multi-diagnostic models that reconstruct masked signals, without an ion temperature

| Work | Finding | Read |
|---|---|---|
| FusionMAE (HL-3), arXiv 2509.12945 | 88 signals; the electron temperature is from ECE; no ion-temperature signal in the table of channels | T |
| TokaMark and TokaMind (MAST), arXiv 2602.10132 and KDD 2026 | 14 tasks; charge exchange is not among their inputs or outputs | P |

## Readers of the spectrum of the frame (need the spectrum; cannot be used where there is none)

NN-CES on KSTAR (Fusion Eng. Des. 222:115518, doi 10.1016/j.fusengdes.2025.115518); real-time CXRS network on HL-3 (Nucl. Fusion 2026, doi 10.1088/1741-4326/aeb037); GAN-augmented network on EAST (Plasma Sci. Technol. 2026, doi 10.1088/2058-6272/aeb5c5); real-time CER at DIII-D by a simplified fit (Rev. Sci. Instrum. 2026, doi 10.1063/5.0350505); x-ray crystal spectra on EAST (Nucl. Fusion 2024, doi 10.1088/1741-4326/ad73e8). All A.

## Not found

- A causal estimate in time of the charge-exchange ion temperature of one chord from its own earlier values and fast diagnostics, scored against hold, interpolation or a GP with intervals.
- An estimate of the charge-exchange value on KSTAR for frames or times without one.
- The error of such an estimate as a function of how old the newest charge-exchange value is.

## What may be written

The lineage is acknowledged first (RTCAKENN, the EAST network), then the extension: time history, the value and level of one chord instead of a fitted profile, references of hold and causal GP with intervals, the dependence on the age of the charge-exchange value, the pedestal top. No sentence says that this work is ahead of NN-CES: the two were never run on the same data and do different things. With a reader such as NN-CES supplying each frame, the condition "the previous frame is visible" becomes real; with the present analysis the schedule rule of B.75 applies.
