# Ten shots to re-acquire at microsecond resolution

Not a controlled experiment — a **data-acquisition selection** made on the existing 641 shot
CSVs, so that a request for raw high-rate KSTAR data is aimed at the discharges that would
actually pay it back. Regenerate everything with:

```bash
py ces_prediction/experiments/hires_shots/select_hires_shots.py
```

Outputs next to the script: `shot_metrics.csv` (all 641), `shot_scored.csv` (ranked),
`FINAL_10.csv`, `FINAL_10.png`.

## Why this needs measuring rather than guessing

The CSVs are a 100 Hz (10 ms) grid, and that grid is where the Mirnov problem lives: MC is a
dB/dt snapshot taken without an anti-aliasing filter, so its lag-1 autocorrelation is ~0.00
while BES/ECEI sit at ~+0.59 (`analyze_data_evidence.py`, claim B). Phase is destroyed — but
for a uniformly random sampling phase `E[x²] = A²/2` still holds, so a **rolling RMS of MC
recovers the mode-amplitude envelope** even from the aliased grid. That envelope is the
instrument used here: it tells us, before spending a single byte on a raw fetch, which
discharges carry strong, sustained, two-coil-coherent magnetic activity.

The same scan exposes a trap. Several shots with headline MC amplitudes are carried by a
handful of samples — #31884 has the largest RMS in the whole dataset (28.9), but drop its five
biggest samples and **93 % of that RMS disappears** (kurtosis 228). That is an electrical
spike, not a mode. Every candidate is therefore gated on `mc_rms_trim_ratio ≥ 0.60` and
`mc_kurt ≤ 80`, which removes 33 of the 154 quality-gated shots.

## Axes

| axis | what it measures |
|---|---|
| `label_value` | clean, independent CES supervision under the confirmed protocol's treatment (T_i fit-failure cut at 3 keV, held/forward-fill removal) plus discharge length |
| `diag_value` | do BES/ECEI move: dynamic range, sustained level steps (L→H-like), repeated fast crashes (ELM-like, 30–400 ms spacing), and the share of variance already aliased into sample-to-sample jitter — that share is exactly what μs sampling would resolve |
| `mc_value` | MC amplitude, spike-robustness, sustained (≥30 ms) hot fraction, two-coil envelope coherence, and coupling to the BES fluctuation level |

Split membership is read from the confirmed W = 2 protocol's frozen manifests
(`data/.b1_w2cut_split_s{42,1,7,123}`), so a shot proposed as a paper test case **is** a test
shot under the protocol the paper reports. `n_test_seeds` counts how many of the four split
seeds also place it on the test side.

## The list

`t_start`–`t_end` is the discharge's contiguous block — the window to request.

| # | shot | split (s42) | n_test_seeds | window [s] | T_i / V_rot labels | MC RMS (trim ratio, kurt) | role |
|---|---|---|---|---|---|---|---|
| 1 | **31921** | test | 2 | 3.51–8.61 | 446 / 296 | 5.0 (0.82, 15) | MC↔turbulence coupling **0.575 — highest of all 641**; sustained mode 21 %; ELM-like 50 ms; late transition with MC, BES, ECEI and T_i rising together |
| 2 | **31902** | test | 2 | 0.50–7.72 | 703 / 412 | 2.9 (0.85, 13) | 7.2 s of dense labels; best seq_v2 T_i skill of the candidates (+0.41 vs PCHIP, +0.49 vs persistence) |
| 3 | **31114** | test | 1 | 4.00–9.08 | 506 / 311 | **8.0** (0.88, 11) | largest clean MC amplitude among test shots; the model gains on **both** targets (T_i +0.26, V_rot +0.16 vs PCHIP) |
| 4 | **32097** | val | 0 | 3.01–9.49 | 631 / 221 | **17.3** (0.87, 17) | strongest Mirnov shot overall — two-coil coherence **0.93**, sustained mode 22 %, and it survives the trim test |
| 5 | **31368** | val | 0 | 0.50–8.89 | 734 / 504 | 5.1 (0.85, 23) | MC value and label richness together; coherence 0.80 over 8.4 s |
| 6 | **31273** | val | 1 | 3.01–10.99 | 795 / **672** | 4.8 (0.75, 54) | richest V_rot sampling in the pool; lively BES/ECEI with a quiet MC → contrast case |
| 7 | **31604** | train | 0 | 6.01–13.39 | 716 / 425 | **21.3** (**0.98**, **−0.9**) | the cleanest large MC in the dataset: near-Gaussian, essentially spike-free, steady-state mode — ideal for spectral/mode-number analysis once phase is restored |
| 8 | **31406** | train | 0 | 0.50–8.89 | 769 / 184 | 10.5 (0.78, 65) | **highest sustained-mode fraction (27 % of the discharge)**; 8.4 s |
| 9 | **31074** | train | 1 | 0.50–7.99 | 736 / 446 | 4.3 (0.74, 36) | balanced all-round: coherence 0.74, 7.5 s, 446 V_rot, ELM-like 320 ms |
| 10 | **31937** | train | 1 | 0.00–15.24 | **1479 / 722** | 1.7 (0.83, 56) | longest discharge by 2×, most labels of any shot; MC is quiet → the negative control that makes "does MC information help?" answerable |

Rank inside the 121 artifact-free, gate-passing shots: #32097 1st, #31921 2nd, #31273 5th,
#31368 8th, #31074 9th, #31937 13th, #31406 18th, #31604 22nd, #31902 25th, #31114 37th.

### Why these ten and not simply the top ten by score

* **Session diversity.** Adjacent shot numbers are repeat discharges from one session; the top
  of the raw ranking clusters into 32088–32099, 31902–31907, 31362–31369, 31406–31409. The
  list takes at most one shot per cluster, spreading across 310xx–320xx.
* **Role balance.** Eight shots carry strong magnetic activity; #31273 and #31937 are
  deliberately MC-quiet. Without a quiet arm there is no contrast against which "raw MC
  restores mode information the 100 Hz grid destroyed" can be tested.
* **Test-side stability.** #31921 and #31902 are test shots under two of the four split seeds,
  so a paper figure drawn on them survives a seed change.

## Caveats

* #31406, #31604, #31074 and #31937 are **training** shots of the confirmed protocol. Raw data
  from them is fine for method development and for physics figures, but a headline performance
  number must not be drawn on them.
* MC amplitudes are comparable across shots (same two coils, MC1T03 / MC1T16 in every file),
  but they are uncalibrated units — treat them as relative.
* `mc_sustained_frac` is defined against each shot's own median envelope, so a genuinely
  steady-state mode (#31604) scores 0 there. Read it together with `mc_rms`.
