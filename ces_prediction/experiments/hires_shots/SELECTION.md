# Ten shots to re-acquire at microsecond resolution

Not a controlled experiment — a **data-acquisition selection**. Two independent screens run
over all 641 shot CSVs, and the final list is the union of what they each say:

```bash
py ces_prediction/experiments/hires_shots/select_hires_shots.py       # data screen (641 shots)
py ces_prediction/experiments/hires_shots/literature_crosscheck.py    # literature screen
py ces_prediction/experiments/hires_shots/literature_crosscheck.py --report   # verified table only
```

Outputs next to the scripts: `shot_metrics.csv` (all 641), `shot_scored.csv` (ranked),
`literature_hits.json`, `FINAL_10.csv`, `FINAL_10.png`.

---

## Screen 1 — has anyone actually studied this discharge?

The dataset spans shots **30801–32751, which is the KSTAR 2022 campaign** — the campaign in
which FIRE mode was characterised. Sweeping arXiv + OpenAlex for every KSTAR paper since
mid-2022, downloading the open-access full text, and grepping for five-digit numbers in that
range turns up **seven of our shots in the published literature** (plus three false positives —
DOI fragments and a page range — rejected by reading the surrounding sentence):

| shot | paper | what the paper does with it |
|---|---|---|
| **31921** | *On FIRE mode in KSTAR*, Nucl. Fusion (10.1088/1741-4326/ae332f) fig.10 · *Experimental identification of I-mode characteristics at the edge of FIRE mode in KSTAR*, Nucl. Fusion (10.1088/1741-4326/adacfc) fig.3, 7–9 | **FIRE mode at 5.40 s vs H-mode at 8.05 s, compared on CES edge ion-temperature profiles** — the very diagnostic this repo predicts. Also BES bispectral analysis of WCM ↔ zonal-density phase coupling |
| **31923** | *I-mode characteristics…* fig.2 | L-mode → FIRE transition; **weakly coherent mode at ~50 kHz measured on `BES_0206` at r/a = 0.95** — that channel is a column in our CSVs |
| **31873** | *Highest fusion performance without harmful edge energy bursts in tokamak*, **Nat. Commun. 15 (2024)** (10.1038/s41467-024-48415-w) fig.5 | fully automated ELM suppression with ML-integrated RMP; Ip = 0.51 MA, q95 ≈ 5.1, optimizer triggered at 4.5 s |
| **31357** | *Tailoring tokamak error fields to control plasma instabilities and transport*, **Nat. Commun. 15 (2024)** (10.1038/s41467-024-45454-1) fig.6 | **with** n = 1 ERMP → the H-mode transition is avoided; BES bicoherence at ρ_N ≈ 0.92 |
| **31359** | same paper, fig.6 | **without** ERMP → density ETB forms at 5.5 s, ELMs appear, core T_i falls to ~5 keV. **The paired control of #31357: same target plasma, one knob changed** |
| 31276 | same paper, fig.3 | optimized path to edge stability; BES density-fluctuation bicoherence — *rejected on data quality, see below* |
| 32027 | *PanoMHD*, arXiv:2603.02672 fig.7 | clear L/H transition, 100–300 kHz cross-power / cross-phase spectrograms — *first alternate* |
| 31888 | *Bayesian NN disruption prediction*, arXiv:2312.12979 | disruption-prediction example — *rejected, see below* |

**The published time markers reproduce in our CSVs.** For #31921 the papers place FIRE mode at
5.40 s and H-mode at 8.05 s; our file covers 3.51–8.61 s and shows exactly that — flat at
T_i ≈ 0.4 keV with MC envelope ≈ 1.7 until 7.3 s, then T_i → 1.25 keV, MC envelope → 13.9, and
`BES_0206` fluctuation up 4×. For #31923 the WCM window (3–6 s) is the quiet stretch in our
file and the transition lands at 6.1 s. For #31873 our window (5.41–13.08 s) covers the whole
suppressed phase, including the late performance decay the paper describes.

**Limits of this screen, stated plainly.** Only open-access full text is searchable: of ~145 OA
links, ~58 PDFs actually downloaded — several publishers (IOP especially) bot-block direct
fetches. The two FIRE-mode papers were read through their article pages instead. So a shot
*absent* from this table is **not** proven absent from the literature; paywalled papers,
proceedings and theses are invisible here.

---

## Screen 2 — would the raw data be worth fetching?

The CSVs are a 100 Hz (10 ms) grid, and that grid is where the Mirnov problem lives: MC is a
dB/dt snapshot taken without an anti-aliasing filter, so its lag-1 autocorrelation is ~0.00
while BES/ECEI sit at ~+0.59 (`analyze_data_evidence.py`, claim B). Phase is destroyed — but
for a uniformly random sampling phase `E[x²] = A²/2` still holds, so a **rolling RMS of MC
recovers the mode-amplitude envelope** even from the aliased grid. That envelope tells us,
before spending a byte on a raw fetch, which discharges carry strong, sustained,
two-coil-coherent magnetic activity.

The same scan exposes a trap. Several shots with headline MC amplitudes are carried by a
handful of samples — #31884 has the largest RMS in the dataset (28.9), but drop its five
biggest samples and **93 % of that RMS disappears** (kurtosis 228). That is an electrical
spike, not a mode. Every candidate is gated on `mc_rms_trim_ratio ≥ 0.60` and `mc_kurt ≤ 80`,
which removes 33 of the 154 quality-gated shots, leaving 121.

| axis | what it measures |
|---|---|
| `label_value` | clean, independent CES supervision under the confirmed protocol's treatment (T_i fit-failure cut at 3 keV, held/forward-fill removal) plus discharge length |
| `diag_value` | do BES/ECEI move: dynamic range, sustained level steps (L→H-like), repeated fast crashes (ELM-like, 30–400 ms spacing), and the share of variance already aliased into sample-to-sample jitter — that share is what μs sampling would resolve |
| `mc_value` | MC amplitude, spike-robustness, sustained (≥30 ms) hot fraction, two-coil envelope coherence, coupling to the BES fluctuation level |

Split membership comes from the confirmed W = 2 protocol's frozen manifests
(`data/.b1_w2cut_split_s{42,1,7,123}`), so a shot proposed as a paper test case **is** a test
shot under the protocol the paper reports.

---

## The list

`t_start`–`t_end` is the discharge's contiguous block — the window to request.

| # | shot | split (s42) | window [s] | T_i / V_rot | MC RMS (trim, kurt) | why |
|---|---|---|---|---|---|---|
| 1 | **31921** | **test** (2 seeds) | 3.51–8.61 | 446 / 296 | 5.0 (0.82, 15) | **published ×2 (FIRE mode, CES edge profiles)** *and* the best data shot we have — rank 2/121, MC↔turbulence coupling 0.575, the highest of all 641 |
| 2 | **31873** | **test** | 5.41–13.08 | 748 / — | 2.6 (0.85, 40) | **published, Nat. Commun. 2024** — automated ELM suppression; 7.7 s window covering the whole suppressed phase |
| 3 | **31114** | **test** | 4.00–9.08 | 506 / 311 | **8.0** (0.88, 11) | largest clean MC amplitude among test shots; model gains on **both** targets (+0.26 T_i, +0.16 V_rot vs PCHIP) |
| 4 | **31923** | val (2 seeds) | 3.51–7.99 | 390 / — | 5.8 (0.83, 12) | **published — WCM ~50 kHz on `BES_0206`**; also the highest sustained-mode fraction of the ten (34 %) |
| 5 | **31359** | val | 4.00–6.98 | 234 / 246 | 8.3 (0.73, 32) | **published, Nat. Commun. 2024** — no ERMP → ETB + ELMs; 246 V_rot values and the liveliest BES/ECEI of the published set |
| 6 | **31357** | val | 3.00–6.98 | 396 / 44 | 5.4 (0.85, 8.8) | **published — the paired control of #31359** (ERMP on, transition avoided); two-coil coherence 0.87 |
| 7 | **32097** | val | 3.01–9.49 | 631 / 221 | **17.3** (0.87, 17) | strongest Mirnov shot overall (rank 1/121): two-coil coherence **0.93**, sustained mode 22 % |
| 8 | **31604** | train | 6.01–13.39 | 716 / 425 | **21.3** (**0.98**, **−0.9**) | the cleanest large MC in the dataset — near-Gaussian, spike-free, steady-state mode: ideal for spectral / mode-number analysis once phase is restored |
| 9 | **31074** | train | 0.50–7.99 | 736 / 446 | 4.3 (0.74, 36) | balanced all-round: coherence 0.74, 7.5 s, 446 V_rot |
| 10 | **31937** | train | 0.00–15.24 | **1479 / 722** | 1.7 (0.83, 56) | longest discharge by 2×, most labels of any shot; MC is quiet → the negative control that makes "does MC information help?" answerable |

Five published shots, five picked by score. Three test shots, so a paper figure has a legitimate
held-out case; #31921 and #31923 are test shots under two of the four split seeds.

### Published shots left out

* **#31276** — MC RMS 12.7 collapses to 32 % of itself when the five largest samples are dropped
  (kurtosis 363). Spike, not mode.
* **#31888** — disruption example; same problem (trim ratio 0.36, kurtosis 105), and a
  disruption tail corrupts the CES labels.
* **#32027** — kept as **first alternate**: the physics (L/H transition, 100–300 kHz
  cross-spectra) fits the μs case well, but it has only 8 independent V_rot values and no MC↔BES
  coupling.

### Why not simply the top ten by score

* **Session diversity.** Adjacent shot numbers are repeat discharges. The list takes at most one
  per session cluster and spreads across 310xx–320xx — except #31357/#31359, which are taken
  *deliberately as a pair* because the paper varies exactly one knob between them.
* **Role balance.** Eight shots carry strong magnetic activity; #31937 and #31873 are MC-quiet.
  Without a quiet arm there is no contrast against which "raw MC restores mode information the
  100 Hz grid destroyed" can be tested.
* **CES_VT is not the point of this fetch.** #31873 and #31923 have no independent V_rot at all
  (a single held value for the whole shot) and would fail the data gate on that alone. The raw
  fetch upgrades the *inputs* (BES/ECEI/MC); the CES labels stay at 10 ms either way, and those
  two shots earn their slot on published physics.

## Caveats

* #31604, #31074 and #31937 are **training** shots of the confirmed protocol. Raw data from
  them is fine for method development and physics figures, but a headline performance number
  must not be drawn on them.
* MC amplitudes are comparable across shots (same two coils, MC1T03 / MC1T16 in every file) but
  are uncalibrated — treat them as relative.
* `mc_sustained_frac` is measured against each shot's own median envelope, so a genuinely
  steady-state mode (#31604) scores 0 there. Read it together with `mc_rms`.
* The WCM is at ~50 kHz and the PanoMHD cross-spectra run to 300 kHz. Nothing at those
  frequencies survives a 100 Hz grid — which is the whole reason to fetch raw data, and the
  reason the fetch should cover **BES and ECEI at full rate, not only the Mirnov coils**.
