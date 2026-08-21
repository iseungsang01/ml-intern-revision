# Twelve shots to re-acquire at microsecond resolution

Not a controlled experiment — a **data-acquisition selection**. Two independent screens run
over all 641 shot CSVs, and the final list is the union of what they each say:

```bash
py ces_prediction/experiments/hires_shots/select_hires_shots.py       # data screen (641 shots)
py ces_prediction/experiments/hires_shots/literature_crosscheck.py    # literature screen
py ces_prediction/experiments/hires_shots/literature_crosscheck.py --report          # verified table only
py ces_prediction/experiments/hires_shots/literature_crosscheck.py --fulltext-index  # index screen (resumable)
```

Outputs next to the scripts: `shot_metrics.csv` (all 641), `shot_scored.csv` (ranked),
`literature_hits.json`, `fulltext_index_hits.json`, `FINAL_12.csv` (the request list),
`FINAL_10.csv` / `FINAL_10.png` (the ten that carry a learning role).

**Twelve shots, ten roles.** Ten shots carry the frozen protocol roles (3 test / 7 pool).
Two more — the same-session partners of #31921 and #31359 — are fetched as *companions*:
never trained on, never in the bootstrap, present so that two published paired comparisons
can be made at full bandwidth. See **The two companions** below.

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
| **31923** | *I-mode characteristics…* fig.2 · *On FIRE mode in KSTAR* fig.11–13 | L-mode → FIRE transition; **weakly coherent mode at ~50 kHz measured on `BES_0206` at r/a = 0.95** — that channel is a column in our CSVs. The second paper uses the same shot for the BES spectrogram, the 30–70 kHz two-channel coherence profile, and the poloidal wave number |
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

**Coverage, measured rather than asserted — and "KSTAR" is not a unique string.** The raw
arXiv + OpenAlex union double-counts every paper that exists as both a preprint and a journal
article; deduplicating by normalised title turns ~600 into **387**, and full text arrives for
**76** of those (143 carried a PDF link, the direct download won some, an Unpaywall pass
recovered 10 more from repository copies OpenAlex had missed, and arXiv-by-title recovered
exactly 1).

But 387 is not the population that matters. **30 of the 76 full texts are not about fusion at
all** — concrete compressive strength, drug toxicity, polymer science, Turkish e-commerce
reviews, top-k planning. They are there because `KSTAR` also names the Weka **K\*** instance-based
classifier and the **K\*** search algorithm, and because a Nature Communications paper about
kinase activity is literally titled *KSTAR*. That contamination is where the hand-maintained
false positives came from: `#30907` is a page range in a biochemistry reference. The scan now
filters on the **body** text (title is not enough — the kinase paper says KSTAR in its title),
which drops `#30907` and `#32017` automatically; only `#31589`, a DOI fragment in a genuine
fusion paper's reference list, still needs listing by hand.

The single arXiv-by-title recovery makes the same point: it found the correct preprint for its
OpenAlex record, and that record was *Ensemble Classifier for Eye State Classification using
EEG Signals*, whose abstract mentions the KStar classifier. The title matcher was right; the
corpus was wrong.

So the number to quote is **46 fusion full texts of ~248 relevant papers (19 %)**, not 76/387,
and the honest statement is not "these shots are not in the literature" but **"these shots are
not in the 46 fusion papers we can read in full."**

### Two routes added 2026-08-18, because "we could not download it" is not a finding

The PDF sweep is bounded by what this machine can fetch, and that bound was doing more work
than the conclusion could bear. Probing 60 of the unread relevant papers showed **38 are open
access**: the obstacle is a bot wall, not a paywall. Two routes get behind it.

**1. Supplementary material (`springer_fulltext`).** Measured, not assumed: nature.com HTML
and `media.springernature.com` SI files both serve this script; IOP answers with its Radware
challenge and AIP with 403. So the crawler covers `10.1038/` DOIs and says so. It reproduces
by machine what had been entered by hand — #31873 in the 43-page supplement of
`s41467-024-48415-w` ("Equilibrium parameters of #31873 with integrated ML-3D optimization"),
and **#31184, #31185, #31189 in the 27-page supplement of `s41467-024-45454-1`**, confirming
both that the SI gap was real and that none of those three is in our 641. It adds no shot we
hold. A dedupe bug had to be fixed first: merging an arXiv preprint with its journal version
kept the arXiv DOI, so the Nature ELM paper — one of this screen's most important hits — was
invisible to any `10.1038/` test.

**2. The full-text index itself (`--fulltext-index`).** OpenAlex indexes the body text of far
more papers than we can download, and it can be asked about a shot number directly. Validated
on four shots already known to be published, it returns exactly the right papers — including
`10.1088/1741-4326/adacfc`, an IOP article this script has never once been able to fetch, which
independently corroborates the hand-read #31921/#31923 claims. Two guards keep it honest: hits
older than 2022 are dropped (a paper predating the campaign cannot cite its discharges), and
`control_n` five-digit numbers from outside the campaign range are queried the same way, so the
screen reports its own coincidence rate instead of assuming it is zero.

This pass is **incomplete**: OpenAlex meters the API and 641 shots exceeds one day's free
allowance (`429 ... Resets at midnight UTC`). The scan is resumable — rerun it and it picks up
where the budget stopped. A pilot covering #30801–#31165 produced one new candidate,
**#31097** (*Observation of edge kink-like modes induced by resonant magnetic perturbations in
KSTAR plasmas*, Phys. Plasmas **32** 012303 (2025), `10.1063/5.0237640`). Its context is
**unverified** — AIP returns 403, so the sentence around the number cannot be read, and this
screen's rule is that a five-digit number is not believed until it is. It changes nothing about
the list in any case: #31097 fails the data gate on `vt_clean_n = 1`.

**Two things this screen still cannot see**, each demonstrated rather than guessed:

* **Bot-blocked publishers.** IOP serves this script a Radware challenge, so the two
  FIRE-mode papers had to be read through their article pages. Doing that is what revealed
  that **#31923 is used by both of them**, which the PDF sweep could never have found. IOP,
  AIP and Elsevier supplements remain unread, and their CAPTCHAs are not to be worked around.
* **The campaign is larger than our sample.** #31184/#31185/#31189 sit inside our shot-number
  range but we hold no CSV for them. Our 641 shots are a sample of the 2022 campaign
  (89 contiguous sessions), not the campaign itself, so "published shots in the campaign" and
  "published shots we could fetch" are different sets.

---

### The batch scan, 2026-08-20: one control, 388 of 641, and three ways to be wrong

Asking OpenAlex about one shot per request puts 641 shots six days away, because the API is
now metered and the free allowance is roughly a hundred requests a day. Batching fixes the
arithmetic — `fulltext.search:KSTAR AND (a OR b OR ...)` rules out 32 shots at once — but it
also introduces a failure mode that looks exactly like a result: if the boolean operators are
not honoured, every batch returns empty and 641 shots read as *absent from the literature*
when they were never asked about.

So the syntax was measured before it was trusted (`--control`, two requests). A batch holding
the four hand-verified shots #31921 / #31359 / #31873 / #32027 among 28 out-of-range decoys
came back with exactly their papers — the FIRE-mode IOP article, the ELM-suppression Nature
Communications paper and its preprint, the error-field paper and its preprint, and PanoMHD —
and a batch of 32 decoys and nothing else came back empty. Positives found, decoys not: a
batch miss can now be read as real absence. **That control is a precondition, not a
formality**, and it is wired to call the same `batch_ask` the sweep uses so it cannot drift
away from what is actually run.

The sweep then answered **388 of 641** and stopped in group 10 when the daily budget went.
It stopped in one request rather than several: a 429 carrying `retry-after` in the hours is
the metered quota, not a burst limit, and no backoff outlasts midnight UTC while every retry
is itself billable. The scan is resumable, so stopping costs the current group, not the pass.

**Eleven hits, and three of them are artefacts.** Screening them turned up three distinct
ways for a five-digit number to be in a fusion paper without being one of our discharges.

* **Another machine's discharge.** #31213 is an **ASDEX Upgrade** shot. The gyrokinetic
  benchmark review says so in as many words: *"the AUG shot considered is the #31213, at
  t = 0.84 s"* (`10.1007/s41614-025-00199-2`, sect. 2 — the NLED-AUG test case). Discharge
  numbers are not unique across machines, and AUG, DIII-D, EAST, JET and JT-60 all number in
  five digits over ranges that overlap the 2022 KSTAR campaign. A number inside a *generic*
  tokamak paper is therefore ambiguous by construction. #31213 had been carried since
  2026-08-19 as one of two new A-tier finds; it is now struck.
* **A cited DOI.** #31589 is the article number of `10.1038/s41467-022-31589-6`, read
  directly out of that same review's reference list. The hand-maintained false-positive table
  had said this since the PDF sweep; the index path simply never consulted it, which is the
  actual defect and is now fixed.
* **A paper that is not about plasma at all.** #31886 and #31913 came from *"Fusexins,
  HAP2/GCS1 and Evolution of Gamete Fusion"*, a cell-biology article that passed the topic
  filter on the word `fusion`. Bare `fusion` is no longer a plasma-physics token; it has to
  appear as `fusion energy`, `fusion performance`, `fusion-born`, `fusion alpha` and so on.

Every index hit now carries a verdict — `confirmed` (hand-verified against the paper),
`kstar` (the citing paper is about KSTAR, so a number in range is attributable), `unverified`
(a fusion paper that never names KSTAR) and `rejected` — so `unverified` can never again be
read as a citation. #31365 is the one `unverified` hit: *Doublet splitting of fusion alpha
particle driven ion cyclotron emission* is D-T alpha physics, which KSTAR does not do, and
IOP will not serve the sentence around the number. It is not selectable on that basis, and at
`vt_clean_n = 65` it would not be selected anyway.

**What the ledger says after 388 shots.** Ten shots survive as usable literature: #31097,
#31276, #31357, #31359, #31747, #31873, #31888, #31921, #31923, #32027. That is the same
count as before the sweep, but not the same set — #31213 left and **#31747** arrived
(*Analysis of neoclassical tearing mode stabilization experiment by electron cyclotron
injection in KSTAR*, EPJ Web Conf. **313** 02005). The anti-correlation that Screen 1 first
measured is unchanged: of the ten, only **#31921 (296) and #31359 (246)** clear 200 valid
`V_rot` rows. #31747 is the near miss at 162, and its `vt_held_frac = 0.66` would disqualify
it regardless.

**The 253 shots left are #31937–#32751, and 55 of them carry ≥ 200 valid `V_rot` rows.**
Those 55 are the only shots that can still turn this into a single set rather than two tiers,
so the resumed scan takes them first (`--priority` orders by `vt_clean_n`, richest first).
Ordering changes nothing about which shots get asked — only which answers exist when the
budget runs out again. Bisection was also made cheaper: a paper the first half of a group
cannot account for proves the second half positive, so that half no longer has to be re-bought.
Replayed over six fake-literature cases it recovers every cited shot at 25–36 % fewer requests.

### A defect this screen exposes in the current test triple

The list's three test shots are #31921, #31873 and #31114, with **1**, 296 and 311 valid
`V_rot` rows respectively. #31873 is `vt_held_frac = 1.00` — its rotation channel is stuck,
not sparse, and microsecond re-acquisition cannot fix it, because what is being re-acquired is
BES / ECEI / MC, not the CES target.

So for `CES_TI` the gate has three test clusters, and for `CES_VT` it effectively has **two**.
`power_analysis.json` measured what that costs, and the cost is not lost power but false
confidence: `CES_VT | seqv2_vs_w2control` passes at **0.770 at k = 2 against 0.665 at k = 3**.
The pass rate goes *up* as the evidence gets thinner, because with two clusters half of all
resamples repeat one shot and the interval collapses. `folds.py` already documents this as the
reason test never drops below three — the triple simply was not checked against it per target.
Whatever comes out of the completed scan, the `CES_VT` arm of the test set needs a third shot
that actually has `V_rot`.

---

### Being published and being usable are anti-correlated, and the gate says so

The obvious follow-up to screen 1 is to swap the six shots that carry no paper for published
ones. Running that question through the gate answers it: of the **nine published shots we hold
a CSV for, exactly one passes** — #31921.

| shot | why the gate rejects it | independent V_rot labels |
|---|---|---|
| #31873, #31923 | `vt_clean_n >= 60` | **1** |
| #31097 (new candidate) | `vt_clean_n >= 60` | **1** |
| #31888 | `vt_clean_n`, and MC is a spike (trim 0.36, kurt 105) | 1 |
| #32027 | `vt_clean_n`, `ecei_ac1 > 0.2` | 8 |
| #31276 | `vt_clean_n`, and MC is a spike (trim 0.32, kurt 363) | 10 |
| #31357 | `vt_clean_n >= 60` | 44 |
| #31359 | `ti_clean_frac >= 0.85` (0.79) | 246 |
| #31921 | — passes everything, rank 2/121 | 296 |

Seven of the eight rejections are the same rejection: **the discharges that get written up
have essentially no fitted CES rotation.** That is not a coincidence to be explained away here,
but it is a constraint with teeth — selecting for publications selects against V_rot
supervision, which is the target this project is already weakest on.

So no substitution is made. Every published shot we can reach is either already in the list
(#31921, #31873, #31359, #32027 with roles; #31923, #31357 as companions) or was rejected on a
measured artifact (#31276, #31888) and would be a downgrade. The six unpublished shots stay:
they are rank 1 (#32097), 9 (#31074), 13 (#31937), 22 (#31604), 37 (#31114) and the
highest two-coil coherence in the dataset (#31745). **The 3 test / 7 pool structure, and the
train 6 / val 1 / test 3 folds it produces, are unchanged** — `folds.py::_check` asserts both.

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

## Screen 3 — one shot per session

Adjacent shot numbers are repeat discharges from a single session: same plasma setup, same
diagnostic gain and offset, same wall state. `session_similarity.py` measures what that is
worth over all 641 shots (176k pairs):

| gap | pairs | summary distance | \|ΔT_i\| [eV] | **calibration distance** |
|---|---|---|---|---|
| 1–2 | 738 | 1.95 | **91.5** | **0.82** |
| 3–6 | 1,211 | 2.53 | 137.7 | 1.49 |
| 7–20 | 3,140 | 3.53 | 227.1 | 2.95 |
| 61–200 | 27,101 | 4.05 | 257.2 | 3.85 |
| all | 176,121 | 4.14 | 264.9 | 4.28 |

Shots two apart differ in mean T_i by a third of what random pairs do, and their per-channel
BES/ECEI calibration is five times closer (one-sided permutation p < 1e-4). Since the model
z-scores its inputs, that calibration channel is how a session leaks **even when the physics
differs** — the ERMP pair #31357/#31359 differ by 202 eV in mean T_i (the paper's whole point)
yet sit at the 2.2nd percentile of calibration distance.

So the list takes **at most one shot per session**. Two same-session pairs were removed:

* **#31923** dropped, **#31921** kept — the pair sat at the *0.0th percentile* of all 176k
  summary distances (|ΔT_i| = 21 eV). #31921 wins on 296 independent V_rot instead of 1 and
  on data rank (2/121). It does **not** win on publication count: a later read of the IOP
  article pages showed #31923 is used by *both* FIRE-mode papers too, so that axis is a tie —
  the earlier "two papers instead of one" was wrong. The decision stands on the other two
  axes, and #31923 is fetched anyway as a companion. Replaced by **#32027**, also published,
  also on the val side.
* **#31357** dropped, **#31359** kept — 246 vs 44 V_rot, MC 8.3 vs 5.4, MC↔BES coupling +0.32
  vs −0.08. Replaced by **#31745**, the highest two-coil envelope coherence of any candidate.

Gap is a proxy, not the criterion: the measured distance is. #31273 sits 86 shots away from
#31604 yet lands at the 3.6th calibration percentile, so it was passed over; #31745 is
128 away *and* at the 23rd percentile.

**Split membership is never reassigned.** Every shot keeps the role the frozen seed-42
manifest gives it, and both replacements were drawn from the same (val) side, so the list is
still 3 test / 4 val / 3 train.

---

## The list

`t_start`–`t_end` is the discharge's contiguous block — the window to request.

| # | shot | split (s42) | window [s] | T_i / V_rot | MC RMS (trim, kurt) | paper | why |
|---|---|---|---|---|---|---|---|
| 1 | **31921** | **test** (2 seeds) | 3.51–8.61 | 446 / 296 | 5.0 (0.82, 15) | [P1] fig.10 · [P2] fig.3, 7–9 | **published ×2 (FIRE mode, CES edge profiles + BES bispectral WCM)** *and* the best data shot we have — rank 2/121, MC↔turbulence coupling 0.575, the highest of all 641 |
| 2 | **31873** | **test** | 5.41–13.08 | 748 / — | 2.6 (0.85, 40) | [P3] fig.5 | **published, Nat. Commun. 2024** — automated ELM suppression; 7.7 s window covering the whole suppressed phase |
| 3 | **31114** | **test** | 4.00–9.08 | 506 / 311 | **8.0** (0.88, 11) | — | largest clean MC amplitude among test shots; model gains on **both** targets (+0.26 T_i, +0.16 V_rot vs PCHIP) |
| 4 | **31359** | val | 4.00–6.98 | 234 / 246 | 8.3 (0.73, 32) | [P4] fig.6 | **published, Nat. Commun. 2024** — no ERMP → ETB + ELMs; 246 V_rot values and the liveliest BES/ECEI of the published set |
| 5 | **32027** | val | 2.21–6.19 | 396 / 8 | 4.2 (0.86, 7.8) | [P5] fig.7 | **published (PanoMHD)** — clear L/H transition with **100–300 kHz cross-power / cross-phase**: exactly the band a μs fetch buys. Largest level step of the ten (0.98) |
| 6 | **32097** | val | 3.01–9.49 | 631 / 221 | **17.3** (0.87, 17) | — | strongest Mirnov shot overall (rank 1/121): two-coil coherence **0.93**, sustained mode 22 % |
| 7 | **31745** | val | 3.01–5.99 | 234 / 216 | 16.6 (0.79, 11) | — | **two-coil envelope coherence 0.96 — the highest of any candidate**, sustained mode 32 %, level step 0.97. Carries the coherent-mode role the dropped shots played |
| 8 | **31604** | train | 6.01–13.39 | 716 / 425 | **21.3** (**0.98**, **−0.9**) | — | the cleanest large MC in the dataset — near-Gaussian, spike-free, steady-state mode: ideal for spectral / mode-number analysis once phase is restored |
| 9 | **31074** | train | 0.50–7.99 | 736 / 446 | 4.3 (0.74, 36) | — | balanced all-round: coherence 0.74, 7.5 s, 446 V_rot |
| 10 | **31937** | train | 0.00–15.24 | **1479 / 722** | 1.7 (0.83, 56) | — | longest discharge by 2×, most labels of any shot; MC is quiet → the negative control that makes "does MC information help?" answerable |

Four published shots, six picked by score; smallest shot-number gap among these ten is
**16**. Three test shots, so a paper figure has a legitimate held-out case.

## The two companions (+2 = 12)

Screen 3 demoted two published shots for same-session overlap. With room for twelve they
come back — not as extra training data, but as the second half of two paired comparisons.

| # | shot | pairs with | window [s] | T_i / V_rot | MC RMS (trim, kurt) | paper | why |
|---|---|---|---|---|---|---|---|
| 11 | **31923** | 31921 (test) | 3.51–7.99 | 390 / 1 | 5.8 (0.83, 12) | [P1] fig.11–13 · [P2] fig.2 | **published** — L-mode → FIRE transition with a weakly coherent mode at **~50 kHz on `BES_0206`**, r/a = 0.95. Highest sustained-mode fraction of any shot in this document (34 %), MC↔BES coupling 0.42 |
| 12 | **31357** | 31359 (pool) | 3.00–6.98 | 396 / 44 | 5.4 (0.85, 8.8) | [P4] fig.6 | **published** — **with** n = 1 ERMP, so the H-mode transition is avoided. The paper's own controlled contrast against #31359: same target plasma, one knob changed. Two-coil coherence 0.87 |

**Why the redundancy argument does not survive the fetch.** Screen 3's distances are computed on the 100 Hz grid. #31921/#31923 sit at the 0.03rd percentile there — but what the papers actually use to tell those two discharges apart is a mode at ~50 kHz, three orders of magnitude above that grid's Nyquist frequency. *Redundant in the band we already have* is not *redundant in the band we are buying*. The ERMP pair is not even redundant at 100 Hz: #31357/#31359 sit at the **25.8th** percentile of summary distance and 202 eV apart in mean T_i, while their calibration distance is at the **2.2nd** — physically different, instrumentally identical, which is exactly what a controlled contrast wants.

**They stay out of the learning structure.** A companion in train while its partner is test would put near-identical calibration on both sides of the split — the leakage screen 3 exists to prevent. They stay out of the bootstrap too: two of k test clusters drawn from one session is the k = 2 artifact again. `folds.py` asserts both.

**What they buy, beyond the physics.** Screen 3 *inferred* session leakage from summary distances. Holding both members of a 0.03rd-percentile pair turns that inference into a measurement: train with the companion in, and the change in test skill on its partner is the leakage, in the units the gate reports.

### Papers referenced

Every record below was re-resolved 2026-08-18 — the four journal articles through the Crossref
API (title, journal, volume, article number and year as printed here) and the two preprints
through the arXiv API — so the links are checked, not transcribed.

* **[P1]** *On FIRE mode in KSTAR*, Nucl. Fusion **66** 026049 (2026) —
  <https://doi.org/10.1088/1741-4326/ae332f>
* **[P2]** *Experimental identification of I-mode characteristics at the edge of FIRE mode in
  KSTAR*, Nucl. Fusion **65** 036003 (2025) — <https://doi.org/10.1088/1741-4326/adacfc>
* **[P3]** *Highest fusion performance without harmful edge energy bursts in tokamak*,
  Nat. Commun. **15** 3990 (2024) — <https://doi.org/10.1038/s41467-024-48415-w>
  (preprint: <https://arxiv.org/abs/2405.05452>)
* **[P4]** *Tailoring tokamak error fields to control plasma instabilities and transport*,
  Nat. Commun. **15** 1275 (2024) — <https://doi.org/10.1038/s41467-024-45454-1>
* **[P5]** *PanoMHD: Multimodal Modelling of Plasma Dynamics towards Tokamak Control*,
  arXiv:2603.02672 (2026) — <https://arxiv.org/abs/2603.02672>
* **[P6]** *Enhancing Disruption Prediction through Bayesian Neural Network in KSTAR*,
  arXiv:2312.12979 (2023) — <https://arxiv.org/abs/2312.12979> (source of the rejected #31888)

Which shot each link belongs to:

| shot | papers |
|---|---|
| **31921** | [P1] fig.10 · [P2] fig.3, 7–9 |
| **31923** | [P1] fig.11–13 · [P2] fig.2 |
| **31873** | [P3] fig.5 (and its Supplementary Information) |
| **31359** | [P4] fig.6 (without ERMP) |
| **31357** | [P4] fig.6 (with ERMP) |
| **32027** | [P5] fig.7 |
| 31114, 31745, 32097, 31604, 31074, 31937 | none found — see the coverage statement above |

### Published shots left out

* **#31276** — MC RMS 12.7 collapses to 32 % of itself when the five largest samples are dropped
  (kurtosis 363). Spike, not mode.
* **#31888** — disruption example; same problem (trim ratio 0.36, kurtosis 105), and a
  disruption tail corrupts the CES labels.
* **#31923**, **#31357** — demoted for same-session overlap, not for quality, and now
  restored as **companions** (see above): fetched, but with no learning role.

### Why not simply the top ten by score

* **One shot per session** (screen 3), measured rather than assumed.
* **Role balance.** Eight shots carry strong magnetic activity; #31937 and #31873 are MC-quiet.
  Without a quiet arm there is no contrast against which "raw MC restores mode information the
  100 Hz grid destroyed" can be tested.
* **CES_VT is not the point of this fetch.** #31873 has no independent V_rot at all (a single
  held value for the whole shot) and would fail the data gate on that alone. The raw fetch
  upgrades the *inputs* (BES/ECEI/MC); the CES labels stay at 10 ms either way, and that shot
  earns its slot on published physics.

---

## How the ten role holders are used: 7-fold rotation, and why test stays at 3

`folds.py` fixes the structure; `power_analysis.py` is why it looks like this.

```
test      = 31921 · 31873 · 31114    frozen, never trained on, never used for selection
pool      = the other seven, rotated leave-one-shot-out
fold      = train 6 / val 1, seven folds, each pool shot is the val shot exactly once
companion = 31923 · 31357           fetched only; in no fold and in no bootstrap
```

Final model: run all seven folds, take the **median stopping epoch**, refit on all seven
pool shots at that epoch. The fold-to-fold spread of the val metric is the model-selection
stability estimate — which is the thing a single fixed val shot cannot give you.

### Why not 2 test shots

The gate resamples **shots**, so k test shots means k bootstrap clusters. Replaying the real
gate on the real 96-shot test set (seq_v2 vs the W = 2 control — the same shape as "does a
μs-MC feature help?"):

| k test shots | 2 | 3 | 6 | 10 | 12 |
|---|---|---|---|---|---|
| CES_TI, shot clusters | *40.5 %* | 28.7 % | 34.5 % | 34.0 % | 36.0 % |
| CES_TI, shot × 500 ms blocks | 36.2 % | 41.2 % | 43.2 % | **49.5 %** | — |
| CES_VT, shot clusters | *77.0 %* | 66.5 % | 84.2 % | **90.8 %** | 92.2 % |
| CES_VT, shot × 500 ms blocks | 60.2 % | 62.5 % | 71.2 % | 85.0 % | — |

The k = 2 numbers in italics are **not power**. With two clusters the resample space is four
draws, half of which repeat one shot, so the CI collapses and the pass rate *rises* going
from k = 3 to k = 2 while nothing has improved. Those extra passes are false positives.
k = 3 is the smallest size that is not sitting on that artifact.

Three further things the sweep says:

* **Effect size barely moves power.** Shrinking the arm difference to a quarter leaves
  CES_TI at 28.7 → 34.0 % and CES_VT at 90.8 → 89.8 %. `skill` is a ratio, so attenuating
  the effect attenuates the spread with it. A small μs-MC effect is not, by itself, a reason
  the experiment cannot conclude.
* **Consistency is what decides.** The fraction of shots where the arm actually wins is 0.66
  for CES_TI and 0.88 for CES_VT — and that is exactly the ordering of the power columns.
  Ten shots cannot fix an inconsistent effect; nothing can, at this scale.
* **CES_VT is the detectable target here.** That is convenient for a Mirnov-focused fetch:
  mode activity acts on rotation through NTV, so μs MC → V_rot is both the physically
  motivated direction and the statistically reachable one. CES_TI at ~30–50 % power should
  be treated as exploratory.

### Bootstrap policy

* **primary** — shot-clustered, identical to every batch in `THESIS_RESULTS.md` §8.
* **secondary** — shot × 500 ms blocks, pre-registered, always reported next to the primary.

Both are always shown together so a reader can check that the block assumption (which buys
CES_TI 34 % → 49.5 %) did not manufacture the conclusion.

---

## Caveats

* #31923 and #31357 are **companions**: fetch them, analyse them, but never train on them
  and never put them in the gate's test set. `folds.py::_check` enforces both.
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

---

# The frozen list (2026-08-21) — scan complete, literature first, test grows to four

Everything above describes the 2026-08-18/20 provisional twelve. This section replaces it.
The list below is FROZEN: it is the one `folds.py` asserts, `fetch_windows.csv` prices, and
`PREREGISTRATION_B6.md` §1.1 binds to.

## The scan finished, and its main product was a new way to be wrong

The batched OpenAlex sweep completed **641/641** (2026-08-21, one budget day, `--priority`
V_rot-richest first). It produced 15 new index hits — and hand-verification rejected every
one of them, by discovering a **fourth false-positive class: AIP article numbers.**
Physics of Plasmas article ids are `iissnn` (issue, section, sequence) — 032302, 032111,
032309 … — and the full-text index matches the bare five digits inside them, so any KSTAR
paper whose reference list cites a PoP issue-3 article "hits" the matching campaign shot
number. Proof by hand, six for six: the readable (arXiv/Springer/Nature) versions of the
indexed works contain the ZERO-PADDED id in their bibliographies and never the bare number
(e.g. "Phys. Plasmas 28(3) 032305 (2021)" inside arXiv:2201.07941 — indexed as a hit on
"#32305"). Rejected on direct or same-mechanism evidence: 32004, 32111, 32115, 32151
(malaria paper, the 31886 class), 32301, 32302, 32303, 32304, 32305, 32308, 32309, 32310
(15 total with the unverified ones; the `FALSE_POSITIVES` table in
`literature_crosscheck.py` carries per-shot evidence).

Two hits survived screening:

* **#32092** — *Spatiotemporal structure of edge harmonic oscillation…*, Nucl. Fusion 2026
  (10.1088/1741-4326/ae8679), a KSTAR EHO paper. `032092` is structurally impossible as an
  AIP id (section-sequence 20/92 does not occur), no other FP class applies, and the same
  kstar-grade standard already admitted #31747 and #31097. IOP's bot wall keeps the
  sentence around the number unread — an SNU institutional login would settle it.
* #32004 came from the same unreadable PoP paper as #31097 but `032004` IS a valid AIP id
  shape (issue 3, section 20, seq 04) and a DIII-D paper sits in its work list — rejected,
  with a note to reverse if the paper is ever read and names the shot.

So the usable literature ledger is **eleven**: the previous ten plus #32092.

## Two decisions (승상님, 2026-08-21)

1. **test = 4.** Literature-first keeps #31873 (V_rot held for the whole shot) in test, so
   a 3-shot test set has TWO effective `CES_VT` clusters — the measured false-positive
   regime (pass rate 0.665 at k = 3 → 0.770 at k = 2). Adding **#31902** (412 valid V_rot,
   the most of any gate-passing s42-test candidate) restores three effective clusters and
   lifts measured power to **0.750 (`CES_VT`) / 0.368 (`CES_TI`)**. Price: pool 7 → 6
   (data shot #31914, 542 V_rot, lost its slot), folds 7 → 6.
2. **#32092 is included** (pool), on rule consistency with #31747/#31097. It also happens
   to be a top-tier Mirnov shot: RMS 20.5 with trim ratio 0.94, kurtosis 1.8, two-coil
   coherence 0.92 — a spike-free strong mode.

The B.6 preregistration's §1.2 eligibility gate (≥ 3 effective V_rot test clusters)
**PASSES**: 31921 (296), 31114 (311), 31902 (412).

## The frozen twelve

| # | shot | role | src | window [s] | span | T_i / V_rot | why (one line) |
|---|---|---|---|---|---:|---|---|
| 1 | **31921** | test | LIT | 3.51–8.61 | 5.10 | 446 / 296 | FIRE mode ×2 papers; best gate shot (score_v2 rank 1/121) |
| 2 | **31873** | test | LIT | 5.41–13.08 | 7.67 | 748 / 1 | Nat. Commun. ELM suppression; V_rot held → the reason test has a 4th shot |
| 3 | **31114** | test | data | 4.00–9.08 | 5.08 | 506 / 311 | largest clean MC of the test picks; gains on both targets |
| 4 | **31902** | test | data | 1.75–7.72 | 5.97 | 703 / 412 | NEW — the k=2 fix: most V_rot of any s42-test candidate |
| 5 | **31097** | pool | LIT | 3.01–10.99 | 7.98 | 796 / 1 | Phys. Plasmas 2025 RMP edge-kink (kstar-grade) |
| 6 | **31359** | pool | LIT | 4.00–6.98 | 2.98 | 234 / 246 | Nat. Commun. ERMP OFF → ETB + ELMs |
| 7 | **31747** | pool | LIT | 3.01–8.49 | 5.48 | 481 / 162 | EPJ WoC NTM stabilisation by ECCD (kstar-grade) |
| 8 | **32027** | pool | LIT | 2.21–6.19 | 3.98 | 396 / 8 | PanoMHD L/H transition, 100–300 kHz band |
| 9 | **32092** | pool | LIT | 3.01–9.49 | 6.48 | 497 / 98 | NEW — NF 2026 EHO (kstar-grade, unread) + top-tier clean Mirnov |
| 10 | **32097** | pool | data | 3.01–9.49 | 6.48 | 631 / 221 | strongest gate-passing Mirnov (coherence 0.93); top score_v2 fill |
| 11 | 31923 | comp | LIT | 3.51–7.99 | 4.48 | 390 / 1 | FIRE companion of #31921 — 50 kHz WCM lives above the old grid |
| 12 | 31357 | comp | LIT | 3.00–6.98 | 3.98 | 396 / 44 | ERMP ON — the paper's own paired control of #31359 |

Request volume: **57.20 s** over the ten role shots + **8.46 s** companions = **65.66 s**
total (`fetch_windows.csv`). Windows open at the V_rot onset where a beam-phase proxy
exists, else at the labelled block. That rule trims #31902 to 1.75–7.72 s and leaves its
121 pre-onset `T_i` labels outside the window (`ti_outside` column) — the only shot where
the proxy costs labels; acceptable because the shot's job is the `V_rot` test arm.

What changed against the provisional twelve: **in** 31902 (test), 32092 (pool), 32097
stays; **out** 31745, 31604, 31074, 31937 (the score_v2 re-rank dropped the length axis
and put V_rot first), 31914/31368/31686 (interim fill candidates). Six of ten roles are
literature shots (five of them gate-failing — the rule is literature first, and the gate
fills only what is left). The one-shot-per-session rule is gone (2026-08-20); the in-pool
pair #32092/#32097 (gap 5) is deliberate and never straddles the test boundary.
