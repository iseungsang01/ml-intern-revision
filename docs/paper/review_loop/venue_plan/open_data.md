# Open data for a second device / public benchmark

Compiled 2026-11-15. Question: on which openly accessible data can the task of the paper be
replicated, namely causal estimation of a slow, sparse, noisy charge-exchange measurement
(ion temperature and toroidal rotation) at times without a measurement, from faster co-located
diagnostics plus the slow diagnostic's own past.

**How to read this file.** Every statement is tagged by how it was obtained:

- **[queried]** read directly from an anonymous public endpoint on 2026-11-15 (S3 listing, zarr
  metadata, parquet catalogue, array values, HTTP API).
- **[page]** stated on a web page that was fetched on 2026-11-15.
- **[search]** from a search-result summary only; the primary page was not opened.
- **[memory]** not checked in this session; verify before citing.

Total transfer in this session was about 235 MB (45 MB MAST catalogue parquet, about 37 MB MAST
per-shot zarr metadata, about 150 MB of LHD analysed-data text for one shot, the rest small arrays),
slightly above the 200 MB guideline. Nothing was written into the repository except this file; the
downloads and a scratch copy of `numcodecs` live in the session scratchpad.

---

## 1. Ranked recommendation

| Rank | Dataset | Why | Licence | Effort to first skill number |
|---|---|---|---|---|
| 1 | **FAIR-MAST level 2** (UKAEA MAST, campaigns M6 to M9) | Same measurement (carbon CXRS `t_i`, `v_i`), 5 ms cadence, 4,278 shots, with Thomson, D-alpha, Mirnov, soft X-ray, interferometer in the same store; BES (32 channels, 2 MHz) in about 29 % of those shots. Anonymous S3. | CC BY-SA 4.0 | 3 to 5 working days (converter 1 to 2 d, download 0.5 to 1 d, training and scoring 1 to 2 d) |
| 2 | **LHD analysed-data repository** (NIFS, heliotron) | Second confinement concept. `cxsmap7` gives Ti and flow velocity on a 20 ms frame grid; `ece_fast` (1 ms, about 40 to 70 radial channels overlapping the CXS radii), FIR interferometer, H-alpha, bolometer in the same HTTP API. Anonymous. | No standard licence; "Rights and Terms" ask for consent of instrument staff and an offer of co-authorship | 5 to 8 working days plus the consent step |
| 3 | **PhysioNet/CinC Challenge 2012 or 2019** (ICU labs from vitals) | Non-fusion structural analogue for an ML venue: sparse noisy laboratory value, denser bedside vitals, the label's own past. Small (7.6 MB to 42 MB), open. | ODC-By 1.0 (2012), CC BY 4.0 (2019) | 2 to 3 working days |

Not usable today as an open second device: MAST-U, JET, DIII-D (full archive), EAST, WEST, TCV,
W7-X, HL-2A/HL-3, Alcator C-Mod, KSTAR. Details in section 4.

Recommended order: do FAIR-MAST first. It is the only source found where the target is the same
physical measurement, the licence is a standard open one, and the fast inputs include BES. LHD is the
stronger "different device class" argument but carries a consent requirement that a public benchmark
claim has to disclose.

---

## 2. FAIR-MAST (rank 1)

### 2.1 Access

- Bucket `s3://mast`, endpoint `https://s3.echo.stfc.ac.uk`, anonymous. **[queried]** A plain HTTPS
  `GET https://s3.echo.stfc.ac.uk/mast/?prefix=...&delimiter=/` returns the S3 listing without
  credentials.
- Top-level prefixes: `level1/`, `level2/`, `fairmast-meta/`, `tokamark/`, `fairmast-docs/`, `dev/`,
  `fdc/`, `gkw/`, `mhd/`, `simulations/`, `test/`. **[queried]**
- Per-shot stores: `level1/shots/{shot}.zarr` (zarr v2, blosc-lz4, consolidated `.zmetadata` of 1 to
  2 MB) and `level2/shots/{shot}.zarr` (zarr v3, zstd, consolidated `zarr.json` of about 0.8 MB).
  Level 2 also exists as one netCDF file per shot under `level2/netcdf/{shot}.nc` (about 20 to 170 MB
  each). **[queried]**
- Catalogue as parquet: `fairmast-meta/l1_shots.parquet` (13.4 MB), `l1_sources.parquet` (14.6 MB),
  `l2_shots.parquet` (12.6 MB), `l2_sources.parquet` (4.1 MB), and one small parquet per shot under
  `fairmast-meta/l1_signals/` and `l2_signals/` (about 18 kB each). **[queried]**
- The REST API at `mastapp.site` did not complete a TLS handshake from this machine (curl and
  Python both failed). UKAEA's service page says the catalogue API is being refreshed and the old
  system is no longer provided **[search]**, so plan on S3 plus the parquet catalogue, not the API.
- Documentation: https://ukaea.github.io/fair-mast (states it is under construction). **[page]**
- Reference to cite: Jackson et al., "An Open Data Service for Supporting Research in Machine
  Learning on Tokamak Data", IEEE Trans. Plasma Sci. (2025) **[search]**; the docs also ask for a
  citation of the service. **[page]**

### 2.2 Licence

CC BY-SA 4.0. Stated on the documentation site **[page]** and embedded in the level-2 group
attributes: `license_name = "Creative Commons 4.0 BY-SA"`,
`license_url = https://creativecommons.org/licenses/by-sa/4.0/` **[queried]**. Share-alike means a
derived per-shot CSV set that we redistribute must carry the same licence.

### 2.3 What is public

- Facility MAST only; shot numbers 11695 to 30473; campaigns M5 (2,679 shots), M6 (3,848),
  M7 (5,168), M8 (2,595), M9 (1,679); 15,969 level-1 shots and 11,573 level-2 shots (11766 to
  30471). **[queried]** No MAST-U shot is in the bucket's shot range.
- Level 1 = the original MAST sources by three-letter code. Level 2 = harmonised, IMAS-named,
  unit-checked groups.

### 2.4 The slow target: charge exchange

Level 1, source `act` ("Time resolved Charge Exchange") **[queried]**:

- 7,970 shots (15020 to 30471): M6 1,372, M7 3,959, M8 1,751, M9 888, M5 none.
- Catalogue quality flag for `act`: "Not Checked" 4,860, "Bad" 3,110.
- Signals for shot 30147: `act/ss_temperature` (eV), `act/ss_temperature_error`,
  `act/ss_velocity` (m/s), `act/ss_velocity_error`, all `[time=130, majorradius=96]`; plus
  `ss_cx_emissivity`, `ss_chisq`, `ss_cx2bgratio`, raw and fitted spectra `[130, 96, 60]`, and
  per-beam summaries (`c_ss_*`, `c_pla_*`). MDS name `\TOP.ANALYSED.ACT.SS:TEMPERATURE`.
- Values read for shot 30147: 130 frames, t = -0.050 to 0.595 s, frame spacing 5.00 ms; R = 0.781 to
  1.430 m; 64 chords valid in at least half the frames; 86 frames with at least 10 valid chords;
  median Ti 761 eV; median relative Ti error 0.039; median |V| 50.5 km/s; median V error 1.16 km/s.

Level 2, group `charge_exchange` **[queried]**:

- 4,278 shots (17555 to 30471): M6 694, M7 1,743, M8 1,016, M9 617, campaign "Unknown" 208.
- Arrays: `t_i` (eV, IMAS `charge_exchange.channel[:].ion[:].t_i`, source `ACT_SS_TEMPERATURE`),
  `v_i` (m/s), `major_radius`, `time`. Shot 29891: `[major_radius=161, time=179]`, 5 ms spacing,
  t = -0.100 to 0.790 s, 64 chords valid in at least half the frames, 109 frames with at least 10
  valid chords. Level 2 does **not** carry the error arrays; take those from level 1 `act`.
- Not every level-2 shot has the group (shot 30420 has none).

Random sample of level-1 `act` shots, 19 read successfully across M6 to M9 **[queried]**:

- Frame spacing 5.0 ms in 18 shots, 10.0 ms in 1 (shot 22676, 65 frames).
- Frames with at least 10 valid chords per shot: range 1 to 109, median 66.
- Median relative Ti error per shot: 0.034 to 0.217 in 18 shots. Shot 28981 has median Ti
  15,778 eV and relative error 0.70, i.e. failed fits stored as values. This is the same artifact
  class as the > 3 keV fit failures in the KSTAR set, so the fit-failure rule of the paper has a
  direct counterpart here.

### 2.5 The fast inputs

Level 2, shot 29891, sampling rate computed from the stored time base **[queried]**:

| Group / array | Shape | Rate | Stored size |
|---|---|---|---|
| `spectrometer_visible/filter_spectrometer_bes_voltage` (BES, source `xbt/channel`) | 32 x 1,591,601 | 2 MHz | 87.5 MB |
| `spectrometer_visible/filter_spectrometer_dalpha_voltage` | 3 x 39,791 | 50 kHz | 0.13 MB |
| `thomson_scattering/t_e`, `n_e`, `p_e` | 120 x 166 | 240 Hz | 0.42 MB (group) |
| `interferometer/n_e_line` | 44,291 | 50 kHz | 0.20 MB |
| `magnetics/b_field_pol_probe_omv_voltage` (outer vertical Mirnov array) | 500 kHz time base | 500 kHz | 1.5 MB |
| `magnetics/b_field_tor_probe_cc_field` (centre-column toroidal Mirnov) | 3 x 442,901 | 500 kHz | 2.4 MB |
| `magnetics/b_field_tor_probe_omaha_voltage` (OMAHA high-frequency coils) | 4 x 1,771,601 | 2 MHz | 4.6 MB |
| `magnetics/b_field_tor_probe_saddle_field` | 12 x 44,291 | 50 kHz | 0.25 MB |
| `soft_x_rays/horizontal_cam_upper`, `_lower` (with line-of-sight R, Z) | 18 x 442,901 each | 500 kHz | 27.6 MB (group) |
| `summary/power_nbi`, `ip`, `line_average_n_e` | 887 | about 1 kHz | 0.03 MB |
| whole level-2 shot | | | 150 MB |

Other level-2 groups: `bolometer`, `equilibrium`, `gas_injection`, `pf_active`, `pf_passive`,
`pulse_schedule`, `tf`, `wall`, `controllers`.

Level-1 rates for shot 30147 **[queried]**: `xmo` OMAHA 14 coils at 10 MHz (7,000,000 samples, 313 MB),
`xsx` soft X-ray at 500 kHz (88 MB), `ama` inboard Mirnov n=odd / n=2 amplitude at 500 kHz, `xmb` at
50 kHz, `xim` D-alpha and CII lines at 50 kHz, `ane` CO2 interferometer at 25 kHz, `anb` NBI power at
20 kHz, `ayc` core Thomson 266 frames, `ada` linear D-alpha camera at about 1.1 kHz, `abm` bolometer
at 2.5 kHz, `amc` coil currents at 5 kHz. A whole level-1 shot is 1.1 GB. There is no `xbt` (BES)
source in level 1; BES is exposed in level 2 only.

ECE does not exist on MAST (overdense spherical tokamak). The electron-temperature stand-in is
Thomson at 240 Hz, which is itself a sampled diagnostic and not a dense fast channel.

### 2.6 Joint coverage

From `l1_sources.parquet` **[queried]**: shots with `act` and OMAHA and soft X-ray and interferometer
and D-alpha: 6,919. `act` and core Thomson `ayc`: 3,690. `act` and MSE: 4,642.

From a random sample of 80 of the 4,278 level-2 charge-exchange shots, reading each shot's signal
catalogue **[queried]**:

| Signal present together with `t_i`, `v_i` | Shots of 80 |
|---|---|
| Thomson `t_e` | 80 |
| D-alpha | 80 |
| Interferometer `n_e_line` | 80 |
| Outer vertical Mirnov (OMV) | 80 |
| Soft X-ray upper camera | 79 |
| Centre-column toroidal Mirnov | 62 |
| OMAHA | 27 |
| BES | 23 |

BES fraction 23/80 = 0.29, Wilson 95 % interval [0.20, 0.39], i.e. roughly 850 to 1,650 shots of the
4,278 (point estimate about 1,230) with charge exchange, BES and Thomson together.

### 2.7 Co-location

- CXRS: 64 midplane chords on the SS beam, R = 0.78 to 1.43 m **[queried]**; the instrument paper
  (Conway et al., Rev. Sci. Instrum. 77, 10F131, 2006) reports about 1 cm radial resolution
  **[search]**.
- Thomson: 120 to 131 points along the midplane with a per-frame `major_radius` **[queried]**, so a
  Thomson point within about 1 cm of any chosen CXRS chord exists.
- Soft X-ray: line-integrated chords with origin and endpoint R, Z in the store **[queried]**.
- BES: 32 channels, viewing the same SS beam; the level-2 store gives only a channel index, **no
  radial position** **[queried]**. The BES view radius on MAST was movable shot to shot **[memory]**,
  so per-shot co-location with a CXRS chord cannot be established from level 2 alone. This is the one
  gap that needs a question to the FAIR-MAST maintainers or a per-shot inference from the beam
  emission profile.
- Mirnov, D-alpha, interferometer: global or edge signals, the same status as the Mirnov channels in
  the KSTAR set.

### 2.8 Download size and time

- Slim set per shot (charge exchange, Thomson, D-alpha, interferometer, OMV, centre-column Mirnov,
  saddle, summary): about 5 MB stored, about 120 objects. For 4,278 shots: about 21 GB and about
  0.5 million requests.
- Adding soft X-ray: about 28 MB per shot more, about 120 GB more in total.
- Adding BES for the estimated 1,230 shots: about 87 MB per shot, about 105 GB.
- Level-1 `act` error arrays: about 0.06 MB per shot plus the 1 to 2 MB `.zmetadata` (avoidable by
  building keys directly).
- Observed from this machine: about 1 s latency per small object and 0.1 to 0.5 MB/s single-stream on
  the 13 MB catalogue files. Throughput will be set by parallelism; with 32 parallel requests the
  slim set is on the order of 4 to 8 hours, the BES subset on the order of 1 to 3 days. These two
  durations are extrapolations, not measurements. The docs recommend `s5cmd` for bulk transfer
  **[search]**.

### 2.9 Mapping to our pipeline

`KSTAR_CES_Dataset` infers features from CSV headers by prefix (`BES_*`, `ECEI_*`, `MC*`) and needs
`time`, `CES_TI`, `CES_VT`; `seq_data.py` keeps every input-complete row on the grid and masks
missing targets in the loss. A MAST converter therefore only has to write per-shot CSVs:

| Our column | MAST level-2 source | Reduction to the grid |
|---|---|---|
| `time` | 5 ms charge-exchange frame times, or a finer uniform grid | grid choice below |
| `CES_TI` | `charge_exchange/t_i` at one fixed chord (or the mean of 3 adjacent chords) | NaN where the fit is absent or fails the error cut |
| `CES_VT` | `charge_exchange/v_i`, same chord, m/s to km/s | same |
| `BES_*` | `filter_spectrometer_bes_voltage`, 32 channels | causal window mean and band power per grid step (BES subset only) |
| `ECEI_*` | no ECE; use soft X-ray chords nearest the chosen radius, or leave the prefix empty and report an input-set difference | causal window mean |
| `MC*` | OMV / centre-column Mirnov | causal window RMS or a raw snapshot, to mirror the KSTAR 100 Hz snapshot |
| extra | D-alpha, `n_e_line`, `power_nbi` | causal window mean; needs a new prefix or reuse of an existing one |

Code needed: one script of about 200 to 300 lines (`zarr` v3 + `s3fs`, or the plain HTTPS reader used
in this session with `numcodecs`), no change to `dataset.py` or `model_seq_v2.py`. Open points:

1. **Grid.** MAST charge exchange is already at 5 ms. A 5 ms grid with native NaNs reproduces the
   KSTAR setting (missing frames from failed fits and beam-off periods). MAST also allows a test KSTAR
   cannot give: decimate the target to 10 or 20 ms for the model's history and score on the withheld
   5 ms frames, which gives measured ground truth inside the gap.
2. **Discharge length.** MAST shots last about 0.3 to 0.7 s, so about 60 to 110 usable frames per
   shot against several thousand rows per KSTAR shot. The per-file cap of 500 never binds; total
   labelled frames are on the order of 4,278 x 66 = about 280,000.
3. **Routing.** `seq_v2` withholds the fast channels from the rotation branch. That routing carries
   over unchanged because it is defined by column position (`N_FAST_CHANNELS`), but the constant 15
   must match the new channel count.
4. **Quality.** Apply the `act` error arrays and a fit-failure rule from the start; shot 28981 shows
   why. The catalogue "Bad" flag on 3,110 `act` shots should be a registered exclusion.

### 2.10 TokaMark and other benchmarks on FAIR-MAST

- **TokaMark** (arXiv:2602.10132, UKAEA / IBM / STFC; Hugging Face `UKAEA-IBM-STFC/tokamark-v1`,
  CC BY 4.0, 565 GB; also at `s3://mast/tokamark/v1/`): 39 signals from 11,573 shots, 14 tasks.
  **No charge-exchange, ion-temperature or rotation signal**; the kinetic signals are Thomson `t_e`,
  `n_e` and interferometer density. **[page]** So TokaMark cannot host our task directly, and a
  charge-exchange gap-filling task on FAIR-MAST would be a task TokaMark does not contain. Its task
  3-1 (Thomson forecasting, 5 ms input, 50 ms horizon) is the nearest neighbour to cite.
- **Fusion Equilibrium Challenge** (NeurIPS 2026 competition; Hugging Face
  `Sophelio/fusion-equilibrium-challenge`, CC BY 4.0, 104 GB): 9,113 DIII-D and 2,443 MAST shots,
  EFIT flux maps from coil currents, Ip and Thomson; no charge exchange, ECE, BES or Mirnov listed.
  **[page]** Useful only as evidence that a DIII-D subset has been released openly.
- **RL4F** (arXiv:2606.07550): offline RL benchmark built on DIII-D discharges **[search]**; contents
  not checked.

---

## 3. LHD (rank 2)

### 3.1 Access and terms

- Analysed data: `https://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?cmd=getfile&diag={name}&shotno={n}&subno=1`,
  anonymous, returns a commented ASCII table. **[queried]**
- Raw diagnostic data: `s3://nifs-lhd` (AWS `ap-northeast-1`, no account needed), organised by
  campaign (`1998_1st/` onward), about 2 PB. **[queried / search]**
- Portal and name search: https://www-lhd.nifs.ac.jp/pub/Repository_en.html,
  https://exp.lhd.nifs.ac.jp/opendata/LHD/dataexplog/ ; DOI https://doi.org/10.57451/lhd.analyzed-data.
  **[page]**
- Terms (https://www-lhd.nifs.ac.jp/pub/RightsTerms.html) **[page]**: no proprietary period and open
  to the public, but data "should not be used for scientific analysis or publication without express
  consent" of instrument staff; submit an analysis proposal early; offer co-authorship; cite the DOI
  and shot numbers. No Creative Commons licence is named. A paper can use the data, but the consent
  step must be completed before submission and a derived public benchmark needs their agreement.

### 3.2 Verified signals, shot 160000

| `diag` | Content | Grid | File |
|---|---|---|---|
| `cxsmap7` | `Ti` (keV), `Tier`, `Vc` (km/s), `Ver`, `Vr`, `reff`, plus mapped `Te`, `ne` | 400 frames x 100 radii, 20 ms spacing, R = 2.91 to 4.81 m | 12.1 MB |
| `ece_fast` | `Te` (keV) per channel with `R`, `rho_vacuum` | 9,999 x 41, 1 ms, R = 3.58 to 4.70 m | 45.9 MB |
| `fir_nel` | line density, 13 chords | 131,072 samples, 0.1 ms | 30.1 MB |
| `ha1` | H-alpha, HeI (4 signals) | 12,001, 1 ms | 0.9 MB |
| `wp` | stored energy, beta | 139,102, 50 us | 8.5 MB |
| `bolo` | radiated power | 13,105, 1 ms | 0.6 MB |
| `fmp_psd` | magnetic-probe spectrogram and toroidal mode content | 2,000 x 500, 5 ms x 1 kHz | 41.0 MB |
| `tsmap_calib` | Thomson Te, ne profiles | 112 x 141 | 6.6 MB |
| `nb1pwr_temporal`, `ech`, `ip`, `imp02`, `shotinfo` | heating, current, impurity lines | 1 ms to 10 ms | small |

`mp`, `bes`, `sxfluc`, `cxsmap3`, `cxsmap6` returned 404 for this shot. **[queried]**

In shot 160000 only 20 of the 400 `cxsmap7` frames hold a valid Ti (3.91 to 4.67 s, 40 ms apart,
about 76 radii each); median Ti 0.79 keV, median relative error 0.17, median |Vc| 22 km/s, median
velocity error 3.8 km/s. The valid fraction of other shots was not measured (each file is 12 MB).
The relative error is about four times the MAST value, and the valid cadence in this shot is 40 ms,
not 5 to 10 ms.

### 3.3 Coverage

Random sample of 48 shots between 110000 and 190000, reading file headers only **[queried]**:
`cxsmap7` exists for 42 of 48 (0.88, Wilson 95 % [0.75, 0.94]); `cxsmap7` and `ece_fast` together for
34 of 48 (0.71, [0.57, 0.82]); none of the 6 sampled shots below 120000 has `ece_fast`. File existence
is an upper bound on usable shots because of the valid-frame fraction above. The ECE channel count
varies by campaign (13 to 70).

### 3.4 Co-location and effort

ECE radii (3.58 to 4.70 m) lie inside the CXS radii (2.91 to 4.81 m) and both files carry a flux
coordinate (`reff`, `rho_vacuum`), so the `ECEI_*` analogue is co-located by construction. There is
no BES. Mirnov content is available only as a spectrogram product in the analysed repository (raw
probes are in the S3 bucket). Text files run about 100 MB per shot for the set above, so 2,000 shots
is about 200 GB; transfer from this machine was fast (150 MB in under a minute). Converter effort is
similar to MAST (ASCII parsing is simpler), plus a per-shot valid-frame scan and the consent step.
Physics caveat for the paper: `Vc` is the flow along the CXS sightline in a heliotron, where
toroidal flow is strongly damped by helical ripple **[memory]**, so the rotation claim does not
transfer one to one; the Ti claim does.

---

## 4. Other fusion sources

| Source | Status for our purpose | Evidence |
|---|---|---|
| MAST-U | Not in the public bucket (shot range ends at 30473). The FAIR-MAST paper describes MAST-U and JET as covered by the same system, but no open MAST-U shot store was found. UKAEA's portal handles requests per experiment. | **[queried]**, **[search]** |
| JET | Not public. EUROfusion data-management goals for 2026-27 list JET mappings as a risk. | **[search]** |
| WEST, TCV | Named as candidates for an open-data release "by end of 2027" in EUROfusion planning slides; about 1,200 WEST discharges and a near-full TCV set are available to EUROfusion users only. No public portal found. | **[search]** |
| DIII-D | Full archive requires the non-proprietary user agreement. An open subset exists in the Fusion Equilibrium Challenge (no CER). A "Fusion Data Platform" is described in a 2024 APS abstract; its contents were not confirmed. DIII-D would be the closest physics match (CER + BES + ECE), so an access request is the realistic route. | **[search]**, **[page]** |
| Alcator C-Mod | Only the MIT-PSFC Open Density Limit Database (CC BY data; 10 ms grid; density, shape, Ip, Bt; no Ti or rotation). MDSplus access needs a collaboration agreement. | **[page]**, **[search]** |
| W7-X | Internal data-access policy; team use only. | **[search]** |
| EAST, HL-2A / HL-3 | No public data route found in this search. Not checked on the institutes' own sites. | **[search]** |
| KSTAR | No published open-data policy found; our own set is held under the provider's terms. | **[search]** |

The TokaMark paper states that FAIR-MAST "represents the only openly available dataset of real
tokamak diagnostics" **[page]**. With LHD added for helical devices, that matches what this search
found.

---

## 5. Non-fusion datasets with the same structure

| Dataset | Slow sparse label | Faster co-located inputs | Size, licence | Fit |
|---|---|---|---|---|
| PhysioNet/CinC Challenge 2012 (https://physionet.org/content/challenge-2012/1.0.0/) | laboratory values (creatinine, bilirubin, glucose, troponin, ...) at irregular intervals | bedside vitals (heart rate, blood pressure, respiration, temperature) over 48 h | 12,000 ICU stays, 37 time-series variables; set A 7.6 MB, full ZIP 1.8 GB; ODC Attribution 1.0, open **[page]** | Standard irregular-time-series benchmark, so baselines exist (GRU-D, mTAN, and others **[memory]**). Vitals are minute-to-hour, labs hours apart. |
| PhysioNet/CinC Challenge 2019 (https://physionet.org/content/challenge-2019/1.0.0/) | 26 laboratory values, mostly missing per hour | 8 vital signs on an hourly grid | 40,336 patients, 42 MB; CC BY 4.0, open **[page]** | Closest to our CSV layout: one row per grid step, dense vitals, NaN labs. A converter is a column rename. |
| Low-cost PM sensors against a reference monitor (Bulot et al., https://zenodo.org/record/7198378) | reference-grade Fidas 200S PM mass at 2-minute resolution | 8 Sensirion SPS30 and 8 Plantower PMS5003 sensors co-located for about one year, plus weather | 763 MB; CC BY 4.0 **[page]** | Physical co-location is exact. The low-cost sampling interval is not stated on the record page. One site, so the "second device" axis is the sensor model. |
| Soft-sensor benchmarks (Fortuna et al.: debutanizer column, sulfur recovery unit) | product composition from a slow analyser | process temperatures, pressures, flows | about 2,400 and 10,000 rows **[search]**; licence not found | Classic, but small and already resampled to one rate; weakest of the four. |
| MIMIC-IV | laboratory values | charted vitals, waveforms | credentialed access with training and a data-use agreement **[memory]** | Larger, but not open in the sense a benchmark claim needs. |

For an ML-venue generality section, Challenge 2019 is the least effort (hourly grid, 42 MB, open
licence) and Challenge 2012 has the most published baselines.

---

## 6. Effort summary

| Step | FAIR-MAST | LHD | PhysioNet 2019 |
|---|---|---|---|
| Access formalities | none | consent of instrument staff, co-authorship offer | access-request click-through |
| Converter to per-shot CSV | 1 to 2 d | 1 to 2 d plus valid-frame scan 0.5 d | 0.5 d |
| Download | 21 GB slim (hours); +105 GB for BES (days) | about 100 MB per shot | 42 MB |
| Baselines (PCHIP, persistence, causal GP) | existing code, new column names | same | same; "GP per discharge" becomes per patient |
| Training and scoring `seq_v2` | 1 to 2 d GPU | 1 to 2 d GPU | under 1 d |
| Open risk | BES radial position not in level 2; short discharges | sparse valid frames, 40 ms cadence, flow is not tokamak toroidal rotation | weak physical coupling between vitals and labs |

The day counts are estimates from the converter scope above, not measured.

## 7. Checks not done

- `mastapp.site` API could not be reached; all MAST facts come from S3.
- BES radial positions on MAST and their per-shot variation.
- The valid-frame fraction of LHD `cxsmap7` beyond shot 160000.
- Five of the 24 randomly drawn level-1 MAST shots produced no output line in the sampling run; the
  cause was not investigated, so the level-1 sample is 19 shots.
- EAST, HL-2A/HL-3, W7-X and KSTAR policies were checked through search summaries only.
- Download durations for the full MAST sets are extrapolated from single small transfers.
