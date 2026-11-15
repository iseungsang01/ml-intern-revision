# MAST hardware timeline, campaigns M6-M9 (2006-2013) — what is confirmed and what is not (2026-11-15)

Purpose: the FAIR-MAST gate mixes four campaigns. This note lists the machine and diagnostic changes
inside that range that could move a charge-exchange readout, with the source for each. It does not
enter any registered rule (`PREREGISTRATION_MAST_GATE_ADDENDUM2.md` section 7).

Campaign ranges are from the FAIR-MAST catalogue (`data/mast/meta/l1_shots.parquet`, queried):

| campaign | shots | first / last time stamp |
|---|---|---|
| M5 | 11695-14706 | 2004-12 / 2005-12 |
| M6 | 14708-19026 | 2006-01 / 2007-08 |
| M7 | 19031-25403 | 2008-04 / 2011-01 |
| M8 | 25404-28355 | 2011-01 / 2012-01 |
| M9 | 28390-30473 | 2012-03 / 2013-09 |

## 1. Confirmed from a source read today (search result text; full papers not opened)

| item | what the source says | date it supports | source |
|---|---|---|---|
| CXRS system | "A major upgrade to the charge exchange recombination spectroscopy system on MAST has recently been implemented"; 224 spatial channels, toroidal views 64 channels per beam, poloidal views 32 per beam, about 1 cm radial resolution, C5+ line at 529 nm, CCD reads the 224 channels in under 4 ms | in place by 2006 (Rev. Sci. Instrum. 77, 10F131, 2006), i.e. by M6 | https://www.osti.gov/biblio/20861329 ; https://scientific-publications.ukaea.uk/papers/high-throughput-charge-exchange-recombination-spectroscopy-system-on-mast/ |
| CXRS time resolution | a 2008 EPS paper (Wisse et al.) describes about 1 cm and 5 ms resolution, 224 chords with one set of 64 toroidal chords | 5 ms operation by 2008 (M7) | https://lac913.epfl.ch/epsppd3/2008/pdf/P2_088.pdf |
| NBI | until summer 2003 two injectors with ORNL duopigatron sources (up to 3 MW, 300 ms); a 2005 paper says they "are now being replaced" by JET-type PINIs (75 kV, 65 A nominal) | replacement in progress in 2005; completion date of each beam line not found | https://opact.sdu.edu.tr/vufind/SummonRecord/FETCH-LOGICAL-c345t-dac97ab537c7479e6da83e4f9ac5112a61dffa77eaa8d0f5797d59fc35cc2dcc3 |
| Thomson scattering | core Nd:YAG system raised from 19 to 130 spatial points, lasers doubled from four to eight, 240 Hz (Fusion Eng. Des. 2010, Shibaev et al.; design paper Scannell et al., Rev. Sci. Instrum. 2008, 120 points planned) | operational by the 2010 publication (inside M7); installation shot not found | https://scientific-publications.ukaea.uk/wp-content/uploads/Published/RSIVOL81p10d521.pdf ; https://impact.ornl.gov/en/publications/design-of-a-new-ndyag-thomson-scattering-system-for-mast/ |
| ELM (RMP) coils, first set | in-vessel coils in two rows of six, above and below the midplane; a 2007 abstract says 3 of 12 coils were installed in October 2006; a 2008 APS abstract says the set of 12 was implemented "during the last shutdown" | phased: partial in late 2006 (M6), full 6 + 6 set by 2008 (start of M7) | https://tec.ipp.kfa-juelich.de/sfp/abstracts/2007/pdf/kirk.pdf ; https://meetings-archive.aps.org/dpp/2008/co3/8 ; https://scientific-publications.ukaea.uk/wp-content/uploads/Published/NFVOL50p034008.pdf |
| ELM (RMP) coils, second set | upgraded from 6 + 6 to 18 coils (6 upper, 12 lower); 2011 workshop slides say the new coils were "installed and tested" | about 2010-2011 (M7 / M8 boundary); approximate | https://nstx.pppl.gov/nstxhome/DragNDrop/Scientific_Conferences/ST_Workshop/STW11-Toki/presentations/Lloyd.pdf ; https://arxiv.org/pdf/1307.1573 |

## 2. From the data and the catalogue (queried, gate shots only)

- `divertor_config` is "Conventional" for all 300 gate shots: no divertor change is recorded in the
  catalogue inside M6-M9. No literature source for the divertor history was found today.
- The stored CXRS arrays have 96 chord columns in 99 of the 264 gate shots with data and 160 in 165;
  three shots have 10 ms frames. Which hardware or processing change this corresponds to was not
  found in the literature; the breakdown by column count is in `docs/paper/mast_gate_addendum2.md`.
- All 17 gate shots tagged "SW Beam" carry no `act` arrays, and none of the shots with arrays is
  SW-beam-only: the stored charge-exchange data exist only where the SS beam fired (or where the
  heating tag is missing).
- `rmp_coil` is recorded for 108 of the 300 gate shots (true 36, false 72).

## 3. From memory only, not confirmed today

- The year each PINI (SS, SW) first injected, and whether SW became a PINI later than SS (memory:
  SS first, SW about 2008-2009). A 2009 SOFE paper "Overview of MAST Neutral Beam System
  Performance" is the likely source; it was not retrieved.
- The BES system installation (memory: 2010) and its movable view radius.
- The name of the CXRS spectrometer system (memory: CELESTE-3). The search did not return this
  name; one thesis uses "ECELESTE".
- Any change of the CXRS fit software or of the background-subtraction scheme between campaigns.

## 4. Pages that did not open or were not opened

- iopscience.iop.org and pubs.aip.org full texts were not requested (paywalled in earlier sessions).
- No page returned an error today; every item above is from search-result text, so wording should be
  checked against the paper before it is cited.

## 5. What this means for the breakdown

Between M6 and M9 the catalogue shows no divertor change, the CXRS system is the same 2006
instrument, and the changes that touch the plasma are the beam sources, the RMP coil sets (2006,
2008, about 2011) and operation itself. The campaign breakdown therefore mixes instrument history
(array layout, fit processing) with operating space; it cannot attribute a difference to either.
