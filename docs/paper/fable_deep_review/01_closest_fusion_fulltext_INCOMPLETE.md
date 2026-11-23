Incomplete output; not a completed review.

# Evidence dossier: closest fusion literature versus the KSTAR pedestal-top CES nowcasting task

## 0. Access, conventions, caveats

- **RTCAKENN** (Shousha et al., Nucl. Fusion 64 026006): all 19 PDF pages read. Locators use printed page numbers (PDF page = printed + 1).
- **COMPASS** (Imrisek et al., PPCF 68 065049): all 18 PDF pages read. Same page convention.
- **Diag2Diag** (Jalalvand et al., Nat. Commun. 16 8506): EuropePMC body text only. There are no page numbers, so locators are section headings. Figures, Supplementary Information and the peer-review file were not supplied and are not assessed.
- **Diag2Diag code**: the supplied archive has two files, `README.md` and `diag2diag_natcomm_model_train.py`. Line numbers are my count from the escaped string and are approximate (≈L); function names and statements are exact. `diag2diag_shotlist.txt`, which the script loads, is not among the supplied files.
- **Tags**: [R] directly read, [A] author claim or interpretation, [H] my application hypothesis, [U] not established in the supplied material.
- **Figure values**: numbers quoted from figures are read from printed annotations or axes and are approximate where marked ≈.
- **Scope**: nothing below is a claim about prior art outside these three sources.

## 1. Locator table

| Dimension | RTCAKENN | COMPASS | Diag2Diag paper | Diag2Diag code |
|---|---|---|---|---|
| Target | Seven CAKE profiles on 121 points: p, j, 1/q, ne, Te, Ti, Vtor (Table 1, Fig. 1, p.3) | TS Te and ne profile vector (§2.1 pp.2–3; §2.3 pp.5–6) | 80 TS channels, 40 ne + 40 Te (Methods, "Feature extraction") | `extract_features`: Y = `ts_core_density` + `ts_core_temperature` (≈L298–300) |
| Ti role | Both input (CER Ti, Ωtor) and target (CAKE Ti); C6+ throughout; edge handled by Ti/Te "stiffness" (§2 p.2) | None; C III line emission only as impurity context (Table 1 p.4) | "CER", 58 channels, input only, 200 Hz, held constant (Results; Methods) | `cer_ti_q.ti.t01–t29`, `v01–v29` (≈L269–273) |
| Own target as present input | Yes: measured ne/Te/Ti/Ω in, fitted profiles out (Table 1; Fig. 1) | No (§1 p.2) | No (Introduction, "Contribution"; Results) | TS absent from X list (≈L289–295) |
| History input | None; "no memory" (§4.2.2 p.12); latest valid sample, ring buffers rejected (§3 p.5) | None; per-sample MLP (§2.3 p.5) | None; first and second derivatives of ECE and interferometer instead (Results) | `calculate_derivatives` (≈L121–169) |
| Temporal preprocessing | Linear interpolation to 33-point ψN grid; no temporal
