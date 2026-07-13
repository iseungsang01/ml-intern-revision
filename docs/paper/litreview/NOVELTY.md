# Novelty verdict (adversarial prior-art search, 2026-07-03)

**Verdict: BORDERLINE-POSITIVE — no same-task prior work found; a close "temporal
densification from other diagnostics" family exists and must be cited + differentiated.**

Search: 4 parallel discovery agents (targeted / fusion-ML / adversarial novelty / ML methods)
with 25+ query variants, all candidates verified against OpenAlex (S2 keyless pool saturated;
fallback documented). Full query list in the novelty agent transcript.

## Closest prior works ("close", none "same")

| Paper | What it does | Why ours is different |
|---|---|---|
| **Diag2Diag** (Jalalvand et al., Nat. Commun. 2025, DIII-D) | Multimodal NN reconstructs Thomson Te/ne at fluctuation-diagnostic rates from other simultaneous diagnostics (CER as *input*) | Target is Te/ne, never CES Ti/V_rot; memory-less non-causal MLP, no CES history; no interpolation benchmark; no information-asymmetry analysis |
| **COMPASS temporal super-resolution** (Imrisek et al., PPCF 2026) | NN upsamples sparse Thomson Te/ne between samples using fast SXR/AXUV/magnetics | Electron quantities only; non-causal; no causal-vs-offline-interpolation bar |
| **EAST missing-Te reconstruction** (Wang et al., NF 2025) | Time-series extrinsic regression fills missing Te from other signals | Te, failed-signal recovery framing; no CXRS targets, no causality benchmark |
| **FusionMAE** (2025, HL-3) | Masked autoencoder reconstructs masked diagnostic channels ("virtual backup diagnosis") | Generic channel dropout, non-causal within window; not sparse-CES gap-filling |
| **RTCAKENN** (Shousha et al., NF 2024, DIII-D) | Real-time kinetic profiles (incl. impurity Ti/rotation) robust to missing CER/TS | Profile reconstruction for control; missing-diagnostic robustness is a side property; no fluctuation-diagnostic inputs |
| **EAST XICS→Ti/rotation NN** (Lin et al., NF 2024) | NN infers Ti+rotation profiles from x-ray crystal spectrometer | Same targets but input is another Doppler spectrometer (same physical channel); instantaneous mapping, not temporal gap-filling |

## What remains unclaimed by prior art (our defensible novelty)

1. Temporal gap-filling/nowcasting of **sparse CES Ti + V_rot** (not electron quantities,
   not spectra fitting) from **fluctuation diagnostics + irregular past-CES history**.
2. **Causal-vs-future-using-interpolation benchmark** (pre-registered, shot-clustered
   bootstrap) as the evaluation bar for a sparse fusion diagnostic.
3. The **Ti↔V_rot information asymmetry** finding (fast diagnostics carry Ti but ~no
   rotation information at 10 ms), physics-predicted and ablation-confirmed.

## Consequences applied to the draft

- Novelty wording narrowed to the three claims above; "temporal densification from other
  diagnostics" family explicitly cited (Diag2Diag, COMPASS, FusionMAE, EAST) and differentiated.
- RTCAKENN corrected: it is a **DIII-D** paper (draft TODO wrongly said KSTAR).
- No KSTAR-specific NN-CES paper exists; the NN-CES lineage is JET (Bishop & Roach 1993;
  Svensson & von Hellermann 1999) + EAST fast-Ti (Chai et al. 2019).
- 2025–2026 close works treated as concurrent (no claims of beating them), per the
  orchestra writer prompt's timeline rule.
