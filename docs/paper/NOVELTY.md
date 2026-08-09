> **표현 규칙 (승상님, 2026-08-05): 부재-프레임 금지.** 이 파일의 탐색 결과("같은
> 태스크는 발견되지 않았다")는 내부 증거 기록이며, 독자용 문서(논문·발표·방어 자료)
> 에서는 절대 "관련 논문이 없다"를 앞세우지 않는다. 올바른 서술은 **계보 인정 →
> 3축 확장(전자→이온 타겟 / 동시각→인과 이력 / 가정→사전등록 검정) → "우리가 아는
> 한 이 결합은 아직 다뤄지지 않았고, 계열의 자연스러운 다음 단계로 제시한다"**의
> 순서다. 부재-프레임은 논문의 일반적 흐름과 어긋나고 반례 하나에 무너지지만,
> 확장-프레임은 계보를 인정할수록 강해진다.

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

---

# Re-verification (2026-08-05, web re-search incl. post-July publications)

**Verdict: all three novelty claims STAND.** 18 distinct queries (incl. Korean) +
12 primary-source fetches; no same-task work found through Aug 2026.

Bibliographic status of the 9 known works: mostly already correct in `refs.bib`
(Bishop = PPCF 35 (1993) w/ von Hellermann; Wang title "Time series extrinsic
regression…"; Imríšek PPCF 68, 065049 (2026); FusionMAE = HL-3, Comm. Phys.;
Diag2Diag's formal title "Multimodal super-resolution…"; Svensson PPCF 41).
**Fixed**: Shousha year 2023 → 2024 (NF 64, 026006). Lin's instrument is XCS
(x-ray crystal spectrometer) — paper text already says so; avoid "XICS".
Diag2Diag confirmed from the arXiv HTML: CER is an *input*, target is Thomson TS
only, model is explicitly memory-less. (Caution: PDF-extraction reads of that
paper hallucinate — use the HTML.)

New finds, none overlapping (added to refs.bib / cited where noted):

| Work | What | Why no overlap / use |
|---|---|---|
| **Jung, Kim & Kang, JKPS 88, 1079 (2026)** (= existing key `Jung2026MachineLearningbased`) | KSTAR ML profile reconstruction, CES among inputs | CES is an **input**, spatial fitting at measured times; **rotation explicitly named future work** → cited with that clause (strengthens N1) |
| **Kim et al., NF 64, 106052 (2024)** (`Kim2024KineticProfile`, added) | KSTAR CES Ti profile GPR + SVMR outlier rejection | Spatial profile fitting, not temporal gap-filling; cited in the GP paragraph |
| **Char et al., arXiv:2404.12416 (2024)** (`Char2024FullShot`, added) | DIII-D full-shot recurrent simulator; predicts rotation from actuators incl. NBI | Positive control for the §Headroom NBI-torque lever; cited there |
| TokaMind / TokaMark (MAST, 2026) | Benchmarks/simulators, no ion diagnostics, no interpolation baselines | Not cited (optional) |
| Pyragius et al., arXiv:2407.18741 (ST40) | SXR → Thomson Te/ne | Electron channel only (optional) |

Claim-level notes from the adversarial pass:
- **N1** stands but state it narrowly: "first causal between-measurement filling
  of a sparse *ion* diagnostic from fluctuation diagnostics + the target's own
  irregular past history." Biggest threat = RTCAKENN (outputs impurity Ti +
  rotation, robust to missing CER). Three defenses: inputs are control-room
  signals not BES/ECEI/Mirnov; no past-target-history channel (Diag2Diag is
  memory-less, Imríšek static); never benchmarked vs interpolation.
- **N2** is the strongest: no surveyed work sets future-using interpolation as a
  statistical bar; frame it as "a protocol that can return negatives" (it denied
  our own V_rot claim).
- **N3**: sell as *measured quantification*, not discovery — the physics is
  predictable a priori; Char et al. makes the "NBI torque is the lever" claim
  falsifiable (rotation IS learnable when torque inputs exist).
- CXRS practice note: beam-blip CXRS routinely uses linear interpolation across
  blips → the future-anchor-interpolation bar mirrors real practice (defends N2).
