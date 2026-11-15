# Development plan toward a main venue (draft for adversarial review, 2026-11-15)

Inputs: nf_bar.md, neurips_bar.md, open_data.md, assets.md (same folder), round_1.md / round_2.md.

## Pivot: add FAIR-MAST as a second, public device
FAIR-MAST level 2 (anonymous S3, CC BY-SA 4.0) has `charge_exchange` t_i / v_i for 4,278 shots
(campaigns M6-M9), 5 ms cadence, 64 midplane chords, plus Thomson, D-alpha, interferometer, Mirnov,
soft X-ray on ~all shots and BES on ~29 %. One addition closes, without waiting for KSTAR data:
- one chord -> 64 chords (radial dependence);
- one campaign / one device -> four campaigns, second device;
- no ground truth at filled times -> decimate real 5 ms CXRS to 10 ms and score the hidden frames
  (a measured answer at the filled instant, which KSTAR cannot give);
- no event tags -> D-alpha gives ELM / L-H tags for event-conditional scoring;
- private data -> reviewers can reach the data (NeurIPS Evaluations & Datasets requirement).
Known limits: no ECE on MAST; BES radial position not in level 2; fit failures stored as values.

## Phase 0 (about 1 day, local GPU, KSTAR only) -- settle Claim 1
1. History-only seq_v2 retrain, 4 splits x 2 populations (~1 h): isolates the fast-channel share.
2. `CES_PER_SHOT_NORM=0` rerun incl. temporal split (~2 h): is the result causal/deployable.
3. Round-2 CPU checks (a)-(d), memoryless MLP/GBM baselines (< 2 h).
Kill/branch rule: if the fast-channel increment over the history-only backbone has a 95 % interval
including 0, the T_i story becomes "denoising of the CES past" and the physics route loses its
main axis; the evaluation-audit route is unaffected.

## Phase 1 (2-3 weeks) -- MAST replication
4. Converter (zarr -> per-shot table; 200-300 lines), slim set ~21 GB.
5. Decimation task: hide alternate CXRS frames, estimate them causally from past CXRS + fast
   diagnostics; baselines persistence, linear, PCHIP, causal GP, offline GP, running mean.
6. Per-chord skill vs radius; event-conditional skill (ELM-tagged, L-H) from D-alpha.
7. Replicate the audit: tail-vs-median readout, influence of top 0.1 % rows, fit-failure anchors.
Registered before scoring, same protocol family as KSTAR.

## Fork A (primary recommendation): physics paper, Nuclear Fusion Letter / PPCF
Claim: on two devices, what co-located fast diagnostics add to the ion-temperature past at
5-10 ms, as a function of radius and event class; and a noise-corrected bound on their coupling to
toroidal rotation. Needs Phase 0 item 1 positive and Phase 1 items 5-6. In parallel, request from
the KSTAR team (lead time): all CES chords, per-exposure chi-square/photon counts, D-alpha, NBI,
EFIT, a 2023/2024 subset, co-authorship and data statement. NF if KSTAR multi-chord + event result
arrive; otherwise PPCF with MAST as the multi-chord arm; FED as floor.

## Fork B (secondary): NeurIPS 2027 Evaluations & Datasets -- evaluation audit
Claim: neural gap-filling skill over interpolation is tail-carried, sign-flips at the median, and
is largely recovered by 1-4-parameter causal smoothers; shown on KSTAR error arrays (if release is
allowed), FAIR-MAST, PhysioNet 2012/2019 and 1-2 more public sets with SAITS/BRITS-class imputers.
Start only if the 1-2 week kill criteria in neurips_bar.md pass. Deadline assumed early May 2027.

## Why A is primary
The student's thesis and assets are physics-framed; Phase 0-1 are shared by both forks; B's odds
(12-18 % if it replicates) require 14-18 weeks of generic ML benchmarking outside the thesis theme.


> Superseded 2026-11-15: the adversarial review (PLAN_review.md, section 5) measured 13 MAST shots and replaced this draft. MAST cannot test "co-located fast diagnostics" (Thomson is one sample per CXRS frame, no ECE); it is used for a model-free two-device regime and coupling study behind a 300-shot gate. The adopted plan is PLAN_review.md section 5.
