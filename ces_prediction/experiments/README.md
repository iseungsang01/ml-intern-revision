# Controlled experiment batches

Every directory here is one **controlled experiment** behind a numbered section of
`THESIS_RESULTS.md` §8. Each has its own runner; none of them is imported by the
training pipeline. Run everything **from the repo root** (the runners put
`ces_prediction/` on `sys.path` themselves).

Two files at this level are shared, not experiments of their own:

- **`runner_common.py`** — the frozen split dirs (`SPLIT_SRC`), the frozen control run
  dirs (`BASELINE_OUT`), the pinned training env (`FULL_ENV` / `SMOKE_ENV`), `run_step`,
  and `prepare_split_copy`. Import these rather than re-declaring them; a batch that
  copies the constants is a batch that can silently drift out of pairing.
- **`paired_model_compare.py`** — the shot-clustered paired bootstrap between two arms'
  `comparison_errors` npz files. It refuses to compute anything until it has verified
  the two arms scored the same rows in the same order.

The published architecture is **`ces_prediction/model_iter009.py`**, and
`ces_prediction/model.py` re-exports it, so `from model import MultimodalCESPredictor`
already gives every runner the paper's model. A batch that varies the architecture
points **`CES_MODEL_FILE`** at its own file in the subprocess env (see
`anchor/run_anchor_experiments.py`) — nothing is ever copied over `model.py`. See
`PROJECT_KNOWLEDGE.md` "Checkpoint / Architecture Provenance" before scoring anything.

| dir | §8 section | question it answers | verdict |
|---|---|---|---|
| `stuckfree/` | §8c | do held (forward-filled) `CES_VT` values hurt *training*, not just evaluation? | **KEEP** — `CES_DROP_STUCK_TARGETS=1` is now the default for new training runs (4/4 seeds improved, 3/4 significant) |
| `seq/` | §8d, **§8t** | does reframing to a full-grid sequence model beat the per-sample framing? (`--arm seq`/`seq_sf`/`seq_v2`/`seq_v2_nops`) | §8d: `CES_TI` 4/4 improved, `CES_VT` gap = held contamination + missing routing. **§8t (`seq_v2`): all four structural ingredients together — best `CES_TI` on 4/4 splits, `V_rot` deficit closed, 1/10 the training cost** |
| `window_sweep/` | §8f | how much history does the model actually need? | one past observation is the whole story; `W=4` is not justified by skill, only by coverage |
| `largegap/` | §8g | is the model weak at large gaps, or is that interpolation's territory? | large gaps belong to interpolation — vs a *causal* baseline the model still wins |
| `mnar/` | §8i | does the result survive reweighting onto the genuinely-missing points? | causal claim survives 4/4 (+0.29); offline claim does not (1/4) |
| `anchor/` | §8k | what does the model's complexity buy over a 1,258-parameter named-terms anchor+Δ? | the anchor recovers ≈31.5% of the `CES_TI` margin — the rest prices the opacity |
| `latency/` | §8l | is real-time inference feasible, and on which device? | CPU batch-1 p99 = 6.4 ms; **CUDA is 8× slower** at this model size |
| `uq/` | §8m | can we get calibrated intervals without retraining? | split conformal beats both baselines 8/8 by Winkler; coverage is marginal, not conditional |
| `campaign/` | §8n | does it generalize across a strictly temporal (campaign) split? | offline advantage dies, causal advantage survives (+0.22); cause measured (BES drifts 1.22σ vs targets 0.115σ) |
| `gp/` | §8p | what happens against the strongest offline arm the fusion community actually uses? | GP beats PCHIP 4/4; **the model ties GP** — never claim "beats every offline method" |
| `fitfail/` | §8q | do CES spectral-fit failures (`CES_TI` > 3 keV) inflate the headline? | they **deflate** it — dropping them roughly doubles skill, so the headline is conservative |
| `heldpeak/` | §8r | are the `CES_VT` peak result and the instrument-hold pattern entangled? | yes, and backwards from the hypothesis — peaks are hold-**richer**; `CES_VT` splits into three regimes (genuine peak wins, genuine bulk ties, hold rows are structurally unwinnable) |
| `pershot/` (+ `campaign/ --arm per_shot`) | §8s | does per-shot input standardization repair the campaign-transfer failure, and what does it cost? | **ADOPT** — campaign `CES_TI` +0.155 mean, 4/4 significant; headline shows 0/4 significant losses (mean −0.036) |
| `b1_gate/` | §8x | does the seq_v2 backbone beat the W = 2 window control under the confirmed protocol, and does the causal GP restate claim 2? | **backbone = seq_v2** (16/16 positive, pooled +0.081, budget-equalised 4/4); causal GP beaten 1/4 by the window model → claim 2 restated, then reinstated by §8y for the backbone (4/4) |
| `b2_explore/` | §8y | B.2 val-only candidate search + rule-guarded TEST confirmation (`v3` attention readout) | promotion **FAILS** (4/4 positive, 1/4 significant); the backbone itself beats the causal GP 4/4 → claim 2 reinstated |
| `b3_interp/` | **§8z** | how much of the backbone's `T_i` skill survives compression to a K-dim probeable latent + persistence anchor + exact linear decomposition? | **all of it**: b3k8 (21,498 params, K = 8) paired `T_i` vs seq_v2 mean **+0.002**, vs retrained anchor +0.35–0.42 4/4\*, vs causal GP PASS 4/4; the §8k anchor collapses onto persistence at W = 2; `V_rot` gap vs backbone is mostly spike-anchor rows; **cut-population conditional** — inclusive population −0.16…−0.21\* vs the backbone (§8ab) |
| `b4_scale/` | **§8aa** | does more capacity buy `T_i` skill on this input set? (seq_v2 `hidden_ti` 24…260 = 34k…879k) | **flat**: mean skill +0.230/+0.236/+0.235/+0.236/+0.230, w260 vs w160 significant 1/4, no knee down to 24 — the input set, not the model, is the ceiling |
| `b5_rescore/` | **§8ab** | re-score every W = 4-based analysis at W = 2 in both populations (cut 3 keV / inclusive), no new decision rule | **unconditional** (both populations): backbone `T_i` vs PCHIP 4/4 + 4/4, vs causal GP 4/4 + 4/4, peak stratum 4/4 + 4/4, conformal Winkler best 32/32, **campaign split 4/4 + 4/4 vs PCHIP and causal GP** (window model collapses 2/4 / 0/4 — §8n reproduced, §8s per-shot repair re-confirmed), cut threshold 2.5–4 keV insensitive, `V_rot` routing bit-identical; **conditional**: b3k8 = backbone only under the cut (−0.19\* inclusive), MNAR-reweighted vs PCHIP 2/4 cut / 4/4 inclusive; `V_rot` vs PCHIP still 1/4 / 2/4. Also `spike_structure_audit.py` (B.7 follow-up memo) |
| `reach/` | **§8ac** | how many contiguous past steps does the trained backbone actually use, and what does each causal arm cost against the 10 ms budget? (no retraining — the frozen B.1 checkpoints re-scored with the recurrent state truncated) | `T_i` saturates at **50 steps = 500 ms**, `V_rot` at **20 = 200 ms**; `ctx = 2` (the window family's contiguous context) loses to `gp_causal` by **−0.34** and `ctx = 1` is worse than persistence — §8f's plateau was about past *observations*, not context. On cost, every causal arm fits the budget; `seq_v2` stateful 1-step p99 **1.49 ms (15%)** beats `gp_causal` (+0.396 vs +0.319 `T_i`) at 1.6× lower tail, ties the *acausal* GP, and is 3.0× cheaper than the window model (4.46 ms). **Quote the ordering, not the absolutes** — see the section's accuracy note. **§8ae (`trained_vs_truncated.py`) narrows this row:** the ctx rungs measure truncation, not information — a model *trained* at reach 2 recovers +0.260 of the 0.310, 4/4 significant, so ≥ 84% of the `ctx = 2` deficit is cold start |
| `wslim/` | **§8ad** | `W = 2` is a smaller input — can the window model's *structure* shrink to match, and what does that cost? (one variable: `CES_MODEL_FILE`, paired against B.1 stage A) | **25,602 params / 21 leaf ops / 0.66 ms** vs 201,258 / 57 / 3.02 ms — but `T_i` paired **−0.087, 4/4 significant deficits** (drops below PCHIP on s42), so the structure earns its keep on `T_i`. `V_rot` is **indifferent** (+0.005, 0/4 either way): a bidirectional GRU + 4-head attention over 2 steps is fully replaceable there. Cheaper-but-worse operating point; **do not adopt** — §8ac shows latency never binds. Structure, not size (cf. §8aa, §8z) |
| `b8_minimal/` | — (pending) | how far down can the parameter count go before skill breaks, when EVERY part shrinks? (B.4 moved only `hidden_ti`, so its 34k floor was set by the parts held fixed) | **pending** — v2m12k…v2m2k, b3m7k…b3m1k, each paired against its split's backbone and W = 2 control |
| `b9_reach/` | — (pending) | what is full-block context worth once the model is TRAINED at each reach, not truncated after the fact? (§8ae) | **pending** — `seq_v2` at reach 2/7/15/31/63 via `CES_SEQ_TRAIN_CTX`, rows-per-batch held constant |
| `b9_family/` | — (pending) | does the sequence-operator family matter once the reach is matched? (§8ac's named-but-untested TCN, plus causal attention) | **pending** — tcn15/tcn63/xfmr63, each paired against the axis A rung at its own reach |
| `b9_latency/` | — (pending) | which arms clear a 10 ms budget, and which clear 1 ms? | **pending** — 5-session protocol, streaming step per family, boundary band instead of a coin-flip verdict |
| `quantum/` | — (`docs/ionq_qpu_실험기록.md`) | side track: variational quantum classifier / IonQ QPU inference | exploratory; not part of the thesis claim chain |

## Non-negotiables for any new batch

Both were learned the hard way (§8f cost a wrong published conclusion, and the
`anchor/` runner shipped with the same bug before it was caught):

1. **Pin the data treatment explicitly** in the runner's env dict — never inherit,
   never `pop`. A silent fallback to `train.py`'s default `CES_DROP_STUCK_TARGETS=0`
   produced a window-sweep conclusion that was pure artifact.
2. **Pair against a control trained under the same treatment**
   (`.sf_iter009_s*` is held-free, `.vt_repro_*` is held-kept) and verify the scored
   populations match row-for-row.
3. When re-scoring frozen runs, add keys **additively** and verify every pre-existing
   npz key reproduces **bit-identically** before trusting the new one (§8g, §8i, §8p
   all did this).
4. Change **one** variable at a time, and record the run in `THESIS_RESULTS.md` §8
   whichever way it turns out.
5. **The confirmed protocol (2026-08-12, `THESIS_RESULTS.md` §8v) governs every new batch:**
   `W = 2`, held-free (`genuine`), and the pre-registered `CES_TI` fit-failure exclusion applied
   identically to every arm in training targets / history inputs / evaluation population.
   `W = 4` artifacts are provisional — usable for historical reproduction, never as the control
   arm of a new confirmatory claim. Before designing a batch, read `PROTOCOL_AUDIT.md` and
   `PREREGISTRATION_W2.md` in this directory.
6. **B.9 batches additionally follow `PREREGISTRATION_B9.md`** (reach ladder × architecture
   family × real-time budget). It inherits §1–§3 of `PREREGISTRATION_W2.md` and adds the
   `PRACTICAL_EPS = 0.02` tie rule and the 5-session latency protocol that a 1 ms budget needs
   (absolutes decide there, so §8ac's "ordering only" rule is not sufficient).
