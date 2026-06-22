# Handoff — 2026-06-22 (framing cleanup, GPU enablement, input-modality ablation)

## Current state
- **Data**: all 641 shot CSVs present locally in `data/` (~34.5 MB). Columns: `time, CES_TI,
  CES_VT`, BES×9, ECEI×4, **MC×2 = `MC1T03`,`MC1T16` (raw Mirnov coil signals)**. No NBI-torque column.
- **GPU enabled**: RTX 5060 Laptop (Blackwell, sm_120, 8 GB), `torch 2.11.0+cu128`, CUDA 12.8.
  Real GPU matmul verified; `train.py`/`evaluate.py` print `Using device: cuda`.
- **Code**: `train.py` + `evaluate.py` are device-agnostic (auto CUDA) with
  pin_memory / non_blocking / cudnn.benchmark and optional AMP (`CES_AMP=1`, default off).
  New **`CES_ABLATE`** flag (`none`/`no_history`/`no_fast`) zeroes a modality group for controlled
  ablation; persistence baseline always uses the real history. `pytest` 3/3 green; CPU + GPU smoke pass.
- **Docs**: `RESEARCH_SUMMARY.md` / `ML_WORKFLOW_ARCHITECTURE.md` re-framed from the abandoned
  super-resolution framing (A) to **gap-filling/nowcasting (B)**; strict-NaN "both targets" corrected
  to per-target masked. Leftover (not yet fixed): `ML_WORKFLOW` §1.2 CNN dims 64/16/208 are stale vs
  `model.py` 96/32/320; `RESEARCH_SUMMARY` verb-form mix + "(수정 반영)" artifact.

## Latest experiment — input-modality ablation (real data, GPU)
Settings: 40k train / 8k val, 12 epochs, batch 1024, window 4, temporal subsets on. Shared split
`data/.ablation_splits`; outputs `data/.ablation_out_{none,no_history,no_fast}`.

| Variant     | TI skill | TI R² | VT skill | VT R² | final val (aug) |
|-------------|---------:|------:|---------:|------:|----------------:|
| Full        |   +0.359 | 0.347 |   +0.180 | 0.788 |          0.3180 |
| no_fast     |   +0.393 | 0.382 |   +0.234 | 0.802 |          0.3134 |
| no_history  |   +0.162 | 0.146 |   −3.31  | −0.116|          0.5860 |

**Conclusion**: full model beats persistence on both targets; **V_rot skill is entirely from CES
history** (fast-only is worse than the mean); **T_i fast-only still beats persistence** (fast
diagnostics carry T_i info, not V_rot); **fast diagnostics are redundant given history**.
Full reading + physics backing: `PROJECT_KNOWLEDGE.md` → "Input-Modality Ablation Finding".

## Next actions (recommended order)
1. **De-noise small effects**: rerun the 3 variants with ≥3 seeds and at full scale (200k/40k) to
   confirm "fast diagnostics redundant given history" (~0.03 Full-vs-no_fast gap) is real, not seed
   noise. The big V_rot fast-only effect (−3.31) needs no confirmation.
2. **MNAR check**: stratify eval by regime (low collisionality / NBI-modulation / ELM-near) and by
   CES drop-out reason; report skill on "missing-like" intervals, not just the aggregate.
3. **Best-checkpoint + early stopping**: training saves the *final* epoch (no early stop). Wire the
   clean (non-aug) val into checkpoint selection before any quality claim.
4. **Doc cosmetics (parked)**: fix `ML_WORKFLOW` §1.2 dims → 96/32/320, time = 4 ch; unify
   `RESEARCH_SUMMARY` verb forms; drop "(수정 반영)".
5. **To make fast diagnostics help V_rot** (if pursued): would need the *fast* (kHz) Mirnov stream
   (extract mode frequency upstream) and/or NBI torque as an input — both are data-contract changes
   (controlled experiments, user approval).

Reproduce one variant (PowerShell, from repo root):
```powershell
$env:CES_EPOCHS="12"; $env:CES_MAX_TRAIN_SAMPLES="40000"; $env:CES_MAX_VAL_SAMPLES="8000"
$env:CES_BATCH_SIZE="1024"; $env:CES_SPLIT_DIR="data\.ablation_splits"
$env:CES_ABLATE="no_history"; $env:CES_OUTPUT_DIR="data\.ablation_out_no_history"
py ces_prediction/train.py ; py ces_prediction/evaluate.py
```

---

# AutoML Session Handoff

## Latest Briefing (Iteration 1)

```text
--- Briefing Report (Iteration 1) ---
Current Val Loss: 0.7845 (Best: 0.7845)
Plateau Count: 0/3
STATUS: NO ARCHITECTURE CHANGE ALLOWED.
DIRECTION: Keep evaluating/tuning the current controlled baseline until plateau criteria are met.

```

## Data Contract

```text
Dataset/training contract that every generated model.py must preserve:
- train.py builds KSTAR_CES_Dataset with temporal subset augmentation.
- BES, ECEI, and MC inputs are per-channel z-score normalized using train-file-only statistics.
- CES_TI and CES_VT are per-channel z-score normalized with train-file-only target statistics.
- ces_history has shape (batch, window, 3): normalized previous CES_TI, normalized previous CES_VT, observed mask.
- The target timestep CES values are masked in ces_history as [0, 0, 0] to avoid leakage.
- model.forward must accept forward(self, bes, ecei, mc, time_features=None, ces_history=None).
- Model outputs must remain normalized CES_TI/CES_VT with shape (batch, 2); train.py compares them to normalized targets.
- Do not denormalize inside model.py. Any inverse transform belongs outside training/evaluation.
```

## Latest Metrics

- Train Loss: 0.4966
- Val Loss: 0.7845
- Epochs: 10
- Samples: train=2000, val=500
- Temporal Subsets: True
- Min Subset Size: 2
- Normalization: per_channel_zscore, scope=train_files_only
- Normalization Groups: bes, ecei, mc, target
- Feature Dims: `{"bes": 9, "ecei": 4, "mc": 2, "time": 4, "ces_history": 3}`

## History

- Iteration 1: train=0.4966, val=0.7845, samples=2000/500, stage=n/a
