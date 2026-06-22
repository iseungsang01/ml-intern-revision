# AutoML Session Handoff (autoresearch)

## Latest Briefing (Iteration 1)

```text
--- Briefing (Iteration 1) ---
Decision on last proposal: BASELINE
Clean skill (mean CES_TI/VT skill_vs_persistence): -2.6869 (best: -2.6869)
  CES_TI: skill=0.3366, R2_vs_mean=0.0022, RMSE=530.6318 (n=917)
  CES_VT: skill=-5.7105, R2_vs_mean=0.0176, RMSE=49.6692 (n=795)
Stale rounds (no new best): 0
Propose ONE controlled architecture change that builds on the current best model.py and raises the clean skill. Preserve the data/model contract; avoid known failed paths.
```

- Best clean skill (mean skill_vs_persistence): -2.6869
- Stale rounds: 0
- Best model snapshot: `C:\Users\lss\Documents\GitHub\ml-intern-revision\ces_prediction\.automl_state\best_model.py`

## Data Contract

```text
Dataset/training contract that every generated model.py must preserve:
- train.py builds KSTAR_CES_Dataset with temporal subset augmentation.
- BES, ECEI, and MC inputs are per-channel z-score normalized using train-file-only statistics.
- CES_TI and CES_VT are per-channel z-score normalized with train-file-only target statistics (NaN-aware, over observed values only).
- CES_TI and CES_VT are missing independently; rows are kept when inputs are complete and at least one CES target is observed.
- ces_history has shape (batch, window, 4): normalized previous CES_TI, normalized previous CES_VT, CES_TI observed flag, CES_VT observed flag.
- The target timestep is fully masked in ces_history (both values and both observed flags set to 0) to avoid leakage.
- Each batch provides target_mask of shape (batch, 2); train.py uses per-target masked MSE so a row with only one observed CES target still supervises that target.
- model.forward must accept forward(self, bes, ecei, mc, time_features=None, ces_history=None).
- Model outputs must remain normalized CES_TI/CES_VT with shape (batch, 2); train.py compares them to normalized targets.
- Do not denormalize inside model.py. Any inverse transform belongs outside training/evaluation.
```

## History

- Iter 1: baseline, skill=-2.6869, best=-2.6869, val_loss=0.9316, stage=ok
