# AutoML Session Handoff (autoresearch)

## Latest Briefing (Iteration 10)

```text
--- Briefing (Iteration 10) ---
Decision on last proposal: ROLLED_BACK
Clean skill (mean CES_TI/VT skill_vs_persistence): 0.1140 (best: 0.2075)
  CES_TI: skill=0.2710, R2_vs_mean=0.0949, RMSE=527.9168 (n=37636)
  CES_VT: skill=0.3171, R2_vs_mean=0.7784, RMSE=30.1258 (n=31721)
Stale rounds (no new best): 5
Propose ONE controlled architecture change that builds on the current best model.py and raises the clean skill. Preserve the data/model contract; avoid known failed paths.
```

- Best clean skill (mean skill_vs_persistence): 0.2075
- Stale rounds: 5
- Best model snapshot: `data\.improve_out\.automl_state\best_model.py`

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

- Iter 1: baseline, skill=0.1792, best=0.1792, val_loss=0.3884, stage=ok
- Iter 2: kept, skill=0.1835, best=0.1835, val_loss=0.4022, stage=ok
- Iter 3: kept, skill=0.1867, best=0.1867, val_loss=0.3903, stage=ok
- Iter 4: kept, skill=0.1927, best=0.1927, val_loss=0.3769, stage=ok
- Iter 5: kept, skill=0.2075, best=0.2075, val_loss=0.3792, stage=ok
- Iter 6: rolled_back, skill=0.1843, best=0.2075, val_loss=0.3856, stage=ok
- Iter 7: rolled_back, skill=0.1849, best=0.2075, val_loss=0.3943, stage=ok
- Iter 8: rolled_back, skill=0.2001, best=0.2075, val_loss=0.3780, stage=ok
- Iter 9: rolled_back, skill=0.2008, best=0.2075, val_loss=0.3773, stage=ok
- Iter 10: rolled_back, skill=0.1140, best=0.2075, val_loss=0.4305, stage=ok
