# AutoML Session Handoff (autoresearch)

## Latest Briefing (Iteration 20)

```text
--- Briefing (Iteration 20) ---
Decision on last proposal: FAILED
Clean skill (mean CES_TI/VT skill_vs_persistence): n/a (best: 0.2541)
Stale rounds (no new best): 16
FAILURE at training: Command '['python', 'C:\\Users\\lss\\Documents\\GitHub\\ml-intern-revision\\ces_prediction\\train.py']' returned non-zero exit status 1.
Your previous change was discarded and model.py restored to the best version. Propose a DIFFERENT controlled change that avoids this failure.
```

- Best clean skill (mean skill_vs_persistence): 0.2541
- Stale rounds: 16
- Best model snapshot: `data\.automl_sub1m_run\.automl_state\best_model.py`

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

- Iter 1: baseline, skill=0.2526, best=0.2526, val_loss=0.3893, stage=ok
- Iter 2: rolled_back, skill=0.2216, best=0.2526, val_loss=0.4019, stage=ok
- Iter 3: rolled_back, skill=0.2418, best=0.2526, val_loss=0.3914, stage=ok
- Iter 4: kept, skill=0.2541, best=0.2541, val_loss=0.3892, stage=ok
- Iter 5: rolled_back, skill=0.2432, best=0.2541, val_loss=0.3895, stage=ok
- Iter 6: rolled_back, skill=0.2467, best=0.2541, val_loss=0.3901, stage=ok
- Iter 7: rolled_back, skill=0.2499, best=0.2541, val_loss=0.3898, stage=ok
- Iter 8: rolled_back, skill=0.2289, best=0.2541, val_loss=0.3906, stage=ok
- Iter 9: rolled_back, skill=0.2496, best=0.2541, val_loss=0.3883, stage=ok
- Iter 10: rolled_back, skill=0.2410, best=0.2541, val_loss=0.3874, stage=ok
- Iter 11: rolled_back, skill=0.1107, best=0.2541, val_loss=0.3951, stage=ok
- Iter 12: rolled_back, skill=0.1005, best=0.2541, val_loss=0.3945, stage=ok
- Iter 13: rolled_back, skill=0.2121, best=0.2541, val_loss=0.4042, stage=ok
- Iter 14: rolled_back, skill=0.1241, best=0.2541, val_loss=0.3979, stage=ok
- Iter 15: rolled_back, skill=0.2301, best=0.2541, val_loss=0.3975, stage=ok
- Iter 16: rolled_back, skill=0.1275, best=0.2541, val_loss=0.3975, stage=ok
- Iter 17: rolled_back, skill=0.2040, best=0.2541, val_loss=0.4032, stage=ok
- Iter 18: rolled_back, skill=0.1446, best=0.2541, val_loss=0.3864, stage=ok
- Iter 19: rolled_back, skill=0.1925, best=0.2541, val_loss=0.3838, stage=ok
- Iter 20: failed, skill=n/a, best=0.2541, val_loss=inf, stage=training
