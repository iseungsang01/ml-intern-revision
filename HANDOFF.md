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
