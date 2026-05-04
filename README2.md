# KSTAR CES Prediction Streamline

This repository is now centered on one workflow: train and evaluate a KSTAR CES multimodal predictor from CSV shot files.

Older generic `ml-intern` agent/web documentation is intentionally not part of this streamlined path. The active AutoML entrypoint is `ces_prediction/automl_agent_loop.py`, which must run controlled experiments instead of uncontrolled architecture rewrites.

## Main Flow

```text
data/*.csv
  -> ces_prediction.dataset.KSTAR_CES_Dataset
  -> train-file-only normalization stats
  -> fixed file-level train/validation split in data/splits/
  -> ces_prediction.model.MultimodalCESPredictor
  -> ces_prediction.train training loop
  -> ces_prediction/weights/multimodal_ces.pth
  -> ces_prediction/metrics.json
```

## Important Contract

- `model.forward` must accept `forward(self, bes, ecei, mc, time_features=None, ces_history=None)`.
- Model output is normalized `[CES_TI, CES_VT]` with shape `(batch, 2)`.
- Do not denormalize in `model.py`; inverse transforms belong in reporting or evaluation code.
- BES, ECEI, MC, and target values use train-file-only per-channel z-score normalization.
- `ces_history` has shape `(batch, window, 3)` containing previous normalized `CES_TI`, previous normalized `CES_VT`, and an observed mask.
- The target timestep in `ces_history` must stay masked as `[0, 0, 0]` to avoid leakage.
- Time features have 4 channels: lookback seconds, delta seconds, `log1p` lookback, and `log1p` delta.

## Files That Matter

```text
ces_prediction/
  dataset.py        # CSV loading, sample indexing, temporal subsets, normalization
  model.py          # baseline multimodal CES predictor
  train.py          # fixed split, training loop, metrics and weights output
  automl_agent_loop.py  # main controlled experiment loop
  inspect_split.py  # writes/prints the current fixed split manifest
tests/
  test_architecture.py
data/
  *.csv
  splits/
```

Supporting project memory:

- `PROJECT_KNOWLEDGE.md`: prior results, failed directions, baseline guidance.
- `HANDOFF.md`: latest experiment handoff.
- `AGENTS.md`: rules for future coding agents.

## Setup

Use Python 3.11 or newer.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The core runtime dependencies are `torch`, `numpy`, and `pandas`. `pytest` is only needed for tests.

## Verify

```bash
python -m pytest -q
```

The tests check the dataset/model interface, real CSV sample loading, time feature shape, CES-history target masking, and forward/backward compatibility.

## Smoke Test

```powershell
.\ces_prediction\run_smoke_test.ps1
```

The smoke test runs the pytest suite and then performs a tiny 1-epoch training run with reduced sample caps. Use it to catch shape, DataLoader, loss, metrics, and split-path breakages before running a full experiment. Do not use smoke-test loss as model-quality evidence.

The script writes temporary split files under `data/.smoke_splits` and temporary metrics/weights under `data/.smoke_outputs`, so it does not overwrite canonical experiment files.

## Train

```bash
python ces_prediction/train.py
```

## Controlled AutoML Loop

```bash
python ces_prediction/automl_agent_loop.py --max-iterations 300
```

Each iteration runs smoke validation before full training. `model.py` is rewritten only when the plateau rule in `AGENTS.md` is met, or when the loop needs to fix a smoke/training failure.

Slack notifications are required for this loop. Missing `slack_sdk`, `SLACK_BOT_TOKEN`, or `SLACK_CHANNEL_ID` stops the run before training starts.

For a quick development check, keep temporary splits and outputs separate:

```bash
python ces_prediction/automl_agent_loop.py --max-iterations 1 --train-samples 2000 --val-samples 500 --split-dir data/.automl_dev_splits --output-dir data/.automl_dev_outputs
```

Default environment variables:

```text
CES_WINDOW_SIZE=4
CES_BATCH_SIZE=512
CES_EPOCHS=10
CES_LR=1e-3
CES_SEED=42
CES_VAL_FRACTION=0.2
CES_MAX_TRAIN_SAMPLES=200000
CES_MAX_VAL_SAMPLES=40000
CES_TEMPORAL_SUBSETS=1
CES_MIN_SUBSET_SIZE=2
CES_CPU_WORKERS=<detected CPU count>
CES_DATALOADER_WORKERS=0
CES_TORCH_THREADS=<derived from CPU budget, capped at 16>
CES_TORCH_INTEROP_THREADS=<derived from torch threads>
```

Outputs:

```text
data/splits/fixed_train_split.csv
data/splits/fixed_val_split.csv
data/splits/split_manifest.json
ces_prediction/weights/multimodal_ces.pth
ces_prediction/metrics.json
```

## Inspect The Split

```bash
python ces_prediction/inspect_split.py
```

This recreates or validates the fixed train/validation split and writes `data/splits/split_manifest.json`.

## Experiment Discipline

Before changing training code, model architecture, data handling, or experiment strategy, read:

```text
PROJECT_KNOWLEDGE.md
HANDOFF.md
```

Recommended order:

1. Preserve the current baseline.
2. Reproduce the baseline with fixed seed and saved best checkpoint.
3. Change only one variable per experiment.
4. Track validation loss and per-target TI/VT error.
5. Avoid repeating known degraded paths: uncontrolled architecture rewrites, wider/deeper scaling without a controlled reason, complex skip variants, and local temporal conv additions that duplicate prior failed attempts.
