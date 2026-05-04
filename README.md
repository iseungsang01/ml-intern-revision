# ML Intern Revision: KSTAR CES Prediction

This revision is focused on **KSTAR CES (Charge Exchange Spectroscopy) multimodal prediction**. The active code path trains a PyTorch model that predicts normalized low-resolution CES targets from higher-resolution BES, ECEI, MC, irregular-time, and previous-CES-history features.

The old generic `ml-intern` CLI/web runtime is not present in this streamlined repository. The current executable paths are the KSTAR CES training pipeline and the controlled AutoML loop under `ces_prediction/`.

## Quick Start

```bash
git clone https://github.com/iseungsang01/ml-intern-revision.git
cd ml-intern-revision
python -m pip install -e ".[dev]"
python -m pytest -q
.\ces_prediction\run_smoke_test.ps1
python ces_prediction/train.py
python ces_prediction/automl_agent_loop.py --max-iterations 300
```

Before running `automl_agent_loop.py`, set Slack notification variables. The loop intentionally fails at startup if Slack is not configured.

```text
SLACK_BOT_TOKEN=<your-slack-bot-token>
SLACK_CHANNEL_ID=<target-channel-id>
```

Training writes:

```text
data/splits/fixed_train_split.csv
data/splits/fixed_val_split.csv
data/splits/split_manifest.json
ces_prediction/weights/multimodal_ces.pth
ces_prediction/metrics.json
```

## Current Repository Map

```text
ml-intern-revision/
|-- ces_prediction/
|   |-- dataset.py        # CSV loading, window/subset indexing, normalization
|   |-- model.py          # MultimodalCESPredictor baseline
|   |-- train.py          # Training, fixed split, metrics, weights
|   |-- automl_agent_loop.py  # Main controlled experiment loop
|   |-- inspect_split.py  # Split validation/manifest helper
|   `-- __init__.py
|-- data/
|   |-- *.csv
|   `-- splits/
|-- tests/
|   `-- test_architecture.py
|-- PROJECT_KNOWLEDGE.md  # Long-term lessons and failed directions
|-- HANDOFF.md            # Latest experiment/result handoff
|-- AGENTS.md             # Rules for future coding agents
|-- README2.md            # Streamline-focused workflow doc
`-- pyproject.toml
```

## Main Training Streamline

```text
Raw KSTAR shot CSVs
        |
        v
+--------------------------+
| KSTAR_CES_Dataset        |
| - infer BES/ECEI/MC cols |
| - build target windows   |
| - temporal subsets       |
| - time features          |
| - CES history masking    |
+--------------------------+
        |
        v
+--------------------------+
| Fixed file-level split   |
| data/splits/*.csv        |
| avoids row leakage       |
+--------------------------+
        |
        v
+--------------------------+
| Train-file-only stats    |
| BES/ECEI/MC/target       |
| per-channel z-score      |
+--------------------------+
        |
        v
+--------------------------+
| MultimodalCESPredictor   |
| BES branch               |
| ECEI branch              |
| MC branch                |
| time branch              |
| fusion MLP               |
+--------------------------+
        |
        v
+--------------------------+
| train.py                 |
| MSE + negative-TI guard  |
| AdamW + LR scheduler     |
| gradient clipping        |
+--------------------------+
        |
        v
+--------------------------+
| metrics.json             |
| weights/*.pth            |
+--------------------------+
```

## Data And Model Contract

Every training/data/model change must preserve this contract unless the change is explicitly approved as a controlled experiment:

- `model.forward` accepts `forward(self, bes, ecei, mc, time_features=None, ces_history=None)`.
- Output is normalized `[CES_TI, CES_VT]` with shape `(batch, 2)`.
- `model.py` must not denormalize predictions.
- BES, ECEI, MC, and target values use train-file-only per-channel z-score normalization.
- `ces_history` shape is `(batch, window, 3)`: previous normalized `CES_TI`, previous normalized `CES_VT`, observed mask.
- The target timestep in `ces_history` stays masked as `[0, 0, 0]` to avoid leakage.
- Time features have 4 channels: lookback seconds, delta seconds, `log1p` lookback, `log1p` delta.

## Dataset Flow

```text
data/*.csv
   |
   | required columns:
   | - time
   | - CES_TI, CES_VT
   | - BES_* columns
   | - ECEI_* columns
   | - MC* columns
   v
KSTAR_CES_Dataset
   |
   | sample output:
   | - bes:           (window, 9)
   | - ecei:          (window, 4)
   | - mc:            (window, 2)
   | - time_features: (window, 4)
   | - ces_history:   (window, 3)
   | - input_mask:    (window,)
   | - target:        (2,)
   v
DataLoader
```

Temporal subset augmentation is enabled by default. A target row can be paired with multiple irregular subsets of previous rows, while the target row itself remains masked in CES history.

## Model Flow

```text
bes + time + ces_history  ---> BES encoder  ----+
ecei + time + ces_history ---> ECEI encoder ----+--> concat --> fusion MLP --> [CES_TI, CES_VT]
mc + time + ces_history   ---> MC encoder   ----+
time only                 ---> time encoder ----+
```

The current baseline is a late-fusion multimodal CNN-style model. It keeps each diagnostic stream separate until fusion, then predicts the two normalized CES targets.

## Controlled AutoML Loop

`ces_prediction/automl_agent_loop.py` is the main experiment orchestrator. It keeps the old Evaluation/Briefing/Research roles, but it no longer rewrites `model.py` after every successful evaluation. It first runs smoke validation, then full training, then decides whether a model update is allowed.

```text
Evaluation role
   |
   | run smoke validation
   | run full train.py
   | collect metrics
   v
Briefing role
   |
   | update HANDOFF.md every run
   | detect plateau or regression
   | summarize every 10 runs into PROJECT_KNOWLEDGE.md
   v
Research role
   |
   | propose one controlled change
   | only change model architecture after plateau rules are met
   | preserve data/model contract
   v
Implementation
   |
   | edit scoped files
   | run tests
   | compare against baseline
   v
Next run
```

Run the main loop:

```bash
python ces_prediction/automl_agent_loop.py --max-iterations 300
```

Slack is required for this loop. Missing `slack_sdk`, `SLACK_BOT_TOKEN`, or `SLACK_CHANNEL_ID` stops the run before training starts.

Useful development run:

```bash
python ces_prediction/automl_agent_loop.py --max-iterations 1 --train-samples 2000 --val-samples 500 --split-dir data/.automl_dev_splits --output-dir data/.automl_dev_outputs
```

### Plateau Rule

A model-architecture change is allowed only after at least **3 consecutive evaluated runs** fail to improve the best validation loss by a meaningful margin.

Recommended margin: **relative improvement below 3%** counts as no meaningful improvement.

```text
relative_improvement = (best_val_loss_before - current_val_loss) / best_val_loss_before

if relative_improvement < 0.03:
    plateau_like_run += 1
else:
    plateau_like_run = 0
```

Why 3%: the earlier experiment history stayed around `0.49-0.53`, so tiny changes around 1% can easily be run noise. A 3% threshold is strict enough to avoid architecture churn, but not so strict that real progress is ignored. If future repeated-seed runs show lower noise, this threshold can be tightened.

## Commands

### Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### Test

```bash
python -m pytest -q
```

### Smoke Test

Use this after code changes that touch data loading, model inputs, training, metrics, or packaging. It verifies that tests pass and a tiny training run completes end to end. It is not a performance benchmark.

```powershell
.\ces_prediction\run_smoke_test.ps1
```

### Train

```bash
python ces_prediction/train.py
```

### Inspect Split

```bash
python ces_prediction/inspect_split.py
```

## Training Configuration

`train.py` is configured through environment variables:

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

The smoke-test script is equivalent to lowering the sample caps before training:

```powershell
$env:CES_MAX_TRAIN_SAMPLES="2000"
$env:CES_MAX_VAL_SAMPLES="500"
$env:CES_EPOCHS="1"
$env:CES_BATCH_SIZE="128"
$env:CES_SPLIT_DIR="data\.smoke_splits"
$env:CES_OUTPUT_DIR="data\.smoke_outputs"
python ces_prediction/train.py
```

The separate `CES_SPLIT_DIR` and `CES_OUTPUT_DIR` keep smoke-test splits, metrics, and weights from overwriting canonical experiment files.

## Documentation Responsibilities

```text
HANDOFF.md
  - update after every meaningful training/evaluation run
  - include latest metrics, settings, and next action

PROJECT_KNOWLEDGE.md
  - update with a compact summary every 10 runs or after a major finding
  - record failed paths so they are not repeated

AGENTS.md
  - defines what future coding agents may change
  - defines plateau and experiment discipline rules
```

## What Was Removed From The Active Path

The following old concepts are no longer part of the active streamline:

- Generic `ml-intern` CLI entrypoint.
- FastAPI/web frontend runtime.
- Slack notification scripts.
- Uncontrolled `model.py` rewriting after every iteration.
- Agent/web dependencies unrelated to KSTAR CES training.

The project still uses `automl_agent_loop.py`, but the loop must follow `AGENTS.md`, preserve the data/model contract, run smoke validation before full training, and compare one experiment change at a time.
