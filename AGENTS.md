# Agent Notes

Before changing training code, model architecture, data handling, or experiment strategy, read `PROJECT_KNOWLEDGE.md` and `HANDOFF.md`.

Key rule: avoid uncontrolled architecture churn. Preserve the documented data/model contract, freeze a baseline, and compare one experiment change at a time.

## Scope And Permissions

Agents may refactor files inside the `ml-intern-revision` repository when the change supports the KSTAR CES prediction workflow. Do not edit files outside this repository unless the user explicitly asks.

Dead code, broken documentation, stale generated artifacts, and dependencies that no longer support the active streamline may be removed. Preserve user-created uncommitted changes and do not revert them without explicit permission.

## Active Streamline

The current executable workflows are:

```text
data/*.csv
  -> ces_prediction.dataset.KSTAR_CES_Dataset
  -> fixed file-level train/validation split in data/splits/
  -> train-file-only normalization stats
  -> ces_prediction.model.MultimodalCESPredictor
  -> ces_prediction.train
  -> ces_prediction/metrics.json and weights/

ces_prediction/automl_agent_loop.py
  -> validate required Slack notification setup
  -> smoke validation
  -> full training evaluation
  -> HANDOFF.md update
  -> model rewrite only after plateau
  -> code repair only after smoke/training failure
```

The old generic `ml-intern` CLI/web runtime is not the active path. `automl_agent_loop.py` is active, but uncontrolled model rewriting after every iteration is not allowed. Smoke or training failure is a repair signal, not architecture-quality evidence.

Slack notifications are required for `automl_agent_loop.py`. If `slack_sdk`, `SLACK_BOT_TOKEN`, or `SLACK_CHANNEL_ID` is missing, fail before training starts.

## Data And Model Contract

Every generated or edited model/training/data change must preserve this contract unless the user explicitly approves a contract-changing experiment:

- `model.forward` accepts `forward(self, bes, ecei, mc, time_features=None, ces_history=None)`.
- Outputs are normalized `[CES_TI, CES_VT]` with shape `(batch, 2)`.
- Do not denormalize inside `model.py`.
- BES, ECEI, MC, and targets use train-file-only per-channel z-score normalization.
- `ces_history` has shape `(batch, window, 3)` with previous normalized `CES_TI`, previous normalized `CES_VT`, and observed mask.
- The target timestep in `ces_history` remains masked as `[0, 0, 0]`.
- Time features use 4 channels: lookback seconds, delta seconds, `log1p` lookback, and `log1p` delta.

## Model Architecture Changes

Do not change model architecture just because a single run did not improve.

A model-architecture change is allowed only when all are true:

1. The current baseline has been evaluated with fixed split and documented settings.
2. At least 3 consecutive evaluated runs show plateau behavior.
3. Plateau behavior means relative validation-loss improvement is less than 3% versus the best known validation loss before that run.
4. The proposed change is one controlled variable, not a bundle of architecture, data, loss, and optimizer changes.
5. The proposed change does not repeat a known failed/degraded path from `PROJECT_KNOWLEDGE.md` unless the difference is explicitly justified.

Use this plateau calculation:

```text
relative_improvement = (best_val_loss_before - current_val_loss) / best_val_loss_before
plateau_like_run = relative_improvement < 0.03
```

The 3% threshold is a conservative default. It exists to avoid reacting to small noisy validation movements around the known `0.49-0.53` band. If repeated-seed evidence later shows lower run noise, update this threshold in `PROJECT_KNOWLEDGE.md` and this file together.

## Experiment Logging

Update `HANDOFF.md` after every meaningful training/evaluation run. It should contain latest metrics, settings, status, and the recommended next action.

Update `PROJECT_KNOWLEDGE.md` every 10 runs, or after a major result, with a compact summary of what was learned. This file is long-term memory, not a per-run log.

## Training Run Limits

Before running a full training job, check whether the user expects a smoke test or a full experiment. If unclear, prefer a small smoke test first.

Default smoke-test limits:

```text
CES_EPOCHS=1
CES_MAX_TRAIN_SAMPLES=2000
CES_MAX_VAL_SAMPLES=500
CES_BATCH_SIZE=128
CES_SPLIT_DIR=data/.smoke_splits
CES_OUTPUT_DIR=data/.smoke_outputs
```

Use the checked-in smoke-test command when possible:

```powershell
.\ces_prediction\run_smoke_test.ps1
```

Full experiments may use the repository defaults or user-provided environment variables, but the run settings must be recorded in `HANDOFF.md`.

When running capped AutoML checks with `--train-samples` or `--val-samples`, use separate `--split-dir` and `--output-dir` paths so canonical experiment splits and outputs are not overwritten.

## Verification

After code changes, run:

```bash
python -m pytest -q
```

If training/data/model behavior changed, also run at least a smoke training command and record whether it passed.
