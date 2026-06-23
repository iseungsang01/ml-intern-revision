# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git / commit policy (do not auto-commit)

Never create git commits or pushes automatically. Do **not** run `git commit` / `git push`, and do
**not** invoke any OMC commit step (the `team` / `autopilot` / `ralph` commit protocol, or
`git-master` auto-commit) unless the user explicitly asks to commit in that same message. Making and
verifying file changes is fine; turning them into commits requires an explicit per-request go-ahead.

## What this repo is

A single PyTorch pipeline that predicts normalized low-resolution KSTAR CES (Charge
Exchange Spectroscopy) targets `[CES_TI, CES_VT]` from higher-resolution BES, ECEI, MC
diagnostics plus irregular-time features and previous-CES history. Everything executable
lives under `ces_prediction/`. The old generic `ml-intern` CLI/FastAPI runtime is gone —
do not reintroduce it.

## Commands

```bash
python -m pip install -e ".[dev]"   # install (requires-python >=3.11)
python -m pytest -q                  # tests (tests/test_architecture.py)
.\ces_prediction\run_smoke_test.ps1  # pytest + 1-epoch tiny train, end-to-end (PowerShell)
python ces_prediction/train.py       # full training run
python ces_prediction/evaluate.py    # clean eval: model vs persistence/mean, physical units, per target
python ces_prediction/inspect_split.py        # validate fixed split / manifest
python ces_prediction/automl_agent_loop.py --max-iterations 300   # controlled experiment loop
```

Run a single test: `python -m pytest tests/test_architecture.py::test_dry_run -q`

**Always run scripts from the repo root.** `train.py`, `model.py`, and `automl_agent_loop.py`
import sibling modules by bare name (`from dataset import ...`, `from slack_notifier import ...`),
relying on the script's own directory being on `sys.path`. There is no `ces_prediction.*`
package import path in the runtime code (`tests/` patches `sys.path` manually).

**You need real data to do almost anything.** The shot CSVs are **not** committed (`data/*`
is gitignored). Point `CES_DATA_DIR` at the real folder (the thesis `data/`, e.g.
`C:\Users\lss\Desktop\원핵 졸논\data`) or copy the CSVs into `data/`. Of the three tests,
only `test_dry_run` runs without data — `test_real_csv_sample_forward` and
`test_temporal_subset_sample_uses_previous_ces_only` load real CSVs and fail without
`CES_DATA_DIR` pointing at them. `train.py`, `evaluate.py`, and the smoke script all need it too.
The local interpreter is invoked as `py` (Python 3.14); the commands above also work as `python`
where a 3.11+ `python` is on PATH.

## The data/model contract (do not break)

This is the project's central invariant. `model.py`, `train.py`, and `dataset.py` all depend
on it, and `tests/test_architecture.py` asserts parts of it. Changing it requires explicit
user approval as a controlled experiment.

- `model.forward(self, bes, ecei, mc, time_features=None, ces_history=None)` — signature is fixed.
- Output is normalized `[CES_TI, CES_VT]`, shape `(batch, 2)`. **Never denormalize inside `model.py`.**
- BES, ECEI, MC, and the target are per-channel z-scored using **train-file-only** statistics
  (`fit_normalization_stats` on the train file set, then `set_normalization_stats`). Target stats
  are **NaN-aware** (over observed CES values only), since CES is sparse.
- `ces_history` is `(batch, window, 4)`: previous normalized `CES_TI`, previous normalized
  `CES_VT`, `CES_TI` observed flag, `CES_VT` observed flag. `CES_TI` and `CES_VT` go missing
  **independently** (≈8% / ≈24% of rows), so observation is tracked per target. The **target
  timestep is fully masked (both values and both flags → 0)** to prevent leakage.
- Each sample carries **`target_mask` `(batch, 2)`**; train/val use **per-target masked MSE**, so a
  row with only one observed CES target still supervises it. A row is kept when its inputs are
  complete and **at least one** of `CES_TI`/`CES_VT` is observed (the old code required both,
  silently discarding ≈28% of labeled rows).
- Time features are 4 channels: lookback seconds, delta seconds, `log1p` lookback, `log1p` delta.

`MultimodalCESPredictor.from_dataset(dataset, ...)` reads channel counts from
`dataset.feature_dims`, so feature-dim changes flow through automatically — but the contract
shapes above are still hard-coded expectations in tests and `train.py`.

## Architecture flow

`KSTAR_CES_Dataset` (`dataset.py`) → fixed file-level split → train-file-only normalization →
`MultimodalCESPredictor` (`model.py`) → training loop (`train.py`) → `metrics.json` + `weights/`.

Key non-obvious behaviors:

- **File-level split, not row-level.** `train.py` splits by whole CSV shot file (seeded) to
  avoid train/val leakage across adjacent rows, then caps each side to a seeded random subset.
- **Fixed splits are pinned to disk** at `data/splits/fixed_{train,val}_split.csv`. On reload,
  `load_fixed_split_csv` re-derives each sample's file/row and **raises if the dataset no longer
  matches** — delete the split CSVs to regenerate. Changing sample caps also regenerates them.
- **Dataset disk cache** lives at `data/.ces_cache/*.npz`, keyed by a hash of window size,
  augmentation flags, columns, and file size/mtime. It auto-invalidates when those change.
- **Temporal subset augmentation** (`CES_TEMPORAL_SUBSETS=1`, default on) enumerates
  `combinations` of previous rows within contiguous time blocks — this is combinatorially
  explosive, which is why sample counts are capped (`CES_MAX_TRAIN_SAMPLES` etc.). Contiguous
  blocks are detected by `time` delta `< 0.5`.
- **Feature columns are inferred** from CSV headers: `BES_*`, `ECEI_*`, `MC*` prefixes plus
  required `time`, `CES_TI`, `CES_VT`.
- Training adds a soft penalty discouraging predictions below normalized-zero `CES_TI`
  (`0.1 * relu(...)`) on top of MSE, plus grad clipping and `ReduceLROnPlateau`.

`train.py` is configured entirely through `CES_*` environment variables (window size, batch,
epochs, LR, seed, sample caps, CPU/threading, `CES_SPLIT_DIR`, `CES_OUTPUT_DIR`, `CES_DATA_DIR`).
See the README for the full list. Use separate `CES_SPLIT_DIR`/`CES_OUTPUT_DIR` for throwaway runs
so canonical splits/metrics/weights are not overwritten (the smoke script uses `data/.smoke_*`).
The shot CSVs are **not** committed (`data/*` is gitignored); point `CES_DATA_DIR` at their real
location (e.g. the thesis `data/` folder) or copy them into `data/`.

## AutoML loop and experiment discipline

`automl_agent_loop.py` is a **keep/discard autoresearch loop** (inspired by Karpathy's
`autoresearch`). Each iteration: smoke-validate → full `train.py` → clean `evaluate.py` → score the
model on the **clean mean `skill_vs_persistence`** (not the augmented val loss) → **keep** the change
if it beats the best so far, else **discard and roll back `model.py`** to the best snapshot. So the
loop never builds on a regression and never loses the best architecture.

- **Researcher = Claude.** The model rewrite uses the official `anthropic` SDK (`claude-opus-4-8`,
  adaptive thinking, streaming). Needs `ANTHROPIC_API_KEY`; a missing key or SDK skips the rewrite
  and the loop keeps running. Override the model with `AUTOML_RESEARCH_MODEL`.
- **`program.md`** (repo root) is the editable agent "skill" — objective, keep/discard rules,
  discipline, and physics priors. Humans tune `program.md`; the loop injects it plus
  `PROJECT_KNOWLEDGE.md` and the hard `DATA_CONTRACT` into the prompt. Output must be raw `model.py`
  preserving the contract (validated: must contain `class MultimodalCESPredictor` and `def forward`).
- **Slack is mandatory.** Missing `slack_sdk`, `SLACK_BOT_TOKEN`, or `SLACK_CHANNEL_ID` makes the
  loop fail before training starts (`validate_slack_config`).
- **State** (best snapshot `best_model.py` + per-iteration archive) lives under
  `<output_dir>/.automl_state/` (default `ces_prediction/.automl_state/`). The loop stops early after
  `--max-consecutive-failures` (default 5).
- Smoke/training failure is a **repair** signal, not architecture-quality evidence: the failed
  proposal is discarded (rolled back) and the agent is asked for a different controlled change.

## Working agreements (from AGENTS.md)

- Read `PROJECT_KNOWLEDGE.md` (long-term lessons, failed paths) and `HANDOFF.md` (latest
  run state) before changing training/model/data code.
- Change **one controlled variable at a time**; don't bundle architecture + data + loss +
  optimizer changes.
- **Update `HANDOFF.md` after every meaningful run** (latest metrics, settings, next action).
  Summarize into `PROJECT_KNOWLEDGE.md` every ~10 runs or after a major finding — it's long-term
  memory, not a per-run log. `automl_agent_loop.py` writes `HANDOFF.md` automatically each run.
- After code changes run `python -m pytest -q`; if training/data/model behavior changed, also
  run a smoke training command and record whether it passed.
