# ML Intern Revision: KSTAR CES Prediction

This revision is focused on **KSTAR CES (Charge Exchange Spectroscopy) multimodal prediction**. The active code path trains a PyTorch model that predicts normalized low-resolution CES targets from higher-resolution BES, ECEI, MC, irregular-time, and previous-CES-history features.

The old generic `ml-intern` CLI/web runtime is not present in this streamlined repository. The current executable paths are the KSTAR CES training/evaluation pipeline and the controlled experiment batches under `ces_prediction/`.

## Quick Start

```bash
git clone https://github.com/iseungsang01/ml-intern-revision.git
cd ml-intern-revision
python -m pip install -e ".[dev]"
python -m pytest -q
.\ces_prediction\run_smoke_test.ps1
python ces_prediction/train.py
python ces_prediction/compare_baselines.py   # model vs interpolation, the thesis comparison
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
|   |-- dataset.py            # CSV loading, window/subset indexing, normalization
|   |-- model.py              # Import surface: re-exports model_iter009 (or CES_MODEL_FILE)
|   |-- model_iter009.py      # THE published architecture -- SHA-256 pinned, do not edit
|   |-- model_iter002.py      # iter2 "before" model of the progression figure -- pinned
|   |-- train.py              # Training, fixed split, metrics, weights
|   |-- evaluate.py           # Clean eval vs persistence/mean
|   |-- compare_baselines.py  # Model vs interpolation (the thesis comparison)
|   |-- bootstrap_compare.py  # Shot-clustered paired bootstrap
|   |-- peak_analysis.py      # Where the model earns its skill
|   |-- collect_paper_numbers.py  # Frozen artifacts -> docs/paper/paper_numbers.json
|   |-- inspect_split.py      # Split validation/manifest helper
|   |-- experiments/          # Controlled batches, one per THESIS_RESULTS.md section 8
|   |                         #   entry, plus runner_common.py / paired_model_compare.py
|   `-- __init__.py
|-- data/                     # gitignored: shot CSVs, splits, run outputs
|-- docs/                     # paper (tex + pdf), presentation decks, reference notes
|-- tests/
|   |-- test_architecture.py
|   |-- test_baselines_interpolation.py
|   |-- test_bootstrap_compare.py
|   `-- test_peak_analysis.py
|-- THESIS_RESULTS.md         # The written-up result + every controlled experiment
|-- PROJECT_KNOWLEDGE.md      # Long-term lessons and failed directions
|-- RESEARCH_CONTEXT.md       # Single-file handoff digest of the whole research
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
- BES, ECEI, MC, and target values use train-file-only per-channel z-score normalization (target stats NaN-aware, over observed values).
- CES_TI and CES_VT are missing independently; a row is kept when inputs are complete and at least one CES target is observed.
- `ces_history` shape is `(batch, window, 4)`: previous normalized `CES_TI`, previous normalized `CES_VT`, `CES_TI` observed flag, `CES_VT` observed flag.
- The target timestep in `ces_history` stays fully masked (both values and both observed flags `0`) to avoid leakage.
- Each sample provides `target_mask` `(batch, 2)`; training/validation use per-target masked MSE.
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
   | - ces_history:   (window, 4)
   | - input_mask:    (window,)
   | - target:        (2,)
   v
DataLoader
```

Temporal subset augmentation is enabled by default. A target row can be paired with multiple irregular subsets of previous rows, while the target row itself remains masked in CES history.

## Model Flow

```text
bes  --+
ecei --+--> time-aware sensor CNN encoders --+
mc   --+                                     |
time ------> time encoder -------------------+--> CES_TI head --> CES_TI
                                             |
ces_history -> GRU -> observation-masked  ---+--> CES_VT head --> CES_VT
               multi-head attention pool ----'
```

A late-fusion multimodal model: each diagnostic stream stays separate until fusion. Two details are
load-bearing rather than decorative:

- **Observation-masked attention pooling.** Each target's history readout can only place weight on
  timesteps where *that* target was actually measured — the inductive bias that makes interpolation
  strong, added at zero parameter cost.
- **Target-aware routing.** `CES_TI` sees the fast diagnostics, history, and time; `CES_VT` sees
  history and time only. The fast channels carry ion-temperature information but not rotation, and
  routing this explicitly is what closes the `V_rot` deficit (THESIS_RESULTS.md §8t).

## Controlled Experiments

Each experiment lives under `ces_prediction/experiments/<name>/` with its own runner, and each is
written up in `THESIS_RESULTS.md` section 8 with its design, result, and verdict.

```text
stuckfree/     held (forward-filled) CES removed from training     -> KEEP  (§8c)
seq/           full-grid sequence reframing (seq / seq_v2)         -> KEEP  (§8d, §8t)
window_sweep/  W in {2,3,4,6,8} x 4 seeds + history-0              -> W=4 not justified (§8f)
largegap/      large gaps vs a CAUSAL baseline                     -> interpolation's territory (§8g)
mnar/          reweighting onto the genuinely-missing points       -> causal claim survives (§8i)
anchor/        1,258-parameter anchor + delta ladder rung          -> recovers ~31.5% of margin (§8k)
latency/       batch-1 inference against the 10 ms CES grid        -> CPU p99 6.4 ms (§8l)
uq/            split conformal intervals, no retraining            -> beats both baselines 8/8 (§8m)
campaign/      strictly temporal (campaign) split                  -> offline claim dies (§8n)
gp/            Gaussian-process arm, the strongest offline opponent-> model TIES it (§8p)
fitfail/       CES spectral-fit failures (T_i > 3 keV)             -> headline is conservative (§8q)
heldpeak/      peak x held crosstab for CES_VT                     -> three regimes (§8r)
pershot/       per-shot input standardization                      -> ADOPT (§8s)
quantum/       IonQ VQC vs a classical MLP                         -> side track, not a thesis claim
```

Plus two shared files at `experiments/` level: `runner_common.py` (frozen split dirs, control run
dirs, pinned training env, `run_step`) and `paired_model_compare.py` (the shot-clustered paired
bootstrap, which refuses to compare arms that did not score the same rows).

The discipline these follow: one controlled variable per experiment, a pre-registered verdict rule,
four independent split seeds, and a shot-clustered paired bootstrap. A single-seed result is not
evidence. An earlier `automl_agent_loop.py` (a keep/discard autoresearch loop) produced the thesis
architecture and was retired on 2026-06-24; recover it from git history if needed. Its archived
output is still the published architecture and now lives in the repo as
`ces_prediction/model_iter009.py` (with the iter2 "before" baseline as
`ces_prediction/model_iter002.py`), pinned by SHA-256 in `tests/test_architecture.py`.
`ces_prediction/model.py` re-exports it, so training and scoring use it by default; a batch that
varies the architecture points `CES_MODEL_FILE` at its own file instead.

A continuous-time batch (`ct/`, four alternative history encoders) was run and rejected; the code
was removed on 2026-08-09 once the paper stopped citing it. The verdict survives in
`THESIS_RESULTS.md` §8e — read it before re-opening the question.

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

### Evaluate (clean, baseline-relative)

```bash
python ces_prediction/evaluate.py
```

Scores the trained model on a **non-augmented** validation set (real most-recent history)
restricted to the manifest's validation shots, in **denormalized physical units per target**,
against **persistence** (reuse the last observed CES) and a **mean** baseline. The headline
`skill_vs_persistence > 0` means the multimodal model beats carrying the last CES forward.
Reads `CES_OUTPUT_DIR/metrics.json`, `CES_OUTPUT_DIR/weights/`, and
`CES_SPLIT_DIR/split_manifest.json`; writes `CES_OUTPUT_DIR/eval_metrics.json`.

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
CES_DROP_STUCK_TARGETS=1   # NaN out held/forward-filled CES targets at load (pin it explicitly)
CES_PER_SHOT_NORM=0        # 1 = z-score BES/ECEI/MC within each shot (targets untouched)
CES_MAX_SAMPLES_PER_FILE=0 # 0 = off; per-shot cap applied before the global caps
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
THESIS_RESULTS.md
  - the written-up result; section 8 records every controlled experiment
  - add a section per experiment: design, result table, verdict

PROJECT_KNOWLEDGE.md
  - long-term memory: lasting lessons, rejected paths, reproducibility traps
  - summarize into it after a major finding -- not a per-run log

CLAUDE.md
  - how to work in this repo: commands, the data/model contract, agreements

RESEARCH_CONTEXT.md
  - single-file handoff digest of the whole research: claim, data, contract,
    results, the experiment ledger, methodology traps, framing rules, open items
  - DERIVED, never the origin of a fact -- the three files above win on conflict
```

## What Was Removed From The Active Path

The following are no longer part of the active streamline:

- Generic `ml-intern` CLI entrypoint and the FastAPI/web frontend runtime.
- Agent/web dependencies unrelated to KSTAR CES training.
- The AutoML autoresearch loop (`automl_agent_loop.py`), its Slack notifier, and `program.md`,
  removed 2026-08-05 after 6 weeks unused -- experiments are now hand-written batches under
  `ces_prediction/experiments/`. Recover from git history if needed.
- Superseded docs: `README2.md`, `AGENTS.md`, `PROGRESS.md`, `HANDOFF.md`, `RESEARCH_PLAN.md`,
  `RESEARCH_SUMMARY.md`, `ML_WORKFLOW_ARCHITECTURE.md` -- their live content now lives in
  `THESIS_RESULTS.md`, `PROJECT_KNOWLEDGE.md`, and `CLAUDE.md`.
