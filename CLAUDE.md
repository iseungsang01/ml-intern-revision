# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The research code and the manuscript of one study: **causal nowcasting of the KSTAR charge-exchange
(CES) ion temperature on a pedestal-top chord** from its own history and the fast diagnostics at the
same location (BES, ECEI), 2022 campaign. Three layers:

- the core PyTorch pipeline under `ces_prediction/` (`dataset.py`, `model.py`, `train.py`,
  `evaluate.py`, `baselines_interpolation.py`): the data contract below and the windowed model;
- the registered experiment batches under `ces_prediction/experiments/` (one directory per batch);
- the manuscript in PPCF format under `docs/paper/ppcf/`, whose numbers are generated from the result
  files of the batches.

`README.md` holds the wiring (what reads what), `HANDOVER.md` the handover (environment, data and
caches, the procedure of a batch, the manuscript build, open work), `AGENTS.md` the short form for other
coding agents. The old generic `ml-intern` CLI/FastAPI runtime is gone; do not reintroduce it. Earlier
lines of this research were removed on 2026-11-21 (git history, last commit before `e86adaf`); do not
bring them back.

## Commands

```bash
python -m pip install -e ".[dev]"   # install (requires-python >=3.11); the batches also need scikit-learn and matplotlib
python -m pytest -q                  # tests
.\ces_prediction\run_smoke_test.ps1  # pytest + 1-epoch tiny train, end-to-end (PowerShell)
python ces_prediction/train.py       # training run of the core pipeline (windowed model)
python ces_prediction/evaluate.py    # clean eval: model vs persistence/mean, physical units, per target
python ces_prediction/inspect_split.py          # validate fixed split / manifest
python ces_prediction/analyze_file_rules.py     # how the 10 ms files relate to the raw 1 us deliveries (grid, snapshots, beam blips)
python ces_prediction/analyze_nbi_files.py      # what the NBI files (data/nbi/) are: rows, sources, notches, CES values by source state
python ces_prediction/check_records.py          # numbers of PROJECT_KNOWLEDGE.md and PAPER_DESIGN.md against their sources
python docs/paper/ppcf/make_tables_v2.py        # manuscript tables and number macros from the result json files (gen_v2.tex)
python docs/paper/ppcf/fig_v2.py                # manuscript figures (figv2_*)
python docs/paper/ppcf/check_numbers_v2.py      # numbers typed in main_ppcf.tex / supp_ppcf.tex against their sources
```

Run a single test: `python -m pytest tests/test_architecture.py::test_dry_run -q`

**Always run scripts from the repo root.** `train.py`, `evaluate.py`, `compare_baselines.py` and
the experiment runners import sibling modules by bare name (`from dataset import ...`), relying on
the script's own directory being on `sys.path`. There is no `ces_prediction.*` package import path
in the runtime code (`tests/` and the runners patch `sys.path` manually).

**You need real data to do almost anything.** The shot CSVs are **not** committed (`data/*`
is gitignored). Point `CES_DATA_DIR` at the real folder or copy the CSVs into `data/` (layout in
`HANDOVER.md` section 4). Tests that load real CSVs fail without them; `test_dry_run` runs without
data. The local interpreter is invoked as `py` (Python 3.14); the commands above also work as
`python` where a 3.11+ `python` is on PATH.

Experiment batches live under `ces_prediction/experiments/`, one directory per batch, each behind a
registration `PREREGISTRATION_B{NN}.md`, plus two shared files (`runner_common.py`,
`paired_model_compare.py`). Read `ces_prediction/experiments/README.md` and `HANDOVER.md` section 5
before adding one.

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
- **What a CSV row is** (`docs/paper/file_rules.md`): the first row (t = 0) and the last (t = 30 s)
  of every file carry no CES value and are not grid rows — drop them in any tally over rows. The
  fast channels of a row are single raw samples at the time stamp; the CES value is the frame whose
  10 ms exposure starts at the stamp. A time step of 20-40 ms inside a discharge is a beam blip
  (the frame without beam is absent); `T_i` and the ECEI level stay low for about 90 ms after it.
- Training adds a soft penalty discouraging predictions below normalized-zero `CES_TI`
  (`0.1 * relu(...)`) on top of MSE, plus grad clipping and `ReduceLROnPlateau`.

`train.py` is configured entirely through `CES_*` environment variables (window size, batch,
epochs, LR, seed, sample caps, CPU/threading, `CES_SPLIT_DIR`, `CES_OUTPUT_DIR`, `CES_DATA_DIR`).
See the README for the full list. Use separate `CES_SPLIT_DIR`/`CES_OUTPUT_DIR` for throwaway runs
so canonical splits/metrics/weights are not overwritten (the smoke script uses `data/.smoke_*`).
The shot CSVs are **not** committed (`data/*` is gitignored); point `CES_DATA_DIR` at their real
location (e.g. the thesis `data/` folder) or copy them into `data/`.

## Estimators and architecture files

- **The estimator of the paper** is not a network: a causal Gaussian process on the past CES values
  followed by a boosted regression on scalars of that history and of the fast channels
  (`experiments/b73_targets/stage_trees.py` for the nowcast condition,
  `experiments/b75_availability/stage_fits.py` for a late or sparse CES history). Definitions are fixed
  in `PAPER_DESIGN.md` section 5.
- **The recurrent network** `experiments/seq/model_seq_v2.py` (a full-grid two-branch causal LSTM,
  trained by `experiments/seq/train_seq.py`, scored by `experiments/seq/eval_seq.py`) and its
  two-stage form are comparison rows of the paper. Their stored trainings are re-scored on the data
  definition D1 by `experiments/b70_d1/`.
- **The windowed model** of the core pipeline: `ces_prediction/model.py` re-exports
  `model_iter009.py`. `model_iter009.py` and `model_iter002.py` are frozen architecture files pinned by
  SHA-256 in `tests/test_architecture.py` — **do not edit them**. The windowed model is not in the
  paper; it is kept for the data contract and its tests.

A controlled experiment that varies the architecture of the core pipeline sets **`CES_MODEL_FILE`**
to its own `.py` file in the subprocess env, like every other `CES_*` knob. The file must define
`MultimodalCESPredictor` with the contract signature. No runner writes to tracked source; never copy
a variant over `model.py`.

The defaults of `train.py` (W = 4, held values kept, no cut) are **not** the data definition of the
paper. Every run pins its `CES_*` variables explicitly; the reference env of the network trainings is
in `experiments/b64_two_stage/run_b64.py` and `experiments/b39_primary/run_b39.py`.

## Working agreements

- **Author preference (2026-11-23):** substantive design/interpretation/scouting/code-review/research-prioritization tasks should use the official Claude API by default, with Fable for economical work and Opus for adversarial synthesis. Redacted explicit evidence tasks only; no secrets, identity, raw observations or private reasoning. Anonymized aggregate evaluated results are authorized for post-evaluation planning. Preserve every paid attempt/usage, independently check facts, and bound tool rounds. Run the committed post-evaluation workflow only after the original final audit and normal completion; TEST-informed follow-ups are exploratory. A Claude proposal does not authorize bypassing preregistration or editing frozen running code.

- **The records.** `README.md` is the wiring of the repository. `HANDOVER.md` is the handover.
  `PAPER_DESIGN.md` fixes the purpose, terms, data definition, targets, estimators, validation
  designs, wording rules and the paper map. `PROJECT_KNOWLEDGE.md` holds the data facts and the
  results in use, one subsection per batch. `THESIS_RESULTS.md` is the chronological log (from B.70).
  Read `PAPER_DESIGN.md` and `PROJECT_KNOWLEDGE.md` before changing training/model/data code or
  writing paper text; open the log for the detail of one experiment.
- **The framing of the paper is the author's** (`PAPER_DESIGN.md` section 1) and is not rebuilt in
  answer to a critique or a grading.
- A number or conclusion that is only in the log is **not** grounds for a next step and is not
  quoted. If it is needed, re-measure it and put it into `PROJECT_KNOWLEDGE.md` section 4.
- **Register before running**; change **one controlled variable at a time**; real data only.
- **Record every meaningful run at the end of `THESIS_RESULTS.md`** (design, result, verdict). What
  will be used goes into `PROJECT_KNOWLEDGE.md` section 4 with its source; a change of design or
  method goes into the change log of `PAPER_DESIGN.md`. Then run
  `python ces_prediction/check_records.py` (every number of the two files against its source).
- **Pin the data treatment explicitly in every run** — never let `CES_DROP_STUCK_TARGETS` and
  friends fall back to a default.
- **Leave no orphan.** A new batch gets a row in `ces_prediction/experiments/README.md` and in the
  wiring table of `README.md`; a file that nothing reads is removed, not parked.
- After code changes run `python -m pytest -q`; if training/data/model behavior changed, also
  run a smoke training command and record whether it passed.
