# AGENTS.md

Instructions for coding agents (Codex and others) working in this repository. `CLAUDE.md` holds the same rules in full; `HANDOVER.md` (Korean) says where everything is and what to do next. Read both before changing anything.

## What this repository is

Causal nowcasting of the KSTAR charge-exchange (CES) ion temperature on one pedestal-top chord from its own history and fast diagnostics at the same location (BES, ECEI), with a thesis manuscript in PPCF format under `docs/paper/ppcf/`. Executable code is under `ces_prediction/`; controlled experiment batches under `ces_prediction/experiments/` (one directory per batch, each behind a registration `PREREGISTRATION_B{NN}.md`).

## Read first

0. `README.md` — the wiring of the repository: what reads what.
1. `HANDOVER.md` — state, environment, data layout, the batch procedure, open work (section 9), proposed extensions (section 10).
2. `PAPER_DESIGN.md` sections 1 to 10 — purpose, terms, targets, estimators, validation designs, wording rules, the paper map.
3. `PROJECT_KNOWLEDGE.md` sections 1 to 3, 4 (from 4.48) and 6 — the data facts, the results in use and the rules.
4. `ces_prediction/experiments/README.md` — batch directories and their status.

## Commands

```bash
py -m pytest -q                                  # tests (89 pass); run after code changes
py ces_prediction/check_records.py               # every number of the two record files against its source
py docs/paper/ppcf/make_tables_v2.py             # result json -> gen_v2.tex (tables, number macros)
py docs/paper/ppcf/fig_v2.py                     # figures of the manuscript
py docs/paper/ppcf/check_numbers_v2.py           # numbers typed in the manuscript against their sources
# build, from docs/paper/ppcf: latexmk -pdf supp_ppcf ; latexmk -pdf main_ppcf ; latexmk -pdf supp_ppcf
```

Always run scripts from the repository root with `py` (Python 3.14 on Windows). The shot data are not in git (`data/*` is ignored); see `HANDOVER.md` section 4.

## Rules that must not be broken

- **Claude API preference (author, 2026-11-23).** Connect substantive design, interpretation, research scouting, implementation review and next-study prioritization to the official Claude API by default. Prefer Fable for routine analysis and Opus for important adversarial synthesis. Send explicit redacted tasks/evidence, never credentials, identity, raw observations or private reasoning. The author also authorizes sending anonymized aggregate evaluated results for post-evaluation research planning. Keep numerical/source verification independent; Claude advice is not verified fact. Bound paid calls and preserve every attempt/usage. Use the committed post-evaluation workflow after the final independent audit, and mark TEST-informed new directions exploratory. Do not modify running frozen code or execute new scientific fits without preregistration merely because Claude proposes them.

- **The framing of the paper is the author's and is not rebuilt.** Purpose: nowcasting at the pedestal top; find the best model against interpolation and the like; 10 ms real time; a demonstration of use is a bonus. Do not rebuild the storyline in answer to a critique or a grading; place new results inside the five questions of `PAPER_DESIGN.md` section 1.
- **Register before running.** Every batch gets `PREREGISTRATION_B{NN}.md` (statements, rule of judgement, expectation, the sentences of the paper for either outcome) committed before any run; open choices go into `..._ADDENDUM1.md` before the readout; the registration is never edited afterwards. Follow `HANDOVER.md` section 5.
- **Real data only.** No synthetic or mock data. Fail loudly if data are missing.
- **Earlier lines of this research were removed on 2026-11-21** (git history, last commit before the cleanup `e86adaf`). Do not restore them and do not quote their numbers. Leave no orphan: a new file is read by something, and that is written into the wiring table of `README.md`.
- **Pin the data treatment** in every run (`CES_*` variables in the runner's env); never rely on defaults.
- **Do not edit** `ces_prediction/model_iter009.py`, `model_iter002.py` (pinned by SHA-256), a registration file after it is committed, `docs/paper/ppcf/earlier_line/`, or existing training code while a GPU batch runs. New work goes into new files.
- **Numbers are never typed by hand** into the manuscript; they come from `gen_v2.tex`. Every number in `PROJECT_KNOWLEDGE.md` and `PAPER_DESIGN.md` must pass `check_records.py`.
- **Wording.** Results are a point estimate, a 95 % interval and whether it excludes zero. No "wins", "beats", "outperforms", "first", "novel". A negative result names the measurement that would overturn it. Report all four validation designs; none is called primary.
- **GPU.** Do not start GPU work unless the author says so. It is paused; the next GPU batch is specified in `HANDOVER.md` section 9.1. Do not stop other processes that hold the GPU.
- **Git.** Commit and push verified work on the checked-out branch with real messages. Never force-push, never rewrite pushed history, never skip hooks. Never open or merge a pull request unless asked. Do not commit anything under `data/` or secrets.
- **Outward actions** (mail, messages, publishing) only when the author asks. Do not put the author's personal details into requests to outside services.
- **Acceptance estimates** for the manuscript are produced only by an independent grader agent that is given the two PDFs and the fixed prompt in `docs/paper/review_loop/GRADER_PROMPT.md`, never by the working agent.
- Answer the author in Korean.

## Pitfalls on this machine

- The console is cp949: ASCII only in `print()`.
- Backslashes inside Bash heredocs get altered: write Python that handles LaTeX or regular expressions to a file and run the file. `sed -i` turns `\usepackage` into `\Sepackage`.
- The sandbox hides the GPU; GPU jobs must run outside it.
