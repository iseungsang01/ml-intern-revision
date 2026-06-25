import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# Load environment variables from a repo-root .env so ANTHROPIC_API_KEY,
# SLACK_BOT_TOKEN, and SLACK_CHANNEL_ID can live in one place. Subprocesses
# (train.py / evaluate.py) inherit them because the loop spawns them with
# os.environ.copy(). Does not override variables already set in the shell.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


PROJECT_KNOWLEDGE_FILE = "PROJECT_KNOWLEDGE.md"
PROGRAM_FILE = "program.md"
RESEARCH_MODEL_DEFAULT = "claude-opus-4-8"
MAX_RESEARCH_TOKENS = 32000
CLI_RESEARCH_TIMEOUT = 1200  # seconds to wait for the `claude` CLI rewrite
DEFAULT_WINDOW_SIZE = 4   # history window used when model.py declares no WINDOW_SIZE
MAX_WINDOW_SIZE = 32      # clamp so a runaway WINDOW_SIZE can't explode augmentation
TARGET_NAMES = ("CES_TI", "CES_VT")


DATA_CONTRACT = """
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
""".strip()


DEFAULT_PROGRAM = """
You are an autonomous ML researcher improving ces_prediction/model.py for KSTAR CES gap-filling.
Objective: maximize the clean mean(CES_TI, CES_VT) skill_vs_persistence from evaluate.py (higher is
better) -- NOT the augmented training val loss. The loop trains your model.py at a fixed budget,
evaluates the clean skill, and keeps your change only if it beats the best so far; otherwise it is
discarded and model.py is restored to the best version. Change ONE controlled variable per
iteration, preserve the data/model contract, avoid known failed paths from PROJECT_KNOWLEDGE.md,
and return ONLY the complete raw model.py code (no prose, no fences).
""".strip()


def root_dir():
    return Path(__file__).resolve().parents[1]


def script_dir():
    return Path(__file__).resolve().parent


def load_text(filename, fallback=""):
    path = root_dir() / filename
    try:
        text = path.read_text(encoding="utf-8").strip()
        print(f"[AutoML Loop] Loaded {filename}: {path}")
        return text
    except FileNotFoundError:
        print(f"[AutoML Loop] {filename} not found at {path}; using fallback.")
        return fallback


def load_project_knowledge():
    return load_text(PROJECT_KNOWLEDGE_FILE, fallback="(No PROJECT_KNOWLEDGE.md content was available.)")


def load_program():
    return load_text(PROGRAM_FILE, fallback=DEFAULT_PROGRAM)


def _fmt(value, digits=4):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return "n/a" if value != value else f"{value:.{digits}f}"  # NaN -> n/a
    if isinstance(value, int):
        return str(value)
    if value is None:
        return "n/a"
    return str(value)


def combined_skill(eval_report):
    """Mean of per-target skill_vs_persistence over targets with observed samples.

    This is the clean, baseline-relative score the loop optimizes (higher is better),
    the KSTAR analog of Karpathy autoresearch's single comparable val_bpb metric.
    """
    if not eval_report:
        return None
    per_target = eval_report.get("per_target", {})
    skills = []
    for stats in per_target.values():
        if not stats.get("n"):
            continue
        skill = stats.get("skill_vs_persistence")
        if isinstance(skill, (int, float)) and skill == skill:  # exclude None / NaN
            skills.append(float(skill))
    if not skills:
        return None
    return sum(skills) / len(skills)


def comparison_skill(comparison_report):
    """Mean per-target skill_vs_pchip from compare_baselines (higher is better).

    This is the metric the loop optimizes when the goal is to BEAT conventional
    past+future interpolation (the thesis claim), as opposed to merely beating
    persistence. Selection is on the val split; the test split stays held out.
    """
    if not comparison_report:
        return None
    skills = []
    for stats in comparison_report.get("per_target", {}).values():
        skill = stats.get("skill_vs_pchip")
        if isinstance(skill, (int, float)) and skill == skill:  # exclude None / NaN
            skills.append(float(skill))
    if not skills:
        return None
    return sum(skills) / len(skills)


def _format_peak_line(name, peak):
    """Format the per-target headline peak line for the briefing (AC8).

    The headline lives at peak["input_only"]["ces_activity"] (Family i sub-signal
    i-a: high-local-activity CES neighborhoods, a target-independent proxy, NOT
    pointwise extrema). Returns None when no peak block is present so the briefing
    is unchanged on iterations without peak data. Surfaces peak skill_vs_pchip and
    its shot-clustered 95% CI, plus an insufficient/n.s. tag so the researcher can
    see when the peak CI straddles 0 (esp. for CES_VT)."""
    if not isinstance(peak, dict):
        return None
    head = peak.get("input_only")
    if isinstance(head, dict):
        head = head.get("ces_activity")
    if not isinstance(head, dict):
        return None
    skill = head.get("peak_skill_vs_pchip")
    ci = head.get("peak_skill_ci95")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        ci_str = f"CI95=[{_fmt(ci[0])}, {_fmt(ci[1])}]"
    else:
        ci_str = "CI95=n/a"
    if head.get("insufficient_shots") or head.get("insufficient_rows"):
        tag = "insufficient"
    elif head.get("pass"):
        tag = "PASS"
    else:
        tag = "n.s."
    n_rows = head.get("n_peak_rows")
    n_shots = head.get("n_peak_shots")
    return (
        f"  {name} PEAK (input-only CES-activity): skill_vs_pchip={_fmt(skill)} "
        f"{ci_str} {tag} (n_rows={_fmt(n_rows)}, n_shots={_fmt(n_shots)})"
    )


def strip_code_fences(text):
    code = text.strip()
    if code.startswith("```"):
        newline = code.find("\n")
        code = code[newline + 1:] if newline != -1 else code[3:]
    if code.endswith("```"):
        code = code[: code.rfind("```")]
    return code.strip()


def extract_model_code(text):
    """Pull model.py source out of a possibly chatty model/CLI response.

    Prefers a fenced code block that defines MultimodalCESPredictor, falls back
    to the largest fenced block, then to fence-stripping the whole response. The
    `claude` CLI runs in agent mode and can wrap output in prose, so plain
    fence-stripping is not enough.
    """
    text = text.strip()
    blocks = re.findall(r"```(?:python)?[ \t]*\r?\n(.*?)```", text, re.DOTALL)
    for block in blocks:
        if "class MultimodalCESPredictor" in block:
            return block.strip()
    if blocks:
        return max(blocks, key=len).strip()
    return strip_code_fences(text)


def read_model_window(model_path, default=DEFAULT_WINDOW_SIZE):
    """History window the model.py requests via a module-level ``WINDOW_SIZE = N``.

    This makes the window length part of the loop's search space (it is a
    data-pipeline knob that lives outside the model class): the loop reads it and
    trains/evaluates at that window via CES_WINDOW_SIZE. Falls back to ``default``
    when not declared, and is clamped to [2, MAX_WINDOW_SIZE] so a runaway value
    cannot explode the temporal-subset augmentation.
    """
    try:
        text = Path(model_path).read_text(encoding="utf-8")
    except OSError:
        return default
    match = re.search(r"^[ \t]*WINDOW_SIZE[ \t]*=[ \t]*(\d+)", text, re.MULTILINE)
    if not match:
        return default
    window = int(match.group(1))
    if window < 2:
        return default
    return min(window, MAX_WINDOW_SIZE)


def run_subprocess(command, cwd, env):
    return subprocess.run(command, cwd=cwd, env=env, check=True)


class EvaluationAgent:
    def __init__(
        self,
        cpu_workers=None,
        dataloader_workers=None,
        train_samples=None,
        val_samples=None,
        run_smoke_test=True,
        split_dir=None,
        output_dir=None,
        param_budget=None,
    ):
        self.cpu_workers = cpu_workers
        self.dataloader_workers = dataloader_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.run_smoke_test = run_smoke_test
        self.split_dir = split_dir
        self.output_dir = output_dir
        self.param_budget = param_budget

    def _subprocess_env(self):
        env = os.environ.copy()
        # Window is searchable: model.py may declare WINDOW_SIZE. The fixed split is
        # validated against the dataset (and so against the window), so each window
        # gets its own split dir to avoid load_fixed_split_csv mismatches.
        window = read_model_window(script_dir() / "model.py")
        env["CES_WINDOW_SIZE"] = str(window)
        # Reserve a held-out test split so selection (on val) never touches it.
        env.setdefault("CES_TEST_FRACTION", "0.15")
        if self.cpu_workers is not None:
            workers = str(self.cpu_workers)
            env["CES_CPU_WORKERS"] = workers
            env.setdefault("OMP_NUM_THREADS", workers)
            env.setdefault("MKL_NUM_THREADS", workers)
            env.setdefault("NUMEXPR_NUM_THREADS", workers)
        if self.dataloader_workers is not None:
            env["CES_DATALOADER_WORKERS"] = str(self.dataloader_workers)
        if self.train_samples is not None:
            env["CES_MAX_TRAIN_SAMPLES"] = str(self.train_samples)
        if self.val_samples is not None:
            env["CES_MAX_VAL_SAMPLES"] = str(self.val_samples)
        split_base = Path(self.split_dir) if self.split_dir is not None else root_dir() / "data" / "splits"
        env["CES_SPLIT_DIR"] = str(split_base / f"w{window}")
        if self.output_dir is not None:
            env["CES_OUTPUT_DIR"] = str(self.output_dir)
        return env

    def _output_dir(self, env):
        return Path(env.get("CES_OUTPUT_DIR", str(script_dir())))

    def _count_params(self):
        """Count trainable params of the current model.py by building it via
        from_dataset on the fixed contract feature_dims, in a clean subprocess.

        Returns an int, or None if the model could not be built (in which case the
        smoke stage will surface the real error -- we do not block on a None count).
        """
        window = read_model_window(script_dir() / "model.py")
        code = (
            "import sys; sys.path.insert(0, r'{d}'); "
            "from types import SimpleNamespace; "
            "from model import MultimodalCESPredictor as M; "
            "ds = SimpleNamespace(feature_dims={{'bes':9,'ecei':4,'mc':2,'time':4,'ces_history':4}}); "
            "m = M.from_dataset(ds, window_size={w}); "
            "print(sum(p.numel() for p in m.parameters()))"
        ).format(d=str(script_dir()), w=window)
        try:
            proc = subprocess.run(
                ["python", "-c", code], cwd=str(root_dir()),
                capture_output=True, text=True, timeout=120,
            )
        except Exception as exc:
            print(f"[Evaluation Agent] Param count failed to run ({exc}); skipping budget gate.")
            return None
        if proc.returncode != 0:
            print(
                "[Evaluation Agent] Param count errored (model may be invalid; smoke will catch): "
                f"{(proc.stderr or '').strip()[:200]}"
            )
            return None
        try:
            return int(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None

    def _smoke_env(self):
        env = os.environ.copy()
        # Smoke must train at the same window the model declares, with its own
        # per-window split dir (the model expects that many timesteps).
        window = read_model_window(script_dir() / "model.py")
        env["CES_WINDOW_SIZE"] = str(window)
        env["CES_EPOCHS"] = "1"
        env["CES_MAX_TRAIN_SAMPLES"] = "2000"
        env["CES_MAX_VAL_SAMPLES"] = "500"
        env["CES_BATCH_SIZE"] = "128"
        env["CES_SPLIT_DIR"] = str(root_dir() / "data" / ".smoke_splits" / f"w{window}")
        env["CES_OUTPUT_DIR"] = str(root_dir() / "data" / ".smoke_outputs")
        return env

    @staticmethod
    def _failure(stage, exc, **extra):
        result = {
            "error": str(exc),
            "error_stage": stage,
            "final_train_loss": None,
            "final_val_loss": float("inf"),
            "eval": None,
            "score": None,
        }
        result.update(extra)
        return result

    def run_smoke_validation(self):
        if not self.run_smoke_test:
            return None

        print("[Evaluation Agent] Running smoke validation before full training...")
        try:
            run_subprocess(["python", "-m", "pytest", "-q"], cwd=root_dir(), env=self._smoke_env())
            run_subprocess(["python", str(script_dir() / "train.py")], cwd=root_dir(), env=self._smoke_env())
        except subprocess.CalledProcessError as exc:
            print(f"[Evaluation Agent] Smoke validation failed: {exc}")
            return self._failure("smoke_test", exc)
        return None

    def run_clean_eval(self, env):
        run_subprocess(["python", str(script_dir() / "evaluate.py")], cwd=root_dir(), env=env)
        eval_path = self._output_dir(env) / "eval_metrics.json"
        return json.loads(eval_path.read_text(encoding="utf-8"))

    def run_comparison(self, env):
        """Model-vs-interpolation comparison on the VAL split (selection metric).
        The held-out test split is never read here."""
        compare_env = dict(env)
        compare_env["CES_SPLIT_TAG"] = "val"
        run_subprocess(["python", str(script_dir() / "compare_baselines.py")], cwd=root_dir(), env=compare_env)
        path = self._output_dir(env) / "comparison_metrics.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _merge_peak_block(self, env, eval_report):
        """Merge the cheap headline peak block into eval_report IN MEMORY (AC8).

        Reuses the val-split comparison_errors npz that run_comparison just wrote
        (no extra forward pass). For each target with a peak block, sets
        eval_report["per_target"][name]["peak"]. Fully guarded: any failure logs,
        leaves peak=None, and never fails the iteration or flips error_stage.
        """
        try:
            from peak_analysis import run_peak_eval

            peak_block = run_peak_eval(env) or {}
            per_target = eval_report.get("per_target") if isinstance(eval_report, dict) else None
            if not isinstance(per_target, dict):
                return
            for name in TARGET_NAMES:
                stats = per_target.get(name)
                if isinstance(stats, dict):
                    stats["peak"] = peak_block.get(name)
        except Exception as exc:
            print(f"[Evaluation Agent] Peak evaluation skipped (non-fatal): {exc}")

    def run_evaluation(self, iteration):
        print(f"\n[Evaluation Agent] Starting iteration {iteration}...")

        # Hard parameter-budget gate: reject over-budget proposals BEFORE spending any
        # training/smoke compute. Treated as a normal failure -> tracker rolls back to
        # the best model and the briefing tells the researcher to shrink.
        if self.param_budget and self.param_budget > 0:
            n_params = self._count_params()
            if n_params is not None and n_params > self.param_budget:
                print(
                    f"[Evaluation Agent] Param budget exceeded: {n_params:,} > "
                    f"{self.param_budget:,}; discarding WITHOUT training."
                )
                return self._failure(
                    "param_budget",
                    ValueError(f"{n_params} params exceeds budget {self.param_budget}"),
                    params=n_params, param_budget=self.param_budget,
                )
            if n_params is not None:
                print(f"[Evaluation Agent] Model params: {n_params:,} (budget {self.param_budget:,}).")

        smoke_error = self.run_smoke_validation()
        if smoke_error is not None:
            return smoke_error

        env = self._subprocess_env()
        try:
            print("[Evaluation Agent] Running full training...")
            run_subprocess(["python", str(script_dir() / "train.py")], cwd=root_dir(), env=env)
            metrics = json.loads((self._output_dir(env) / "metrics.json").read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[Evaluation Agent] Full training failed: {exc}")
            return self._failure("training", exc)

        try:
            print("[Evaluation Agent] Running clean (baseline-relative) evaluation...")
            eval_report = self.run_clean_eval(env)
            print("[Evaluation Agent] Running interpolation comparison (val skill_vs_pchip)...")
            comparison_report = self.run_comparison(env)
        except Exception as exc:
            print(f"[Evaluation Agent] Clean evaluation/comparison failed: {exc}")
            return {**metrics, "eval": None, "score": None, "error": str(exc), "error_stage": "evaluation"}

        # AC8: merge the CHEAP headline peak block (high-local-activity neighborhood
        # skill + bootstrap CI) into the IN-MEMORY eval_report so it reaches the
        # researcher via build_briefing. This reuses the val-split comparison_errors
        # npz that run_comparison just wrote (NO extra forward pass) and is purely
        # additive: a peak failure logs + leaves peak=None and never fails the
        # iteration or flips error_stage. The keep/discard gate (comparison_skill)
        # is unaffected -- it reads only comparison_report.
        self._merge_peak_block(env, eval_report)

        score = comparison_skill(comparison_report)
        persistence_skill = combined_skill(eval_report)
        print(
            f"[Evaluation Agent] Iteration {iteration} val skill_vs_pchip (mean): {_fmt(score)} "
            f"(skill_vs_persistence: {_fmt(persistence_skill)})"
        )
        return {
            **metrics,
            "eval": eval_report,
            "comparison": comparison_report,
            "skill_vs_persistence": persistence_skill,
            "score": score,
            "error": None,
            "error_stage": None,
        }


class ExperimentTracker:
    """Karpathy-style keep/discard with automatic rollback to the best model.py.

    The score is the clean mean skill_vs_persistence (higher is better). A proposal is kept only
    if it beats the best so far; otherwise model.py is restored from the best snapshot, so the
    loop never builds on a regression and never loses the best-known architecture.
    """

    def __init__(self, state_dir):
        self.state_dir = Path(state_dir)
        self.best_model_path = self.state_dir / "best_model.py"
        self.history_dir = self.state_dir / "history"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = None
        self.stale_rounds = 0
        self.consecutive_failures = 0
        self.history = []

    def _snapshot(self, model_path):
        shutil.copyfile(model_path, self.best_model_path)

    def _rollback(self, model_path):
        if self.best_model_path.exists():
            shutil.copyfile(self.best_model_path, model_path)
            return True
        return False

    def record(self, iteration, result, model_path):
        # Archive the proposal that was just evaluated (reviewable diffs).
        try:
            shutil.copyfile(model_path, self.history_dir / f"model_iter{iteration:03d}.py")
        except OSError:
            pass

        score = result.get("score")
        error_stage = result.get("error_stage")

        if error_stage or score is None:
            status = "failed"
            self.stale_rounds += 1
            self.consecutive_failures += 1
            if self._rollback(model_path):
                print(f"[Tracker] Iteration {iteration} {error_stage or 'no-score'} -> rolled back to best.")
        else:
            self.consecutive_failures = 0
            if self.best_score is None:
                status = "baseline"
                self.best_score = score
                self.stale_rounds = 0
                self._snapshot(model_path)
                print(f"[Tracker] Iteration {iteration} baseline clean skill = {_fmt(score)}.")
            elif score > self.best_score:
                status = "kept"
                print(f"[Tracker] Iteration {iteration} improved: {_fmt(score)} > best {_fmt(self.best_score)} -> kept.")
                self.best_score = score
                self.stale_rounds = 0
                self._snapshot(model_path)
            else:
                status = "rolled_back"
                print(f"[Tracker] Iteration {iteration} not better ({_fmt(score)} <= best {_fmt(self.best_score)}) -> discarded.")
                self.stale_rounds += 1
                self._rollback(model_path)

        self.history.append({
            "iteration": iteration,
            "status": status,
            "score": score,
            "best_score": self.best_score,
            "stale_rounds": self.stale_rounds,
            "error_stage": error_stage,
            "error": result.get("error"),
            "train_loss": result.get("final_train_loss"),
            "val_loss": result.get("final_val_loss"),
            "eval": result.get("eval"),
        })
        return status

    def build_briefing(self, iteration, result, status):
        eval_report = result.get("eval") or {}
        per_target = eval_report.get("per_target", {})
        lines = [
            f"--- Briefing (Iteration {iteration}) ---",
            f"Decision on last proposal: {status.upper()}",
            f"Clean skill (mean CES_TI/VT skill_vs_persistence): {_fmt(result.get('score'))} "
            f"(best: {_fmt(self.best_score)})",
        ]
        for name in TARGET_NAMES:
            stats = per_target.get(name, {})
            if stats.get("n"):
                lines.append(
                    f"  {name}: skill={_fmt(stats.get('skill_vs_persistence'))}, "
                    f"R2_vs_mean={_fmt(stats.get('r2_vs_mean'))}, "
                    f"RMSE={_fmt(stats.get('rmse_model'))} (n={stats.get('n')})"
                )
            peak_line = _format_peak_line(name, stats.get("peak"))
            if peak_line:
                lines.append(peak_line)
        lines.append(f"Stale rounds (no new best): {self.stale_rounds}")
        if result.get("error_stage"):
            lines.append(f"FAILURE at {result['error_stage']}: {str(result.get('error'))[:300]}")
            if result.get("error_stage") == "param_budget":
                lines.append(
                    f"HARD PARAMETER BUDGET EXCEEDED: your last model had {result.get('params')} "
                    f"params, over the {result.get('param_budget')} limit, so it was discarded "
                    "WITHOUT training (wasted iteration). model.py was restored to the best version. "
                    "Propose a SMALLER architecture well under budget (target ~0.4-0.9M params): "
                    "shrink hidden width / layers / d_model rather than adding capacity."
                )
            else:
                lines.append(
                    "Your previous change was discarded and model.py restored to the best version. "
                    "Propose a DIFFERENT controlled change that avoids this failure."
                )
        else:
            lines.append(
                "Propose ONE controlled architecture change that builds on the current best "
                "model.py and raises the clean skill. Preserve the data/model contract; avoid "
                "known failed paths."
            )
        return "\n".join(lines)

    def write_handoff(self, iteration, briefing):
        content = "# AutoML Session Handoff (autoresearch)\n\n"
        content += f"## Latest Briefing (Iteration {iteration})\n\n```text\n{briefing}\n```\n\n"
        content += f"- Best clean skill (mean skill_vs_persistence): {_fmt(self.best_score)}\n"
        content += f"- Stale rounds: {self.stale_rounds}\n"
        content += f"- Best model snapshot: `{self.best_model_path}`\n\n"
        content += "## Data Contract\n\n```text\n" + DATA_CONTRACT + "\n```\n\n"
        content += "## History\n\n"
        for entry in self.history:
            content += (
                f"- Iter {entry['iteration']}: {entry['status']}, "
                f"skill={_fmt(entry['score'])}, best={_fmt(entry['best_score'])}, "
                f"val_loss={_fmt(entry['val_loss'])}, stage={entry['error_stage'] or 'ok'}\n"
            )
        (root_dir() / "HANDOFF.md").write_text(content, encoding="utf-8")


class ResearcherAgent:
    """Claude-driven controlled rewrite of model.py.

    Uses the logged-in `claude` CLI (Claude Code subscription billing) by default,
    or the Anthropic SDK when a real sk-ant-api03 key is set in ANTHROPIC_API_KEY.
    """

    def __init__(self, program="", project_knowledge="", param_budget=None):
        self.program = program.strip()
        self.project_knowledge = project_knowledge.strip()
        self.param_budget = param_budget

    def _budget_section(self):
        if not self.param_budget or self.param_budget <= 0:
            return ""
        return (
            "## Hard parameter budget (enforced before training)\n"
            f"The total trainable parameter count of MultimodalCESPredictor -- built via "
            f"from_dataset on feature_dims bes=9, ecei=4, mc=2, time=4, ces_history=4 -- MUST be "
            f"strictly under {self.param_budget} parameters. Any model over budget is discarded "
            "WITHOUT training (a wasted iteration), so do not exceed it. Empirically the sweet spot "
            "is ~0.45-0.8M params; bigger (>1M) overfits and scored WORSE here. Aim for a compact, "
            "well-regularized model, not raw capacity.\n\n"
        )

    def _build_prompt(self, briefing, current_code):
        return (
            f"{self.program or DEFAULT_PROGRAM}\n\n"
            "## Project knowledge (avoid these failed paths)\n"
            f"{self.project_knowledge or '(none)'}\n\n"
            "## Hard data/model contract (never violate)\n"
            f"{DATA_CONTRACT}\n\n"
            f"{self._budget_section()}"
            f"## Current briefing\n{briefing}\n\n"
            "## Current model.py (build on this)\n"
            f"```python\n{current_code}\n```\n\n"
            "Return the complete updated model.py as a SINGLE Python code block fenced "
            "with ```python ... ``` and output nothing outside that code block. Keep "
            "class MultimodalCESPredictor and the signature "
            "forward(self, bes, ecei, mc, time_features=None, ces_history=None)."
        )

    def research_and_update(self, briefing, model_path):
        current_code = model_path.read_text(encoding="utf-8")
        prompt = self._build_prompt(briefing, current_code)
        model_id = os.getenv("AUTOML_RESEARCH_MODEL", RESEARCH_MODEL_DEFAULT)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key.startswith("sk-ant-api"):
            # Real API key -> Anthropic SDK, billed via API credits.
            raw = self._request_via_sdk(prompt, model_id, api_key)
        else:
            # Default: drive the logged-in `claude` CLI -> billed via the Claude
            # subscription (no separate API credits). Works whenever `claude` is
            # logged in, or CLAUDE_CODE_OAUTH_TOKEN is set from `claude setup-token`.
            raw = self._request_via_cli(prompt, model_id)
        if raw is None:
            return False

        code = extract_model_code(raw)
        if "class MultimodalCESPredictor" not in code or "def forward" not in code:
            print("[Researcher Agent] Response missing required class/forward; keeping current model.py.")
            return False

        model_path.write_text(code.strip() + "\n", encoding="utf-8")
        print("[Researcher Agent] Updated model.py")
        return True

    def _request_via_cli(self, prompt, model_id):
        claude = shutil.which("claude")
        if not claude:
            print("[Researcher Agent] `claude` CLI not found on PATH; skipping model update.")
            return None

        env = os.environ.copy()
        # The CLI authenticates via the Claude Code login (subscription). A token
        # left in ANTHROPIC_API_KEY would force API-credit billing, and an
        # sk-ant-oat token is not a valid API key, so move it to the OAuth slot.
        leftover = env.pop("ANTHROPIC_API_KEY", "")
        if leftover.startswith("sk-ant-oat") and not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            env["CLAUDE_CODE_OAUTH_TOKEN"] = leftover

        cmd = [
            claude,
            "-p",
            "--model",
            model_id,
            "--output-format",
            "text",
            "--max-turns",
            "1",
            "--append-system-prompt",
            "Respond with exactly one ```python fenced code block containing the "
            "complete file and no text outside the block.",
        ]
        print(
            f"[Researcher Agent] Requesting controlled model update via Claude CLI "
            f"({model_id}, subscription login)..."
        )
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=CLI_RESEARCH_TIMEOUT,
            )
        except Exception as exc:
            print(f"[Researcher Agent] Claude CLI failed: {exc}. Keeping current model.py.")
            return None
        if result.returncode != 0:
            err = (result.stderr or "").strip()[:300]
            print(f"[Researcher Agent] Claude CLI exited {result.returncode}: {err}. Keeping current model.py.")
            return None
        return result.stdout

    def _request_via_sdk(self, prompt, model_id, api_key):
        try:
            import anthropic
        except ImportError:
            print(
                "[Researcher Agent] `anthropic` SDK not installed; cannot use ANTHROPIC_API_KEY. "
                "Install it or use the Claude CLI login. Skipping."
            )
            return None

        print(f"[Researcher Agent] Requesting controlled model update from {model_id} (API key)...")
        try:
            client = anthropic.Anthropic(api_key=api_key)
            with client.messages.stream(
                model=model_id,
                max_tokens=MAX_RESEARCH_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:
            print(f"[Researcher Agent] Claude request failed: {exc}. Keeping current model.py.")
            return None

        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )


def run_auto_ml_loop(
    max_iterations=300,
    cpu_workers=None,
    dataloader_workers=None,
    train_samples=None,
    val_samples=None,
    run_smoke_test=True,
    split_dir=None,
    output_dir=None,
    max_consecutive_failures=5,
    param_budget=1_000_000,
):
    from slack_notifier import (
        send_iteration_result,
        send_loop_complete,
        send_loop_start,
        validate_slack_config,
    )

    validate_slack_config()
    program = load_program()
    project_knowledge = load_project_knowledge()

    eval_agent = EvaluationAgent(
        cpu_workers=cpu_workers,
        dataloader_workers=dataloader_workers,
        train_samples=train_samples,
        val_samples=val_samples,
        run_smoke_test=run_smoke_test,
        split_dir=split_dir,
        output_dir=output_dir,
        param_budget=param_budget,
    )
    state_dir = (Path(output_dir) if output_dir else script_dir()) / ".automl_state"
    tracker = ExperimentTracker(state_dir)
    researcher = ResearcherAgent(
        program=program, project_knowledge=project_knowledge, param_budget=param_budget
    )
    model_path = script_dir() / "model.py"

    print("=== Starting autoresearch loop (keep/discard on clean skill) ===")
    print(f"[AutoML Loop] Max iterations: {max_iterations}")
    print(f"[AutoML Loop] Smoke validation: {run_smoke_test}")
    print(f"[AutoML Loop] Param budget: {param_budget:,}" if param_budget else "[AutoML Loop] Param budget: (none)")
    print(f"[AutoML Loop] State dir: {state_dir}")
    send_loop_start(max_iterations, run_smoke_test)

    for iteration in range(1, max_iterations + 1):
        result = eval_agent.run_evaluation(iteration)
        status = tracker.record(iteration, result, model_path)
        briefing = tracker.build_briefing(iteration, result, status)
        print(f"\n[Briefing Agent]\n{briefing}\n")
        tracker.write_handoff(iteration, briefing)
        send_iteration_result(iteration, result, status, tracker.best_score, tracker.stale_rounds)

        if tracker.consecutive_failures >= max_consecutive_failures:
            print(
                f"[AutoML Loop] {tracker.consecutive_failures} consecutive failures "
                f">= limit {max_consecutive_failures}; stopping."
            )
            break

        if iteration < max_iterations:
            researcher.research_and_update(briefing, model_path)

    print("\n=== autoresearch loop completed ===")
    send_loop_complete(tracker.history, max_iterations, tracker.best_score)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the KSTAR CES autoresearch loop (keep/discard).")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--cpu-workers", type=int, default=None)
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
    parser.add_argument("--param-budget", type=int, default=1_000_000,
                        help="Hard max trainable params; over-budget proposals are discarded "
                             "without training. Set 0 to disable.")
    parser.add_argument("--no-smoke-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_auto_ml_loop(
        max_iterations=args.max_iterations,
        cpu_workers=args.cpu_workers,
        dataloader_workers=args.dataloader_workers,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        run_smoke_test=not args.no_smoke_test,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        max_consecutive_failures=args.max_consecutive_failures,
        param_budget=args.param_budget,
    )
