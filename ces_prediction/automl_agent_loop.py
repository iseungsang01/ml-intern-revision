import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_KNOWLEDGE_FILE = "PROJECT_KNOWLEDGE.md"
PROGRAM_FILE = "program.md"
RESEARCH_MODEL_DEFAULT = "claude-opus-4-8"
MAX_RESEARCH_TOKENS = 32000
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


def strip_code_fences(text):
    code = text.strip()
    if code.startswith("```"):
        newline = code.find("\n")
        code = code[newline + 1:] if newline != -1 else code[3:]
    if code.endswith("```"):
        code = code[: code.rfind("```")]
    return code.strip()


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
    ):
        self.cpu_workers = cpu_workers
        self.dataloader_workers = dataloader_workers
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.run_smoke_test = run_smoke_test
        self.split_dir = split_dir
        self.output_dir = output_dir

    def _subprocess_env(self):
        env = os.environ.copy()
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
        if self.split_dir is not None:
            env["CES_SPLIT_DIR"] = str(self.split_dir)
        if self.output_dir is not None:
            env["CES_OUTPUT_DIR"] = str(self.output_dir)
        return env

    def _output_dir(self, env):
        return Path(env.get("CES_OUTPUT_DIR", str(script_dir())))

    def _smoke_env(self):
        env = os.environ.copy()
        env["CES_EPOCHS"] = "1"
        env["CES_MAX_TRAIN_SAMPLES"] = "2000"
        env["CES_MAX_VAL_SAMPLES"] = "500"
        env["CES_BATCH_SIZE"] = "128"
        env["CES_SPLIT_DIR"] = str(root_dir() / "data" / ".smoke_splits")
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

    def run_evaluation(self, iteration):
        print(f"\n[Evaluation Agent] Starting iteration {iteration}...")

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
        except Exception as exc:
            print(f"[Evaluation Agent] Clean evaluation failed: {exc}")
            return {**metrics, "eval": None, "score": None, "error": str(exc), "error_stage": "evaluation"}

        score = combined_skill(eval_report)
        print(f"[Evaluation Agent] Iteration {iteration} clean skill (mean): {_fmt(score)}")
        return {**metrics, "eval": eval_report, "score": score, "error": None, "error_stage": None}


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
        lines.append(f"Stale rounds (no new best): {self.stale_rounds}")
        if result.get("error_stage"):
            lines.append(f"FAILURE at {result['error_stage']}: {str(result.get('error'))[:300]}")
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
    """Claude-driven controlled rewrite of model.py (official Anthropic SDK)."""

    def __init__(self, program="", project_knowledge=""):
        self.program = program.strip()
        self.project_knowledge = project_knowledge.strip()

    def _build_prompt(self, briefing, current_code):
        return (
            f"{self.program or DEFAULT_PROGRAM}\n\n"
            "## Project knowledge (avoid these failed paths)\n"
            f"{self.project_knowledge or '(none)'}\n\n"
            "## Hard data/model contract (never violate)\n"
            f"{DATA_CONTRACT}\n\n"
            f"## Current briefing\n{briefing}\n\n"
            "## Current model.py (build on this)\n"
            f"```python\n{current_code}\n```\n\n"
            "Return ONLY the complete raw Python source for model.py (no prose, no markdown "
            "fences). Keep class MultimodalCESPredictor and the signature "
            "forward(self, bes, ecei, mc, time_features=None, ces_history=None)."
        )

    def research_and_update(self, briefing, model_path):
        try:
            import anthropic
        except ImportError:
            print(
                "[Researcher Agent] `anthropic` SDK not installed. "
                "Install it (`python -m pip install anthropic`) to enable Claude-driven research. Skipping."
            )
            return False
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[Researcher Agent] ANTHROPIC_API_KEY not set; skipping model update.")
            return False

        current_code = model_path.read_text(encoding="utf-8")
        prompt = self._build_prompt(briefing, current_code)
        model_id = os.getenv("AUTOML_RESEARCH_MODEL", RESEARCH_MODEL_DEFAULT)

        print(f"[Researcher Agent] Requesting controlled model update from {model_id}...")
        try:
            client = anthropic.Anthropic()
            with client.messages.stream(
                model=model_id,
                max_tokens=MAX_RESEARCH_TOKENS,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:
            print(f"[Researcher Agent] Claude request failed: {exc}. Keeping current model.py.")
            return False

        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
        code = strip_code_fences(text)
        if "class MultimodalCESPredictor" not in code or "def forward" not in code:
            print("[Researcher Agent] Response missing required class/forward; keeping current model.py.")
            return False

        model_path.write_text(code.strip() + "\n", encoding="utf-8")
        print("[Researcher Agent] Updated model.py")
        return True


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
    )
    state_dir = (Path(output_dir) if output_dir else script_dir()) / ".automl_state"
    tracker = ExperimentTracker(state_dir)
    researcher = ResearcherAgent(program=program, project_knowledge=project_knowledge)
    model_path = script_dir() / "model.py"

    print("=== Starting autoresearch loop (keep/discard on clean skill) ===")
    print(f"[AutoML Loop] Max iterations: {max_iterations}")
    print(f"[AutoML Loop] Smoke validation: {run_smoke_test}")
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
    )
