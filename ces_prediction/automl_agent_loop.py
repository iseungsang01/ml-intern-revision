import argparse
import json
import os
import subprocess
from pathlib import Path


PROJECT_KNOWLEDGE_FILE = "PROJECT_KNOWLEDGE.md"
PLATEAU_RELATIVE_IMPROVEMENT = 0.03
PLATEAU_PATIENCE = 3


DATA_CONTRACT = """
Dataset/training contract that every generated model.py must preserve:
- train.py builds KSTAR_CES_Dataset with temporal subset augmentation.
- BES, ECEI, and MC inputs are per-channel z-score normalized using train-file-only statistics.
- CES_TI and CES_VT are per-channel z-score normalized with train-file-only target statistics.
- ces_history has shape (batch, window, 3): normalized previous CES_TI, normalized previous CES_VT, observed mask.
- The target timestep CES values are masked in ces_history as [0, 0, 0] to avoid leakage.
- model.forward must accept forward(self, bes, ecei, mc, time_features=None, ces_history=None).
- Model outputs must remain normalized CES_TI/CES_VT with shape (batch, 2); train.py compares them to normalized targets.
- Do not denormalize inside model.py. Any inverse transform belongs outside training/evaluation.
""".strip()


def root_dir():
    return Path(__file__).resolve().parents[1]


def script_dir():
    return Path(__file__).resolve().parent


def load_project_knowledge():
    knowledge_path = root_dir() / PROJECT_KNOWLEDGE_FILE
    try:
        knowledge = knowledge_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"[AutoML Loop] Project knowledge file not found: {knowledge_path}")
        return ""

    print(f"[AutoML Loop] Loaded project knowledge: {knowledge_path}")
    return knowledge


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

    def _smoke_env(self):
        env = os.environ.copy()
        env["CES_EPOCHS"] = "1"
        env["CES_MAX_TRAIN_SAMPLES"] = "2000"
        env["CES_MAX_VAL_SAMPLES"] = "500"
        env["CES_BATCH_SIZE"] = "128"
        env["CES_SPLIT_DIR"] = str(root_dir() / "data" / ".smoke_splits")
        env["CES_OUTPUT_DIR"] = str(root_dir() / "data" / ".smoke_outputs")
        return env

    def run_smoke_validation(self):
        if not self.run_smoke_test:
            return None

        print("[Evaluation Agent] Running smoke validation before full training...")
        try:
            run_subprocess(["python", "-m", "pytest", "-q"], cwd=root_dir(), env=self._smoke_env())
            run_subprocess(["python", str(script_dir() / "train.py")], cwd=root_dir(), env=self._smoke_env())
        except subprocess.CalledProcessError as exc:
            print(f"[Evaluation Agent] Smoke validation failed: {exc}")
            return {
                "error": str(exc),
                "error_stage": "smoke_test",
                "final_train_loss": None,
                "final_val_loss": float("inf"),
            }
        return None

    def run_evaluation(self, iteration):
        print(f"\n[Evaluation Agent] Starting iteration {iteration}...")

        smoke_error = self.run_smoke_validation()
        if smoke_error is not None:
            return smoke_error

        env = self._subprocess_env()
        try:
            print("[Evaluation Agent] Running full training evaluation...")
            run_subprocess(["python", str(script_dir() / "train.py")], cwd=root_dir(), env=env)
            metrics_path = script_dir() / "metrics.json"
            with metrics_path.open("r", encoding="utf-8") as f:
                metrics = json.load(f)

            val_loss = metrics.get("final_val_loss", float("inf"))
            print(f"[Evaluation Agent] Iteration {iteration} completed. Val Loss: {val_loss:.4f}")
            return metrics
        except Exception as exc:
            print(f"[Evaluation Agent] Full training failed: {exc}")
            return {
                "error": str(exc),
                "error_stage": "training",
                "final_train_loss": None,
                "final_val_loss": float("inf"),
            }


class BriefingAgent:
    def __init__(self, plateau_threshold=PLATEAU_RELATIVE_IMPROVEMENT, plateau_patience=PLATEAU_PATIENCE):
        self.history = []
        self.best_loss = float("inf")
        self.plateau_count = 0
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience

    @staticmethod
    def _fmt(value, digits=4):
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        if value is None:
            return "n/a"
        return str(value)

    def _summarize_metrics(self, metrics):
        normalization = metrics.get("normalization", {})
        cpu_config = metrics.get("cpu_config", {})
        return {
            "error": metrics.get("error"),
            "error_stage": metrics.get("error_stage"),
            "train_loss": metrics.get("final_train_loss"),
            "val_loss": metrics.get("final_val_loss", float("inf")),
            "epochs": metrics.get("epochs"),
            "train_samples": metrics.get("train_samples"),
            "val_samples": metrics.get("val_samples"),
            "feature_dims": metrics.get("feature_dims"),
            "temporal_subset_augmentation": metrics.get("temporal_subset_augmentation"),
            "min_subset_size": metrics.get("min_subset_size"),
            "normalization_scope": normalization.get("scope"),
            "normalization_method": normalization.get("method"),
            "normalization_groups": sorted(normalization.get("stats", {}).keys()),
            "cpu_workers": cpu_config.get("cpu_workers"),
            "dataloader_workers": cpu_config.get("dataloader_workers"),
        }

    def update_plateau_state(self, val_loss):
        if not isinstance(val_loss, (int, float)) or not val_loss < float("inf"):
            return False

        if self.best_loss == float("inf"):
            self.best_loss = val_loss
            self.plateau_count = 0
            return False

        relative_improvement = (self.best_loss - val_loss) / self.best_loss
        if relative_improvement >= self.plateau_threshold:
            self.best_loss = val_loss
            self.plateau_count = 0
            return False

        if val_loss < self.best_loss:
            self.best_loss = val_loss
        self.plateau_count += 1
        return self.plateau_count >= self.plateau_patience

    def generate_briefing(self, iteration, current_metrics):
        metric_summary = self._summarize_metrics(current_metrics)
        self.history.append({"iteration": iteration, **metric_summary})

        val_loss = metric_summary["val_loss"]
        plateau_detected = self.update_plateau_state(val_loss)
        failed_stage = metric_summary.get("error_stage")

        briefing = f"--- Briefing Report (Iteration {iteration}) ---\n"
        briefing += f"Current Val Loss: {self._fmt(val_loss)} (Best: {self._fmt(self.best_loss)})\n"
        briefing += f"Plateau Count: {self.plateau_count}/{self.plateau_patience}\n"

        if failed_stage == "smoke_test":
            briefing += "STATUS: SMOKE TEST FAILED.\n"
            briefing += "DIRECTION: Fix code/interface/training breakage. Do not treat this as model-quality evidence.\n"
            allow_research = True
        elif failed_stage:
            briefing += f"STATUS: EVALUATION FAILED at {failed_stage}.\n"
            briefing += "DIRECTION: Fix the failing path before architecture exploration.\n"
            allow_research = True
        elif plateau_detected:
            briefing += "STATUS: PLATEAU DETECTED.\n"
            briefing += "DIRECTION: A controlled architecture change is allowed. Change one variable and preserve the data/model contract.\n"
            allow_research = True
        else:
            briefing += "STATUS: NO ARCHITECTURE CHANGE ALLOWED.\n"
            briefing += "DIRECTION: Keep evaluating/tuning the current controlled baseline until plateau criteria are met.\n"
            allow_research = False

        print(f"\n[Briefing Agent] {briefing}")
        self.write_handoff(iteration, briefing, metric_summary)
        return briefing, allow_research

    def write_handoff(self, iteration, briefing, metric_summary):
        content = f"# AutoML Session Handoff\n\n## Latest Briefing (Iteration {iteration})\n\n"
        content += f"```text\n{briefing}\n```\n\n"
        content += "## Data Contract\n\n```text\n"
        content += DATA_CONTRACT
        content += "\n```\n\n"
        content += "## Latest Metrics\n\n"
        content += f"- Train Loss: {self._fmt(metric_summary['train_loss'])}\n"
        content += f"- Val Loss: {self._fmt(metric_summary['val_loss'])}\n"
        if metric_summary.get("error"):
            content += f"- Error Stage: {self._fmt(metric_summary['error_stage'])}\n"
            content += f"- Error: {self._fmt(metric_summary['error'])}\n"
        content += f"- Epochs: {self._fmt(metric_summary['epochs'])}\n"
        content += f"- Samples: train={self._fmt(metric_summary['train_samples'])}, val={self._fmt(metric_summary['val_samples'])}\n"
        content += f"- Temporal Subsets: {self._fmt(metric_summary['temporal_subset_augmentation'])}\n"
        content += f"- Min Subset Size: {self._fmt(metric_summary['min_subset_size'])}\n"
        content += f"- Normalization: {self._fmt(metric_summary['normalization_method'])}, scope={self._fmt(metric_summary['normalization_scope'])}\n"
        content += f"- Normalization Groups: {', '.join(metric_summary['normalization_groups']) or 'n/a'}\n"
        content += f"- Feature Dims: `{json.dumps(metric_summary['feature_dims'], ensure_ascii=False)}`\n\n"

        content += "## History\n\n"
        for entry in self.history:
            content += (
                f"- Iteration {entry['iteration']}: "
                f"train={self._fmt(entry.get('train_loss'))}, "
                f"val={self._fmt(entry.get('val_loss'))}, "
                f"samples={self._fmt(entry.get('train_samples'))}/{self._fmt(entry.get('val_samples'))}, "
                f"stage={self._fmt(entry.get('error_stage'))}\n"
            )

        (root_dir() / "HANDOFF.md").write_text(content, encoding="utf-8")


class ResearcherAgent:
    def __init__(self, project_knowledge=""):
        self.project_knowledge = project_knowledge.strip()

    def research_and_update(self, briefing):
        try:
            import litellm
        except ImportError:
            print("[Researcher Agent] litellm is not installed. Skipping automated model update.")
            return

        model_path = script_dir() / "model.py"
        current_code = model_path.read_text(encoding="utf-8")
        prompt = (
            "You are an expert ML researcher for KSTAR CES prediction.\n"
            "Follow the project knowledge and do not repeat known failed paths.\n\n"
            f"{self.project_knowledge or '(No PROJECT_KNOWLEDGE.md content was available.)'}\n\n"
            f"{DATA_CONTRACT}\n\n"
            f"Latest briefing:\n{briefing}\n\n"
            f"Current model.py:\n```python\n{current_code}\n```\n\n"
            "Return ONLY complete raw Python code for model.py. "
            "Keep class name MultimodalCESPredictor and preserve the forward signature."
        )

        print("[Researcher Agent] Requesting controlled model update from LLM...")
        response = litellm.completion(
            model=os.getenv("AUTOML_RESEARCH_MODEL", "gemini/gemma-4-31b-it"),
            messages=[{"role": "user", "content": prompt}],
        )
        new_code = response.choices[0].message.content.strip()
        if new_code.startswith("```python"):
            new_code = new_code[9:]
        elif new_code.startswith("```"):
            new_code = new_code[3:]
        if new_code.endswith("```"):
            new_code = new_code[:-3]

        model_path.write_text(new_code.strip() + "\n", encoding="utf-8")
        print("[Researcher Agent] Updated model.py")


def run_auto_ml_loop(
    max_iterations=300,
    cpu_workers=None,
    dataloader_workers=None,
    train_samples=None,
    val_samples=None,
    run_smoke_test=True,
    split_dir=None,
    output_dir=None,
):
    from slack_notifier import (
        send_iteration_result,
        send_loop_complete,
        send_loop_start,
        validate_slack_config,
    )

    validate_slack_config()
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
    briefing_agent = BriefingAgent()
    researcher_agent = ResearcherAgent(project_knowledge=project_knowledge)

    print("=== Starting Controlled AutoML Loop ===")
    print(f"[AutoML Loop] Max iterations: {max_iterations}")
    print(f"[AutoML Loop] Smoke validation: {run_smoke_test}")
    send_loop_start(max_iterations, run_smoke_test)

    for iteration in range(1, max_iterations + 1):
        metrics = eval_agent.run_evaluation(iteration)
        briefing, allow_research = briefing_agent.generate_briefing(iteration, metrics)
        send_iteration_result(
            iteration,
            metrics,
            briefing_agent.plateau_count,
            briefing_agent.plateau_patience,
            allow_research,
        )

        if iteration < max_iterations and allow_research:
            researcher_agent.research_and_update(briefing)
        elif iteration < max_iterations:
            print("[AutoML Loop] Skipping model rewrite; plateau criteria not met.")

    print("\n=== Controlled AutoML Loop Completed ===")
    send_loop_complete(briefing_agent.history, max_iterations)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the controlled KSTAR CES AutoML loop.")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--cpu-workers", type=int, default=None)
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
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
    )
