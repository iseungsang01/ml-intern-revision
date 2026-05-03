import argparse
import os
import json
import subprocess
from pathlib import Path
import litellm


FIXED_SPLIT_FILES = ("fixed_train_split.csv", "fixed_val_split.csv")


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
"""


MODEL_ARCHITECTURE_NOTE = """
Current model/data design snapshot:
- Dataset samples are row-window histories ending at a CES target row. With temporal subset augmentation enabled, each target can be paired with multiple irregular subsets of previous rows instead of only one fixed contiguous window.
- Inputs are separated by diagnostic modality: BES has 9 channels, ECEI has 4 channels, MC has 2 channels, time metadata has 4 channels, and CES history has 3 channels.
- The model keeps a late-fusion multimodal design. BES, ECEI, and MC are encoded by separate time-aware 1D CNN branches. Each branch receives its sensor channels concatenated with the same time features and CES-history features.
- Time features encode true irregular sampling: lookback seconds, delta seconds, log1p lookback, and log1p delta. CES history contains previous normalized CES_TI/CES_VT plus an observed mask; the target row is masked as [0, 0, 0] to prevent leakage.
- Each sensor branch uses Conv1d -> BatchNorm -> GELU -> Conv1d -> BatchNorm -> GELU -> AdaptiveAvgPool1d -> Linear -> GELU, producing a 96-dimensional feature vector.
- A separate time-only Conv1d encoder produces a 32-dimensional time feature vector.
- The fusion head concatenates BES/ECEI/MC/time features into 320 dimensions, then predicts normalized [CES_TI, CES_VT] through Linear(320, 160) -> GELU -> Dropout(0.2) -> Linear(160, 64) -> GELU -> Linear(64, 2).
- Training uses MSE on normalized targets plus a small penalty against physically invalid negative TI in normalized space. AdamW, gradient clipping, and ReduceLROnPlateau are used.
- The AutoML loop changes the model because the Researcher Agent rewrites model.py after every evaluated iteration except the last. The dry-run test only checks interface/shape compatibility, not whether the new architecture is scientifically better.
""".strip()


CONVERGENCE_ASSESSMENT_NOTE = """
Convergence assessment from the latest 8-iteration history:
- Validation loss has stayed in a narrow band around 0.49-0.53 for most runs, with one unstable failure at 0.8764. The best observed validation loss is 0.4901 at iteration 5; the latest value is 0.5023.
- Train loss is usually lower than validation loss, but improving train loss has not reliably improved validation loss. Iterations 1-4 had train loss near 0.33 while validation stayed around 0.51-0.53.
- This pattern is more consistent with an approach/validation-generalization plateau than with a run that merely needs many more epochs. Longer training may reduce train loss, but the current evidence does not show a stable path to lower validation loss.
- The main risk is architecture churn without controlled ablations. Because model.py is rewritten between iterations, loss changes mix architecture changes, initialization noise, and training dynamics. A better next step is to freeze one strong baseline, run repeated seeds/longer epochs for that baseline, then test one change at a time.
- Recommended next experiments: keep the current late-fusion CNN as a baseline; run 30-50 epochs with early stopping and best-checkpoint saving; compare against no temporal subset augmentation, no CES-history input, larger/smaller window sizes, and a sequence model that uses input_mask explicitly. Track per-target TI/VT validation error after denormalization, not only aggregate normalized MSE.
""".strip()


class EvaluationAgent:
    """
    Evaluation Agent:
    테스트 코드로 자가 치유(Self-Healing)를 검증한 후, 실제 학습(train.py)을 수행합니다.
    """
    def __init__(self, cpu_workers=None, dataloader_workers=None):
        self.cpu_workers = cpu_workers
        self.dataloader_workers = dataloader_workers

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
        return env

    def run_evaluation(self, iteration):
        print(f"\n[Evaluation Agent] Starting iteration {iteration}...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(script_dir)
        env = self._subprocess_env()
        
        # 1. Dry Run (에러 사전 차단)
        try:
            print("[Evaluation Agent] Running Architecture Dry-Run validation...")
            # tests 디렉토리의 테스트 코드를 실행
            subprocess.run(
                ["python", "-m", "pytest", "-q", os.path.join(root_dir, "tests", "test_architecture.py")],
                cwd=root_dir,
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError:
            print("[Evaluation Agent] Dry-run failed. Researcher must fix the Shape/Architecture mismatch.")
            return {"error": "Dry-run failed", "final_val_loss": float('inf')}

        # 2. Actual Training
        try:
            subprocess.run(
                ["python", os.path.join(script_dir, "train.py")],
                cwd=script_dir,
                env=env,
                check=True,
            )
            with open(os.path.join(script_dir, "metrics.json"), "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            val_loss = metrics.get("final_val_loss", float('inf'))
            print(f"[Evaluation Agent] Iteration {iteration} completed. Val Loss: {val_loss:.4f}")
            return metrics
        except Exception as e:
            print(f"[Evaluation Agent] Training failed: {e}")
            return {"error": str(e), "final_val_loss": float('inf')}


def reset_session_artifacts():
    root_dir = Path(__file__).resolve().parents[1]
    script_dir = Path(__file__).resolve().parent
    paths = [root_dir / "HANDOFF.md"]
    paths.extend(script_dir / name for name in FIXED_SPLIT_FILES)

    for path in paths:
        if path.exists():
            path.unlink()
            print(f"[AutoML Loop] Removed previous session artifact: {path}")

class BriefingAgent:
    """
    Briefing Agent:
    지금까지의 실험 기록(History)을 관리하고, Evaluation 결과를 바탕으로 
    성능 정체(Plateau) 여부를 판단하여 다음 방향성을 Researcher에게 브리핑합니다.
    """
    def __init__(self):
        self.history = []
        self.best_loss = float('inf')
        self.plateau_count = 0

    def _summarize_metrics(self, metrics):
        normalization = metrics.get("normalization", {})
        cpu_config = metrics.get("cpu_config", {})
        return {
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

    @staticmethod
    def _fmt(value, digits=4):
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        if value is None:
            return "n/a"
        return str(value)
        
    def generate_briefing(self, iteration, current_metrics):
        val_loss = current_metrics.get("final_val_loss", float('inf'))
        metric_summary = self._summarize_metrics(current_metrics)
        
        # 기록 업데이트
        self.history.append({"iteration": iteration, **metric_summary})
        
        # Plateau(정체기) 체크
        if val_loss >= self.best_loss * 0.99: # 성능 향상이 1% 미만일 경우
            self.plateau_count += 1
        else:
            self.best_loss = val_loss
            self.plateau_count = 0
            
        briefing = f"--- Briefing Report (Iteration {iteration}) ---\n"
        briefing += f"Current Val Loss: {val_loss:.4f} (Best: {self.best_loss:.4f})\n"
        
        if self.plateau_count >= 3:
            briefing += "STATUS: PLATEAU DETECTED (3+ iterations without improvement).\n"
            briefing += "DIRECTION: Completely discard the current 1D CNN approach. Explore a completely new architecture (e.g., Transformer, Mamba, or Graph Neural Networks for spatial channel interactions).\n"
        else:
            briefing += "STATUS: LEARNING OR STABLE.\n"
            briefing += "DIRECTION: Tweak hyperparameters, add skip connections, or adjust channel dimensions. Keep the current base architecture but optimize it.\n"
            
        print(f"\n[Briefing Agent] {briefing}")
        
        # --- Save to HANDOFF.md ---
        handoff_content = f"# AutoML Session Handoff\n\n## Latest Briefing (Iteration {iteration})\n\n```text\n{briefing}\n```\n\n"
        handoff_content += "## Data Contract\n\n```text\n"
        handoff_content += DATA_CONTRACT.strip()
        handoff_content += "\n```\n\n"
        handoff_content += "## Model Design And Architecture\n\n"
        handoff_content += MODEL_ARCHITECTURE_NOTE
        handoff_content += "\n\n"
        handoff_content += "## Convergence Assessment\n\n"
        handoff_content += CONVERGENCE_ASSESSMENT_NOTE
        handoff_content += "\n\n"
        handoff_content += "## Latest Metrics\n\n"
        handoff_content += f"- Train Loss: {self._fmt(metric_summary['train_loss'])}\n"
        handoff_content += f"- Val Loss: {self._fmt(metric_summary['val_loss'])}\n"
        handoff_content += f"- Epochs: {self._fmt(metric_summary['epochs'])}\n"
        handoff_content += f"- Samples: train={self._fmt(metric_summary['train_samples'])}, val={self._fmt(metric_summary['val_samples'])}\n"
        handoff_content += f"- Temporal Subsets: {self._fmt(metric_summary['temporal_subset_augmentation'])}\n"
        handoff_content += f"- Min Subset Size: {self._fmt(metric_summary['min_subset_size'])}\n"
        handoff_content += f"- Normalization: {self._fmt(metric_summary['normalization_method'])}, scope={self._fmt(metric_summary['normalization_scope'])}\n"
        handoff_content += f"- Normalization Groups: {', '.join(metric_summary['normalization_groups']) or 'n/a'}\n"
        handoff_content += f"- Feature Dims: `{json.dumps(metric_summary['feature_dims'], ensure_ascii=False)}`\n\n"
        handoff_content += "## History\n\n"
        for entry in self.history:
            handoff_content += (
                f"- Iteration {entry['iteration']}: "
                f"train={self._fmt(entry.get('train_loss'))}, "
                f"val={self._fmt(entry.get('val_loss'))}, "
                f"samples={self._fmt(entry.get('train_samples'))}/{self._fmt(entry.get('val_samples'))}, "
                f"norm={self._fmt(entry.get('normalization_method'))}:{self._fmt(entry.get('normalization_scope'))}\n"
            )

        handoff_path = Path(__file__).resolve().parents[1] / "HANDOFF.md"
        with handoff_path.open("w", encoding="utf-8") as f:
            f.write(handoff_content)
        # ---------------------------

        return briefing

class ResearcherAgent:
    """
    Researcher Agent:
    Briefing Agent의 리포트를 받아 새로운 접근법을 고민하고, 
    model.py나 hyperparameter를 실제로 수정(작성)하는 역할을 합니다.
    """
    def research_and_update(self, briefing):
        print(f"\n[Researcher Agent] Analyzing briefing and researching new architecture...")
        
        try:
            with open("model.py", "r", encoding="utf-8") as f:
                current_code = f.read()
        except FileNotFoundError:
            current_code = "# model.py not found"

        prompt = (
            f"You are an expert AI ML Researcher for KSTAR nuclear fusion data.\n"
            f"{DATA_CONTRACT}\n"
            f"Here is the briefing from the latest experiment:\n{briefing}\n\n"
            f"Here is the current 'model.py' code:\n```python\n{current_code}\n```\n\n"
            f"Based on the briefing, rewrite the architecture or hyperparameters in 'model.py'. "
            f"Keep the class name 'MultimodalCESPredictor' and preserve the dataset/training contract exactly. "
            f"Keep the forward pass signature compatible with train.py, including ces_history. "
            f"Return ONLY the complete, raw Python code. Do not include markdown blocks."
        )
        
        try:
            print("[Researcher Agent] Requesting new architecture from LLM (Gemma 4 31B)...")
            response = litellm.completion(
                model="gemini/gemma-4-31b-it",
                messages=[{"role": "user", "content": prompt}]
            )
            new_code = response.choices[0].message.content.strip()
            
            # Clean up potential markdown formatting
            if new_code.startswith("```python"): new_code = new_code[9:]
            elif new_code.startswith("```"): new_code = new_code[3:]
            if new_code.endswith("```"): new_code = new_code[:-3]
                
            with open("model.py", "w", encoding="utf-8") as f:
                f.write(new_code.strip() + "\n")
                
            print("[Researcher Agent] Successfully updated model.py")
        except Exception as e:
            print(f"[Researcher Agent] LLM Error: {e}")

def run_auto_ml_loop(max_iterations=5, cpu_workers=None, dataloader_workers=None):
    reset_session_artifacts()

    eval_agent = EvaluationAgent(
        cpu_workers=cpu_workers,
        dataloader_workers=dataloader_workers,
    )
    briefing_agent = BriefingAgent()
    researcher_agent = ResearcherAgent()
    
    experiment_log = [] # 10회마다의 인사이트 생성을 위해 코드와 성능을 기록
    
    print("=== Starting Autonomous ML R&D Loop ===")
    if cpu_workers is not None:
        print(f"[AutoML Loop] CPU worker budget: {cpu_workers}")
    if dataloader_workers is not None:
        print(f"[AutoML Loop] DataLoader workers override: {dataloader_workers}")
    
    # --- Slack 알림 발송 (실험 시작) ---
    try:
        from slack_notifier import send_experiment_start
        send_experiment_start(max_iterations, cpu_workers)
    except Exception as e:
        print(f"[AutoML Loop] Failed to send experiment start notification: {e}")
    # --------------------------------
    
    for i in range(1, max_iterations + 1):
        # 1. Evaluation (현재 모델 평가)
        metrics = eval_agent.run_evaluation(i)
        
        # 현재 평가된 모델 코드 읽기
        try:
            with open("model.py", "r", encoding="utf-8") as f:
                current_code = f.read()
        except FileNotFoundError:
            current_code = "No model.py found"
            
        # 로그 저장
        experiment_log.append({
            "iteration": i,
            "val_loss": metrics.get("final_val_loss", float('inf')),
            "code": current_code
        })
        
        # 2. Briefing (결과 요약 및 방향성 설정)
        briefing = briefing_agent.generate_briefing(i, metrics)
        
        # 3. Research & Update (새로운 모델/파라미터 적용)
        if i < max_iterations: # 마지막 턴이 아니면 다음 모델 준비
            researcher_agent.research_and_update(briefing)
            
        # 4. Slack 알림 발송 (매 이터레이션)
        try:
            from slack_notifier import send_iteration_update, send_insight_report
            # 매번 간단한 업데이트 전송
            send_iteration_update(i, metrics)
            
            # 10회마다 상세 LLM 인사이트 분석 전송
            if i % 10 == 0:
                send_insight_report(experiment_log[-10:], i)
        except Exception as e:
            print(f"[AutoML Loop] Failed to send slack notification: {e}")
            
    print("\n=== Autonomous ML R&D Loop Completed ===")
    
    # Slack 알림 전송 (종료 요약)
    try:
        from slack_notifier import send_slack_summary
        send_slack_summary(briefing_agent.history, max_iterations)
    except Exception as e:
        print(f"[Slack Notifier] Failed to trigger notification: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run the KSTAR CES AutoML loop.")
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="CPU core budget passed to training. Example: --cpu-workers 16",
    )
    parser.add_argument(
        "--dataloader-workers",
        type=int,
        default=None,
        help="Override DataLoader worker processes. Defaults to about half of --cpu-workers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_auto_ml_loop(
        max_iterations=args.max_iterations,
        cpu_workers=args.cpu_workers,
        dataloader_workers=args.dataloader_workers,
    )
