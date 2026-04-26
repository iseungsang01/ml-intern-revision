import os
import json
import subprocess
import litellm

class EvaluationAgent:
    """
    Evaluation Agent:
    테스트 코드로 자가 치유(Self-Healing)를 검증한 후, 실제 학습(train.py)을 수행합니다.
    """
    def run_evaluation(self, iteration):
        print(f"\n[Evaluation Agent] Starting iteration {iteration}...")
        
        # 1. Dry Run (에러 사전 차단)
        try:
            print("[Evaluation Agent] Running Architecture Dry-Run validation...")
            # tests 디렉토리의 테스트 코드를 실행
            subprocess.run(["python", "../tests/test_architecture.py"], check=True)
        except subprocess.CalledProcessError:
            print("[Evaluation Agent] Dry-run failed. Researcher must fix the Shape/Architecture mismatch.")
            return {"error": "Dry-run failed", "final_val_loss": float('inf')}

        # 2. Actual Training
        try:
            subprocess.run(["python", "train.py"], check=True)
            with open("metrics.json", "r") as f:
                metrics = json.load(f)
            
            val_loss = metrics.get("final_val_loss", float('inf'))
            print(f"[Evaluation Agent] Iteration {iteration} completed. Val Loss: {val_loss:.4f}")
            return metrics
        except Exception as e:
            print(f"[Evaluation Agent] Training failed: {e}")
            return {"error": str(e), "final_val_loss": float('inf')}

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
        
    def generate_briefing(self, iteration, current_metrics):
        val_loss = current_metrics.get("final_val_loss", float('inf'))
        
        # 기록 업데이트
        self.history.append({"iteration": iteration, "val_loss": val_loss})
        
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
            f"Here is the briefing from the latest experiment:\n{briefing}\n\n"
            f"Here is the current 'model.py' code:\n```python\n{current_code}\n```\n\n"
            f"Based on the briefing, rewrite the architecture or hyperparameters in 'model.py'. "
            f"Keep the class name 'MultimodalCESPredictor' and the forward pass signature the same. "
            f"Return ONLY the complete, raw Python code. Do not include markdown blocks."
        )
        
        try:
            print("[Researcher Agent] Requesting new architecture from LLM (gpt-4o)...")
            response = litellm.completion(
                model="gpt-4o", 
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

def run_auto_ml_loop(max_iterations=5):
    eval_agent = EvaluationAgent()
    briefing_agent = BriefingAgent()
    researcher_agent = ResearcherAgent()
    
    print("=== Starting Autonomous ML R&D Loop ===")
    
    for i in range(1, max_iterations + 1):
        # 1. Evaluation (현재 모델 평가)
        metrics = eval_agent.run_evaluation(i)
        
        # 2. Briefing (결과 요약 및 방향성 설정)
        briefing = briefing_agent.generate_briefing(i, metrics)
        
        # 3. Research & Update (새로운 모델/파라미터 적용)
        if i < max_iterations: # 마지막 턴이 아니면 다음 모델 준비
            researcher_agent.research_and_update(briefing)
            
    print("\n=== Autonomous ML R&D Loop Completed ===")

if __name__ == "__main__":
    run_auto_ml_loop(max_iterations=300)
