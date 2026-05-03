import os
import difflib
from dotenv import load_dotenv

# .env 파일 로드 (루트 디렉토리 기준)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def _get_slack_client(token):
    try:
        from slack_sdk import WebClient
    except ImportError:
        print("[Slack Notifier] slack_sdk is not installed. Skipping Slack notification.")
        return None
    return WebClient(token=token)

def _generate_diff(old_code, new_code):
    """이전 이터레이션과 현재 이터레이션 간의 코드 차이(Diff)를 생성합니다."""
    if old_code == new_code:
        return "No architectural changes."
    
    diff = difflib.unified_diff(
        old_code.splitlines(), 
        new_code.splitlines(),
        fromfile='previous_model.py',
        tofile='current_model.py',
        lineterm=''
    )
    diff_text = '\n'.join(list(diff)[:50]) # 너무 길면 50줄까지만
    if not diff_text:
        return "No architectural changes."
    return diff_text + ("\n...[truncated]" if len(list(diff)) > 50 else "")

def send_experiment_start(max_iterations, cpu_workers):
    """실험이 시작되었음을 Slack으로 전송합니다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    
    if not token or not channel_id:
        return

    client = _get_slack_client(token)
    if client is None:
        return
    
    text = "🚀 *New AutoML Experiment Started*\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
    text += f"• Max Iterations: `{max_iterations}`\n"
    if cpu_workers:
        text += f"• CPU Budget: `{cpu_workers}` cores\n"
    else:
        text += "• CPU Budget: `Default` (Auto)\n"
    text += "Waiting for the first iteration results..."

    try:
        client.chat_postMessage(channel=channel_id, text=text)
        print(f"[Slack Notifier] Experiment start notification sent.")
    except Exception as e:
        print(f"[Slack Notifier] Failed to send experiment start notification: {e}")

def send_insight_report(recent_log, current_iteration):
    """10번의 이터레이션 기록(Diff + 성능)을 LLM으로 분석하여 인사이트를 Slack으로 전송합니다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    
    if not token or not channel_id:
        return

    client = _get_slack_client(token)
    if client is None:
        return
    try:
        import litellm
    except ImportError:
        print("[Slack Notifier] litellm is not installed. Skipping insight report.")
        return
    
    print(f"\n[Slack Notifier] Generating insight report for iterations {current_iteration-9} to {current_iteration}...")

    # LLM에게 넘길 컨텍스트 구성 (전체 코드가 아닌 Diff 전달)
    log_texts = []
    previous_code = ""
    
    for entry in recent_log:
        current_code = entry['code']
        # 이전 턴이 있으면 Diff 생성, 없으면 "Initial Model" 처리
        if previous_code:
            code_diff = _generate_diff(previous_code, current_code)
        else:
            code_diff = "Initial Model Architecture (Base)"
            
        previous_code = current_code
            
        log_texts.append(
            f"--- Iteration {entry['iteration']} ---\n"
            f"Validation Loss: {entry['val_loss']:.4f}\n"
            f"Code Changes (Diff):\n{code_diff}\n"
        )
    
    context_data = "\n".join(log_texts)

    prompt = f"""
    You are a Senior AI ML Scientist monitoring an autonomous AutoML loop.
    Below is the log of the last 10 iterations. Instead of full code, it includes the 'Code Changes (Diff)' 
    showing exactly what was modified from the previous iteration, along with the resulting Validation Loss.

    {context_data}

    Analyze this 10-iteration window and write a concise, engaging Slack message to the team.
    Your message MUST include:
    1. 🛠️ **Approaches Tried**: What specific architectural changes or techniques were explored? (Look at the diffs).
    2. 📈 **Performance Analysis**: What changes improved the loss? Which changes failed or caused plateaus?
    3. 💡 **Key Insight**: One brief concluding thought on what the agent seems to be learning or what direction it should take next.

    Format the output nicely using Slack markdown (*bold*, _italic_, bullet points, emojis).
    DO NOT wrap the final response in markdown code blocks (```). Just output the raw text.
    """

    try:
        response = litellm.completion(
            model="gemini/gemma-4-31b-it", 
            messages=[{"role": "user", "content": prompt}]
        )
        insight_message = response.choices[0].message.content.strip()
        
        # 슬랙 메시지 헤더 조립 (10회 미만일 경우 처리)
        start_iter = max(1, current_iteration - 9)
        header = f"🧠 *AutoML Insight Report (Iter {start_iter} ~ {current_iteration})*\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        final_text = header + insight_message
        
        response = client.chat_postMessage(channel=channel_id, text=final_text)
        print(f"[Slack Notifier] Insight report sent successfully! (ts: {response['ts']})")
        
    except Exception as e:
        print(f"[Slack Notifier] Failed to generate/send insight: {e}")

def send_iteration_update(iteration, metrics):
    """매 이터레이션의 결과를 간단하게 Slack으로 전송합니다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    
    if not token or not channel_id:
        return

    client = _get_slack_client(token)
    if client is None:
        return
    
    val_loss = metrics.get("final_val_loss", float('inf'))
    train_loss = metrics.get("final_train_loss", "n/a")
    
    # Loss 값 포맷팅 (inf일 경우 처리)
    val_loss_str = f"{val_loss:.4f}" if val_loss != float('inf') else "inf"
    train_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) and train_loss != float('inf') else str(train_loss)
    
    status_emoji = "✅" if val_loss != float('inf') else "❌"
    text = f"{status_emoji} *Iteration #{iteration} Result*\n"
    text += f"• Val Loss: `{val_loss_str}`\n"
    text += f"• Train Loss: `{train_loss_str}`"

    try:
        client.chat_postMessage(channel=channel_id, text=text)
        print(f"[Slack Notifier] Iteration update sent successfully.")
    except Exception as e:
        print(f"[Slack Notifier] Failed to send iteration update: {e}")

def send_slack_summary(history, max_iterations):
    """전체 R&D 루프가 종료되었을 때 결과를 Slack으로 전송합니다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    
    if not token or not channel_id:
        print("[Slack Notifier] Token or Channel ID not found in .env. Skipping Slack notification.")
        return

    client = _get_slack_client(token)
    if client is None:
        return
    try:
        from slack_sdk.errors import SlackApiError
    except ImportError:
        SlackApiError = Exception
    
    try:
        if not history:
            text = "🏁 *ML R&D Loop Completed*, but no history was recorded."
        else:
            valid_history = [h for h in history if h.get('val_loss', float('inf')) != float('inf')]
            best_entry = min(valid_history, key=lambda x: x['val_loss']) if valid_history else None
            
            text = "🏁 *ML R&D Loop Completed*\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
            text += f"Total Iterations: {len(history)} / {max_iterations}\n"
            
            if best_entry:
                text += f"Best Val Loss: *{best_entry['val_loss']:.4f}* (Iteration #{best_entry['iteration']})\n\n"
                
                text += "📊 *Top 5 Iterations*\n"
                top_5 = sorted(valid_history, key=lambda x: x['val_loss'])[:5]
                for entry in top_5:
                    marker = " ⭐️ Best" if entry['iteration'] == best_entry['iteration'] else ""
                    text += f"  #{entry['iteration']:<3} | {entry['val_loss']:.4f}{marker}\n"
            else:
                text += "All iterations failed (Loss = inf).\n"
                
            text += "\n📁 Full report saved to `HANDOFF.md`"

        response = client.chat_postMessage(channel=channel_id, text=text)
        print(f"[Slack Notifier] Notification sent successfully! (ts: {response['ts']})")
        
    except SlackApiError as e:
        print(f"[Slack Notifier] Error sending message: {e.response['error']}")
    except Exception as e:
        print(f"[Slack Notifier] Unexpected error: {e}")
