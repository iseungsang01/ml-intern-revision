import os
import litellm
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv

# .env 파일 로드 (루트 디렉토리 기준)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

def send_insight_report(recent_log, current_iteration):
    """10번의 이터레이션 기록(코드 + 성능)을 LLM으로 분석하여 인사이트를 Slack으로 전송합니다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    
    if not token or not channel_id:
        return

    print(f"\n[Slack Notifier] Generating insight report for iterations {current_iteration-9} to {current_iteration}...")

    # LLM에게 넘길 컨텍스트 구성
    log_texts = []
    for entry in recent_log:
        code_snippet = entry['code']
        # 너무 길면 자르기 (보통 1000자 내외면 아키텍처 파악 충분)
        if len(code_snippet) > 1500:
            code_snippet = code_snippet[:1500] + "\n...[truncated]"
            
        log_texts.append(
            f"--- Iteration {entry['iteration']} ---\n"
            f"Validation Loss: {entry['val_loss']:.4f}\n"
            f"Model Code Snippet:\n{code_snippet}\n"
        )
    
    context_data = "\n".join(log_texts)

    prompt = f"""
    You are a Senior AI ML Scientist monitoring an autonomous AutoML loop.
    Below is the log of the last 10 iterations. It includes the Validation Loss and the 'model.py' architecture used for each iteration.

    {context_data}

    Analyze this 10-iteration window and write a concise, engaging Slack message to the team.
    Your message MUST include:
    1. 🛠️ **Approaches Tried**: What specific model architectures or techniques were explored in these 10 iterations? (e.g., CNN, Transformers, specific hyperparameter tweaks).
    2. 📈 **Performance Analysis**: What worked well (which iteration/architecture had the lowest loss) and what failed or caused plateaus?
    3. 💡 **Key Insight**: One brief concluding thought on what the agent seems to be learning or what direction it should take next.

    Format the output nicely using Slack markdown (*bold*, _italic_, bullet points, emojis).
    DO NOT wrap the final response in markdown code blocks (```). Just output the raw text.
    """

    try:
        response = litellm.completion(
            model="gemini/gemini-3.1-pro-preview", 
            messages=[{"role": "user", "content": prompt}]
        )
        insight_message = response.choices[0].message.content.strip()
        
        # 슬랙 메시지 헤더 조립
        header = f"🧠 *AutoML Insight Report (Iter {current_iteration-9} ~ {current_iteration})*\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        final_text = header + insight_message
        
        client = WebClient(token=token)
        response = client.chat_postMessage(channel=channel_id, text=final_text)
        print(f"[Slack Notifier] Insight report sent successfully! (ts: {response['ts']})")
        
    except Exception as e:
        print(f"[Slack Notifier] Failed to generate/send insight: {e}")

def send_slack_summary(history, max_iterations):
    """전체 R&D 루프가 종료되었을 때 결과를 Slack으로 전송합니다."""
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    
    if not token or not channel_id:
        print("[Slack Notifier] Token or Channel ID not found in .env. Skipping Slack notification.")
        return

    try:
        client = WebClient(token=token)
        
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
