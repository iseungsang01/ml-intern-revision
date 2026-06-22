import os


def _client_and_channel():
    try:
        from slack_sdk import WebClient
    except ImportError as exc:
        raise RuntimeError(
            "slack_sdk is required for AutoML notifications. "
            "Install dependencies with `python -m pip install -e .`."
        ) from exc

    token = os.environ.get("SLACK_BOT_TOKEN")
    channel_id = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel_id:
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_CHANNEL_ID must be set before starting AutoML.")

    return WebClient(token=token), channel_id


def _post(text):
    client, channel_id = _client_and_channel()
    client.chat_postMessage(channel=channel_id, text=text)


def validate_slack_config():
    _client_and_channel()


def send_loop_start(max_iterations, smoke_enabled):
    text = (
        "*KSTAR CES AutoML started*\n"
        f"- Max iterations: `{max_iterations}`\n"
        f"- Smoke validation: `{smoke_enabled}`"
    )
    _post(text)


def _fmt(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "n/a" if value != value or value == float("inf") else f"{value:.4f}"
    return "n/a"


def send_iteration_result(iteration, result, status, best_score, stale_rounds):
    eval_report = result.get("eval") or {}
    per_target = eval_report.get("per_target", {})

    lines = [
        f"*AutoML Iteration #{iteration}* — `{status}`",
        f"- Clean skill (mean): `{_fmt(result.get('score'))}`  (best `{_fmt(best_score)}`)",
    ]
    for name in ("CES_TI", "CES_VT"):
        stats = per_target.get(name, {})
        if stats.get("n"):
            lines.append(
                f"- {name}: skill `{_fmt(stats.get('skill_vs_persistence'))}`, "
                f"R² `{_fmt(stats.get('r2_vs_mean'))}`"
            )
    lines.append(
        f"- Train/val loss: `{_fmt(result.get('final_train_loss'))}` / `{_fmt(result.get('final_val_loss'))}`"
    )
    lines.append(f"- Stale rounds: `{stale_rounds}`")
    if result.get("error_stage"):
        lines.append(f"- Error stage: `{result['error_stage']}`")
    if result.get("error"):
        lines.append(f"- Error: `{str(result['error'])[:400]}`")

    _post("\n".join(lines))


def send_loop_complete(history, max_iterations, best_score):
    best_iters = [e["iteration"] for e in history if e.get("status") in ("kept", "baseline")]
    best_text = f"`{_fmt(best_score)}`"
    if best_iters:
        best_text += f" (last improved at iteration `{best_iters[-1]}`)"

    text = (
        "*KSTAR CES AutoML completed*\n"
        f"- Iterations recorded: `{len(history)}/{max_iterations}`\n"
        f"- Best clean skill (mean skill_vs_persistence): {best_text}\n"
        "- Handoff: `HANDOFF.md`"
    )
    _post(text)
