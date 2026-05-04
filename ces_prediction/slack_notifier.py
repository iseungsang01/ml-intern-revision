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


def send_iteration_result(iteration, metrics, plateau_count, plateau_patience, allow_research):
    val_loss = metrics.get("final_val_loss", float("inf"))
    train_loss = metrics.get("final_train_loss")
    error_stage = metrics.get("error_stage")
    error = metrics.get("error")

    val_text = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) and val_loss < float("inf") else "inf"
    train_text = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else "n/a"

    lines = [
        f"*AutoML Iteration #{iteration}*",
        f"- Train loss: `{train_text}`",
        f"- Val loss: `{val_text}`",
        f"- Plateau count: `{plateau_count}/{plateau_patience}`",
        f"- Model rewrite allowed: `{allow_research}`",
    ]
    if error_stage:
        lines.append(f"- Error stage: `{error_stage}`")
    if error:
        lines.append(f"- Error: `{str(error)[:500]}`")

    _post("\n".join(lines))


def send_loop_complete(history, max_iterations):
    valid = [
        entry for entry in history
        if isinstance(entry.get("val_loss"), (int, float)) and entry.get("val_loss") < float("inf")
    ]
    if valid:
        best = min(valid, key=lambda entry: entry["val_loss"])
        best_text = f"`{best['val_loss']:.4f}` at iteration `{best['iteration']}`"
    else:
        best_text = "`n/a`"

    text = (
        "*KSTAR CES AutoML completed*\n"
        f"- Iterations recorded: `{len(history)}/{max_iterations}`\n"
        f"- Best val loss: {best_text}\n"
        "- Handoff: `HANDOFF.md`"
    )
    _post(text)
