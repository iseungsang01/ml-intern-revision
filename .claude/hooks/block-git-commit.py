#!/usr/bin/env python
"""PreToolUse guard: block automatic git commits/pushes.

Enforces the project policy in CLAUDE.md ("Never create git commits or pushes
automatically"). This catches commits made by the assistant or by any OMC skill
(team / autopilot / ralph commit protocol, git-master auto-commit) regardless of
how the command is wrapped (compound `git add && git commit`, `git -C ... push`,
PowerShell `;`-chains, etc.).

How to actually commit when YOU explicitly want to:
  - Run it yourself in your own terminal, or type `! git commit -m "..."` in the
    Claude prompt (the `!` prefix runs in your shell, not through this tool).
  - Or set the env var CLAUDE_ALLOW_GIT_COMMIT=1 (e.g. in .claude/settings.json
    `env`) to disable this guard.

Hooked on the Bash and PowerShell tools via .claude/settings.local.json.
"""
import json
import os
import re
import sys

# Explicit, intentional override.
if os.environ.get("CLAUDE_ALLOW_GIT_COMMIT") == "1":
    sys.exit(0)

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # Can't parse -> don't interfere.

command = (payload.get("tool_input") or {}).get("command") or ""

# Match `git [global flags...] commit|push` anywhere in the command string.
# Global flags allowed between `git` and the subcommand: -C <path>, -c k=v,
# --git-dir=..., -P, --no-pager, etc. A non-flag token (e.g. `log`) stops the
# scan, so `git log --grep=commit` does NOT match.
GIT_WRITE = re.compile(
    r"\bgit\b"
    r"(?:\s+(?:-[A-Za-z]\S*|--[A-Za-z][\w-]*(?:=\S+)?"
    r"|-C\s+\S+|-c\s+\S+))*"
    r"\s+(commit|push)\b",
    re.IGNORECASE,
)

if GIT_WRITE.search(command):
    sub = GIT_WRITE.search(command).group(1).lower()
    reason = (
        f"Blocked by project policy: automatic `git {sub}` is not allowed "
        "(CLAUDE.md: never commit/push automatically; no OMC commit step). "
        "If the user explicitly asked to commit in this same message, tell them "
        "to run it themselves (e.g. `! git "
        f"{sub} ...`) or to set CLAUDE_ALLOW_GIT_COMMIT=1. Do not retry this command."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)

sys.exit(0)
