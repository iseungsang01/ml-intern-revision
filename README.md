<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="160" />
</p>

# 🚀 ML Intern (KSTAR CES Prediction Revision)

An ML intern that autonomously researches, writes, and ships good quality ML related code. This specific revision is highly customized for **KSTAR CES (Charge Exchange Spectroscopy) Multimodal Prediction**, predicting low-resolution CES data from high-resolution BES/ECEI/MC sensors using an Autonomous Multi-Agent R&D loop.

## ⚡ Quick Start

### 🌟 What's New: Autonomous ML R&D Workflow for KSTAR
The system has been upgraded from a generic coding assistant to a **Multi-Agent AI Scientist**. It introduces specialized agents (Evaluation, Briefing, Researcher) that autonomously run training loops, track validation loss histories, detect plateaus, and modify model architectures (e.g., from CNNs to Transformers) to continuously improve performance on the KSTAR dataset.

### 🛠️ Installation

```bash
git clone https://github.com/iseungsang01/ml-intern-revision.git
cd ml-intern-revision
uv sync
uv tool install -e .
```

#### That's it. Now `ml-intern` works from any directory:

```bash
ml-intern
```

Create a `.env` file in the project root (or export these in your shell):

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key> # if using anthropic models
OPENAI_API_KEY=<your-openai-api-key> # if using openai models
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token> 
```
If no `HF_TOKEN` is set, the CLI will prompt you to paste one on first launch. To get a GITHUB_TOKEN follow the tutorial [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token).

### Usage

**Interactive mode** (start a chat session):

```bash
ml-intern
```

**Headless mode** (single prompt, auto-approve):

```bash
ml-intern "fine-tune llama on my dataset"
```

**Options:**

```bash
ml-intern --model anthropic/claude-opus-4-6 "your prompt"
ml-intern --model openai/gpt-5.5 "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

## 🏗️ Architecture

### 🧠 The Multi-Agent R&D Paradigm (New)

The original `ml-intern` operated as a single, general-purpose agent loop. While excellent for general coding, ML research requires iterative experimentation, long-term metric tracking, and the ability to escape local minima (plateaus). To address this, we introduced the **Autonomous Multi-Agent R&D Loop**, inspired by the "AI Scientist" concept.

**Why is this better?**
- **Separation of Concerns**: Evaluating code, tracking historical metrics, and hypothesizing new architectures require different cognitive contexts.
- **Plateau Recovery**: The Briefing Agent tracks history across *multiple runs*. If loss plateaus, it forces the Researcher Agent to abandon minor tweaks and explore radically different architectures (e.g., CNN -> Transformer).
- **True Autonomy**: Instead of asking the user what to do after an experiment finishes, the agents brief each other and immediately start the next experiment.

### 🧩 Component Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        User / CLI / Config                              │
└─────────────────────────────────────────────────────────────────────────┘
             │                                          │
             ▼                                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Multi-Agent ML R&D Loop                          │
│                                                                         │
│  ┌──────────────────┐    Metrics / Loss       ┌──────────────────┐      │
│  │ Evaluation Agent │ ──────────────────────> │  Briefing Agent  │      │
│  │ (Executes code,  │                         │ (Tracks history, │      │
│  │ runs train.py,   │ <───────────────┐       │ detects plateaus)│      │
│  │ logs validation) │                 │       └────────┬─────────┘      │
│  └──────────────────┘                 │                │                │
│                                       │                │ Direction /    │
│                                 Proposed Model         │ Status         │
│                                 & Hyperparams          │                │
│                                       │                ▼                │
│                               ┌───────┴──────────────────────────┐      │
│                               │         Researcher Agent         │      │
│                               │ (Analyzes briefing, edits code,  │      │
│                               │  explores new architectures)     │      │
│                               └──────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘

                     (Underlying Execution Engine)
┌─────────────────────────────────────────────────────────────────────────┐
│        Handlers.run_agent() (General Agentic Loop for execution)        │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ Session Context & Tools                                           │  │
│  │ ├─ ContextManager (Message history, auto-compaction)              │  │
│  │ └─ ToolRouter (HF docs, GitHub search, Sandbox, File I/O)         │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🌊 Agentic Execution Flow

```
User Message
     ↓
[Add to ContextManager]
     ↓
     ╔═══════════════════════════════════════════╗
     ║      Iteration Loop (max 300)             ║
     ║                                           ║
     ║  Get messages + tool specs                ║
     ║         ↓                                 ║
     ║  litellm.acompletion()                    ║
     ║         ↓                                 ║
     ║  Has tool_calls? ──No──> Done             ║
     ║         │                                 ║
     ║        Yes                                ║
     ║         ↓                                 ║
     ║  Add assistant msg (with tool_calls)      ║
     ║         ↓                                 ║
     ║  Doom loop check                          ║
     ║         ↓                                 ║
     ║  For each tool_call:                      ║
     ║    • Needs approval? ──Yes──> Wait for    ║
     ║    │                         user confirm ║
     ║    No                                     ║
     ║    ↓                                      ║
     ║    • ToolRouter.execute_tool()            ║
     ║    • Add result to ContextManager         ║
     ║         ↓                                 ║
     ║  Continue loop ─────────────────┐         ║
     ║         ↑                       │         ║
     ║         └───────────────────────┘         ║
     ╚═══════════════════════════════════════════╝
```

## 📡 Events

The agent emits the following events via `event_queue`:

- `processing` - Starting to process user input
- `ready` - Agent is ready for input
- `assistant_chunk` - Streaming token chunk
- `assistant_message` - Complete LLM response text
- `assistant_stream_end` - Token stream finished
- `tool_call` - Tool being called with arguments
- `tool_output` - Tool execution result
- `tool_log` - Informational tool log message
- `tool_state_change` - Tool execution state transition
- `approval_required` - Requesting user approval for sensitive operations
- `turn_complete` - Agent finished processing
- `error` - Error occurred during processing
- `interrupted` - Agent was interrupted
- `compacted` - Context was compacted
- `undo_complete` - Undo operation completed
- `shutdown` - Agent shutting down

## 💻 Development

### 🔧 Adding Built-in Tools

Edit `agent/core/tools.py`:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### 🔌 Adding MCP Servers

Edit `configs/cli_agent_config.json` for CLI defaults, or
`configs/frontend_agent_config.json` for web-session defaults:

```json
{
  "model_name": "anthropic/claude-sonnet-4-5-20250929",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Note: Environment variables like `${YOUR_TOKEN}` are auto-substituted from `.env`.
