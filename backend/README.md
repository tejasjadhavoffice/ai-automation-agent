# Automation agent (Week 1 + Week 2)

## Week 1 (single step)

- Groq → JSON → validate → one tool → result.

## Week 2 (ReAct loop)

- **Loop:** each step = **Thought** (reason) → **Act** (one tool) → **Observe** (tool output).
- **Task decomposition:** model fills optional `subtasks` in JSON each step.
- **Short-term memory:** list of short text lines (`memory_lines`); appended after each observation, fed back in the next user message.
- **Trimming:** memory string is cut to the last `max_memory_chars` (~context budget).
- **Stop conditions:** `done: true` from model, **max steps** (10), or **stuck** (same tool+args repeated 3 times in a row).
- **Trace:** plain `print` lines per step (`THOUGHT`, `ACT`, `OBSERVE`, etc.) — no extra logging module.

### Loop diagram (concept)

```text
User goal
   │
   ▼
┌──────────────────┐
│ Build user msg   │◄──── short-term memory (trimmed)
│ + system ReAct   │
└────────┬─────────┘
         ▼
    LLM JSON step
         │
    ┌────┴────┐
    │ done?   │──yes──► final message
    └────┬────┘
         no
         ▼
    Execute tool
         │
         ▼
    Append observation to memory
         │
         └──── repeat (max / stuck guards)
```

## Setup

1. `backend/venv` activated.
2. `pip install -r requirements.txt`
3. `backend/.env`: `GROQ_API_KEY`, and SMTP fields if you use `send_email`.

## Run

From `backend/`:

**Week 2 (default):** ReAct multi-step

```bash
python -m app.main --request "Read file sample.txt then summarise the content"
```

**Week 1:** single tool call

```bash
python -m app.main --mode once --prompt-style zero-shot --request "Read file sample.txt"
```

Prompt style applies to both modes (`--prompt-style zero-shot|few-shot|cot`).
