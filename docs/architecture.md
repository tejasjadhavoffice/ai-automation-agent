# Agent Architecture — Week 2

This document describes the architecture of the Automation Agent after Week 2.

---

## High-Level Flow

```
User (CLI)
    │
    ▼
AgentCli (main.py)
    │  parse_args()   --request "..." --mode react --prompt-style few-shot
    │
    ▼
AutomationAgent (automation_agent.py)   ← Facade — owns all sub-components
    │
    ├── AppSettings (config/settings.py)       reads .env → API keys, SMTP config
    ├── GroqChatClient (clients/groq_client.py) wraps Groq API + retry logic
    ├── ToolExecutionService (services/)        dispatches tool by name
    └── AgentOrchestrationService (services/)  brain — runs the loop
```

---

## Mode 1 — `once` (Week 1, single step)

```
User request
     │
     ▼
build_user_prompt(style)          ← adds few-shot example or CoT hint
     │
     ▼
GroqChatClient.complete_chat()    ← calls Groq API, retries on failure
     │
     ▼  raw LLM text
_parse_json() + LlmToolDecision   ← validates JSON schema with Pydantic
     │
     ▼  validated decision
ToolExecutionService.execute_tool_by_name()
     │
     ▼
Tool function (read_file / fetch_data / summarise_text / send_email)
     │
     ▼
Result dict  { success, message, data }
     │
     ▼
Print as JSON to console
```

---

## Mode 2 — `react` (Week 2, ReAct loop)

```
User request
     │
     ▼
┌──────────────────────────────────────────────────────┐
│                   ReAct Loop (max 5 steps)           │
│                                                      │
│  1. REASON                                           │
│     build_react_user_message(goal, style, memory)    │
│     → GroqChatClient.complete_chat()                 │
│     → _parse_json() + ReactStep (Pydantic)           │
│     logs: THOUGHT + SUBTASKS                         │
│                                                      │
│  2. CHECK TERMINATION                                │
│     ├── done == True  → return final message         │
│     ├── same tool+args repeated ≥ 2 times → STUCK   │
│     └── step > max_steps → MAX STEPS reached         │
│                                                      │
│  3. ACT                                              │
│     ToolExecutionService.execute_tool_by_name()      │
│     logs: ACT (tool name + arguments)                │
│                                                      │
│  4. OBSERVE                                          │
│     tool_result → appended to memory_lines           │
│     logs: OBSERVE (first 800 chars of result)        │
│                                                      │
│  5. TRIM MEMORY                                      │
│     _trim_memory() keeps last 4500 chars             │
│     → passed back to LLM in next step               │
│                                                      │
│  repeat from step 1                                  │
└──────────────────────────────────────────────────────┘
     │
     ▼
step_log  [{ step, thought, tool_name, arguments, observation }, ...]
     │
     ▼
Print as JSON to console
```

---

## Tool Registry

| Tool Name | What It Does | Arguments |
|---|---|---|
| `read_file` | Reads a local file | `{ "path": "..." }` |
| `fetch_data` | Makes an HTTP GET request | `{ "url": "..." }` |
| `summarise_text` | Calls Groq LLM to summarise text | `{ "text": "..." }` |
| `send_email` | Sends email via SMTP | `{ "to": "...", "subject": "...", "body": "..." }` |
| `no_tool` | No action, agent explains why | — |

---

## Key Design Decisions

| Decision | Why |
|---|---|
| **Pydantic models for LLM output** | The LLM might return malformed JSON. Pydantic catches that before we act on it (guardrail). |
| **Exponential backoff in GroqClient** | Groq API can return 429 (rate limit) or 5xx errors. Retrying with increasing delays avoids hammering the API. |
| **Memory trimming at 4500 chars** | The LLM has a limited context window (8192 tokens ≈ ~32000 chars for llama3-8b). We trim memory so the prompt never gets too long. |
| **Stuck detection by fingerprint** | `sig = tool_name + sorted(arguments)` — if the agent calls the same tool with the same args twice in a row, it's stuck in an infinite loop. We break out. |
| **Facade pattern (`AutomationAgent`)** | `main.py` doesn't need to know about Groq, Pydantic, or tools. It just calls `agent.run()`. One class owns the wiring. |

---

## Termination Conditions (ReAct Loop)

```
done == True         → Agent decided the goal is complete
stuck                → Same tool+args repeated ≥ 2 times  (stuck_repeat_limit = 2)
max_steps reached    → Loop ran more than 5 steps         (max_steps = 5)
```

---

## How to Run

```bash
# Week 1 mode (single step)
python -m app.main --request "Read file python_info.txt" --mode once --prompt-style zero-shot

# Week 2 mode (ReAct loop)
python -m app.main --request "Read file python_info.txt and summarise it" --mode react --prompt-style few-shot

# Debug logging
python -m app.main --request "..." --mode react --log-level DEBUG
```
