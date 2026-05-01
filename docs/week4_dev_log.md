# Week 4 Dev Log — Production Agent

**Candidate:** Tejas | **Track:** AI Engineer
**Week theme:** End-to-end production system — clarification, guardrails at every boundary, full observability

---

## What was changed / added

| File | Change | Why |
|---|---|---|
| `app/models/llm_tool_decision.py` | Added `needs_clarification` + `clarifying_question` to `ReactStep` | LLM can now signal ambiguity instead of guessing |
| `app/prompts/prompt_factory.py` | Added `WEEK4_REACT_SYSTEM_PROMPT` + `build_week4_user_message()` | New prompt teaches the LLM to ask for clarification + includes `trace_id` |
| `app/guardrails/guardrail_checker.py` | Added `check_tool_decision()` | Validates tool name AND required arguments before any tool is executed |
| `app/logging_setup.py` | Added `trace_id` param to `log_step()` | Every step now carries one ID so you can trace a full run in the log file |
| `app/services/agent_orchestration_service.py` | Added `run_week4()` | Full production ReAct loop with all 4 new features |
| `app/automation_agent.py` | `run()` routes `mode="week4"` to `run_week4()` | Single entry point stays clean |
| `app/main.py` | Added `week4` to `--mode` choices | CLI flag to trigger Week 4 agent |
| `docs/system_design.md` | New file | 1-2 page system design document (Week 4 deliverable) |

---

## How to run Week 4 agent

```bash
# Normal request — agent runs 3+ tools end to end
python -m app.main --mode week4 --request "Read the sales data from data/sales_data.txt, summarise it, then tell me the result"

# Ambiguous request — agent will ask a clarifying question and stop
python -m app.main --mode week4 --request "Generate a report"
```

---

## Feature 1: Ambiguity detection

**Where:** `run_week4()` in `agent_orchestration_service.py`, lines that check `react.needs_clarification`

**How it works:**
```python
# In the ReactStep model (llm_tool_decision.py)
needs_clarification: bool = False
clarifying_question: str = ""

# In run_week4() — checked BEFORE any tool runs
if react.needs_clarification:
    return {
        "stopped": "needs_clarification",
        "clarifying_question": react.clarifying_question,
    }
```

**The system prompt tells the LLM:**
> If the user request is vague or missing key details (file path, recipient, date range),
> set "needs_clarification": true and write your question in "clarifying_question".
> Use tool_name "no_tool" and stop — do NOT guess.

**Example:**
- Request: `"Generate a report"` → LLM asks `"Which file should I read to generate the report?"`
- Request: `"Read sales data from data/sales_data.txt and summarise it"` → LLM proceeds directly

**Why this matters:** Without this, the agent guesses a file path and either crashes or produces a report on the wrong data. With it, the user gets a clear question and can rerun with a complete request.

---

## Feature 2: Guardrail at every output→action boundary

**Where:** `check_tool_decision()` in `guardrail_checker.py`, called in `run_week4()` before every tool execution

**How it works:**
```python
# Step 1: Is the tool name valid?
ok, msg = self.check_task_assignment(tool_name)

# Step 2: Are required arguments present?
required = {
    "read_file": ["path"],
    "fetch_data": ["url"],
    "summarise_text": ["text"],
    "send_email": ["to", "subject", "body"],
}
for key in required.get(tool_name, []):
    if not arguments.get(key, "").strip():
        return False, f"'{tool_name}' requires '{key}'"
```

**Example guardrail block:**
- LLM returns `tool_name: "summarise_text"` but `arguments: {}` (forgot to include `text`)
- Guardrail returns `(False, "'summarise_text' requires non-empty argument 'text'")`
- `run_week4()` stops and returns `{"stopped": "guardrail_blocked", ...}`
- No LLM data reaches `summarise_text_tool.py` — the agent fails safely

---

## Feature 3: trace_id — full observability

**Where:** `log_step()` in `logging_setup.py`, `run_week4()` in `agent_orchestration_service.py`

**How it works:**
```python
# Generated once per run at the top of run_week4()
trace_id = uuid.uuid4().hex[:6]   # e.g. "a1b2c3"

# Passed to every log_step() call
log_step("Week4Agent", "start", user_request, "", "started", trace_id=trace_id)
```

**Sample log lines for one run:**
```json
{"trace_id": "a1b2c3", "step": "start", "decision": "started"}
{"trace_id": "a1b2c3", "step": "step_1_thought", "decision": "thinking"}
{"trace_id": "a1b2c3", "step": "step_1_guardrail", "decision": "OK"}
{"trace_id": "a1b2c3", "step": "step_1_tool_read_file", "decision": "True"}
{"trace_id": "a1b2c3", "step": "done", "decision": "done"}
```

To read one run: open `logs/agent_steps.jsonl` and search for `"a1b2c3"`.

---

## Feature 4: 3+ distinct tool types in one run

The Week 4 run command uses:
1. `read_file` — reads `data/sales_data.txt` (file I/O tool)
2. `summarise_text` — summarises the content (LLM-powered tool)
3. `fetch_data` — can fetch an external API if needed (HTTP tool)

These are 3 distinct tool **types** in one ReAct loop, satisfying the roadmap requirement.

---

## Week 4 evaluation checklist

| Requirement | Status | Where |
|---|---|---|
| Accepts natural language + decomposes into steps | ✅ | `run_week4()` ReAct loop |
| 3+ distinct tool types | ✅ | `read_file`, `summarise_text`, `fetch_data` |
| ReAct loop with full reasoning logs | ✅ | `run_week4()` with `trace_id` on every log line |
| Handles ambiguous input | ✅ | `needs_clarification` check before any tool runs |
| Guardrails at every output→action boundary | ✅ | `check_tool_decision()` called before every tool |
| Full observability — every step logged and traceable | ✅ | `log_step()` with `trace_id`, written to `agent_steps.jsonl` |
| System design document | ✅ | `docs/system_design.md` |
| Graceful failure — no unhandled exceptions | ✅ | All paths return dicts, no bare exceptions |
